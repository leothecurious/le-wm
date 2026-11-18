"""Moving-dot sequences generated on the fly, after Sobal et al. (2022, arXiv 2211.10831), App. A.2.

A dot in the unit square moves by actions of norm at most ``max_step`` and stops ``margin`` short of the
edges. Each sequence draws a heading uniformly; each action's direction is Von Mises around it and its norm
uniform in [0, max_step]. Frames are a Gaussian bump of peak 255, grey in all three channels. Lengths are in
units of the square's side. Geometry follows the reference code (vladisai/JEPA_SSL_NeurIPS_2022,
data/single.py) where it differs from the paper's text. Noise is blended in by noise.blend: one pattern per
sequence in ``episode`` mode, a fresh one per frame in ``frame`` mode.

Every stage (train, val, probe, test, stats) draws from its own pair of generators, one for geometry and one
for noise, seeded from the run seed and the stage name. They never touch the global RNG, so a run's data does
not depend on the model's randomness, objectives at one seed see identical sequences in the same order
whatever the noise mode, and no two stages share a stream. Streams are reproducible on the same device type;
CPU and CUDA generators produce different numbers for one seed.
"""

import math

import numpy as np
import torch
from torch.distributions import VonMises

from noise import blend

STAGES = {"train": 0, "val": 1, "probe": 2, "test": 3, "stats": 4}


def stream_seed(seed, stage, stream):
    """A 63-bit seed for ``stream`` (0 geometry, 1 noise) of ``stage`` under run seed ``seed``."""
    words = np.random.SeedSequence([int(seed), STAGES[stage], stream]).generate_state(1, dtype=np.uint64)
    return int(words[0]) >> 1


class Streams:
    """The geometry and noise generators of one stage."""

    def __init__(self, seed, stage, device):
        self.geometry, self.noise = (
            torch.Generator(device).manual_seed(stream_seed(seed, stage, k)) for k in (0, 1)
        )

    def state_dict(self):
        return {"geometry": self.geometry.get_state(), "noise": self.noise.get_state()}

    def load_state_dict(self, state):
        self.geometry.set_state(state["geometry"])
        self.noise.set_state(state["noise"])


def von_mises(loc, concentration, generator):
    """Von Mises samples around ``loc``. torch's sampler reads the global RNG, so it runs on a fork of
    that RNG seeded from ``generator``, which leaves the model's stream untouched."""
    seed = int(torch.randint(0, 2**62, (), generator=generator, device=generator.device))
    with torch.random.fork_rng(devices=None if loc.is_cuda else []):
        torch.manual_seed(seed)
        return VonMises(loc, torch.tensor(concentration, device=loc.device)).sample()


def render(pos, size, sigma):
    """Frames (..., size, size) in [0, 1] with a bump of peak 1 at pos (..., 2) = (x, y)."""
    grid = (torch.arange(size, dtype=pos.dtype, device=pos.device) + 0.5) / size
    bump = lambda d: torch.exp(-d.square() / (2 * sigma**2))
    return bump(grid - pos[..., 1, None])[..., :, None] * bump(grid - pos[..., 0, None])[..., None, :]


def trajectories(cfg, n, generator):
    """Dot positions (n, steps + 1, 2) and the actions (n, steps, 2) that move it."""
    device = generator.device
    rand = lambda *shape: torch.rand(*shape, generator=generator, device=device)
    heading = rand(n, 1) * 2 * math.pi
    theta = von_mises(heading.expand(n, cfg.steps), cfg.concentration, generator)
    norm = rand(n, cfg.steps, 1) * cfg.max_step
    actions = norm * torch.stack([theta.cos(), theta.sin()], -1)
    lo, hi = cfg.margin, 1 - cfg.margin
    pos = [lo + (hi - lo) * rand(n, 2)]
    for t in range(cfg.steps):
        pos.append((pos[-1] + actions[:, t]).clamp(lo, hi))
    return torch.stack(pos, 1), actions


def render_frames(cfg, pos, noise, alpha, generator):
    """uint8 frames (n, T, size, size) of positions (n, T, 2), with ``noise`` ('off', 'episode' or
    'frame') of strength ``alpha`` blended in."""
    out = (render(pos, cfg.size, cfg.sigma) * 255).round().to(torch.uint8)
    if noise == "off":
        return out
    n, t = pos.shape[:2]
    z = torch.rand(n, 1 if noise == "episode" else t, cfg.size, cfg.size, generator=generator, device=pos.device)
    return blend(out, z, alpha)


def trajectories_with_final_action(cfg, n, generator):
    """Positions and actions (n, steps + 1, 2) of fresh sequences; no action leaves the final frame, so its row is NaN."""
    pos, actions = trajectories(cfg, n, generator)
    return pos, torch.cat([actions, torch.full_like(actions[:, :1], float("nan"))], 1)


def sequences(cfg, n, streams, noise="off", alpha=0.0, chunk=1000):
    """Whole sequences: frames (n, steps + 1, size, size) uint8, state and action (n, steps + 1, 2).
    Frames are rendered ``chunk`` sequences at a time to bound the noise tensor; the chunk size is part of
    what a seed reproduces."""
    pos, actions = trajectories_with_final_action(cfg, n, streams.geometry)
    out = torch.cat([render_frames(cfg, p, noise, alpha, streams.noise) for p in pos.split(chunk)])
    return out, pos, actions


def clips(cfg, n, length, streams, noise="off", alpha=0.0):
    """One clip of ``length`` consecutive frames from each of n fresh sequences, starting uniformly over
    the sequence: frames (n, length, size, size) uint8 and the actions taken from those frames."""
    pos, actions = trajectories_with_final_action(cfg, n, streams.geometry)
    start = torch.randint(0, cfg.steps + 2 - length, (n, 1), generator=streams.geometry, device=pos.device)
    rows = start + torch.arange(length, device=pos.device)
    pos, actions = (torch.gather(x, 1, rows[..., None].expand(-1, -1, 2)) for x in (pos, actions))
    return render_frames(cfg, pos, noise, alpha, streams.noise), actions


def action_stats(cfg, seed, device, n):
    """Mean and std (1, 2) of the actions of ``n`` sequences from the ``stats`` stage."""
    _, actions = trajectories(cfg, n, Streams(seed, "stats", device).geometry)
    actions = actions.flatten(0, 1)
    return {"mean": actions.mean(0, keepdim=True).cpu(), "std": actions.std(0, keepdim=True).cpu()}


def to_batch(frame, action, stats):
    """A training batch: grey frames repeated to RGB, as uint8 (..., T, 3, H, W), and normalized actions."""
    mean, std = (stats[k].to(action.device) for k in ("mean", "std"))
    pixels = frame[..., None, :, :].expand(*frame.shape[:-2], 3, *frame.shape[-2:]).contiguous()
    return {"pixels": pixels, "action": (action - mean) / std}


class ClipStream(torch.utils.data.IterableDataset):
    """An endless stream of training clips cut into epochs of ``steps`` batches. Each batch is drawn fresh,
    so no clip or noise pattern repeats. The generators carry on across epochs and are part of the
    checkpoint state. Clips are generated ``chunk`` batches at a time to keep kernel launches off the
    training loop's path; the epoch ends on a chunk boundary, so no generated clip is held over."""

    def __init__(self, cfg, streams, stats, noise, alpha, length, batch_size, steps, chunk):
        self.cfg, self.streams, self.stats = cfg, streams, stats
        self.noise, self.alpha, self.length = noise, alpha, length
        self.batch_size, self.steps, self.chunk = batch_size, steps, chunk

    def __len__(self):
        return self.steps

    def state_dict(self):
        return self.streams.state_dict()

    def load_state_dict(self, state):
        self.streams.load_state_dict(state)

    def __iter__(self):
        for first in range(0, self.steps, self.chunk):
            k = min(self.chunk, self.steps - first)
            frame, action = clips(self.cfg, k * self.batch_size, self.length, self.streams, self.noise, self.alpha)
            batch = to_batch(frame, action, self.stats)
            for i in range(k):
                s = slice(i * self.batch_size, (i + 1) * self.batch_size)
                yield {key: v[s] for key, v in batch.items()}


class HeldOutClips(torch.utils.data.IterableDataset):
    """Every clip of ``n`` fixed sequences, generated once from the ``val`` stage. A batch holds clips of
    distinct sequences, as training batches do: clip c starts at c // n of sequence c % n."""

    def __init__(self, cfg, seed, device, stats, noise, alpha, length, batch_size, n):
        self.n, self.length, self.batch_size, self.stats = n, length, batch_size, stats
        self.frames, _, self.actions = sequences(cfg, n, Streams(seed, "val", device), noise, alpha)
        self.clips = n * (cfg.steps + 2 - length)

    def __len__(self):
        return -(-self.clips // self.batch_size)

    def __iter__(self):
        for first in range(0, self.clips, self.batch_size):
            c = torch.arange(first, min(first + self.batch_size, self.clips), device=self.frames.device)
            seq = c % self.n
            rows = c[:, None] // self.n + torch.arange(self.length, device=c.device)
            yield to_batch(self.frames[seq[:, None], rows], self.actions[seq[:, None], rows], self.stats)
