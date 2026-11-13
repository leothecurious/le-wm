"""Moving-dot sequences of Sobal et al. (2022, arXiv 2211.10831), App. A.2.

One dot in the unit square is moved by actions of norm at most MAX_STEP, drawn
as a Gaussian bump on a SIZE x SIZE frame. Sequences are a deterministic
function of (seed, split, chunk), so each split is a fixed dataset of the
paper's size that is regenerated on the device instead of stored.
"""

import math

import numpy as np
import torch
from torch.distributions import VonMises

SIZE = 28
STEPS = 16  # actions per sequence; a sequence has STEPS + 1 frames
MAX_STEP = 0.14
SIGMA = 0.05  # blur, in units of the square's side
CONCENTRATION = 1.3  # Von Mises concentration of the action directions
CHUNK = 1000  # sequences generated together
SPLIT_SIZES = {"train": 1_000_000, "probe": 300_000, "val": 10_000, "eval": 10_000}
NOISES = ("none", "fixed", "changing")


def render(pos):
    """Frames (..., SIZE, SIZE) with a Gaussian bump of peak 1 at pos (..., 2) = (x, y)."""
    g = (torch.arange(SIZE, device=pos.device, dtype=pos.dtype) + 0.5) / SIZE
    dx = g - pos[..., 0, None]  # (..., SIZE) over columns
    dy = g - pos[..., 1, None]  # (..., SIZE) over rows
    bump = lambda d: torch.exp(-d.square() / (2 * SIGMA**2))
    return bump(dy)[..., :, None] * bump(dx)[..., None, :]


def trajectory(start, actions):
    """Positions (B, STEPS + 1, 2) from start (B, 2); the dot stops at the square's edges."""
    pos = [start]
    for t in range(actions.shape[1]):
        pos.append((pos[-1] + actions[:, t]).clamp(0, 1))
    return torch.stack(pos, dim=1)


def make_chunk(split, index, noise="none", alpha=0.0, seed=0, n=CHUNK, device="cpu"):
    """Sequences ``index`` of ``split``.

    Returns frames (n, STEPS + 1, SIZE, SIZE) with noise o + alpha * Z, Z uniform
    in [0, 1] per pixel, drawn once per sequence (``fixed``) or per frame
    (``changing``); the noise-free frames; positions (n, STEPS + 1, 2); and
    actions (n, STEPS, 2).
    """
    assert noise in NOISES
    device = torch.device(device)
    key = np.random.SeedSequence([seed, list(SPLIT_SIZES).index(split), index])
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(int(key.generate_state(1, dtype=np.uint64)[0] >> 1))
        omega = torch.rand(n, 1, device=device) * 2 * math.pi
        theta = VonMises(omega.expand(n, STEPS), CONCENTRATION).sample()
        step = torch.rand(n, STEPS, 1, device=device) * MAX_STEP
        actions = step * torch.stack([theta.cos(), theta.sin()], dim=-1)
        pos = trajectory(torch.rand(n, 2, device=device), actions)
        clean = render(pos)
        frames = clean
        if noise != "none":
            z = torch.rand(n, 1 if noise == "fixed" else STEPS + 1, SIZE, SIZE, device=device)
            frames = clean + alpha * z
    return {"frames": frames, "clean": clean, "pos": pos, "actions": actions}


def batches(split, noise, alpha, seed=0, batch_size=500, size=None, epoch=0, shuffle=False, device="cpu"):
    """Batches over ``size`` sequences of ``split`` (default: the paper's size).

    ``batch_size`` divides CHUNK. With ``shuffle`` the chunks and the sequences within
    them are visited in an order drawn from (seed, epoch).
    """
    assert CHUNK % batch_size == 0
    size = size or SPLIT_SIZES[split]
    n = min(CHUNK, size)
    assert size % n == 0
    order = np.arange(math.ceil(size / n))
    rng = np.random.default_rng([seed, epoch])
    if shuffle:
        rng.shuffle(order)
    for index in order:
        chunk = make_chunk(split, int(index), noise, alpha, seed, n, device)
        perm = torch.from_numpy(rng.permutation(n)).to(device) if shuffle else torch.arange(n, device=device)
        for i in range(0, n, min(batch_size, n)):
            yield {k: v[perm[i : i + batch_size]] for k, v in chunk.items()}
