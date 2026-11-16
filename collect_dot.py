"""Record moving-dot episodes as a stable-worldmodel HDF5 dataset.

A dot in the unit square moves by actions of norm at most ``max_step`` and stops
at the edges. Each episode draws a heading uniformly; each action's direction is
Von Mises around it and its norm uniform in [0, max_step]. Frames are a Gaussian
bump of peak 255, grey in all three channels. Rows hold the frame, the dot's
(x, y) as ``state`` and the action taken from it; the last row's action is NaN.
Noise is not recorded: noise.py blends it in when clips are loaded.

Usage: uv run collect_dot.py [episodes=N] [seed=N] [name=dot.h5]
"""

import math
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from torch.distributions import VonMises


def render(pos, size, sigma):
    """Frames (..., size, size) in [0, 1] with a bump of peak 1 at pos (..., 2) = (x, y)."""
    grid = (torch.arange(size, dtype=pos.dtype) + 0.5) / size
    bump = lambda d: torch.exp(-d.square() / (2 * sigma**2))
    return bump(grid - pos[..., 1, None])[..., :, None] * bump(grid - pos[..., 0, None])[..., None, :]


def episodes(cfg, n, generator):
    heading = torch.rand(n, 1, generator=generator) * 2 * math.pi
    theta = VonMises(heading.expand(n, cfg.steps), torch.tensor(cfg.concentration)).sample()
    norm = torch.rand(n, cfg.steps, 1, generator=generator) * cfg.max_step
    actions = norm * torch.stack([theta.cos(), theta.sin()], -1)
    pos = [torch.rand(n, 2, generator=generator)]
    for t in range(cfg.steps):
        pos.append((pos[-1] + actions[:, t]).clamp(0, 1))
    pos = torch.stack(pos, 1)  # (n, steps + 1, 2)
    frames = (render(pos, cfg.size, cfg.sigma) * 255).round().to(torch.uint8)
    actions = torch.cat([actions, torch.full((n, 1, 2), float("nan"))], 1)
    return frames[..., None].expand(-1, -1, -1, -1, 3), pos, actions


@hydra.main(version_base=None, config_path="./config/collect", config_name="dot")
def run(cfg):
    # VonMises samples from the global generator, so seed it along with the explicit one.
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    out = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"), cfg.name)
    with HDF5Writer(out, mode="error") as writer:
        for first in range(0, cfg.episodes, cfg.chunk):
            frames, pos, actions = episodes(cfg, min(cfg.chunk, cfg.episodes - first), generator)
            for i in range(len(frames)):
                writer.write_episode({
                    "pixels": frames[i].numpy(),
                    "state": pos[i].numpy().astype(np.float32),
                    "action": actions[i].numpy().astype(np.float32),
                })
    print(out)


if __name__ == "__main__":
    run()
