"""Uniform pixel noise over every frame, after Sobal et al. (2022, arXiv 2211.10831), App. A.2.

A frame o becomes (o + alpha * 255 * Z) / (1 + alpha), with Z uniform in [0, 1]
per pixel and shared across channels. Dividing by 1 + alpha keeps the frame in
uint8 range with the paper's ratio of signal to noise. In ``episode`` mode Z is
drawn once per episode, so it is perfectly predictable within a clip; in
``frame`` mode it is redrawn every frame. Z is a deterministic hash of
``(seed, episode)`` or ``(seed, episode, step)``, so training and measurement
see the same patterns when they share a seed.
"""

import numpy as np
import torch

from tag import stamp_clips


def blend(pixels, z, alpha):
    """(pixels + alpha * 255 * z) / (1 + alpha), rounded to uint8, for uint8 ``pixels`` and noise ``z`` in [0, 1]."""
    return ((pixels.float() + alpha * 255 * z) / (1 + alpha)).round().to(torch.uint8)


class PixelNoise:
    def __init__(self, mode="off", alpha=0.0, seed=0):
        self.mode = mode
        self.alpha = alpha
        self.seed = seed

    def pattern(self, height, width, *key):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, *key]))
        return torch.from_numpy(rng.random((1, height, width), dtype=np.float32))

    def stamp(self, pixels, episode, start, frameskip):
        """Blend noise into ``pixels`` (T, C, H, W) uint8 in place and return it.

        Frame t of a clip starting at row ``start`` of ``episode`` is row
        ``start + t * frameskip`` of that episode.
        """
        t, _, h, w = pixels.shape
        if self.mode == "episode":
            z = self.pattern(h, w, int(episode))[None]
        else:
            z = torch.stack([self.pattern(h, w, int(episode), int(start) + i * frameskip) for i in range(t)])
        pixels.copy_(blend(pixels, z, self.alpha))
        return pixels

    def wrap(self, dataset):
        """Make ``dataset`` blend noise into every clip it returns; off leaves it alone."""
        return dataset if self.mode == "off" else stamp_clips(dataset, self)
