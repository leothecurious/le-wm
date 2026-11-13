"""A small coloured square stamped in the top-left corner of every frame.

In ``episode`` mode the colour is constant within an episode and differs
across episodes. In ``frame`` mode it changes every frame, so it cannot be
predicted from the past. Colours are a deterministic hash of
``(seed, episode)`` or ``(seed, episode, step)``.

Stamping happens on the raw uint8 frame, before normalisation and resizing, as
in Enigma-Incorporated/The-Obsessed-Encoder@37819425 (``pixel_tag.py``).
Training, measurement and planning must share one seed. Training takes the run
seed; measurement and planning leave it null and read it from the checkpoint's
saved config.
"""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf


def resolve_seed(tag_cfg, checkpoint_dir):
    """Fill a null ``tag_cfg.seed`` from the ``config.yaml`` in ``checkpoint_dir``.

    Seed 0 when there is no saved tag to read: no checkpoint (the random
    policy), no ``config.yaml`` (a downloaded model) or a checkpoint trained
    before the tag existed. The seed is read even while the tag is off, as the
    pair metric stamps the training colours regardless of mode.
    """
    if tag_cfg.get("seed") is not None:
        return
    config = None if checkpoint_dir is None else Path(checkpoint_dir) / "config.yaml"
    saved = OmegaConf.load(config).get("tag") if config and config.exists() else None
    tag_cfg.seed = saved.seed if saved else 0


class PixelTag:
    """Colour hash and corner stamp for one ``mode``, square ``size`` and ``seed``."""

    def __init__(self, mode="off", size=5, seed=0):
        self.mode = mode
        self.size = size
        self.seed = seed

    def color(self, *key):
        """RGB uint8 colour for ``key``, a deterministic hash with the seed."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, *key]))
        return torch.from_numpy(rng.integers(0, 256, size=3, dtype=np.uint8))

    def stamp(self, pixels, episode, start, frameskip):
        """Stamp ``pixels`` (T, C, H, W) uint8 in place and return it.

        Frame t of a clip starting at row ``start`` of ``episode`` is row
        ``start + t * frameskip`` of that episode.
        """
        n = self.size
        if self.mode == "episode":
            pixels[:, :, :n, :n] = self.color(int(episode)).view(3, 1, 1)
        else:
            for t in range(pixels.shape[0]):
                step = int(start) + t * frameskip
                pixels[t, :, :n, :n] = self.color(int(episode), step).view(3, 1, 1)
        return pixels

    def wrap(self, dataset):
        """Make ``dataset`` stamp every clip it returns, in place; off leaves it alone."""
        if self.mode == "off":
            return dataset
        if type(dataset) is not swm.data.HDF5Dataset:
            raise TypeError(f"tag supports HDF5 datasets, got {type(dataset).__name__}")
        dataset.__class__ = TaggedHDF5Dataset
        dataset.tag = self
        return dataset


class TaggedHDF5Dataset(swm.data.HDF5Dataset):
    """HDF5 dataset that stamps the tag on the raw pixels of each clip."""

    tag: PixelTag

    def _load_slice(self, episode, start, end):
        # The base class applies the transform inside ``_load_slice``; hold it
        # back until the frames are stamped.
        transform, self.transform = self.transform, None
        try:
            steps = super()._load_slice(episode, start, end)
        finally:
            self.transform = transform
        self.tag.stamp(steps["pixels"], episode, start, self.frameskip)
        return transform(steps) if transform else steps


class EvalTag:
    """Image transform for planning that stamps its input, then applies ``inner``.

    ``episode`` mode uses the colour of episode 0 for every rollout; ``frame``
    mode draws a new colour on every call. Images are (3, H, W) uint8.
    """

    def __init__(self, tag, inner):
        self.tag = tag
        self.inner = inner
        self.calls = 0

    def __call__(self, image):
        if self.tag.mode == "off":
            return self.inner(image)
        assert image.dtype == torch.uint8 and image.shape[0] == 3, (
            f"tag expects (3, H, W) uint8 frames, got {image.dtype} {tuple(image.shape)}"
        )
        image = image.clone()
        self.tag.stamp(image[None], 0, self.calls, 1)
        self.calls += 1
        return self.inner(image)
