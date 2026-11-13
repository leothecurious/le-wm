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
import torch
from omegaconf import OmegaConf


def resolve_seed(tag_cfg, checkpoint_dir):
    """Fill a null ``tag_cfg.seed`` from the ``config.yaml`` in ``checkpoint_dir``."""
    if tag_cfg.get("seed") is not None:
        return
    if tag_cfg.mode == "off":
        tag_cfg.seed = 0
        return
    tag_cfg.seed = OmegaConf.load(Path(checkpoint_dir) / "config.yaml").tag.seed


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
        """``dataset`` with every clip stamped, or ``dataset`` itself when off."""
        return dataset if self.mode == "off" else TaggedDataset(dataset, self)


class TaggedDataset:
    """Stamps the pixels of each clip a swm episode dataset returns.

    The dataset's transform moves here so it runs on the stamped frames. Other
    attributes are read from the wrapped dataset.
    """

    def __init__(self, dataset, tag):
        self.dataset = dataset
        self.tag = tag
        self.transform = dataset.transform
        dataset.transform = None

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def _clip(self, episode, start, end):
        ds = self.dataset
        steps = ds._load_slice(episode, start, end)
        self.tag.stamp(steps["pixels"], episode, start, ds.frameskip)
        # The transform sees one action per row, so it runs before the reshape
        # to one action per frame, as in stable_worldmodel's HDF5Dataset.
        if self.transform:
            steps = self.transform(steps)
        if "action" in steps:
            steps["action"] = steps["action"].reshape((end - start) // ds.frameskip, -1)
        return steps

    def __getitem__(self, idx):
        episode, start = self.dataset.clip_indices[idx]
        return self._clip(episode, start, start + self.dataset.span)

    def load_chunk(self, episodes, starts, ends):
        return [self._clip(e, s, t) for e, s, t in zip(episodes, starts, ends)]


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
