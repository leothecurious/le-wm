"""A small coloured square stamped in the top-left corner of every frame.

The square carries no information about the task. In ``episode`` mode its
colour is fixed within an episode and differs across episodes, so it is a
feature that is constant within each episode. In ``frame`` mode the colour
changes every frame, so it cannot be predicted from the past and acts as a
control. Colours are a deterministic hash of ``(seed, episode)`` or
``(seed, episode, step)``, so any run can be reproduced and checked. Training,
measurement and planning must use the same seed for the colours to coincide;
all three default to 0.

Stamping happens on the raw uint8 frame, before normalisation and resizing,
as in the obsessed-encoder post and repository (Enigma-Incorporated/
The-Obsessed-Encoder@37819425, ``pixel_tag.py``).
"""

import numpy as np
import torch

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

        Frame t of a clip that starts at row ``start`` of ``episode`` sits at
        row ``start + t * frameskip`` of that episode.
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

    The dataset's transform moves here, so it runs on the stamped uint8
    frames. Every other attribute is read from the wrapped dataset.
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
    """Image transform for planning that stamps its input before ``inner``.

    ``episode`` mode uses the colour of episode 0 for every rollout. ``frame``
    mode draws a new colour on every call. Frames are (C, H, W) uint8.
    """

    def __init__(self, tag, inner):
        self.tag = tag
        self.inner = inner
        self.calls = 0

    def __call__(self, image):
        if self.tag.mode == "off":
            return self.inner(image)
        image = image.clone()
        key = (0,) if self.tag.mode == "episode" else (0, self.calls)
        self.calls += 1
        n = self.tag.size
        image[:, :n, :n] = self.tag.color(*key).view(3, 1, 1).to(image.dtype)
        return self.inner(image)
