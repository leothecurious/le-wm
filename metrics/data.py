"""Evaluation windows: sampling, the training transform and probe targets.

A window is ``(episode, start, n_frames)``: ``n_frames`` observations taken
every ``frameskip`` rows from ``start``, with the actions between them, as in
training. Windows shorter than the requested length are padded and masked.
"""

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from tag import PixelTag
from utils import get_column_normalizer, get_img_preprocessor


def load(cfg, num_frames, target_columns=(), cache_dir=None):
    """Dataset with the training image and action transform.

    Target columns are cached in memory and never loaded per window, so the
    parent process opens no file handle that forked workers would share.
    """
    ds = swm.data.load_dataset(
        cfg.name,
        transform=None,
        cache_dir=cache_dir,
        frameskip=cfg.frameskip,
        num_steps=num_frames,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action", *target_columns],
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", cfg.img_size),
        get_column_normalizer(ds, "action", "action"),
    )
    for c in cfg.get("columns") or []:
        dim = ds.get_dim(c.column)
        assert dim == c.dim, f"{c.column} has {dim} dims, expected {c.dim}"
    return PixelTag(**cfg.tag).wrap(ds)


def sample_windows(ds, num_frames, min_frames, sampler, start, n, seed):
    """Pick windows of at most ``num_frames`` observations.

    ``episodes`` takes one window per episode, from its first row
    (``episode_start``) or a uniform start that leaves ``num_frames`` frames
    where the episode is long enough (``any_start``). ``random_clips`` draws
    full-length clips uniformly over the dataset's clip index.
    """
    rng = np.random.default_rng(seed)
    fs = ds.frameskip

    if sampler == "random_clips":
        assert ds.num_steps == num_frames
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
        return [(int(ds.clip_indices[i][0]), int(ds.clip_indices[i][1]), num_frames) for i in idx]

    assert sampler == "episodes", sampler
    eligible = np.nonzero(np.asarray(ds.lengths) // fs >= min_frames)[0]
    episodes = rng.choice(eligible, min(n, len(eligible)), replace=False)
    windows = []
    for ep in episodes:
        length = int(ds.lengths[ep])
        s = 0
        if start == "any_start":
            s = int(rng.integers(0, max(0, length - num_frames * fs) + 1))
        else:
            assert start == "episode_start", start
        windows.append((int(ep), s, min((length - s) // fs, num_frames)))
    return windows


def split_episodes(windows, train_fraction, seed):
    """Boolean train mask per window; every episode lands on one side."""
    episodes = np.array([w[0] for w in windows])
    unique = np.random.default_rng(seed).permutation(np.unique(episodes))
    train_eps = unique[: round(train_fraction * len(unique))]
    return np.isin(episodes, train_eps)


class Windows(torch.utils.data.Dataset):
    """Transformed pixels and actions of each window, padded to ``num_frames``.

    Padding repeats the last valid frame and uses zero actions; ``valid``
    marks the real frames.
    """

    def __init__(self, ds, windows, num_frames):
        self.ds = ds
        self.windows = windows
        self.num_frames = num_frames

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ep, s, n = self.windows[i]
        item = self.ds.load_chunk([ep], [s], [s + n * self.ds.frameskip])[0]
        pixels, action = item["pixels"], torch.as_tensor(item["action"]).float()
        pad = self.num_frames - n
        if pad:
            pixels = torch.cat([pixels, pixels[-1:].expand(pad, *pixels.shape[1:])])
            action = torch.cat([action, action.new_zeros(pad, action.size(1))])
        return {
            "pixels": pixels,
            "action": torch.nan_to_num(action, 0.0),
            "valid": torch.arange(self.num_frames) < n,
        }


def column_targets(ds, specs):
    """Per-row target values for the whole dataset, shape (rows, len(specs)).

    Each spec reads ``column[:, index]``.
    """
    cols, out = {}, []
    for spec in specs:
        if spec.column not in cols:
            cols[spec.column] = np.asarray(ds.get_col_data(spec.column), dtype=np.float64)
        col = cols[spec.column]
        x = col[:, spec.get("index", 0)] if col.ndim > 1 else col
        assert np.ptp(x) > 0, f"target {spec.name} is constant"
        out.append(x)
    return np.stack(out, 1)


def window_targets(ds, windows, num_frames, specs):
    """Targets at each window frame, shape (N, num_frames, len(specs)).

    Padded frames repeat the last valid frame's target.
    """
    per_row = column_targets(ds, specs)
    j = np.arange(num_frames)
    rows = np.stack([
        ds.offsets[ep] + s + np.minimum(j, n - 1) * ds.frameskip
        for ep, s, n in windows
    ])
    return per_row[rows]


def target_meta(specs):
    """Names, angle flags and groups of the targets, in spec order."""
    names = [s.name for s in specs]
    angle = [bool(s.get("angle", False)) for s in specs]
    groups = [s.get("group", "state") for s in specs]
    return names, angle, groups


def build(cfg, specs, num_frames, min_frames, cache_dir=None):
    """Loader over the sampled windows, plus targets and split for each one."""
    ds = load(cfg, num_frames, sorted({s.column for s in specs}), cache_dir)
    windows = sample_windows(
        ds, num_frames, min_frames, cfg.sampler, cfg.start, cfg.n, cfg.seed
    )
    loader = torch.utils.data.DataLoader(
        Windows(ds, windows, num_frames),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        loader,
        window_targets(ds, windows, num_frames, specs),
        np.array([w[0] for w in windows]),
        split_episodes(windows, cfg.train_fraction, cfg.seed),
    )
