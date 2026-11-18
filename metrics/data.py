"""Evaluation windows: sampling, the training transform and probe targets.

A window is ``(episode, start, n_frames)``: ``n_frames`` observations taken
every ``frameskip`` rows from ``start``, with the actions between them, as in
training. Windows shorter than the requested length are padded and masked.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

import dot
from noise import PixelNoise
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
        get_column_normalizer(ds, "action", "action", stats=cfg.get("action_normalizer")),
    )
    for c in cfg.get("columns") or []:
        dim = ds.get_dim(c.column)
        assert dim == c.dim, f"{c.column} has {dim} dims, expected {c.dim}"
    return PixelNoise(**cfg.noise).wrap(PixelTag(**cfg.tag).wrap(ds))


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


def _sample(cfg, columns, num_frames, min_frames, cache_dir):
    ds = load(cfg, num_frames, columns, cache_dir)
    windows = sample_windows(ds, num_frames, min_frames, cfg.sampler, cfg.start, cfg.n, cfg.seed)
    return ds, windows


class GeneratedWindows:
    """Batches of whole generated sequences, preprocessed like training frames on the device they were
    generated on. Actions and the valid mask stay on the CPU, where ``extract`` collects them."""

    def __init__(self, frames, action, batch_size, img_size):
        self.frames, self.action, self.batch_size = frames, action, batch_size
        self.preprocess = get_img_preprocessor("pixels", "pixels", img_size)

    def __len__(self):
        return -(-len(self.frames) // self.batch_size)

    def __iter__(self):
        for i in range(0, len(self.frames), self.batch_size):
            frame, action = self.frames[i:i + self.batch_size], self.action[i:i + self.batch_size]
            pixels = frame[:, :, None].expand(-1, -1, 3, -1, -1).contiguous()
            yield {
                "pixels": self.preprocess({"pixels": pixels})["pixels"],
                "action": torch.nan_to_num(action, 0.0).cpu(),
                "valid": torch.ones(frame.shape[:2], dtype=torch.bool),
            }


def build_generated(cfg, specs):
    """Loader, targets, episode ids and probe-train mask of generated sequences: ``probe.n`` from the
    ``probe`` stream followed by ``test.n`` from the ``test`` stream, whose windows are whole sequences.
    Stages are drawn from the run seed ``cfg.seed`` and share nothing with training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (torch.tensor(cfg.action_normalizer[k], device=device).reshape(1, 1, -1) for k in ("mean", "std"))
    counts = {"probe": cfg.probe.n, "test": cfg.test.n}
    frames, actions, states = [], [], []
    for stage, n in counts.items():
        f, state, action = dot.sequences(cfg.generator, n, dot.Streams(cfg.seed, stage, device), cfg.noise.mode, cfg.noise.alpha)
        frames.append(f), states.append(state), actions.append((action - mean) / std)
    assert {s.column for s in specs} == {"state"}, "generated sequences have one target column, state"
    targets = torch.cat(states)[..., [s.get("index", 0) for s in specs]].double().cpu().numpy()
    loader = GeneratedWindows(torch.cat(frames), torch.cat(actions), cfg.batch_size, cfg.img_size)
    train_mask = np.arange(sum(counts.values())) < cfg.probe.n
    return loader, targets, np.arange(len(train_mask)), train_mask


def build(cfg, specs, num_frames, min_frames, cache_dir=None):
    """Loader over sampled windows; separate probe/test files when configured."""
    if cfg.get("generator") is not None:
        return build_generated(cfg, specs)
    columns = sorted({s.column for s in specs})
    if cfg.get("probe") is not None:
        probe_cfg = deepcopy(cfg)
        probe_cfg.name = cfg.probe.name
        probe_cfg.n = cfg.probe.n
        probe_cfg.noise.seed = cfg.probe.noise_seed
        if cfg.noise.mode != "off" and probe_cfg.noise.seed == cfg.noise.seed:
            raise ValueError("probe and test noise seeds must differ")
        probe_ds, probe_windows = _sample(probe_cfg, columns, num_frames, min_frames, cache_dir)
        ds, windows = _sample(cfg, columns, num_frames, min_frames, cache_dir)
        if Path(probe_ds.h5_path).resolve() == Path(ds.h5_path).resolve():
            raise ValueError("probe and test must use different dataset paths")
        samples = torch.utils.data.ConcatDataset([
            Windows(probe_ds, probe_windows, num_frames), Windows(ds, windows, num_frames)
        ])
        targets = np.concatenate([
            window_targets(probe_ds, probe_windows, num_frames, specs),
            window_targets(ds, windows, num_frames, specs),
        ])
        episodes = np.array(
            [w[0] for w in probe_windows] + [w[0] + len(probe_ds.lengths) for w in windows]
        )
        train_mask = np.concatenate([
            np.ones(len(probe_windows), dtype=bool), np.zeros(len(windows), dtype=bool)
        ])
    else:
        ds, windows = _sample(cfg, columns, num_frames, min_frames, cache_dir)
        samples = Windows(ds, windows, num_frames)
        targets = window_targets(ds, windows, num_frames, specs)
        episodes = np.array([w[0] for w in windows])
        train_mask = split_episodes(windows, cfg.train_fraction, cfg.seed)
    loader = torch.utils.data.DataLoader(
        samples,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, targets, episodes, train_mask
