"""Guard against episode leakage and evaluation refitting train action statistics."""

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from metrics import data
from noise import PixelNoise
from utils import get_column_normalizer, get_column_stats, split_episodes


def write_dataset(path, lengths, action_shift=0):
    offsets = np.cumsum([0, *lengths[:-1]])
    actions = np.concatenate([
        np.column_stack((np.arange(n) + 10 * ep, 2 * np.arange(n) - 5 * ep))
        for ep, n in enumerate(lengths)
    ]).astype(np.float32) + action_shift
    actions[offsets + lengths - 1] = np.nan
    with h5py.File(path, "w", libver="latest") as f:
        f["ep_len"] = lengths
        f["ep_offset"] = offsets
        f["pixels"] = np.zeros((sum(lengths), 16, 16, 3), dtype=np.uint8)
        f["action"] = actions
        f["position"] = np.arange(sum(lengths), dtype=np.float32)[:, None]
    return actions, offsets


def eval_config(path, stats):
    return OmegaConf.create({
        "name": str(path), "frameskip": 1, "img_size": 16,
        "noise": {"mode": "off", "alpha": 1.0, "seed": 11},
        "tag": {"mode": "off", "size": 5, "seed": 0},
        "action_normalizer": stats,
        "sampler": "episodes", "start": "episode_start", "n": 3,
        "seed": 7, "batch_size": 5, "num_workers": 0, "train_fraction": 0.5,
    })


def test_whole_episode_split_fits_train_rows_and_reuses_saved_stats(tmp_path):
    lengths = np.array([1, 4, 5, 6, 7])
    path = tmp_path / "train.h5"
    actions, offsets = write_dataset(path, lengths)
    ds = swm.data.HDF5Dataset(path=path, num_steps=2, keys_to_load=["pixels", "action"])
    train, val = split_episodes(ds, 0.5, torch.Generator().manual_seed(3))
    repeated, _ = split_episodes(ds, 0.5, torch.Generator().manual_seed(3))
    assert train.indices == repeated.indices
    train_eps = {ds.clip_indices[i][0] for i in train.indices}
    val_eps = {ds.clip_indices[i][0] for i in val.indices}
    assert train_eps.isdisjoint(val_eps)
    assert train_eps | val_eps == {1, 2, 3, 4}
    assert sorted(train.indices + val.indices) == list(range(len(ds)))

    expected_rows = np.concatenate([
        actions[offsets[ep]:offsets[ep] + lengths[ep] - 1] for ep in sorted(train_eps)
    ])
    mean, std = expected_rows.mean(0), expected_rows.std(0, ddof=1)
    stats = get_column_stats(ds, "action", train_eps)
    np.testing.assert_allclose(stats["mean"].numpy()[0], mean)
    np.testing.assert_allclose(stats["std"].numpy()[0], std)
    assert not np.allclose(mean, np.nanmean(actions, axis=0))
    ds.transform = get_column_normalizer(ds, "action", "action", stats)
    PixelNoise("episode", seed=8).wrap(ds)
    ep, start = ds.clip_indices[train.indices[0]]
    expected = (actions[offsets[ep] + start:offsets[ep] + start + 2] - mean) / std
    np.testing.assert_allclose(train[0]["action"].numpy(), expected, rtol=1e-6, equal_nan=True)

    saved = {key: value.flatten().tolist() for key, value in stats.items()}
    config_path = tmp_path / "config.yaml"
    OmegaConf.save({"data": {"action_normalizer": saved}}, config_path)
    shifted_path = tmp_path / "eval.h5"
    shifted, _ = write_dataset(shifted_path, np.array([4, 4]), action_shift=100)
    cfg = eval_config(shifted_path, OmegaConf.load(config_path).data.action_normalizer)
    eval_ds = data.load(cfg, 2, cache_dir=tmp_path)
    np.testing.assert_allclose(
        eval_ds[0]["action"].numpy(), (shifted[:2] - mean) / std, rtol=1e-6
    )
