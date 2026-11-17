"""Guard against episode leakage and evaluation refitting train action statistics."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import collect_dot
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


def test_probe_and_test_use_separate_files_masks_and_noise(tmp_path):
    probe_path, test_path = tmp_path / "probe.h5", tmp_path / "test.h5"
    lengths = np.array([4, 4, 4])
    write_dataset(probe_path, lengths)
    write_dataset(test_path, lengths, action_shift=100)
    cfg = eval_config(test_path, {"mean": [0.0, 0.0], "std": [1.0, 1.0]})
    cfg.noise.mode = "episode"
    cfg.probe = {"name": str(probe_path), "n": 2, "noise_seed": 10}
    specs = OmegaConf.create([{"column": "position", "name": "x"}])
    loader, targets, episodes, train_mask = data.build(cfg, specs, 2, 2, cache_dir=tmp_path)
    np.testing.assert_array_equal(train_mask, [True, True, False, False, False])
    assert set(episodes[:2]).isdisjoint(episodes[2:])
    np.testing.assert_array_equal(targets[:, :, 0], (episodes % 3)[:, None] * 4 + [0, 1])
    batch = next(iter(loader))
    assert batch["action"][:2].max() < 100
    assert batch["action"][2:, :, 0].min() >= 100
    assert batch["valid"].all()
    probe_ds, test_ds = [windows.ds for windows in loader.dataset.datasets]
    probe_pixels, test_pixels = probe_ds[0]["pixels"], test_ds[0]["pixels"]
    assert not torch.equal(probe_pixels, test_pixels)
    torch.testing.assert_close(probe_pixels, probe_ds[0]["pixels"])

    cfg.probe.noise_seed = cfg.noise.seed
    with pytest.raises(ValueError, match="noise seeds"):
        data.build(cfg, specs, 2, 2, cache_dir=tmp_path)
    cfg.probe.noise_seed = 10
    alias = tmp_path / "alias.h5"
    alias.symlink_to(test_path)
    cfg.probe.name = str(alias)
    with pytest.raises(ValueError, match="dataset paths"):
        data.build(cfg, specs, 2, 2, cache_dir=tmp_path)


def test_recording_is_rechunked_without_changing_its_contents(tmp_path):
    path = tmp_path / "dot.h5"
    with initialize_config_dir(str(Path(__file__).resolve().parents[1] / "config/collect"), version_base=None):
        cfg = compose("dot", overrides=["episodes=5", f"name={path}"])
    collect_dot.run.__wrapped__(cfg)

    torch.manual_seed(cfg.seed)
    frames, pos, actions = collect_dot.episodes(cfg, 5, torch.Generator().manual_seed(cfg.seed))
    expected = {
        "pixels": frames.reshape(-1, *frames.shape[2:]).numpy(),
        "state": pos.reshape(-1, 2).numpy(),
        "action": actions.reshape(-1, 2).numpy(),
    }
    with h5py.File(path) as f:
        assert all(d.chunks[0] > 1 for d in f.values())
        assert f["pixels"].chunks[0] == cfg.steps + 1
        for name, array in expected.items():
            np.testing.assert_array_equal(f[name][:], array)

    ds = swm.data.HDF5Dataset(path=path, num_steps=cfg.steps + 1, keys_to_load=["pixels", "action"])
    clip = ds[2]
    np.testing.assert_array_equal(clip["pixels"].permute(0, 2, 3, 1).numpy(), frames[2].numpy())
    np.testing.assert_array_equal(clip["action"].numpy(), actions[2].numpy())
