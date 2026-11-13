import pickle

import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from tag import EvalTag, PixelTag, resolve_seed


def enigma_color(seed, *key):
    """Frozen copy of the colour hash in Enigma-Incorporated/The-Obsessed-Encoder@37819425
    (additional_files/pixel_tag.py)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, *key]))
    return torch.from_numpy(rng.integers(0, 256, size=3, dtype=np.uint8))


def toy_dataset(tmp_path):
    """Two episodes of 8 and 7 rows, clips of 3 frames at frameskip 2.

    Clip i is (episode, start): 0 (0, 0), 1 (0, 1), 2 (0, 2), 3 (1, 0), 4 (1, 1).
    """
    path = tmp_path / "toy.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f["ep_len"] = np.array([8, 7])
        f["ep_offset"] = np.array([0, 8])
        f["pixels"] = np.zeros((15, 16, 16, 3), dtype=np.uint8)
        f["action"] = np.zeros((15, 2), dtype=np.float32)
    return swm.data.HDF5Dataset(path=path, frameskip=2, num_steps=3, keys_to_load=["pixels", "action"])


def corner(pixels, n=5):
    return pixels[:, :, :n, :n]


def test_colors_match_enigma_reference():
    tag = PixelTag("episode", seed=3)
    for episode in (0, 1, 7, 18684):
        torch.testing.assert_close(tag.color(episode), enigma_color(3, episode))
        torch.testing.assert_close(tag.color(episode, 40), enigma_color(3, episode, 40))


def test_episode_mode_is_constant_within_and_distinct_across_episodes(tmp_path):
    ds = PixelTag("episode").wrap(toy_dataset(tmp_path))
    a, b, c = ds[0]["pixels"], ds[1]["pixels"], ds[3]["pixels"]

    assert (corner(a) == corner(a)[:, :, :1, :1]).all()
    assert (a[0, :, 0, 0] == b[0, :, 0, 0]).all()
    assert (a[0, :, 0, 0] != c[0, :, 0, 0]).any()
    assert not a[:, :, 5:, :].any() and not a[:, :, :, 5:].any()


def test_frame_mode_changes_every_frame_and_follows_the_row_in_the_episode(tmp_path):
    ds = PixelTag("frame", seed=1).wrap(toy_dataset(tmp_path))
    a, b = ds[0]["pixels"], ds[1]["pixels"]

    assert len({tuple(a[t, :, 0, 0].tolist()) for t in range(3)}) == 3
    # clip 1 starts at row 1 and clip 0 reaches row 2 at its frame 1
    assert (b[0, :, 0, 0] != a[1, :, 0, 0]).any()
    # clip 4 is episode 1 from row 1; its frame 1 is row 3
    torch.testing.assert_close(ds[4]["pixels"][1, :, 0, 0], enigma_color(1, 1, 3))


def test_transform_sees_stamped_frames_and_load_chunk_matches_getitem(tmp_path):
    ds = toy_dataset(tmp_path)
    ds.transform = lambda steps: {
        **steps,
        "seen": corner(steps["pixels"]).any(),
        "rows_at_transform": steps["action"].shape[0],
    }
    tagged = PixelTag("episode").wrap(ds)

    chunk = tagged.load_chunk([1], [1], [7])[0]
    assert chunk["seen"]
    assert chunk["rows_at_transform"] == 6  # one action per row, as swm transforms them
    assert chunk["action"].shape == (3, 4)
    torch.testing.assert_close(chunk["pixels"], tagged[4]["pixels"])


def test_tagged_dataset_pickles_and_loads_in_batches_from_spawned_workers(tmp_path):
    ds = pickle.loads(pickle.dumps(PixelTag("episode").wrap(toy_dataset(tmp_path))))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=5, num_workers=1, multiprocessing_context="spawn"
    )
    pixels = next(iter(loader))["pixels"]

    assert pixels.shape == (5, 3, 3, 16, 16)
    assert corner(pixels.flatten(0, 1)).any(-1).any(-1).all()


def test_off_leaves_the_dataset_alone_and_other_readers_are_rejected(tmp_path):
    ds = toy_dataset(tmp_path)
    assert PixelTag("off").wrap(ds) is ds
    with pytest.raises(TypeError, match="HDF5"):
        PixelTag("episode").wrap(object())


def test_eval_tag_uses_one_colour_in_episode_mode_and_new_ones_in_frame_mode():
    frame = torch.zeros(3, 16, 16, dtype=torch.uint8)
    episode = EvalTag(PixelTag("episode"), lambda x: x)
    per_frame = EvalTag(PixelTag("frame"), lambda x: x)

    first, second = episode(frame), episode(frame)
    assert (first[:, 0, 0] == second[:, 0, 0]).all()
    torch.testing.assert_close(first[:, 0, 0], enigma_color(0, 0))
    assert (per_frame(frame)[:, 0, 0] != per_frame(frame)[:, 0, 0]).any()
    assert not frame.any()


def test_eval_tag_rejects_non_uint8_frames():
    with pytest.raises(AssertionError, match="uint8"):
        EvalTag(PixelTag("episode"), lambda x: x)(torch.zeros(3, 16, 16))


def test_null_seed_is_read_from_the_checkpoint_config(tmp_path):
    OmegaConf.save({"tag": {"mode": "episode", "seed": 2}}, tmp_path / "config.yaml")
    tag_cfg = OmegaConf.create({"mode": "episode", "size": 5, "seed": None})
    resolve_seed(tag_cfg, tmp_path)
    assert tag_cfg.seed == 2

    explicit = OmegaConf.create({"mode": "episode", "size": 5, "seed": 9})
    resolve_seed(explicit, tmp_path)
    assert explicit.seed == 9


def test_null_seed_without_a_checkpoint_is_zero():
    random_policy = OmegaConf.create({"mode": "episode", "size": 5, "seed": None})
    resolve_seed(random_policy, None)
    assert random_policy.seed == 0


@pytest.mark.parametrize(
    "path, key",
    [
        ("config/train/lewm.yaml", "tag.mode"),
        ("config/eval/pusht.yaml", "tag.mode"),
        ("config/measure/pusht.yaml", "data.tag.mode"),
    ],
)
def test_configs_default_to_the_string_off(path, key):
    # unquoted YAML 1.1 `off` loads as False
    assert OmegaConf.select(OmegaConf.load(path), key) == "off"
