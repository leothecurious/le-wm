"""The generated moving-dot stream: reproducible, never repeating, and separate from every other stage."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import dot
import train
from metrics import data

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config" / "train")
SMALL = ["data.generator.steps_per_epoch=3", "data.generator.chunk=2", "data.generator.val_sequences=20", "data.generator.stats_sequences=200",
         "loader.batch_size=8", "trainer.accelerator=cpu", "noise.alpha=1.5"]


def loaders(*overrides, config="lewm"):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config, overrides=["data=dot_online", *SMALL, *overrides])
    return train.generated_data(cfg)


def first_batches(loader, n=2):
    return [b for _, b in zip(range(n), loader)]


def test_one_seed_gives_every_objective_the_same_sequences_and_a_new_batch_each_epoch():
    latent, pixel = loaders("noise.mode=episode")[0], loaders("noise.mode=episode", config="lewm_pixel")[0]
    again = loaders("noise.mode=episode")[0]
    for a, b, c in zip(first_batches(latent), first_batches(pixel), first_batches(again)):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    assert next(iter(latent))["pixels"].shape == (8, 2, 3, 32, 32)

    # the stream carries on across epochs, and a restored state continues it
    epoch1 = list(latent.dataset)
    state = latent.dataset.state_dict()
    epoch2 = list(latent.dataset)
    assert len(epoch1) == len(epoch2) == 3
    assert not torch.equal(epoch1[0]["pixels"], epoch2[0]["pixels"])
    again.dataset.load_state_dict(state)
    torch.testing.assert_close(next(iter(again))["pixels"], epoch2[0]["pixels"], rtol=0, atol=0)

    # noise changes pixels but not the trajectories the sequences follow
    clean, noisy = loaders("noise.mode=off")[0], loaders("noise.mode=episode")[0]
    torch.testing.assert_close(next(iter(clean))["action"], next(iter(noisy))["action"], rtol=0, atol=0)
    assert not torch.equal(next(iter(clean))["pixels"], next(iter(noisy))["pixels"])


@pytest.mark.parametrize("mode", ["episode", "frame"])
def test_noise_is_held_across_a_sequence_in_episode_mode_only(mode):
    cfg = OmegaConf.create({"size": 32, "steps": 16, "max_step": 0.148, "sigma": 0.048, "margin": 0.096, "concentration": 0.2})
    alpha = 4.0
    kwargs = dict(cfg=cfg, n=6)
    noisy, _, _ = dot.sequences(streams=dot.Streams(0, "train", "cpu"), noise=mode, alpha=alpha, **kwargs)
    clean, _, _ = dot.sequences(streams=dot.Streams(0, "train", "cpu"), **kwargs)
    # recover the pattern from blend: noisy * (1 + alpha) = clean + alpha * 255 * z, up to rounding
    z = (noisy.float() * (1 + alpha) - clean.float()) / (alpha * 255)
    within = (z - z[:, :1]).abs().amax()
    across = (z[:, 0] - z[:1, 0]).abs().mean()
    assert across > 0.1
    assert (within < 0.01) == (mode == "episode")


def test_every_stage_and_seed_draws_its_own_sequences():
    cfg = OmegaConf.create({"size": 8, "steps": 4, "max_step": 0.148, "sigma": 0.048, "margin": 0.096, "concentration": 0.2})
    drawn = {
        (seed, stage): dot.trajectories(cfg, 4, dot.Streams(seed, stage, "cpu").geometry)[0]
        for seed in (0, 1, 2) for stage in dot.STAGES
    }
    keys = list(drawn)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not torch.equal(drawn[a], drawn[b]), (a, b)


def test_probe_and_test_sets_are_regenerated_from_the_run_seed_and_are_separate():
    cfg = OmegaConf.create({
        "generator": {"size": 32, "steps": 16, "max_step": 0.148, "sigma": 0.048, "margin": 0.096, "concentration": 0.2},
        "action_normalizer": {"mean": [0.0, 0.0], "std": [0.06, 0.06]}, "seed": 3, "img_size": 32, "batch_size": 4,
        "noise": {"mode": "episode", "alpha": 1.5}, "probe": {"n": 5}, "test": {"n": 3},
    })
    specs = OmegaConf.create([{"column": "state", "index": 0}, {"column": "state", "index": 1}])
    loader, targets, episodes, train_mask = data.build_generated(cfg, specs)
    again = data.build_generated(cfg, specs)
    assert targets.shape == (8, 17, 2) and train_mask.tolist() == [True] * 5 + [False] * 3
    assert len(set(episodes.tolist())) == 8
    assert (targets[:5] != targets[5:].mean(0)).any() and not (targets[:3] == targets[5:]).all()
    torch.testing.assert_close(next(iter(loader))["pixels"], next(iter(again[0]))["pixels"], rtol=0, atol=0)
