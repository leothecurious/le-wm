import json
import math

import pytest
import torch

from toy.dot import data, probe
from toy.dot.objectives import OBJECTIVES
from toy.dot.train import main

N = 20


def chunk(noise="none", alpha=0.0, index=0, split="train"):
    return data.make_chunk(split, index, noise, alpha, n=N)


def test_chunks_are_a_function_of_seed_split_and_index():
    a, b = chunk(index=3), chunk(index=3)
    assert torch.equal(a["pos"], b["pos"])
    assert not torch.equal(a["pos"], chunk(index=4)["pos"])
    assert not torch.equal(a["pos"], chunk(split="probe", index=3)["pos"])
    assert not torch.equal(a["pos"], data.make_chunk("train", 3, seed=1, n=N)["pos"])


def test_noise_is_fixed_within_a_sequence_or_redrawn_every_frame():
    alpha = 1.5
    clean = chunk()["frames"]
    fixed, changing = chunk("fixed", alpha), chunk("changing", alpha)
    zf = (fixed["frames"] - fixed["clean"]) / alpha
    zc = (changing["frames"] - changing["clean"]) / alpha
    assert torch.equal(fixed["clean"], clean) and torch.equal(changing["clean"], clean)
    assert torch.allclose(zf[:, 0], zf[:, 5]) and not torch.allclose(zf[:, 0], zf[0:1, 0].expand_as(zf[:, 0]))
    assert not torch.allclose(zc[:, 0], zc[:, 5])
    for z in (zf, zc):
        assert z.min() >= 0 and z.max() <= 1
        assert z.mean() == pytest.approx(0.5, abs=0.02)


def test_the_probe_recovers_positions_from_states_that_encode_them_linearly():
    torch.manual_seed(0)
    mix = torch.randn(2, 32)
    features = lambda batch: batch["pos"] @ mix + 0.3
    train = list(data.batches("probe", "none", 0.0, size=1000))
    held_out = list(data.batches("eval", "none", 0.0, size=1000))
    weights = probe.fit_probe(features, train)
    assert probe.rmse(features, weights, held_out).max() < 1e-3
    noise = lambda batch: torch.randn(*batch["pos"].shape[:2], 32)
    assert probe.rmse(noise, probe.fit_probe(noise, train), held_out).min() > 0.2


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_a_run_writes_its_artifacts(objective, tmp_path, monkeypatch):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    main([f"objective={objective}", "noise=changing", "alpha=0.5", "epochs=2", "train_size=100", "probe_size=100", "val_size=100",
          "eval_size=100", "grad_share_every=1", "subdir=smoke"])
    run = tmp_path / "checkpoints" / "smoke"
    result = json.loads((run / "result.json").read_text())
    header = (run / "metrics.csv").read_text().splitlines()[0].split(",")
    assert {"val/pred_loss", "val/copy_loss"} <= set(header)
    assert ("train/latent_grad_share_min" in header) == (objective in ("grounded", "latent_twin"))
    assert math.isfinite(result["rmse"]) and len(result["weights_sha256"]) == 64
