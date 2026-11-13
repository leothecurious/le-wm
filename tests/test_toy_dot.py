import json
import math

import pytest
import torch

from module import SIGReg
from toy.dot import data, jobs, probe
from toy.dot.model import DotModel
from toy.dot.objectives import OBJECTIVES, compute
from toy.dot.train import main

N = 20


def chunk(noise="none", alpha=0.0, index=0, split="train"):
    return data.make_chunk(split, index, noise, alpha, n=N)


def test_the_dot_moves_by_its_actions_and_stays_in_the_square():
    c = chunk()
    assert c["frames"].shape == (N, 17, 28, 28) and c["actions"].shape == (N, 16, 2)
    assert c["actions"].norm(dim=-1).max() <= data.MAX_STEP + 1e-6
    assert c["pos"].min() >= 0 and c["pos"].max() <= 1
    step = c["pos"][:, 1:] - c["pos"][:, :-1]
    inside = (c["pos"][:, 1:] > 0) & (c["pos"][:, 1:] < 1)  # moves that were not clipped
    assert torch.allclose(step[inside], c["actions"][inside], atol=1e-6)


def test_the_blurred_dot_peaks_at_its_position_with_value_one():
    c = chunk()
    flat = c["clean"].flatten(-2).argmax(-1)
    row, col = flat // 28, flat % 28
    assert ((row + 0.5) / 28 - c["pos"][..., 1]).abs().max() <= 0.5 / 28 + 1e-6
    assert ((col + 0.5) / 28 - c["pos"][..., 0]).abs().max() <= 0.5 / 28 + 1e-6
    assert c["clean"].max() == pytest.approx(1.0, abs=0.05)
    spread = data.render(torch.tensor([0.5, 0.5]))
    expected = math.exp(-((0.5 / 28) ** 2 + (0.5 / 28) ** 2) / (2 * data.SIGMA**2))
    assert spread[14, 14] == pytest.approx(expected, rel=1e-5)


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


def test_batches_partition_a_split_into_fixed_chunks():
    seen = [b["pos"] for b in data.batches("val", "none", 0.0, batch_size=500, size=2000, shuffle=True, epoch=1)]
    assert len(seen) == 4 and seen[0].shape[0] == 500
    again = [b["pos"] for b in data.batches("val", "none", 0.0, batch_size=500, size=2000, shuffle=True, epoch=1)]
    assert all(torch.equal(x, y) for x, y in zip(seen, again))


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
def test_each_objective_trains_a_step_with_finite_losses(objective):
    model = DotModel(decoder=objective != "sigreg" and objective != "supervised", position_head=objective == "supervised")
    batch = chunk("fixed", 1.0)
    out = compute(model, batch, objective, SIGReg(num_proj=16), beta=1.0, lam=1.0, shares=True)
    out["loss"].backward()
    assert all(torch.isfinite(v) for v in out.values())
    if objective in ("grounded", "latent_twin"):
        assert float(out["latent_grad_share"] + out["pixel_grad_share"]) == pytest.approx(1.0)
    else:
        assert "latent_grad_share" not in out


def test_a_run_writes_its_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    main(["objective=grounded", "noise=changing", "alpha=0.5", "epochs=2", "train_size=100", "probe_size=100", "val_size=100",
          "eval_size=100", "grad_share_every=1", "subdir=smoke"])
    run = tmp_path / "checkpoints" / "smoke"
    result = json.loads((run / "result.json").read_text())
    header = (run / "metrics.csv").read_text().splitlines()[0].split(",")
    assert {"train/pixel_grad_share", "train/latent_grad_share_min", "val/pred_loss", "val/copy_loss"} <= set(header)
    assert math.isfinite(result["rmse"]) and len(result["weights_sha256"]) == 64


def test_the_grid_has_seventy_runs_that_differ_only_in_objective_noise_alpha_and_seed():
    lr = {o: 1e-3 for o in OBJECTIVES}
    lines = list(jobs.grid_jobs(lr, 1.0, 1.0))
    assert len(lines) == 70 and len({line.split()[0] for line in lines}) == 70
    assert lines[0].split()[0].endswith("seed1") and lines[-1].split()[0].endswith("seed2")
    assert len(list(jobs.lr_jobs())) == 20 and len(list(jobs.weight_jobs(lr))) == 6
