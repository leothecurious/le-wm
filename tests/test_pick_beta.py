import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pick_beta", Path(__file__).parents[1] / "scripts" / "pick_beta.py")
pick_beta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick_beta)

HEADER = "epoch,step,fit/pixel_grad_share,fit/latent_grad_share_min,validate/pred_loss,validate/copy_loss"


def run(tmp_path, beta, rows):
    path = tmp_path / f"beta{beta}"
    path.mkdir()
    (path / "config.yaml").write_text(f"loss:\n  beta: {beta}\n")
    (path / "metrics.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    return str(path)


def test_pick_takes_the_share_closest_to_half_among_runs_beating_copy_last(tmp_path, capsys):
    dirs = [
        run(tmp_path, 0.1, ["0,9,0.1,0.8,,", "0,9,,,0.5,1.0", "1,19,0.2,0.7,,", "1,19,,,0.4,1.0"]),
        run(tmp_path, 1, ["0,9,0.4,0.5,,", "0,9,,,0.5,1.0", "1,19,0.45,0.4,,", "1,19,,,0.4,1.0"]),
        run(tmp_path, 10, ["1,19,0.5,0.05,,", "1,19,,,2.0,1.0"]),
    ]
    assert pick_beta.main(dirs) == 0
    assert "beta = 1;" in capsys.readouterr().out


def test_pick_reports_when_no_beta_beats_copy_last(tmp_path, capsys):
    assert pick_beta.main([run(tmp_path, 1, ["1,19,0.5,0.4,,", "1,19,,,2.0,1.0"])]) == 1
    assert "no beta" in capsys.readouterr().out


def test_pick_uses_the_pixel_share_of_the_last_epoch_only(tmp_path):
    rows = pick_beta.epoch_metrics(run(tmp_path, 1, ["0,9,0.9,0.5,,", "1,19,0.3,0.4,,"]) + "/metrics.csv")
    assert rows[1]["fit/pixel_grad_share"] == pytest.approx(0.3)


def test_pick_finds_the_metrics_in_the_runs_stable_pretraining_cache(tmp_path):
    ckpt = tmp_path / "checkpoints" / "run"
    ckpt.mkdir(parents=True)
    (ckpt / "config.yaml").write_text("loss:\n  beta: 1\n")
    cache = tmp_path / "spt" / "run" / "runs" / "20261113" / "205205" / "abc"
    cache.mkdir(parents=True)
    (cache / "metrics.csv").write_text("\n".join([HEADER, "1,19,0.5,0.4,,", "1,19,,,0.4,1.0"]) + "\n")
    assert pick_beta.summarize(ckpt, "beta")["share"] == pytest.approx(0.5)


def test_pick_with_lam_swaps_the_roles_of_the_terms(tmp_path):
    path = tmp_path / "lam1"
    path.mkdir()
    (path / "config.yaml").write_text("loss:\n  lam: 1\n")
    (path / "metrics.csv").write_text("epoch,fit/latent_grad_share,fit/pixel_grad_share_min,validate/pred_loss,validate/copy_loss\n1,0.3,0.2,,\n1,,,0.4,1.0\n")
    row = pick_beta.summarize(path, "lam")
    assert row["share"] == pytest.approx(0.3) and row["other_share_min"] == pytest.approx(0.2)
