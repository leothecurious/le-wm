import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pick_beta", Path(__file__).parents[1] / "scripts" / "pick_beta.py")
pick_beta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick_beta)

HEADER = "epoch,step,fit/pixel_grad_share,fit/latent_grad_share_min,validate/pred_loss,validate/copy_loss"


def run(tmp_path, beta, rows):
    """A checkpoint directory and its stable-pretraining metrics, laid out as scripts/run.sh leaves them."""
    name = f"beta{beta}"
    ckpt = tmp_path / "checkpoints" / name
    ckpt.mkdir(parents=True)
    (ckpt / "config.yaml").write_text(f"loss:\n  beta: {beta}\n")
    cache = tmp_path / "spt" / name / "runs" / "20261113" / "205205" / "abc"
    cache.mkdir(parents=True)
    (cache / "metrics.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    return str(ckpt)


def test_pick_takes_the_share_closest_to_half_among_runs_beating_copy_last(tmp_path, capsys):
    dirs = [
        run(tmp_path, 0.1, ["0,9,0.1,0.8,,", "0,9,,,0.5,1.0", "1,19,0.2,0.7,,", "1,19,,,0.4,1.0"]),
        run(tmp_path, 1, ["0,9,0.4,0.5,,", "0,9,,,0.5,1.0", "1,19,0.45,0.4,,", "1,19,,,0.4,1.0"]),
        run(tmp_path, 10, ["1,19,0.5,0.05,,", "1,19,,,2.0,1.0"]),
    ]
    assert pick_beta.main(dirs) == 0
    assert "beta = 1;" in capsys.readouterr().out
