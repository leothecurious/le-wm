import pytest
from omegaconf import OmegaConf

import measure
import tracking


def wandb_cfg(**kwargs):
    return OmegaConf.create({"entity": "e", "project": "p", "group": None, "tags": [], "resume": "allow", **kwargs})


@pytest.mark.parametrize(
    "cfg, mode",
    [({"entity": None}, "online"), ({"project": None}, "online"), ({}, "disabled")],
)
def test_enabled_wandb_refuses_to_run_without_logging(monkeypatch, cfg, mode):
    monkeypatch.setenv("WANDB_MODE", mode)
    with pytest.raises((ValueError, RuntimeError)):
        tracking.start(wandb_cfg(**cfg), name="n", run_id="n", job_type="train", run_dir="unused", config={})


def test_measure_writes_into_the_training_run_of_the_checkpoint(tmp_path, monkeypatch):
    # run_dirs gives a checkpoint's directory, whose name need not be the training run's id
    (tmp_path / "dir_name").mkdir()
    OmegaConf.save({"output_model_name": "model", "subdir": "the_run"}, tmp_path / "dir_name" / "config.yaml")
    opened = {}
    monkeypatch.setattr(tracking, "start", lambda cfg, **kwargs: opened.update(kwargs) or "run")
    cfg = OmegaConf.create({"wandb": wandb_cfg(), "policy": "dir_name/weights.pt", "data": {"seed": 0, "tag": {"seed": 0}, "noise": {}}})
    measure.start_wandb(cfg, tmp_path / "dir_name", None)
    assert (opened["name"], opened["run_id"], opened["job_type"]) == ("model", "the_run", "measure")
