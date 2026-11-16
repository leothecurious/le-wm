from pathlib import Path

import h5py
import numpy as np
import pytest
import stable_pretraining as spt
import torch
from hydra import compose, initialize_config_dir

import train
from utils import ResumeCkptCallback

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config" / "train")


def toy_dataset(root):
    rng = np.random.default_rng(0)
    with h5py.File(root / "toy.h5", "w", libver="latest") as f:
        f["ep_len"] = np.array([40, 40])
        f["ep_offset"] = np.array([0, 40])
        f["pixels"] = rng.integers(0, 256, (80, 28, 28, 3), dtype=np.uint8)
        f["action"] = rng.normal(size=(80, 2)).astype(np.float32)


def fit(data, subdir, epochs):
    overrides = [
        f"subdir={subdir}",
        f"data.dataset.name={data}",
        "data.dataset.keys_to_load=[pixels,action]",
        "data.dataset.keys_to_cache=[action]",
        "img_size=28",
        "num_workers=0",
        "loader.batch_size=4",
        "loader.persistent_workers=false",
        "loader.prefetch_factor=null",
        "loader.pin_memory=false",
        "model.predictor.depth=1",
        "model.predictor.heads=2",
        "model.predictor.dim_head=8",
        "model.predictor.mlp_dim=32",
        "trainer.accelerator=cpu",
        "trainer.precision=32",
        f"trainer.max_epochs={epochs}",
        "+trainer.limit_train_batches=3",
        "+trainer.limit_val_batches=1",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        train.run(compose(config_name="lewm", overrides=overrides))


def state(run_dir):
    return torch.load(run_dir / "lewm_weights.ckpt", map_location="cpu", weights_only=False)


def test_resume_after_a_kill_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    spt.set(cache_dir=str(tmp_path / "spt"))
    data = tmp_path / "toy.h5"
    toy_dataset(tmp_path)
    runs = tmp_path / "checkpoints"

    fit(data, "whole", 3)

    save = ResumeCkptCallback.on_train_epoch_end
    epochs_run = []

    def save_then_die(self, trainer, pl_module):
        epochs_run.append(trainer.current_epoch)
        save(self, trainer, pl_module)
        if not killed:
            raise RuntimeError("killed")

    killed = False
    monkeypatch.setattr(ResumeCkptCallback, "on_train_epoch_end", save_then_die)
    with pytest.raises(RuntimeError, match="killed"):
        fit(data, "killed", 3)
    killed = True
    fit(data, "killed", 3)

    # the rerun continues at epoch 1 instead of repeating epoch 0
    assert epochs_run == [0, 1, 2]
    whole, resumed = state(runs / "whole"), state(runs / "killed")
    assert resumed["global_step"] == whole["global_step"] == 9
    assert resumed["lr_schedulers"] == whole["lr_schedulers"]
    for name, tensor in whole["state_dict"].items():
        torch.testing.assert_close(resumed["state_dict"][name], tensor, rtol=0, atol=0, msg=name)
