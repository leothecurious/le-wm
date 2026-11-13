from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

import train
from grounding import copy_last_loss, grad_shares, next_frame_targets
from module import SIGReg


def test_next_frame_targets_are_area_resized_later_frames():
    pixels = torch.arange(2 * 4 * 3 * 8 * 8, dtype=torch.float32).view(2, 4, 3, 8, 8)
    target = next_frame_targets(pixels, n_preds=1, size=(4, 4))
    assert target.shape == (2, 3, 3, 4, 4)
    block = pixels[0, 1, 0, :2, :2].mean()
    assert target[0, 0, 0, 0, 0] == pytest.approx(float(block))


def test_copy_last_loss_is_zero_for_a_constant_embedding_and_positive_otherwise():
    emb = torch.ones(2, 4, 5)
    assert copy_last_loss(emb, ctx_len=3, n_preds=1) == 0
    emb[:, 2:] = 3
    assert copy_last_loss(emb, ctx_len=3, n_preds=1) > 0


def test_grad_shares_split_the_gradient_norm_at_the_embedding():
    emb = torch.randn(3, 4, requires_grad=True)
    shares = grad_shares({"a": (2 * emb).sum(), "b": (6 * emb).sum()}, emb)
    assert float(shares["a"]) == pytest.approx(0.25) and float(shares["b"]) == pytest.approx(0.75)
    assert emb.grad is None


class Recorder(SimpleNamespace):
    def log_dict(self, values, **kw):
        self.logged.update(values)

    def log(self, name, value, **kw):
        self.logged[name] = value


def tiny_module(config, step=0):
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(
            config_name=config,
            overrides=[
                "img_size=28",
                "model.action_encoder.input_dim=2",
                "model.predictor.depth=1",
                "model.predictor.mlp_dim=32",
                "model.projector.hidden_dim=32",
                "model.pred_proj.hidden_dim=32",
            ],
        )
    module = Recorder(model=instantiate(cfg.model), sigreg=SIGReg(num_proj=16), global_step=step, logged={})
    return module, cfg


def batch():
    return {"pixels": torch.rand(4, 4, 3, 28, 28), "action": torch.randn(4, 4, 2)}


def test_grounded_loss_adds_beta_times_the_pixel_error():
    module, cfg = tiny_module("lewm_grounded")
    cfg.loss.beta = 2.0
    out = train.grounded_forward(module, batch(), "fit", cfg)
    expected = out["pred_loss"] + 0.09 * out["sigreg_loss"] + 2.0 * out["pixel_loss"]
    assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert out["loss"].requires_grad
    pred_pixels = module.model.decode(torch.zeros(4, 3, 192))
    assert pred_pixels.shape == (4, 3, 3, 64, 64)


def test_grounded_logs_gradient_shares_on_measured_training_steps_only():
    module, cfg = tiny_module("lewm_grounded", step=0)
    assert cfg.loss.grad_share_every == 50
    train.grounded_forward(module, batch(), "fit", cfg)
    pixel, latent = module.logged["fit/pixel_grad_share"], module.logged["fit/latent_grad_share"]
    assert float(pixel + latent) == pytest.approx(1.0)
    assert module.logged["fit/latent_grad_share_min"] == latent
    assert module.logged["fit/pixel_grad_share_min"] == pixel

    skipped, cfg = tiny_module("lewm_grounded", step=1)
    train.grounded_forward(skipped, batch(), "fit", cfg)
    assert not any("grad_share" in k for k in skipped.logged)


def test_validation_logs_the_copy_last_baseline_next_to_the_prediction_error():
    for config, forward in (("lewm_grounded", train.grounded_forward), ("lewm_pixel", train.pixel_forward)):
        module, cfg = tiny_module(config)
        with torch.no_grad():
            forward(module, batch(), "validate", cfg)
        assert {"validate/pred_loss", "validate/copy_loss", "validate/pixel_loss"} <= set(module.logged)
        assert not any("grad_share" in k for k in module.logged)


def test_the_pixel_twin_loss_is_the_pixel_error_alone():
    module, cfg = tiny_module("lewm_pixel")
    out = train.pixel_forward(module, batch(), "fit", cfg)
    assert out["loss"] is out["pixel_loss"]
