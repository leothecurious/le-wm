import hydra
import pytest
import torch
from hydra import compose, initialize

from module import grad_shares


def test_grad_shares_split_the_gradient_norm_at_the_embedding():
    emb = torch.randn(3, 4, requires_grad=True)
    shares = grad_shares({"a": (2 * emb).sum(), "b": (6 * emb).sum()}, emb)
    assert float(shares["a"]) == pytest.approx(0.25) and float(shares["b"]) == pytest.approx(0.75)
    assert emb.grad is None


def test_pixel_twin_shares_the_world_model():
    def build(name):
        with initialize(version_base=None, config_path="../config/train"):
            cfg = compose(config_name=name, overrides=["model.action_encoder.input_dim=10"])
        return hydra.utils.instantiate(cfg.model)

    latent, pixel = build("lewm"), build("lewm_pixel")
    assert latent.decoder is None
    world = {k: v.shape for k, v in pixel.state_dict().items() if not k.startswith("decoder.")}
    assert world == {k: v.shape for k, v in latent.state_dict().items()}
