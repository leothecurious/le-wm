import hydra
from hydra import compose, initialize

CONFIGS = ("lewm", "lewm_pixel")


def build(name):
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=name, overrides=["model.action_encoder.input_dim=10"])
    return cfg, hydra.utils.instantiate(cfg.model)


def test_pixel_twin_shares_the_world_model():
    (latent_cfg, latent), (pixel_cfg, pixel) = (build(c) for c in CONFIGS)
    assert latent_cfg.loss.kind == "latent" and pixel_cfg.loss.kind == "pixel"
    assert latent.decoder is None and pixel.decoder is not None
    world = {k: v.shape for k, v in pixel.state_dict().items() if not k.startswith("decoder.")}
    assert world == {k: v.shape for k, v in latent.state_dict().items()}
