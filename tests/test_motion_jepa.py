from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from module import SIGReg
from train import lejepa_forward

# The reference below is a frozen copy of code from vivoTechResearch/motionjepa@d732c4f
# (https://github.com/vivoTechResearch/motionjepa, the release accompanying Karmann et al.,
# "MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent
# Space", arXiv:2609.23881), used under the MIT licence reproduced in LICENSE.motionjepa:
# Copyright (c) 2026 Motion-JEPA authors
# Copyright (c) 2026 Lucas Maes (original LeWorldModel implementation)
#
# Their weights come from scripts/_method_benchmark_loop.sh, method "ours".
REFERENCE_WEIGHTS = {"prediction": 1.0, "sigreg": 0.25, "sigreg_diff_image": 2.0, "diff_image_latent_prediction": 0.5}
REFERENCE_DIFF_IMAGE_SIZE = 112


class DiffImageLatentPredictor(nn.Module):
    """module.py: predict diff-image latent from consecutive image latents."""

    def __init__(self, z_dim, diff_dim, hidden_dim=512, depth=2):
        super().__init__()
        layers = []
        dim = 2 * z_dim
        for _ in range(max(depth - 1, 0)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, diff_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, z_t1):
        return self.net(torch.cat([z_t, z_t1], dim=-1))


def reference_losses(model, batch, sigreg, cfg):
    """train.py lejepa_forward with diff_image enabled, without its detached probe and
    reconstruction terms. ``model`` is a MotionJEPA built from our config; its
    diff_predictor weights are loaded into the reference class."""
    weights = REFERENCE_WEIGHTS
    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, : cfg.history_size]
    ctx_act = act_emb[:, : cfg.history_size]
    tgt_emb = emb[:, cfg.num_preds :]
    pred_emb = model.predict(ctx_emb, ctx_act)

    output["loss_prediction"] = F.mse_loss(pred_emb, tgt_emb)
    output["loss_sigreg_z"] = sigreg(emb.transpose(0, 1))

    diff_pixels = batch["pixels"][:, 1:] - batch["pixels"][:, :-1]
    if REFERENCE_DIFF_IMAGE_SIZE != batch["pixels"].size(-1):
        shape = diff_pixels.shape
        diff_pixels = F.interpolate(
            diff_pixels.reshape(-1, *shape[-3:]),
            size=(REFERENCE_DIFF_IMAGE_SIZE, REFERENCE_DIFF_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).view(*shape[:-2], REFERENCE_DIFF_IMAGE_SIZE, REFERENCE_DIFF_IMAGE_SIZE)
    # jepa.py encode_diff
    b = diff_pixels.size(0)
    flat = rearrange(diff_pixels.float(), "b t ... -> (b t) ...")
    cls = model.diff_encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    diff_emb = rearrange(model.diff_projector(cls), "(b t) d -> b t d", b=b)
    output["loss_sigreg_diff_image"] = sigreg(diff_emb.transpose(0, 1))
    reference_predictor = DiffImageLatentPredictor(emb.size(-1), diff_emb.size(-1), hidden_dim=512, depth=3)
    reference_predictor.net.load_state_dict(model.diff_predictor.net.state_dict())
    diff_from_latents = reference_predictor(emb[:, :-1], emb[:, 1:])
    output["loss_diff_image_latent_prediction"] = F.mse_loss(diff_from_latents, diff_emb)

    output["loss"] = (
        weights["prediction"] * output["loss_prediction"]
        + weights["sigreg"] * output["loss_sigreg_z"]
        + weights["sigreg_diff_image"] * output["loss_sigreg_diff_image"]
        + weights["diff_image_latent_prediction"] * output["loss_diff_image_latent_prediction"]
    )
    return output


class SeededSIGReg(SIGReg):
    """SIGReg draws random projections, so each call restarts from the same seed to make the
    value independent of call order."""

    def forward(self, proj):
        torch.manual_seed(0)
        return super().forward(proj)


def compose_train(config_name, *overrides):
    with initialize_config_dir(str(Path("config/train").resolve()), version_base=None):
        return compose(config_name, overrides=list(overrides))


def small_model(config_name, *overrides):
    cfg = compose_train(config_name, "img_size=56", *overrides)
    OmegaConf.update(cfg, "model.action_encoder.input_dim", 10)
    torch.manual_seed(0)
    return cfg, instantiate(cfg.model).eval()


def fixed_batch():
    generator = torch.Generator().manual_seed(1)
    return {
        "pixels": torch.randn(3, 4, 3, 56, 56, generator=generator),
        "action": torch.randn(3, 4, 10, generator=generator),
    }


def trainer_stub(model):
    return SimpleNamespace(model=model, sigreg=SeededSIGReg(), log_dict=lambda *args, **kwargs: None)


def test_losses_match_the_reference_on_a_fixed_batch():
    cfg, model = small_model("lewm_motion")
    ours = lejepa_forward(trainer_stub(model), fixed_batch(), "fit", cfg)
    reference = reference_losses(model, fixed_batch(), SeededSIGReg(), cfg)
    pairs = {
        "loss": "loss",
        "pred_loss": "loss_prediction",
        "sigreg_loss": "loss_sigreg_z",
        "diff_sigreg_loss": "loss_sigreg_diff_image",
        "diff_pred_loss": "loss_diff_image_latent_prediction",
    }
    for name, reference_name in pairs.items():
        torch.testing.assert_close(ours[name], reference[reference_name], msg=name)


def test_lewm_adds_no_motion_terms():
    cfg, model = small_model("lewm")
    output = lejepa_forward(trainer_stub(model), fixed_batch(), "fit", cfg)
    assert not any(key.startswith("diff_") for key in output)
    expected = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
    torch.testing.assert_close(output["loss"], expected)


def flatten(node, prefix=""):
    if isinstance(node, dict):
        return {k: v for key, value in node.items() for k, v in flatten(value, f"{prefix}{key}.").items()}
    return {prefix[:-1]: node}


def test_motion_config_only_adds_keys_and_retargets_the_model_and_sigreg_weight():
    for arm in ("baseline", "tag_episode", "tag_frame"):
        lewm = flatten(OmegaConf.to_container(compose_train("lewm", f"+arm={arm}", "seed=1")))
        motion = flatten(OmegaConf.to_container(compose_train("lewm_motion", f"+arm={arm}", "seed=1")))
        assert lewm.keys() <= motion.keys()
        added = motion.keys() - lewm.keys()
        assert all(key in ("run_prefix", "diff_dim") or key.startswith(("model.diff_", "loss.motion.")) for key in added)
        changed = {key for key in lewm if lewm[key] != motion[key]}
        assert changed == {"model._target_", "loss.sigreg.weight"}
        assert motion["loss.sigreg.weight"] == 0.25


def test_motion_runs_are_named_apart_from_the_lewm_arms():
    for arm in ("baseline", "tag_episode", "tag_frame"):
        lewm = compose_train("lewm", f"+arm={arm}", "seed=1")
        motion = compose_train("lewm_motion", f"+arm={arm}", "seed=1")
        assert lewm.output_model_name == lewm.subdir == f"{arm}_seed1"
        assert motion.output_model_name == motion.subdir == f"motion_{arm}_seed1"


def test_diff_branch_is_built_as_in_the_reference_config():
    _, model = small_model("lewm_motion")
    assert model.diff_size == REFERENCE_DIFF_IMAGE_SIZE
    config = model.diff_encoder.config
    assert (config.hidden_size, config.num_hidden_layers, config.patch_size, config.image_size) == (192, 12, 14, 112)
    linear_in, norm, _, linear_out = model.diff_projector.net
    assert (linear_in.in_features, linear_in.out_features) == (192, 2048)
    assert isinstance(norm, nn.BatchNorm1d) and norm.num_features == 2048
    assert (linear_out.in_features, linear_out.out_features) == (2048, 8)
