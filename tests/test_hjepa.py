"""Focused behavioral coverage for the explicit H-JEPA interfaces."""

from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from hydra.utils import instantiate

import module
import train
from metrics.latents import extract

ROOT = Path(__file__).parents[1]


def make_cfg():
    with hydra.initialize_config_dir(str(ROOT / "config" / "train"), version_base=None):
        cfg = hydra.compose(
            "hjepa",
            overrides=[
                "img_size=28", "+model.encoder.num_hidden_layers=1",
                "loss.sigreg.weight=0.07", "loss.hjepa.weight=0.37", "loss.hjepa.sigreg_weight=0.19",
                "model.predictor.depth=1", "model.predictor.heads=2",
                "model.predictor.mlp_dim=32", "model.predictor.dim_head=8",
                "model.upper.predictor.depth=1", "model.upper.predictor.heads=2",
                "model.upper.predictor.mlp_dim=32", "model.upper.predictor.dim_head=8",
                "model.upper.encoder.hidden_dim=32",
                "model.projector.hidden_dim=32", "model.pred_proj.hidden_dim=32",
                "model.upper.pred_proj.hidden_dim=32", "model.action_encoder.input_dim=10",
            ],
        )
    return cfg


def make_model():
    cfg = make_cfg()
    model = instantiate(cfg.model)
    # AdaLN is zero-initialized by design; nonzero it so the action-conditioned
    # predictor participates in this gradient-path check.
    with torch.no_grad():
        for block in list(model.predictor.transformer.layers) + list(model.upper.predictor.transformer.layers):
            block.adaLN_modulation[-1].weight.fill_(0.01)
            block.adaLN_modulation[-1].bias.fill_(0.01)
    return cfg, model


def batch():
    gen = torch.Generator().manual_seed(7)
    return {"pixels": torch.randn(2, 7, 3, 28, 28, generator=gen), "action": torch.randn(2, 7, 10, generator=gen)}


def test_explicit_hierarchy_training_updates_both_levels_and_roundtrips_state():
    cfg, model = make_model()
    model.train()
    sigreg = module.SIGReg(knots=5, num_proj=8)
    out = train.training_forward(
        SimpleNamespace(model=model, sigreg=sigreg, log_dict=lambda *a, **k: None),
        batch(), "train", cfg,
    )
    expected = (out["pred_loss"] + cfg.loss.sigreg.weight * out["sigreg_loss"]
                + cfg.loss.hjepa.weight * (out["pred_loss_upper"]
                + cfg.loss.hjepa.sigreg_weight * out["sigreg_loss_upper"]))
    torch.testing.assert_close(out["loss"], expected)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
    # Upper prediction must train its targets and the lower observation/action paths.
    paths = [out["upper_emb"], model.encoder.embeddings.patch_embeddings.projection.weight,
             model.action_encoder.patch_embed.weight, model.upper.action_pool.step_mlp.fc1.weight]
    target, *grads = torch.autograd.grad(out["pred_loss_upper"], paths, retain_graph=True)
    assert target[:, -1].abs().sum() > 0
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    out["loss"].backward()
    assert model.upper.encoder.fc1.weight.grad.abs().sum() > 0
    assert model.upper.latent_action_queue.shape == (2, 4)

    model.eval()
    pixels = batch()["pixels"]
    before = model.encode({"pixels": pixels})["emb"]
    state = model.state_dict()
    _, restored = make_model()
    restored.load_state_dict(state, strict=True)
    restored.eval()
    after = restored.encode({"pixels": pixels})["emb"]
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(restored.upper.latent_action_queue, model.upper.latent_action_queue)
    torch.testing.assert_close(restored.levels(pixels)[1], model.levels(pixels)[1])


def test_metrics_extraction_at_training_window_uses_full_encode():
    cfg, model = make_model()
    model.eval()
    gen = torch.Generator().manual_seed(11)
    pixels = torch.randn(2, 7, 3, 28, 28, generator=gen)
    actions = torch.randn(2, 7, 10, generator=gen)
    loader = [{"pixels": pixels, "action": actions, "valid": torch.ones(2, 7, dtype=torch.bool)}]
    before = model.upper.latent_action_queue.clone()
    out = extract(model, loader)
    assert out["z"].shape == (2, 7, 192)
    assert torch.equal(model.upper.latent_action_queue, before)
    expected = model.encode({"pixels": pixels})["emb"]
    torch.testing.assert_close(out["z"], expected)


def test_projected_planner_action_cost_uses_only_available_subgoal_actions():
    from hjepa import UpperLevel
    from hjepa_planner import LevelCostModel
    from jepa import JEPA

    class Add(torch.nn.Module):
        def forward(self, emb, action):
            return emb + action

    class MeanPool(torch.nn.Module):
        def forward(self, chunks, padding_mask):
            return chunks.mean(dim=2)

    lower = JEPA(torch.nn.Identity(), Add(), torch.nn.Identity())
    upper = UpperLevel(torch.nn.Identity(), Add(), torch.nn.Identity(),
                       torch.nn.Identity(), MeanPool(), stride=2, queue_size=0).eval()
    cost = LevelCostModel(lower, upper)
    actions = torch.tensor([[[[1., 0.], [1., 0.], [3., 0.], [3., 0.]]]])
    info = {"embed_0": torch.zeros(1, 1, 1, 2), "goal_embed_0": torch.tensor([[[[8., 0.]]]]),
            "goal_action": torch.tensor([[[[2., 0.]]]]), "project": True,
            "cost_last_n": 2, "action_cost_weight": 0.2}
    # Projected states are 0,2,8; only the final pooled action (3,0) has a target.
    torch.testing.assert_close(cost.get_cost(info, actions), torch.tensor([[36.2]]))
