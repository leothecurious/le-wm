import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import hydra
import pytest
import stable_pretraining  # registers the eval resolver the configs use
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import open_dict

import hjepa_planner
import module
import train
from scripts import convert_hjepa

ROOT = Path(__file__).parent.parent
RELEASED = Path(
    os.environ.get("HJEPA_CKPT", ROOT / "cache/hjepa_hf/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt")
)

spec = importlib.util.spec_from_file_location("hjepa_reference", Path(__file__).parent / "reference/hjepa_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def compose(group, name, *overrides):
    with hydra.initialize_config_dir(str(ROOT / "config" / group), version_base=None):
        return hydra.compose(name, overrides=list(overrides))


def train_cfg(*overrides):
    cfg = compose("train", "lewm", "data=pusht", *overrides)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * 2
    return cfg


def randomise(model):
    """Nonzero weights everywhere, so no branch of the model is silent (adaLN is zero at init)."""
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05 + (1.0 if p.ndim == 1 else 0.0))
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_(0, 0.1, generator=gen)
                m.running_var.uniform_(0.5, 1.5, generator=gen)
        model.upper.latent_action_queue = torch.randn(4000, 4, generator=gen)
    return model.eval()


def to_reference(state, mask_token):
    """Names of the released ``HJEPA.jepas`` for a ``hjepa.HJEPA`` state dict."""
    out = {"jepas.0.encoder.encoder.embeddings.mask_token": mask_token}
    for key, value in state.items():
        old = next(old + key[len(new) :] for old, new in convert_hjepa._PREFIXES if key.startswith(new))
        out[old] = value
    return out


@pytest.fixture(scope="module")
def pair():
    """The port and the reference with the same random weights, both in eval mode."""
    ours = randomise(instantiate(train_cfg("model=hjepa").model))
    ref = reference.build_pusht_hjepa_l2().eval()
    mask = ref.state_dict()["jepas.0.encoder.encoder.embeddings.mask_token"]
    ref.load_state_dict(to_reference(ours.state_dict(), mask))
    return ours, ref


def batch(B=2, T=7):
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, T, 3, 224, 224, generator=g), torch.randn(B, T, 10, generator=g)


def reference_encode(ref, pixels, action, seed):
    torch.manual_seed(seed)
    return ref.encode_hierarchical({"pixels_level1": pixels, "action_level1": action})


def assert_encode_matches(ours, ref):
    pixels, action = batch()
    torch.manual_seed(0)
    info = ours.encode({"pixels": pixels, "action": action})
    out = reference_encode(ref, pixels, action, 0)
    for mine, theirs in [("emb", "embed_1"), ("act_emb", "action_1"), ("upper_emb", "embed_2"), ("upper_act_emb", "action_2")]:
        assert info[mine].shape == out[theirs].shape == (2, 4, 192), mine
        torch.testing.assert_close(info[mine], out[theirs], atol=1e-4, rtol=1e-4)


def test_encode_matches_reference(pair):
    assert_encode_matches(*pair)


def test_losses_match_reference(pair):
    ours, ref = pair
    cfg = train_cfg("model=hjepa")
    pixels, action = batch()

    torch.manual_seed(2)
    sigreg = module.SIGReg(**cfg.loss.sigreg.kwargs)
    stub = SimpleNamespace(model=ours, sigreg=sigreg, log_dict=lambda *a, **k: None)
    out = train.lejepa_forward(stub, {"pixels": pixels, "action": action}, "val", cfg)

    enc = reference_encode(ref, pixels, action, 2)
    sig = reference.SIGReg(**cfg.loss.sigreg.kwargs)
    expected = {}
    for level in (1, 2):
        emb, act = enc[f"embed_{level}"], enc[f"action_{level}"]
        pred = ref.get_level(level).predict(emb[:, :3], act[:, :3])
        expected[f"pred{level}"] = F.mse_loss(pred, emb[:, 1:4])
        expected[f"sigreg{level}"] = sig(emb.transpose(0, 1))
    total = sum(expected[f"pred{i}"] + 0.09 * expected[f"sigreg{i}"] for i in (1, 2))

    for mine, theirs in [("pred_loss", "pred1"), ("sigreg_loss", "sigreg1"), ("pred_loss_upper", "pred2"), ("sigreg_loss_upper", "sigreg2"), ("loss", None)]:
        want = total if theirs is None else expected[theirs]
        torch.testing.assert_close(out[mine], want, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not RELEASED.exists(), reason="released H-JEPA Push-T checkpoint not downloaded")
def test_released_checkpoint_matches_reference():
    ours = instantiate(train_cfg("model=hjepa").model)
    state = convert_hjepa.load_released(RELEASED)
    ours.load_state_dict(state)
    ours.eval()
    ref = reference.build_pusht_hjepa_l2().eval()
    mask = torch.zeros(1, 1, 192)
    ref.load_state_dict(to_reference(ours.state_dict(), mask))
    assert ours.upper.latent_action_queue.shape == (4000, 4)
    assert_encode_matches(ours, ref)

    pixels, action = batch(B=2)
    torch.manual_seed(0)
    info = ours.encode({"pixels": pixels, "action": action})
    pred = ours.predict(info["emb"][:, :3], info["act_emb"][:, :3])
    theirs = reference_encode(ref, pixels, action, 0)
    ref_pred = ref.get_level(1).predict(theirs["embed_1"][:, :3], theirs["action_1"][:, :3])
    torch.testing.assert_close(pred, ref_pred, atol=1e-4, rtol=1e-4)


def level_cost_inputs(level, B=2, S=3):
    g = torch.Generator().manual_seed(3)
    horizon, dim = (2, 10) if level == 1 else (4, 4)
    # one observation and goal per environment, shared by its S candidates as in the solver
    return (
        torch.randn(B, 1, 1, 192, generator=g).expand(B, S, 1, 192),
        torch.randn(B, 1, 1, 192, generator=g).expand(B, S, 1, 192),
        torch.randn(B, 1, 1, 4, generator=g).expand(B, S, 1, 4),
        torch.randn(B, S, horizon, dim, generator=g),
    )


def test_level_2_cost_matches_reference(pair):
    ours, ref = pair
    emb, goal, _, actions = level_cost_inputs(2)
    info = {"pixels": torch.zeros(2, 3, 1, 1), "embed_0": emb, "goal_embed_0": goal, "cost_last_n": 1}
    want = reference.LevelCostModel(model=ref.get_level(2), upper_models=[]).get_cost(dict(info), actions)
    got = hjepa_planner.LevelCostModel(ours.upper).get_cost({**info, "project": False, "action_cost_weight": 0.0}, actions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_level_1_cost_matches_reference_on_upper_goal_and_on_own_goal(pair):
    ours, ref = pair
    emb, goal, goal_action, actions = level_cost_inputs(1)
    info = {"pixels": torch.zeros(2, 3, 1, 1), "embed_0": emb, "goal_embed_0": goal, "goal_action": goal_action, "cost_last_n": 2}
    ref_cost = reference.LevelCostModel(model=ref.get_level(1), upper_models=[ref.get_level(2)])
    cost = hjepa_planner.LevelCostModel(ours, ours.upper)

    up = {"goal_embed_0_in_upper_space": True, "action_cost_weight": 0.2}
    want = ref_cost.get_cost({**info, **up}, actions)
    got = cost.get_cost({**info, "project": True, "action_cost_weight": 0.2}, actions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

    own = {"goal_embed_0_in_upper_space": False, "action_cost_weight": 0.2}
    want = ref_cost.get_cost({**info, **own}, actions[:, :, :1])
    got = cost.get_cost({**info, "project": False, "action_cost_weight": 0.0}, actions[:, :, :1])
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_horizon_schedule_matches_reference():
    for calls in range(1, 14):
        for horizon in range(1, 10):
            assert hjepa_planner.decreasing_horizon_schedule(calls, horizon) == reference.decreasing_horizon_schedule(
                num_calls=calls, configured_horizon=horizon
            )


def eval_cfg(*overrides):
    return compose("eval", "pusht_hjepa", "policy=run/weights.pt", *overrides)


def test_planner_runs_every_call_of_a_budget_on_the_model(pair):
    ours, _ = pair
    cfg = eval_cfg(
        "eval.eval_budget=90", "solver.solvers.level1.device=cpu", "solver.solvers.level2.device=cpu",
        "solver.solvers.level1.num_samples=3", "solver.solvers.level2.num_samples=3",
        "solver.solvers.level1.n_steps=2", "solver.solvers.level2.n_steps=2",
    )
    solver = instantiate(cfg.solver, model=ours)
    plan = instantiate(cfg.plan_config)
    solver.configure(action_space=gym.spaces.Box(-1, 1, (2, 2)), n_envs=2, config=plan)
    g = torch.Generator().manual_seed(4)
    info = {"pixels": torch.randn(2, 1, 3, 224, 224, generator=g), "goal": torch.randn(2, 1, 3, 224, 224, generator=g)}

    kept = []
    while solver.steps_taken < cfg.eval.eval_budget:
        actions = solver(info)["actions"]
        kept.append((actions.shape[1], plan.receding_horizon, plan.receding_horizon * 5))
        assert actions.shape == (2, kept[-1][0], 10)
    assert [k[1] for k in kept] == [2] * 8 + [1, 1]
    assert [k[0] for k in kept] == [2] * 8 + [1, 1]


def test_training_step_updates_both_levels_and_the_action_queue():
    cfg = train_cfg("model=hjepa")
    model = instantiate(cfg.model).train()
    pixels, action = batch(B=2)
    stub = SimpleNamespace(model=model, sigreg=module.SIGReg(**cfg.loss.sigreg.kwargs), log_dict=lambda *a, **k: None)
    out = train.lejepa_forward(stub, {"pixels": pixels, "action": action}, "train", cfg)
    out["loss"].backward()
    assert model.projector.net[0].weight.grad.abs().sum() > 0
    assert model.upper.encoder.fc1.weight.grad.abs().sum() > 0
    assert model.upper.latent_action_queue.shape == (2, 4)
    for name in ("pred_loss", "sigreg_loss", "pred_loss_upper", "sigreg_loss_upper", "loss"):
        assert torch.isfinite(out[name])


def test_vit_checkpointing_changes_memory_not_the_maths():
    cfg = train_cfg("model=hjepa")
    assert cfg.model.checkpoint_vit is True
    with open_dict(cfg):
        cfg.model.checkpoint_vit = False
    plain = instantiate(cfg.model)
    assert plain.encoder.config.hidden_dropout_prob == plain.encoder.config.attention_probs_dropout_prob == 0
    with open_dict(cfg):
        cfg.model.checkpoint_vit = True
    saved = instantiate(cfg.model)
    saved.load_state_dict(plain.state_dict())

    layer_calls = []
    layer = saved.encoder.encoder.layer[0]
    layer_forward = layer.forward
    layer.forward = lambda *a, **k: (layer_calls.append(1), layer_forward(*a, **k))[1]
    pixels, action = batch()
    results = []
    for model in (plain, saved):
        model.train()
        torch.manual_seed(5)
        stub = SimpleNamespace(model=model, sigreg=module.SIGReg(**cfg.loss.sigreg.kwargs), log_dict=lambda *a, **k: None)
        out = train.lejepa_forward(stub, {"pixels": pixels, "action": action}, "train", cfg)
        out["loss"].backward()
        results.append((out["loss"].detach(), {n: p.grad for n, p in model.named_parameters()}))
    assert len(layer_calls) == 2  # forward, then again in the backward pass
    (loss, grads), (loss_saved, grads_saved) = results
    torch.testing.assert_close(loss_saved, loss, atol=1e-6, rtol=1e-5)
    for name, grad in grads.items():
        torch.testing.assert_close(grads_saved[name], grad, atol=1e-5, rtol=1e-4)
