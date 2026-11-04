import types

import numpy as np
import torch
from torch import nn

from jepa import JEPA
from metrics.latents import Latents, extract
from metrics.suite import rollout_consistency

B, T, D, CONTEXT = 64, 10, 4, 3


class FlatEncoder(nn.Module):
    """CLS token = the flattened frame, so a frame is its own latent."""

    def forward(self, x, interpolate_pos_encoding=False):
        return types.SimpleNamespace(last_hidden_state=x.flatten(1)[:, None])


class AddAction(nn.Module):
    """Exact dynamics z_{t+1} = z_t + a_t at every position."""

    def forward(self, emb, act_emb):
        return emb + act_emb


def stub_model():
    model = JEPA(FlatEncoder(), AddAction(), nn.Identity())
    model.register_parameter("anchor", nn.Parameter(torch.zeros(())))
    return model.eval()


def trajectories(n_valid):
    g = torch.Generator().manual_seed(0)
    action = torch.randn(B, T, D, generator=g)
    z = torch.randn(B, 1, D, generator=g) + torch.cat(
        [torch.zeros(B, 1, D), action[:, :-1].cumsum(1)], 1
    )
    valid = torch.arange(T)[None] < n_valid[:, None]
    noise = torch.randn(B, T, D, generator=g) * 100
    z = torch.where(valid[..., None], z, noise)
    action = torch.where(valid[..., None], action, torch.zeros_like(action))
    return z, action, valid


def latents(z, action, valid, model_action=None):
    batch = {
        "pixels": z[..., None, None],
        "action": action if model_action is None else model_action,
        "valid": valid,
    }
    out = extract(stub_model(), [batch], CONTEXT, history_size=3)
    return Latents(
        **out,
        target=torch.zeros(B, T, 1, dtype=torch.float64),
        episode=torch.arange(B),
        train=torch.arange(B) < B // 2,
        target_names=["x"],
        target_angle=[False],
        target_group=["state"],
        context=CONTEXT,
        history_size=3,
    )


def test_rollout_consistency_is_exact_for_exact_dynamics():
    z, action, valid = trajectories(torch.full((B,), T))
    res = rollout_consistency(latents(z, action, valid))

    assert res["k"] == list(range(1, T - CONTEXT + 1))
    np.testing.assert_allclose(res["lrc"], 1.0, atol=1e-5)
    assert max(res["persistence"]) < 0.9

    other_episode = action[torch.roll(torch.arange(B), 1)]
    res = rollout_consistency(latents(z, action, valid, model_action=other_episode))
    assert res["lrc"][-1] < 0.5


def test_padded_frames_are_masked():
    n_valid = torch.randint(CONTEXT + 1, T + 1, (B,), generator=torch.Generator().manual_seed(1))
    z, action, valid = trajectories(n_valid)
    res = rollout_consistency(latents(z, action, valid))

    expected_n = [(n_valid >= CONTEXT + k).sum().item() for k in res["k"]]
    assert res["n"] == expected_n
    np.testing.assert_allclose(res["lrc"], 1.0, atol=1e-5)
