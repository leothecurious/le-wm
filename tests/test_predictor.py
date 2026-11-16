import types

import torch
from torch import nn

from metrics import predictor

HISTORY = 3


class Shift(nn.Module):
    """Predictor that adds ``step`` to the embedding at every position."""

    def __init__(self, step):
        super().__init__()
        self.step = step
        self.predictor = types.SimpleNamespace(pos_embedding=torch.zeros(1, HISTORY, 1))
        self.anchor = nn.Parameter(torch.zeros(()))

    def action_encoder(self, action):
        return action

    def predict(self, emb, act_emb):
        return emb + self.step


def latents():
    """Frame t of window n embeds as t, for every dimension. Frames after a window's
    valid prefix, and the probe-train windows, hold values that would dominate any
    error they leaked into."""
    n, t = 6, 9
    z = torch.arange(t).float().expand(n, t)[..., None].repeat(1, 1, 4)
    valid = torch.ones(n, t, dtype=torch.bool)
    valid[1, 5:] = False
    z[1, 5:] = 1e3
    train = torch.zeros(n, dtype=torch.bool)
    train[2:4] = True
    z[2:4] = 1e3
    return types.SimpleNamespace(z=z, action=torch.zeros(n, t, 2), valid=valid, train=train)


def test_next_frame_alignment_and_masking():
    lat = latents()
    copy = predictor.measure(Shift(0.0), lat)
    assert copy["ratio"] == 1
    # frames advance by 1, so the shifted predictor is exact and copying is off by 1
    ahead = predictor.measure(Shift(1.0), lat)
    assert ahead["pred_mse"] == 0 and ahead["copy_mse"] == 1
