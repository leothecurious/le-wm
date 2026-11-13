import types

import torch
from torch import nn

from jepa import JEPA
from metrics.latents import Latents, extract

B, T, D = 64, 10, 4


class FlatEncoder(nn.Module):
    """CLS token = the flattened frame, so a frame is its own latent."""

    def forward(self, x, interpolate_pos_encoding=False):
        return types.SimpleNamespace(last_hidden_state=x.flatten(1)[:, None])


def stub_model():
    model = JEPA(FlatEncoder(), nn.Identity(), nn.Identity())
    model.register_parameter("anchor", nn.Parameter(torch.zeros(())))
    return model.eval()


def test_extract_encodes_every_frame_and_keeps_labels():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(B, T, D, generator=g)
    action = torch.randn(B, T, D, generator=g)
    valid = torch.arange(T)[None] < torch.randint(2, T + 1, (B, 1), generator=g)
    batch = {"pixels": z[..., None, None], "action": action, "valid": valid}

    out = extract(stub_model(), [batch])
    lat = Latents(
        **out,
        target=torch.zeros(B, T, 1, dtype=torch.float64),
        episode=torch.arange(B),
        train=torch.arange(B) < B // 2,
        target_names=["x"],
        target_angle=[False],
        target_group=["state"],
    )

    torch.testing.assert_close(lat.z, z)
    torch.testing.assert_close(lat.action, action)
    assert torch.equal(lat.valid, valid)
