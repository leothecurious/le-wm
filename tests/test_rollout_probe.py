from types import SimpleNamespace

import numpy as np
import torch

from jepa import JEPA

from metrics import suite
from metrics.latents import Latents, extract


class PixelEncoder(torch.nn.Module):
    """Reads a position stored in a frame's channels; counts encoded frames."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.encoded_frames = []

    def forward(self, pixels, interpolate_pos_encoding):
        self.encoded_frames.append(len(pixels))
        return SimpleNamespace(last_hidden_state=pixels[:, None, :, 0, 0])


class Translate(torch.nn.Module):
    def forward(self, emb, act_emb):
        return emb + act_emb


def translation_model():
    return JEPA(PixelEncoder(), Translate(), torch.nn.Identity())


def test_first_frame_rollout_joint_position_rmse():
    # Train trajectories span two coordinates. Their time-dependent target
    # scales average to one, so only a shared probe learns the identity map.
    start = torch.tensor([[.2, .2], [.6, .2], [.2, .6], [.25, .25], [.75, .75]])
    pixels = torch.full((5, 3, 2, 1, 1), 99.)
    pixels[:, 0, :, 0, 0] = start
    action = torch.zeros(5, 3, 2)
    action[3:, 0] = torch.tensor([.1, 0.])
    action[3:, 1] = torch.tensor([0., .2])
    action[:, 2] = 1000.  # The action after the last frame has no target.
    valid = torch.ones(5, 3, dtype=torch.bool)
    valid[4, 2] = False
    batch = {"pixels": pixels, "action": action, "valid": valid}
    model = translation_model()
    extracted = extract(model, [batch], rollout=True)
    future_changed = {**batch, "pixels": batch["pixels"].clone()}
    future_changed["pixels"][:, 1:] = -7.
    torch.testing.assert_close(
        extract(translation_model(), [future_changed], rollout=True)["predicted_z"],
        extracted["predicted_z"],
    )
    expected = start[:, None].repeat(1, 3, 1)
    expected[3:, 1] += torch.tensor([.1, 0.])
    expected[3:, 2] += torch.tensor([.1, .2])
    torch.testing.assert_close(extracted["predicted_z"], expected)
    assert model.encoder.encoded_frames == [5, 15]  # first frames for the rollout, then every frame
    assert torch.all(extracted["z"][:, 1:] == 99.)
    assert set(batch) == {"pixels", "action", "valid"}

    target = expected.clone()
    target[:3] *= torch.tensor([.5, 1., 1.5])[None, :, None]
    target[3:, 1] += torch.tensor([.03, .04])
    target[3:, 2] += torch.tensor([.06, .08])
    target[4, 2] = 1000.  # Padded labels must not affect scores.
    lat = Latents(
        **extracted, target=target.flip(-1), episode=torch.arange(5),
        train=torch.tensor([True, True, True, False, False]),
        target_names=["dot_y", "dot_x"], target_angle=[False, False],
        target_group=["state", "state"],
    )
    result = suite.run(lat, [{"name": "rollout_probe"}])["rollout_probe"]
    assert result["units"] == "coordinate side length"
    assert result["n_train"] == 9
    assert result["n_test"] == 5
    assert result["n_test_per_step"] == [2, 2, 1]
    np.testing.assert_allclose(result["rmse_per_step"], [0., .05, .1], atol=1e-6)
    np.testing.assert_allclose(result["mean_rmse"], .05, atol=1e-6)
    np.testing.assert_allclose(result["pooled_rmse"], np.sqrt(.003), atol=1e-6)
    center_steps = np.sqrt([.125, .1435, .009])
    np.testing.assert_allclose(result["center_control"]["rmse_per_step"], center_steps, atol=1e-6)
    np.testing.assert_allclose(result["center_control"]["mean_rmse"], center_steps.mean(), atol=1e-6)
    np.testing.assert_allclose(result["center_control"]["pooled_rmse"], np.sqrt(.1092), atol=1e-6)
    np.testing.assert_allclose(result["r"], .05 / center_steps.mean(), atol=1e-6)
