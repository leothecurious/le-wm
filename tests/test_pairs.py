import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metrics import pairs
from tag import PixelTag

N, SIZE = 400, 32


class Reader(nn.Module):
    """Encoder whose CLS token is ``features(frames)``; the projector is the identity."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.projector = nn.Identity()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encoder(self, x, interpolate_pos_encoding=False):
        return types.SimpleNamespace(last_hidden_state=self.features(x)[:, None])


# sees only the 5x5 corner, where the tag is stamped
tag_reader = Reader(lambda x: x[:, :, :5, :5].mean((2, 3)))
# sees only the area away from the corner
scene_reader = Reader(lambda x: F.avg_pool2d(x[:, :, 8:, 8:], 6).flatten(1))


def frames():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (N, 3, SIZE, SIZE), dtype=torch.uint8, generator=g)


def measure(model):
    return pairs.measure(model, frames(), PixelTag("episode", 5, 0), SIZE, kinds=("linear",))


def test_partner_is_one_cycle_without_fixed_points():
    partner = pairs.cyclic_derangement(50, np.random.default_rng(0))
    visited, j = {0}, partner[0]
    while j != 0:
        visited.add(j)
        j = partner[j]
    assert len(visited) == 50


def test_pair_cosines_separate_tag_readers_from_scene_readers():
    tag_pair = measure(tag_reader)["pair"]["backbone"]
    scene_pair = measure(scene_reader)["pair"]["backbone"]

    assert tag_pair["same_tag"] > 0.9
    assert abs(tag_pair["same_content"]) < 0.2
    assert scene_pair["same_content"] > 0.99
    assert abs(scene_pair["same_tag"]) < 0.2
    assert abs(tag_pair["baseline"]) < 0.2 and abs(scene_pair["baseline"]) < 0.2


def test_tag_probe_decodes_colour_only_from_the_tag_reader():
    on_tag = measure(tag_reader)["tag_probe"]["backbone"]["linear"]["eps"]
    on_scene = measure(scene_reader)["tag_probe"]["backbone"]["linear"]["eps"]

    assert on_tag < 20
    assert on_scene > 80
