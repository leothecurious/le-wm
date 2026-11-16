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
    return pairs.measure(model, frames(), PixelTag("episode", 5, 0), SIZE)


class Episodes:
    """Dataset of episodes of the given lengths whose frames are all filled with the episode index."""

    def __init__(self, lengths):
        self.lengths = np.array(lengths)

    def load_chunk(self, episodes, starts, ends):
        assert (starts < self.lengths[episodes]).all()
        return [{"pixels": torch.full((1, 3, 4, 4), int(ep), dtype=torch.uint8)} for ep in episodes]


def test_sample_frames_takes_one_frame_per_episode():
    ds = Episodes([5, 9, 7, 6, 8])
    assert len(set(pairs.sample_frames(ds, 3, 0)[:, 0, 0, 0].tolist())) == 3
    assert sorted(pairs.sample_frames(ds, 99, 0)[:, 0, 0, 0].tolist()) == [0, 1, 2, 3, 4]


def test_partner_is_one_cycle_without_fixed_points():
    partner = pairs.cyclic_derangement(50, np.random.default_rng(0))
    visited, j = {0}, partner[0]
    while j != 0:
        visited.add(j)
        j = partner[j]
    assert len(visited) == 50


def test_pair_cosines_separate_tag_readers_from_scene_readers():
    tag_pair = measure(tag_reader)["backbone"]
    scene_pair = measure(scene_reader)["backbone"]

    assert tag_pair["same_tag"] > 0.9
    assert abs(tag_pair["same_content"]) < 0.2
    assert scene_pair["same_content"] > 0.99
    assert abs(scene_pair["same_tag"]) < 0.2
    assert abs(tag_pair["baseline"]) < 0.2 and abs(scene_pair["baseline"]) < 0.2


class Scenes:
    """Renders a frame with the destination in one patch and the block pose in another."""

    def render(self, state, dest_pose):
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        frame[8:16, 8:16] = (np.asarray(dest_pose) / [372, 372, 2 * np.pi] * 255).astype(np.uint8)
        frame[20:28, 8:16] = (np.asarray(state)[2:5] / [372, 372, 2 * np.pi] * 255).astype(np.uint8)
        return frame


dest_reader = Reader(lambda x: x[:, :, 8:16, 8:16].mean((2, 3)))
block_reader = Reader(lambda x: x[:, :, 20:28, 8:16].mean((2, 3)))


def measure_t_position(model):
    return pairs.measure_t_position(model, N, PixelTag("episode", 5, 0), SIZE, renderer=Scenes())


def test_t_position_separates_destination_readers_from_block_readers():
    dest_pair = measure_t_position(dest_reader)["backbone"]
    block_pair = measure_t_position(block_reader)["backbone"]

    assert dest_pair["same_tag"] > 0.99
    assert abs(dest_pair["same_content"]) < 0.2
    assert block_pair["same_content"] > 0.99
    assert abs(block_pair["same_tag"]) < 0.2
    assert abs(dest_pair["baseline"]) < 0.2 and abs(block_pair["baseline"]) < 0.2


def test_colour_probe_reads_the_corner_and_sits_at_chance_without_it():
    # 192 features, the width of the latent, so that OLS overfits the 320 train frames
    masked_reader = Reader(lambda x: F.avg_pool2d(x[:, :, 8:, 8:], 3).flatten(1))
    kinds = ("linear", "ridge")
    args = (frames(), PixelTag("episode", 5, 0), SIZE, kinds)
    on_corner = pairs.measure_colour(tag_reader, *args)["backbone"]
    masked = pairs.measure_colour(masked_reader, *args)["backbone"]

    assert on_corner["ridge"]["eps"] < 5
    assert abs(masked["ridge"]["eps"] - 100) < 10
    assert masked["linear"]["eps"] > masked["ridge"]["eps"]
