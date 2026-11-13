import numpy as np
import pytest
import torch

from tag import EvalTag, PixelTag


def enigma_color(seed, *key):
    """The colour hash of Enigma-Incorporated/The-Obsessed-Encoder@37819425,
    additional_files/pixel_tag.py, kept as the reference to match."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, *key]))
    return torch.from_numpy(rng.integers(0, 256, size=3, dtype=np.uint8))


class Clips:
    """Episode dataset with the swm interface the tag relies on."""

    frameskip, num_steps, transform = 2, 3, None
    span = frameskip * num_steps
    clip_indices = [(0, 0), (0, 1), (7, 4)]

    def __len__(self):
        return len(self.clip_indices)

    def _load_slice(self, episode, start, end):
        n = (end - start) // self.frameskip
        return {
            "pixels": torch.zeros(n, 3, 16, 16, dtype=torch.uint8),
            "action": torch.zeros(end - start, 2),
        }


def corner(pixels, n=5):
    return pixels[:, :, :n, :n]


def test_colors_match_enigma_reference():
    tag = PixelTag("episode", seed=3)
    for episode in (0, 1, 7, 18684):
        torch.testing.assert_close(tag.color(episode), enigma_color(3, episode))
        torch.testing.assert_close(tag.color(episode, 40), enigma_color(3, episode, 40))


def test_episode_mode_is_constant_within_and_distinct_across_episodes():
    ds = PixelTag("episode").wrap(Clips())
    a, b, c = ds[0]["pixels"], ds[1]["pixels"], ds[2]["pixels"]

    assert (corner(a) == corner(a)[:, :, :1, :1]).all()
    assert (a[0, :, 0, 0] == a[2, :, 0, 0]).all()
    assert (a[0, :, 0, 0] == b[0, :, 0, 0]).all()
    assert (a[0, :, 0, 0] != c[0, :, 0, 0]).any()


def test_frame_mode_changes_every_frame_and_follows_the_row_in_the_episode():
    ds = PixelTag("frame", seed=1).wrap(Clips())
    a, b = ds[0]["pixels"], ds[1]["pixels"]

    colors = {tuple(a[t, :, 0, 0].tolist()) for t in range(3)}
    assert len(colors) == 3
    # frame t of the clip starting at row 1 is row 1 + 2t; row 2 of clip 0 is its frame 1
    assert (b[0, :, 0, 0] != a[1, :, 0, 0]).any()
    expected = enigma_color(1, 7, 4 + 2)
    torch.testing.assert_close(ds[2]["pixels"][1, :, 0, 0], expected)


def test_only_the_corner_square_changes():
    pixels = PixelTag("episode", size=5).wrap(Clips())[0]["pixels"]

    outside = pixels.clone()
    outside[:, :, :5, :5] = 0
    assert not outside.any()
    assert corner(pixels).any()


def test_off_returns_the_dataset_unchanged():
    ds = Clips()
    assert PixelTag("off").wrap(ds) is ds


def test_transform_sees_stamped_frames_and_load_chunk_matches_getitem():
    ds = Clips()
    ds.transform = lambda steps: {**steps, "seen": corner(steps["pixels"]).any()}
    tagged = PixelTag("episode").wrap(ds)

    chunk = tagged.load_chunk([7], [4], [10])[0]
    assert chunk["seen"]
    assert chunk["action"].shape == (3, 4)
    torch.testing.assert_close(chunk["pixels"], tagged[2]["pixels"])


def test_eval_tag_uses_one_colour_in_episode_mode_and_new_ones_in_frame_mode():
    frame = torch.zeros(3, 16, 16, dtype=torch.uint8)
    episode = EvalTag(PixelTag("episode"), lambda x: x)
    per_frame = EvalTag(PixelTag("frame"), lambda x: x)

    first, second = episode(frame), episode(frame)
    assert (first[:, 0, 0] == second[:, 0, 0]).all()
    torch.testing.assert_close(first[:, 0, 0], enigma_color(0, 0))
    assert (per_frame(frame)[:, 0, 0] != per_frame(frame)[:, 0, 0]).any()
    assert not frame.any()


def test_unknown_mode_is_rejected():
    with pytest.raises(AssertionError):
        PixelTag("video")
