import os

import numpy as np
from stable_worldmodel.envs.pusht.env import PushT

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from eval import align_destinations  # noqa: E402


class Dataset:
    def __init__(self, cols):
        self.cols = cols

    def get_col_data(self, name):
        return self.cols[name]


class World:
    def __init__(self, envs):
        self.envs = type("Pool", (), {"envs": envs})()


def test_each_env_draws_its_episodes_destination_across_resets():
    poses = np.array([[100.0, 100.0, 0.0], [340.0, 210.0, -1.0], [150.0, 150.0, 2.0]])
    envs = [PushT(), PushT()]
    assert align_destinations(World(envs), Dataset({"goal_pose": poses}), rows=[2, 1])
    for env, pose in zip(envs, poses[[2, 1]]):
        for seed in (7, 8):
            env.reset(seed=seed)
            assert np.allclose(env.goal_pose, pose)

    fixed = PushT()
    assert not align_destinations(World([fixed]), Dataset({"state": np.zeros((4, 7))}), rows=[0])
    fixed.reset(seed=7)
    assert np.allclose(fixed.goal_pose, [256.0, 256.0, np.pi / 4])
