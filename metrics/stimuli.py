"""Freshly rendered Push-T scenes with a controllable block pose and destination outline.

Port of Enigma-Incorporated/The-Obsessed-Encoder@37819425
(``leworldmodel/additional_files/stimuli.py``), MIT licence, the parts the
t_position suite uses. Frames come from the simulator, never from a dataset.
"""

import os

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401  registers swm/PushT-v1

# p1..p99 of the pusht_expert_train state column [agent_xy].
AGENT_LO = np.array([38.0, 75.0])
AGENT_HI = np.array([449.0, 482.0])

# A tee whose centre lies in [140, 372]^2 stays inside the walls at any angle.
# Used for the block and for the destination outline.
TEE_LO = np.array([140.0, 140.0])
TEE_HI = np.array([372.0, 372.0])

RENDER_SEED = 0  # fixes the background, agent and block colours


class StimulusRenderer:
    """swm/PushT-v1, created on first use, rendering (state, destination) pairs."""

    def __init__(self):
        self._env = None

    def render(self, state, dest_pose):
        """(H, W, 3) uint8 frame with block and agent at ``state`` and the outline at ``dest_pose``."""
        if self._env is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            self._env = gym.make("swm/PushT-v1", render_mode="rgb_array")
        env = self._env.unwrapped
        env.reset(seed=RENDER_SEED, options={"state": np.asarray(state, float)})
        env.goal_pose = np.asarray(dest_pose, float)
        return np.ascontiguousarray(env.render())


def sample_orig(rng, k):
    """``k`` env states [agent_xy, block_xy, block_angle, velocity_xy = 0]."""
    agent = rng.uniform(AGENT_LO, AGENT_HI, size=(k, 2))
    block = rng.uniform(TEE_LO, TEE_HI, size=(k, 2))
    angle = rng.uniform(0.0, 2 * np.pi, size=(k, 1))
    return np.concatenate([agent, block, angle, np.zeros((k, 2))], axis=1)


def sample_dest(rng, k):
    """``k`` destination poses [x, y, angle]."""
    xy = rng.uniform(TEE_LO, TEE_HI, size=(k, 2))
    angle = rng.uniform(0.0, 2 * np.pi, size=(k, 1))
    return np.concatenate([xy, angle], axis=1)
