"""
Gymnasium-compatible wrapper for Jad environment.
Used with Stable-Baselines3 for parallelized training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from env import JadEnv, Observation, TerminationState
from observations import obs_to_array, OBS_DIM
from rewards import compute_reward


MAX_EPISODE_LENGTH = 300  # Hard cap to prevent stalling


class JadGymnasiumEnv(gym.Env):
    """
    Gymnasium wrapper for the Jad environment.

    This allows using Stable-Baselines3 and vectorized environments.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.env = JadEnv()

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9)

        # State tracking
        self.prev_obs: Observation | None = None
        self.current_obs: Observation | None = None
        self.episode_length = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_obs = self.env.reset()
        self.prev_obs = None
        self.episode_length = 0

        # Return raw observation - normalization handled by SelectiveVecNormalize wrapper
        obs_array = obs_to_array(self.current_obs)
        return obs_array, {}

    def step(self, action):
        self.prev_obs = self.current_obs

        # Convert numpy int to Python int for JSON serialization
        action = int(action)
        result = self.env.step(action)
        self.current_obs = result.observation
        self.episode_length += 1

        # Determine termination state
        truncated = False
        if result.terminated:
            if self.current_obs.jad_hp <= 0:
                termination = TerminationState.JAD_KILLED
            else:
                termination = TerminationState.PLAYER_DIED
        elif self.episode_length >= MAX_EPISODE_LENGTH:
            termination = TerminationState.TRUNCATED
            truncated = True
        else:
            termination = TerminationState.ONGOING

        # Compute reward (all reward logic is in compute_reward)
        # Reward normalization is handled by SelectiveVecNormalize wrapper
        reward = compute_reward(
            self.current_obs,
            self.prev_obs,
            termination,
            self.episode_length,
        )

        # Return raw observation - normalization handled by SelectiveVecNormalize wrapper
        obs_array = obs_to_array(self.current_obs)

        # Build info dict with outcome for tracking
        info = {"raw_reward": reward}
        if termination == TerminationState.JAD_KILLED:
            info["outcome"] = "kill"
        elif termination in (TerminationState.PLAYER_DIED, TerminationState.TRUNCATED):
            info["outcome"] = "death"

        return obs_array, reward, result.terminated, truncated, info

    def close(self):
        self.env.close()


def make_jad_env():
    """Factory function for creating JadGymnasiumEnv instances wrapped with Monitor."""
    def _init():
        env = JadGymnasiumEnv()
        # Monitor wrapper adds episode stats (r, l, t) to info dict
        return Monitor(env)
    return _init
