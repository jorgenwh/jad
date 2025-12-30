"""
Gymnasium-compatible wrapper for Jad environment.
Used with Stable-Baselines3 for parallelized training.
Supports 1-6 Jads with per-Jad healers.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from config import JadConfig, get_action_count
from env import JadEnv, Observation, TerminationState
from observations import obs_to_array, get_obs_dim, get_normalize_mask


MAX_EPISODE_LENGTH = 300  # Hard cap to prevent stalling


class JadGymnasiumEnv(gym.Env):
    """
    Gymnasium wrapper for the Jad environment.

    This allows using Stable-Baselines3 and vectorized environments.
    Supports configurable number of Jads (1-6).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: JadConfig | None = None, reward_type: str = "default"):
        super().__init__()

        self._config = config or JadConfig()
        self._reward_type = reward_type
        self.env = JadEnv(config=self._config)

        # Get dynamic dimensions
        obs_dim = get_obs_dim(self._config)
        action_count = get_action_count(self._config)

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_count)

        # State tracking
        self.prev_obs: Observation | None = None
        self.current_obs: Observation | None = None
        self.episode_length = 0

    @property
    def config(self) -> JadConfig:
        return self._config

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

        # Determine termination state - check if ALL Jads are dead
        truncated = False
        if result.terminated:
            # Check if all Jads are dead
            all_jads_dead = all(jad.hp <= 0 for jad in self.current_obs.jads)
            if all_jads_dead:
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
        from rewards import compute_reward
        reward = compute_reward(
            self.current_obs,
            self.prev_obs,
            termination,
            self.episode_length,
            reward_type=self._reward_type,
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


def make_jad_env(config: JadConfig | None = None, reward_type: str = "default"):
    """Factory function for creating JadGymnasiumEnv instances wrapped with Monitor."""
    def _init():
        env = JadGymnasiumEnv(config=config, reward_type=reward_type)
        # Monitor wrapper adds episode stats (r, l, t) to info dict
        return Monitor(env)
    return _init
