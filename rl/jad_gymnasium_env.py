"""
Gymnasium-compatible wrapper for Jad environment.
Used with Stable-Baselines3 for parallelized training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from env import JadEnv, Observation
from observations import obs_to_array, OBS_DIM, NORMALIZE_MASK
from normalizer import RunningNormalizer


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

        # Observation normalizer (only for continuous features, like train.py)
        n_continuous = int(np.sum(NORMALIZE_MASK))
        self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))
        self.normalize_mask = NORMALIZE_MASK

    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalize only continuous features, leaving one-hot features unchanged.
        Matches train.py's normalization approach.
        """
        result = obs.copy()
        continuous = obs[self.normalize_mask]

        if update:
            self.obs_normalizer.update(continuous)

        normalized_continuous = self.obs_normalizer.normalize(continuous)
        result[self.normalize_mask] = normalized_continuous
        return result

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_obs = self.env.reset()
        self.prev_obs = None
        self.episode_length = 0

        obs_array = obs_to_array(self.current_obs)
        obs_array = self.normalize_obs(obs_array)
        return obs_array, {}

    def step(self, action):
        self.prev_obs = self.current_obs

        # Convert numpy int to Python int for JSON serialization
        action = int(action)
        result = self.env.step(action)
        self.current_obs = result.observation
        self.episode_length += 1

        # Compute reward
        reward = self._compute_reward(
            self.current_obs,
            self.prev_obs,
            result.terminated,
            self.episode_length,
        )

        # Check for timeout
        truncated = False
        if self.episode_length >= MAX_EPISODE_LENGTH and not result.terminated:
            truncated = True
            reward -= 100.0  # Timeout penalty

        # Scale reward to stabilize training (matches train.py)
        reward = reward / 100.0

        obs_array = obs_to_array(self.current_obs)
        obs_array = self.normalize_obs(obs_array)

        # Build info dict with outcome for tracking
        info = {}
        if result.terminated or truncated:
            if self.current_obs.jad_hp <= 0:
                info["outcome"] = "kill"
            else:
                info["outcome"] = "death"

        return obs_array, reward, result.terminated, truncated, info

    def _compute_reward(
        self,
        obs: Observation,
        prev_obs: Observation | None,
        terminated: bool,
        episode_length: int,
    ) -> float:
        """
        Reward function matching the original train.py implementation.
        """
        reward = 0.0

        # Prayer switching feedback
        if prev_obs is not None and prev_obs.jad_attack != 0:
            if obs.active_prayer == prev_obs.jad_attack:
                reward += 1
            else:
                reward -= 1

        # Survival bonus
        reward += 0.1

        # Penalty for not attacking Jad
        if not obs.player_aggro:
            reward -= 1

        if prev_obs is not None:
            # Damage taken penalty
            damage_taken = prev_obs.player_hp - obs.player_hp
            if damage_taken > 0:
                reward -= damage_taken * 0.1

            # Damage dealt reward
            damage_dealt = prev_obs.jad_hp - obs.jad_hp
            if damage_dealt > 0:
                reward += damage_dealt * 0.1

        # Terminal rewards
        if terminated:
            if obs.player_hp <= 0:
                reward -= 100.0  # Death
            elif obs.jad_hp <= 0:
                reward += 100.0  # Win
                reward -= episode_length * 0.25  # Faster kills are better

        return reward

    def close(self):
        self.env.close()


def make_jad_env():
    """Factory function for creating JadGymnasiumEnv instances wrapped with Monitor."""
    def _init():
        env = JadGymnasiumEnv()
        # Monitor wrapper adds episode stats (r, l, t) to info dict
        return Monitor(env)
    return _init
