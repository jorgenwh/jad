import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from jad_types import JadConfig
from utils import get_action_count
from env_process_wrapper import EnvProcessWrapper
from observations import obs_to_array, get_obs_dim


BASE_EPISODE_LENGTH = 300  # Per-jad episode length cap during training

TRUNCATION_PENALTIES = {
    "default": -150.0,
    "sparse": -1.0,
    "multijad": -150.0,
}


class JadGymnasiumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: JadConfig | None = None,
        reward_func: str = "default",
        training: bool = True
    ):
        super().__init__()

        self._config = config or JadConfig()
        self._reward_func = reward_func
        self._training = training

        # Scale episode length by jad count; no limit for evaluation
        self._max_episode_length = (
            BASE_EPISODE_LENGTH * self._config.jad_count if training else float('inf')
        )

        self.env = EnvProcessWrapper(config=self._config, reward_func=reward_func)

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
        self.current_obs = None
        self.episode_length = 0

    @property
    def config(self) -> JadConfig:
        return self._config

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_obs = self.env.reset()
        self.episode_length = 0

        # Return raw observation - normalization handled by SelectiveVecNormalize wrapper
        obs_array = obs_to_array(self.current_obs, self._config)
        return obs_array, {}

    def step(self, action):
        # Convert numpy int to Python int for JSON serialization
        action = int(action)
        result = self.env.step(action)
        self.current_obs = result.observation
        self.episode_length += 1

        # Use reward from TypeScript (source of truth)
        reward = result.reward

        # Check for truncation (training-only concept)
        truncated = False
        if not result.terminated and self.episode_length >= self._max_episode_length:
            truncated = True
            reward += TRUNCATION_PENALTIES.get(self._reward_func, -150.0)

        # Return raw observation - normalization handled by SelectiveVecNormalize wrapper
        obs_array = obs_to_array(self.current_obs, self._config)

        # Build info dict with outcome for tracking
        info = {"raw_reward": reward}
        if result.terminated:
            all_jads_dead = all(jad.hp <= 0 for jad in self.current_obs.jads)
            info["outcome"] = "kill" if all_jads_dead else "death"
        elif truncated:
            info["outcome"] = "death"

        return obs_array, reward, result.terminated, truncated, info

    def close(self):
        self.env.close()


def make_jad_env(config: JadConfig | None = None, reward_func: str = "default",
                 training: bool = True):
    """Factory function for creating JadGymnasiumEnv instances wrapped with Monitor"""
    def _init():
        env = JadGymnasiumEnv(config=config, reward_func=reward_func, training=training)
        # Monitor wrapper adds episode stats (r, l, t) to info dict
        return Monitor(env)
    return _init
