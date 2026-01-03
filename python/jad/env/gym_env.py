import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from jad.types import JadConfig
from jad.actions import get_action_count
from jad.env.process_wrapper import EnvProcessWrapper
from jad.env.observations import obs_to_array, get_observation_dim


BASE_EPISODE_LENGTH = 300  # Per-jad episode length cap during training

TRUNCATION_PENALTIES = {
    "default": -150.0,
    "sparse": -1.0,
    "multijad": -150.0,
}


class JadGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: JadConfig | None = None,
        reward_func: str = "default",
    ):
        super().__init__()

        self._config = config or JadConfig()
        self._reward_func = reward_func
        self._max_episode_length = BASE_EPISODE_LENGTH * self._config.jad_count

        self.env = EnvProcessWrapper(config=self._config, reward_func=reward_func)

        obs_dim = get_observation_dim(self._config)
        action_count = get_action_count(self._config)

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_count)

        self.episode_length = 0

    @property
    def config(self) -> JadConfig:
        return self._config

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = self.env.reset()
        self.episode_length = 0

        obs_array = obs_to_array(obs, self._config)
        return obs_array, {}

    def step(self, action):
        action = int(action)
        result = self.env.step(action)
        self.episode_length += 1

        reward = result.reward
        obs = result.observation

        # Check for truncation (training-only concept)
        truncated = False
        if not result.terminated and self.episode_length >= self._max_episode_length:
            truncated = True
            reward += TRUNCATION_PENALTIES.get(self._reward_func, -150.0)

        obs_array = obs_to_array(obs, self._config)

        info = {"raw_reward": reward}
        if result.terminated:
            all_jads_dead = all(jad.hp <= 0 for jad in obs.jads)
            info["outcome"] = "kill" if all_jads_dead else "death"
        elif truncated:
            info["outcome"] = "death"

        return obs_array, reward, result.terminated, truncated, info

    def close(self):
        self.env.close()


def make_jad_env(config: JadConfig | None = None, reward_func: str = "default"):
    """Factory function for creating JadGymEnv instances wrapped with Monitor."""
    def _init():
        env = JadGymEnv(config=config, reward_func=reward_func)
        return Monitor(env)
    return _init
