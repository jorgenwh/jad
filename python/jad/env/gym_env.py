import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from jad.types import JadConfig
from jad.actions import get_action_dims
from jad.env.process_wrapper import EnvProcessWrapper
from jad.env.observations import obs_to_array, get_observation_dim


BASE_EPISODE_LENGTH = 300  # Per-jad episode length cap during training

TRUNCATION_PENALTIES = {
    "sparse": -1.0,
    "jad1": -150.0,
    "jad2": -50.0,
    "jad3": -50.0,
}


class JadGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: JadConfig | None = None,
        *,
        reward_func: str,
    ):
        super().__init__()

        self._config = config or JadConfig()
        self._reward_func = reward_func
        self._max_episode_length = BASE_EPISODE_LENGTH * self._config.jad_count

        self.env = EnvProcessWrapper(config=self._config, reward_func=reward_func)

        obs_dim = get_observation_dim(self._config)
        self._action_dims = get_action_dims(self._config)

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(self._action_dims)

        self.episode_length = 0
        # Per-head action masks
        self._current_action_masks: tuple[np.ndarray, ...] = tuple(
            np.ones(dim, dtype=np.bool_) for dim in self._action_dims
        )

    @property
    def config(self) -> JadConfig:
        return self._config

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.env.reset()
        self.episode_length = 0

        if result.valid_action_mask:
            self._current_action_masks = tuple(
                np.array(mask, dtype=np.bool_) for mask in result.valid_action_mask
            )

        obs_array = obs_to_array(result.observation, self._config)
        return obs_array, {}

    def step(self, action):
        # Convert numpy array to list for JSON serialization
        action_list = [int(a) for a in action]
        result = self.env.step(action_list)
        self.episode_length += 1

        reward = result.reward
        obs = result.observation

        if result.valid_action_mask:
            self._current_action_masks = tuple(
                np.array(mask, dtype=np.bool_) for mask in result.valid_action_mask
            )

        # Check for truncation (training-only concept)
        truncated = False
        if not result.terminated and self.episode_length >= self._max_episode_length:
            truncated = True
            reward += TRUNCATION_PENALTIES.get(self._reward_func, -150.0)

        obs_array = obs_to_array(obs, self._config)

        info = {"raw_reward": reward}
        if result.terminated or truncated:
            jads_killed = sum(1 for jad in obs.jads if jad.hp <= 0)
            all_jads_dead = jads_killed == len(obs.jads)
            info["outcome"] = "kill" if all_jads_dead else "death"
            info["jads_killed"] = jads_killed
            info["jad_count"] = len(obs.jads)

        return obs_array, reward, result.terminated, truncated, info

    def action_masks(self) -> tuple[np.ndarray, ...]:
        """Return per-head action masks for MultiDiscrete masking."""
        return self._current_action_masks

    def close(self):
        self.env.close()


def make_jad_env(config: JadConfig | None = None, *, reward_func: str):
    """Factory function for creating JadGymEnv instances wrapped with Monitor."""
    def _init():
        env = JadGymEnv(config=config, reward_func=reward_func)
        return Monitor(env)
    return _init
