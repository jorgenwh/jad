import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from jad.types import JadConfig
from jad.env.observations import get_normalize_mask, NORMALIZE_MASK


class RunningNormalizer:
    """
    Maintains running mean and variance for online normalization.
    Uses Welford's algorithm for numerically stable online computation.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new observation(s)."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape == self.shape:
            x = x.reshape(1, *self.shape)

        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        # Parallel Welford algorithm
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        return (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + self.epsilon)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class SelectiveVecNormalize(VecEnvWrapper):
    """
    Vectorized environment wrapper that normalizes observations and rewards.

    Observation normalization:
    - Uses a shared RunningNormalizer across all parallel environments
    - Only normalizes continuous features, leaving one-hot encodings unchanged

    Reward normalization (VecNormalize-style):
    - Tracks rolling discounted returns per environment
    - Normalizes rewards by the std of returns (not mean-centered)
    - Resets returns at episode boundaries
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        config: JadConfig | None = None,
    ):
        """
        Args:
            venv: Vectorized environment to wrap
            training: Whether to update running statistics
            norm_obs: Whether to normalize observations
            norm_reward: Whether to normalize rewards
            clip_obs: Max absolute value for normalized observations
            clip_reward: Max absolute value for normalized rewards
            gamma: Discount factor for return calculation
            config: JadConfig for determining normalize mask (default: 1 Jad, 3 healers)
        """
        super().__init__(venv)

        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma

        # Get normalize mask for this config
        if config is not None:
            self.normalize_mask = get_normalize_mask(config)
        else:
            self.normalize_mask = NORMALIZE_MASK

        # Observation normalizer for continuous features
        n_continuous = int(np.sum(self.normalize_mask))
        self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))

        # Reward normalizer: tracks std of discounted returns
        self.ret_normalizer = RunningNormalizer(shape=())
        self.returns = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self):
        """Reset all environments and normalize observations."""
        obs = self.venv.reset()
        # Reset returns for all environments
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        return self._normalize_obs(obs)

    def step_wait(self):
        """Step all environments, normalize observations and rewards."""
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._normalize_obs(obs)
        rewards = self._normalize_reward(rewards, dones)
        return obs, rewards, dones, infos

    def _normalize_reward(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using running std of discounted returns.

        Args:
            rewards: Raw rewards from all envs, shape (n_envs,)
            dones: Done flags from all envs, shape (n_envs,)

        Returns:
            Normalized rewards with same shape
        """
        if not self.norm_reward:
            return rewards

        # Update rolling discounted returns
        self.returns = self.returns * self.gamma + rewards

        # Update running statistics with current returns
        if self.training:
            self.ret_normalizer.update(self.returns)

        # Normalize by std of returns (NOT mean-centered)
        # This is the key difference from observation normalization
        normalized = rewards / np.sqrt(self.ret_normalizer.var + 1e-8)
        normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)

        # Reset returns for finished episodes
        self.returns[dones] = 0

        return normalized.astype(np.float32)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize only continuous features of observations.

        Args:
            obs: Batch of observations from all envs, shape (n_envs, obs_dim)

        Returns:
            Normalized observations with same shape
        """
        if not self.norm_obs:
            return obs

        # Extract continuous features from all envs
        continuous = obs[:, self.normalize_mask]  # (n_envs, n_continuous)

        # Update running stats with batch from all envs
        if self.training:
            self.obs_normalizer.update(continuous)

        # Normalize continuous features
        normalized_continuous = self.obs_normalizer.normalize(continuous)

        # Clip to prevent extreme values
        normalized_continuous = np.clip(
            normalized_continuous, -self.clip_obs, self.clip_obs
        )

        # Reconstruct full observation with normalized continuous features
        result = obs.copy()
        result[:, self.normalize_mask] = normalized_continuous

        return result.astype(np.float32)

    def get_normalizer_state(self) -> dict:
        """Get all normalizer states for saving."""
        return {
            "obs": self.obs_normalizer.state_dict(),
            "ret": self.ret_normalizer.state_dict(),
        }

    def set_normalizer_state(self, state: dict) -> None:
        """Load all normalizer states."""
        if "obs" in state:
            self.obs_normalizer.load_state_dict(state["obs"])
        if "ret" in state:
            self.ret_normalizer.load_state_dict(state["ret"])

    def save_normalizer(self, path: str) -> None:
        """Save normalizer stats to file."""
        obs_state = self.obs_normalizer.state_dict()
        ret_state = self.ret_normalizer.state_dict()
        np.savez(
            path,
            # Observation normalizer
            obs_mean=obs_state["mean"],
            obs_var=obs_state["var"],
            obs_count=obs_state["count"],
            # Reward normalizer
            ret_mean=ret_state["mean"],
            ret_var=ret_state["var"],
            ret_count=ret_state["count"],
        )

    @staticmethod
    def load_normalizer(path: str) -> dict:
        """Load normalizer stats from file."""
        data = np.load(path)
        return {
            "obs": {
                "mean": data["obs_mean"],
                "var": data["obs_var"],
                "count": int(data["obs_count"]),
            },
            "ret": {
                "mean": data["ret_mean"],
                "var": data["ret_var"],
                "count": int(data["ret_count"]),
            },
        }
