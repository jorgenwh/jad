"""
Running normalizer for observation normalization.
Uses Welford's online algorithm for numerical stability.
"""

import numpy as np


class RunningNormalizer:
    """
    Maintains running mean and variance for online normalization.

    Uses Welford's algorithm for numerically stable online computation.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8):
        """
        Args:
            shape: Shape of observations to normalize
            epsilon: Small constant for numerical stability
        """
        self.shape = shape
        self.epsilon = epsilon

        # Running statistics
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with a new observation.

        Args:
            x: New observation (can be single or batch)
        """
        x = np.asarray(x, dtype=np.float64)

        # Handle single observation
        if x.shape == self.shape:
            x = x.reshape(1, *self.shape)

        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        # Update using parallel Welford algorithm
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ) -> None:
        """Update statistics from batch moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / total_count

        # Update variance (parallel Welford)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize observation using running statistics.

        Args:
            x: Observation to normalize

        Returns:
            Normalized observation with ~zero mean and ~unit variance
        """
        return (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + self.epsilon)

    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from dict."""
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]
