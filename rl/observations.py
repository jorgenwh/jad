"""
Observation encoding for Jad RL environment.
"""

import numpy as np
from env import Observation

# Observation dimensions
OBS_DIM = 15

# Indices for each feature in the observation array
# Continuous features (will be normalized)
IDX_PLAYER_HP = 0
IDX_PLAYER_PRAYER = 1
IDX_JAD_HP = 2

# One-hot features (should NOT be normalized)
IDX_ACTIVE_PRAYER_START = 3   # 3 values: none, mage, range
IDX_JAD_ATTACK_START = 6      # 3 values: none, mage, range
IDX_RESTORE_DOSES_START = 9   # 5 values: 0, 1, 2, 3, 4
IDX_PLAYER_AGGRO = 14         # Binary: 0 or 1

# Masks for normalization (True = normalize, False = don't normalize)
NORMALIZE_MASK = np.array([
    True,   # player_hp
    True,   # player_prayer
    True,   # jad_hp
    False, False, False,  # active_prayer (one-hot)
    False, False, False,  # jad_attack (one-hot)
    False, False, False, False, False,  # restore_doses (one-hot)
    False,  # player_aggro (binary)
], dtype=bool)


def one_hot(index: int, size: int) -> np.ndarray:
    """Create one-hot encoded array."""
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        arr[index] = 1.0
    return arr


def obs_to_array(obs: Observation) -> np.ndarray:
    """
    Convert Observation dataclass to numpy array with one-hot encoding.

    Returns:
        Array of shape (15,):
        - [0]: player_hp (continuous)
        - [1]: player_prayer (continuous)
        - [2]: jad_hp (continuous)
        - [3:6]: active_prayer (one-hot: none, mage, range)
        - [6:9]: jad_attack (one-hot: none, mage, range)
        - [9:14]: restore_doses (one-hot: 0, 1, 2, 3, 4)
        - [14]: player_aggro (binary: 0 or 1)
    """
    return np.concatenate([
        # Continuous features
        np.array([obs.player_hp, obs.player_prayer, obs.jad_hp], dtype=np.float32),
        # One-hot features
        one_hot(obs.active_prayer, 3),  # 0=none, 1=mage, 2=range
        one_hot(obs.jad_attack, 3),     # 0=none, 1=mage, 2=range
        one_hot(obs.restore_doses, 5),  # 0, 1, 2, 3, 4 doses
        # Binary features
        np.array([float(obs.player_aggro)], dtype=np.float32),
    ])
