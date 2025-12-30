"""
Observation encoding for Jad RL environment.
"""

import numpy as np
from env import Observation

# Observation dimensions
# 3 continuous + 4 active_prayer + 4 jad_attack + 5 restore + 5 super_combat + 5 sara_brew + 1 piety + 1 aggro + 4 healer_count = 32
OBS_DIM = 32

# Indices for each feature in the observation array
# Continuous features (will be normalized)
IDX_PLAYER_HP = 0
IDX_PLAYER_PRAYER = 1
IDX_JAD_HP = 2

# One-hot features (should NOT be normalized)
IDX_ACTIVE_PRAYER_START = 3   # 4 values: none, mage, range, melee
IDX_JAD_ATTACK_START = 7      # 4 values: none, mage, range, melee
IDX_RESTORE_DOSES_START = 11  # 5 values: 0, 1, 2, 3, 4
IDX_SUPER_COMBAT_DOSES_START = 16  # 5 values: 0, 1, 2, 3, 4
IDX_SARA_BREW_DOSES_START = 21  # 5 values: 0, 1, 2, 3, 4
IDX_PIETY_ACTIVE = 26         # Binary: 0 or 1
IDX_PLAYER_AGGRO = 27         # Binary: 0 or 1
IDX_HEALER_COUNT_START = 28   # 4 values: 0, 1, 2, 3 healers

# Masks for normalization (True = normalize, False = don't normalize)
NORMALIZE_MASK = np.array([
    True,   # player_hp
    True,   # player_prayer
    True,   # jad_hp
    False, False, False, False,  # active_prayer (one-hot, 4 values)
    False, False, False, False,  # jad_attack (one-hot, 4 values)
    False, False, False, False, False,  # restore_doses (one-hot, 5 values)
    False, False, False, False, False,  # super_combat_doses (one-hot, 5 values)
    False, False, False, False, False,  # sara_brew_doses (one-hot, 5 values)
    False,  # piety_active (binary)
    False,  # player_aggro (binary)
    False, False, False, False,  # healer_count (one-hot, 4 values: 0, 1, 2, 3)
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
        Array of shape (32,):
        - [0]: player_hp (continuous)
        - [1]: player_prayer (continuous)
        - [2]: jad_hp (continuous)
        - [3:7]: active_prayer (one-hot: none, mage, range, melee)
        - [7:11]: jad_attack (one-hot: none, mage, range, melee)
        - [11:16]: restore_doses (one-hot: 0, 1, 2, 3, 4)
        - [16:21]: super_combat_doses (one-hot: 0, 1, 2, 3, 4)
        - [21:26]: sara_brew_doses (one-hot: 0, 1, 2, 3, 4)
        - [26]: piety_active (binary: 0 or 1)
        - [27]: player_aggro (binary: 0 or 1)
        - [28:32]: healer_count (one-hot: 0, 1, 2, 3)
    """
    return np.concatenate([
        # Continuous features
        np.array([obs.player_hp, obs.player_prayer, obs.jad_hp], dtype=np.float32),
        # One-hot features
        one_hot(obs.active_prayer, 4),  # 0=none, 1=mage, 2=range, 3=melee
        one_hot(obs.jad_attack, 4),     # 0=none, 1=mage, 2=range, 3=melee
        one_hot(obs.restore_doses, 5),  # 0, 1, 2, 3, 4 doses
        one_hot(obs.super_combat_doses, 5),  # 0, 1, 2, 3, 4 doses
        one_hot(obs.sara_brew_doses, 5),  # 0, 1, 2, 3, 4 doses
        # Binary features
        np.array([float(obs.piety_active)], dtype=np.float32),
        np.array([float(obs.player_aggro)], dtype=np.float32),
        # Healer count (one-hot: 0, 1, 2, 3)
        one_hot(min(obs.healer_count, 3), 4),
    ])
