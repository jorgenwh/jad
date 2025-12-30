"""
Observation encoding for Jad RL environment.

Observation structure (46 dimensions total):
- 22 continuous features (normalized to [0,1])
- 22 one-hot features
- 2 binary features
"""

import numpy as np
from env import Observation

# Observation dimensions
# 22 continuous + 5 player_aggro + 4 active_prayer + 4 jad_attack + 3*3 healer_aggro + 2 binary = 46
OBS_DIM = 46

# Normalization constants
MAX_PLAYER_HP = 115      # Sara brew can boost above 99
MAX_PRAYER = 99
MAX_STAT = 118           # Super combat boosted
MAX_COORD = 19           # 20x20 grid (0-19)
MAX_JAD_HP = 350
MAX_HEALER_HP = 90

# Indices for continuous features (will be normalized)
# These are the first 22 values in the array
CONTINUOUS_FEATURES = 22

# Masks for normalization (True = normalize, False = don't normalize)
# First 22 are continuous, rest are one-hot/binary
NORMALIZE_MASK = np.array(
    [True] * CONTINUOUS_FEATURES +  # All continuous features normalized
    [False] * (OBS_DIM - CONTINUOUS_FEATURES),  # One-hot and binary not normalized
    dtype=bool
)


def one_hot(index: int, size: int) -> np.ndarray:
    """Create one-hot encoded array."""
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        arr[index] = 1.0
    return arr


def safe_divide(value: float, divisor: float) -> float:
    """Safely divide, returning 0 if divisor is 0."""
    if divisor <= 0:
        return 0.0
    return value / divisor


def obs_to_array(obs: Observation) -> np.ndarray:
    """
    Convert Observation dataclass to numpy array with proper encoding.

    Returns:
        Array of shape (46,):

        Continuous features (22, normalized to [0,1]):
        - [0]: player_hp / 115
        - [1]: player_prayer / 99
        - [2]: player_attack / 118
        - [3]: player_strength / 118
        - [4]: player_defence / 118
        - [5]: super_combat_doses / starting_doses
        - [6]: sara_brew_doses / starting_doses
        - [7]: super_restore_doses / starting_doses
        - [8]: player_x / 19
        - [9]: player_y / 19
        - [10]: jad_hp / 350
        - [11]: jad_x / 19
        - [12]: jad_y / 19
        - [13]: healer_1_hp / 90
        - [14]: healer_1_x / 19
        - [15]: healer_1_y / 19
        - [16]: healer_2_hp / 90
        - [17]: healer_2_x / 19
        - [18]: healer_2_y / 19
        - [19]: healer_3_hp / 90
        - [20]: healer_3_x / 19
        - [21]: healer_3_y / 19

        One-hot features (22):
        - [22:27]: player_aggro (5 values: none, jad, healer1, healer2, healer3)
        - [27:31]: active_prayer (4 values: none, mage, range, melee)
        - [31:35]: jad_attack (4 values: none, mage, range, melee)
        - [35:38]: healer_1_aggro (3 values: not_present, jad, player)
        - [38:41]: healer_2_aggro (3 values: not_present, jad, player)
        - [41:44]: healer_3_aggro (3 values: not_present, jad, player)

        Binary features (2):
        - [44]: piety_active
        - [45]: healers_spawned
    """
    # Continuous features (normalized to [0,1])
    continuous = np.array([
        # Player state
        obs.player_hp / MAX_PLAYER_HP,
        obs.player_prayer / MAX_PRAYER,
        obs.player_attack / MAX_STAT,
        obs.player_strength / MAX_STAT,
        obs.player_defence / MAX_STAT,
        # Inventory (normalized by starting doses)
        safe_divide(obs.super_combat_doses, obs.starting_super_combat_doses),
        safe_divide(obs.sara_brew_doses, obs.starting_sara_brew_doses),
        safe_divide(obs.super_restore_doses, obs.starting_super_restore_doses),
        # Player position
        obs.player_x / MAX_COORD,
        obs.player_y / MAX_COORD,
        # Jad state
        obs.jad_hp / MAX_JAD_HP,
        obs.jad_x / MAX_COORD,
        obs.jad_y / MAX_COORD,
        # Healer 1
        obs.healer_1_hp / MAX_HEALER_HP,
        obs.healer_1_x / MAX_COORD,
        obs.healer_1_y / MAX_COORD,
        # Healer 2
        obs.healer_2_hp / MAX_HEALER_HP,
        obs.healer_2_x / MAX_COORD,
        obs.healer_2_y / MAX_COORD,
        # Healer 3
        obs.healer_3_hp / MAX_HEALER_HP,
        obs.healer_3_x / MAX_COORD,
        obs.healer_3_y / MAX_COORD,
    ], dtype=np.float32)

    # One-hot features
    player_aggro_onehot = one_hot(obs.player_aggro, 5)  # none, jad, h1, h2, h3
    active_prayer_onehot = one_hot(obs.active_prayer, 4)  # none, mage, range, melee
    jad_attack_onehot = one_hot(obs.jad_attack, 4)  # none, mage, range, melee
    healer_1_aggro_onehot = one_hot(obs.healer_1_aggro, 3)  # not_present, jad, player
    healer_2_aggro_onehot = one_hot(obs.healer_2_aggro, 3)
    healer_3_aggro_onehot = one_hot(obs.healer_3_aggro, 3)

    # Binary features
    binary = np.array([
        float(obs.piety_active),
        float(obs.healers_spawned),
    ], dtype=np.float32)

    return np.concatenate([
        continuous,
        player_aggro_onehot,
        active_prayer_onehot,
        jad_attack_onehot,
        healer_1_aggro_onehot,
        healer_2_aggro_onehot,
        healer_3_aggro_onehot,
        binary,
    ])
