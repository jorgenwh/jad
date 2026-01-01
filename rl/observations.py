"""
Observation encoding for Jad RL environment.
Supports dynamic observation sizes based on Jad count.

Observation structure for N Jads:
- Continuous features (normalized to [0,1])
- One-hot features for player target, prayer, jad attacks, healer targets
- Binary features (piety, healers_spawned)
"""

import numpy as np
from env import Observation
from config import JadConfig


# Normalization constants
MAX_PLAYER_HP = 115      # Sara brew can boost above 99
MAX_PRAYER = 99
MAX_MELEE_STAT = 118     # Super combat boosted (attack, strength, defence)
MAX_RANGED_STAT = 112    # Bastion boosted (99 + 13)
MAX_MAGIC_STAT = 109     # Imbued heart boosted (99 + 10)
MAX_COORD = 26           # 27x27 grid (0-26)
MAX_JAD_HP = 350
MAX_HEALER_HP = 90


def get_obs_dim(config: JadConfig) -> int:
    """
    Calculate observation dimension for given Jad configuration.

    Structure:
    - Player continuous: 9 (hp, prayer, ranged, def, 3 potion doses, x, y)
    - Player aggro one-hot: 1 + jad_count + jad_count * healers_per_jad
    - Active prayer one-hot: 4
    - Per-Jad continuous: 3 (hp, x, y) * jad_count
    - Per-Jad attack one-hot: 4 * jad_count (none/mage/range/melee)
    - Per-Healer continuous: 3 (hp, x, y) * jad_count * healers_per_jad
    - Per-Healer target one-hot: 3 * jad_count * healers_per_jad
    - Binary: 2 (rigour_active, healers_spawned)
    """
    jad_count = config.jad_count
    healers_per_jad = config.healers_per_jad
    total_healers = jad_count * healers_per_jad

    player_continuous = 9
    player_target_onehot = 1 + jad_count + total_healers
    active_prayer_onehot = 4
    jad_continuous = 3 * jad_count  # hp, x, y per jad
    jad_attack_onehot = 4 * jad_count
    healer_continuous = 3 * total_healers
    healer_target_onehot = 3 * total_healers
    binary = 2

    return (player_continuous + player_target_onehot + active_prayer_onehot +
            jad_continuous + jad_attack_onehot + healer_continuous +
            healer_target_onehot + binary)


def get_continuous_feature_count(config: JadConfig) -> int:
    """Get the number of continuous features that should be normalized."""
    jad_count = config.jad_count
    healers_per_jad = config.healers_per_jad
    total_healers = jad_count * healers_per_jad

    player_continuous = 9
    jad_continuous = 3 * jad_count
    healer_continuous = 3 * total_healers

    return player_continuous + jad_continuous + healer_continuous


def get_normalize_mask(config: JadConfig) -> np.ndarray:
    """Get mask for which features should be normalized (True = normalize)."""
    obs_dim = get_obs_dim(config)
    continuous_count = get_continuous_feature_count(config)

    # Continuous features first, then one-hot/binary
    mask = np.zeros(obs_dim, dtype=bool)
    mask[:continuous_count] = True
    return mask


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


def obs_to_array(obs: Observation, config: JadConfig) -> np.ndarray:
    """
    Convert Observation dataclass to numpy array with proper encoding.

    The array structure is:
    1. Continuous features (player, jads, healers)
    2. One-hot features (player_target, active_prayer, jad_attacks, healer_targets)
    3. Binary features (rigour_active, healers_spawned)

    Args:
        obs: The observation dataclass
        config: Jad configuration (jad_count, healers_per_jad)

    Returns:
        Array of shape (obs_dim,) where obs_dim = get_obs_dim(config)
    """
    jad_count = config.jad_count
    healers_per_jad = config.healers_per_jad
    total_healers = jad_count * healers_per_jad

    # ========== CONTINUOUS FEATURES ==========

    # Player continuous (9 features)
    player_continuous = np.array([
        obs.player_hp / MAX_PLAYER_HP,
        obs.player_prayer / MAX_PRAYER,
        obs.player_ranged / MAX_RANGED_STAT,
        obs.player_defence / MAX_MELEE_STAT,  # Defence boosted by super combat or sara brew
        safe_divide(obs.bastion_doses, obs.starting_bastion_doses),
        safe_divide(obs.sara_brew_doses, obs.starting_sara_brew_doses),
        safe_divide(obs.super_restore_doses, obs.starting_super_restore_doses),
        obs.player_location_x / MAX_COORD,
        obs.player_location_y / MAX_COORD,
    ], dtype=np.float32)

    # Jad continuous (3 * jad_count features)
    jad_continuous = []
    for jad in obs.jads:
        jad_continuous.extend([
            jad.hp / MAX_JAD_HP,
            jad.x / MAX_COORD,
            jad.y / MAX_COORD,
        ])
    jad_continuous = np.array(jad_continuous, dtype=np.float32)

    # Healer continuous (3 * total_healers features)
    healer_continuous = []
    for healer in obs.healers:
        healer_continuous.extend([
            healer.hp / MAX_HEALER_HP,
            healer.x / MAX_COORD,
            healer.y / MAX_COORD,
        ])
    healer_continuous = np.array(healer_continuous, dtype=np.float32)

    # ========== ONE-HOT FEATURES ==========

    # Player target one-hot (1 + jad_count + total_healers)
    target_size = 1 + jad_count + total_healers
    player_target_onehot = one_hot(obs.player_target, target_size)

    # Active prayer one-hot (4: none, mage, range, melee)
    active_prayer_onehot = one_hot(obs.active_prayer, 4)

    # Per-Jad attack one-hot (4 * jad_count)
    jad_attack_onehot = []
    for jad in obs.jads:
        jad_attack_onehot.append(one_hot(jad.attack, 4))
    jad_attack_onehot = np.concatenate(jad_attack_onehot) if jad_attack_onehot else np.array([], dtype=np.float32)

    # Per-Healer target one-hot (3 * total_healers)
    healer_target_onehot = []
    for healer in obs.healers:
        healer_target_onehot.append(one_hot(healer.target, 3))
    healer_target_onehot = np.concatenate(healer_target_onehot) if healer_target_onehot else np.array([], dtype=np.float32)

    # ========== BINARY FEATURES ==========

    binary = np.array([
        float(obs.rigour_active),
        float(obs.healers_spawned),
    ], dtype=np.float32)

    # Concatenate all features
    return np.concatenate([
        player_continuous,
        jad_continuous,
        healer_continuous,
        player_target_onehot,
        active_prayer_onehot,
        jad_attack_onehot,
        healer_target_onehot,
        binary,
    ])


# For backwards compatibility with 1-Jad config
def get_default_obs_dim() -> int:
    """Get observation dimension for default 1-Jad, 3-healer config."""
    return get_obs_dim(JadConfig(jad_count=1, healers_per_jad=3))


# Legacy constant for backwards compatibility
OBS_DIM = get_default_obs_dim()

# Legacy constant for backwards compatibility
CONTINUOUS_FEATURES = get_continuous_feature_count(JadConfig(jad_count=1, healers_per_jad=3))

# Legacy mask for backwards compatibility
NORMALIZE_MASK = get_normalize_mask(JadConfig(jad_count=1, healers_per_jad=3))
