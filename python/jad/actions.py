from jad.types import JadConfig

# Action head indices
PROTECTION_PRAYER_HEAD = 0
OFFENSIVE_PRAYER_HEAD = 1
POTION_HEAD = 2
TARGET_HEAD = 3

# Head sizes (fixed)
PROTECTION_PRAYER_SIZE = 4  # no-op + 3 prayers
OFFENSIVE_PRAYER_SIZE = 2   # no-op + rigour
POTION_SIZE = 4             # none + 3 potions


def get_target_head_size(config: JadConfig) -> int:
    """Get the size of the target head: 1 (no-op) + N (jads) + N*H (healers)."""
    return 1 + config.jad_count + config.jad_count * config.healers_per_jad


def get_action_dims(config: JadConfig) -> list[int]:
    """
    Get dimensions for MultiDiscrete action space.
    Returns [protection_prayer_size, offensive_prayer_size, potion_size, target_size].
    """
    return [
        PROTECTION_PRAYER_SIZE,
        OFFENSIVE_PRAYER_SIZE,
        POTION_SIZE,
        get_target_head_size(config),
    ]
