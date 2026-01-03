from jad.types import JadConfig


def get_action_count(config: JadConfig) -> int:
    """
    Get action count for given Jad configuration.
    Actions: DO_NOTHING + N*AGGRO_JAD + N*NUM_HEALERS_PER_JAD*AGGRO_HEALER + 7 prayers/potions
    """
    return 1 + config.jad_count + config.jad_count * config.healers_per_jad + 7


def get_action_name(action: int, config: JadConfig) -> str:
    """Get human-readable name for an action index."""
    if action == 0:
        return "DO_NOTHING"

    idx = 1

    # Aggro Jad actions
    for i in range(config.jad_count):
        if action == idx:
            return f"AGGRO_JAD_{i + 1}"
        idx += 1

    # Aggro healer actions
    for jad_idx in range(config.jad_count):
        for healer_idx in range(config.healers_per_jad):
            if action == idx:
                return f"AGGRO_H{jad_idx + 1}.{healer_idx + 1}"
            idx += 1

    # Prayer/potion actions
    fixed_actions = [
        "TOGGLE_PROTECT_MELEE",
        "TOGGLE_PROTECT_MISSILES",
        "TOGGLE_PROTECT_MAGIC",
        "TOGGLE_RIGOUR",
        "DRINK_BASTION",
        "DRINK_SUPER_RESTORE",
        "DRINK_SARA_BREW",
    ]
    fixed_idx = action - idx
    if 0 <= fixed_idx < len(fixed_actions):
        return fixed_actions[fixed_idx]

    return "UNKNOWN"
