import os
from jad_types import JadConfig


def get_action_count(config: JadConfig) -> int:
    """
    Get action count for given Jad configuration
    Actions: DO_NOTHING + N*AGGRO_JAD + N*NUM_HEALERS_PER_JAD*AGGRO_HEALER + 7 prayers/potions
    """
    return 1 + config.jad_count + config.jad_count * config.healers_per_jad + 7


def parse_config_from_env() -> JadConfig:
    jad_count = int(os.environ.get("JAD_COUNT", "1"))
    healers_per_jad = int(os.environ.get("HEALERS_PER_JAD", "3"))
    return JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)
