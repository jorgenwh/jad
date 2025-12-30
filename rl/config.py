"""
Configuration for multi-Jad environment.
"""

from dataclasses import dataclass
import os


@dataclass
class JadConfig:
    """Configuration for Jad environment."""

    jad_count: int = 1
    healers_per_jad: int = 3

    def __post_init__(self):
        if not 1 <= self.jad_count <= 6:
            raise ValueError(f"jad_count must be 1-6, got {self.jad_count}")
        if not 0 <= self.healers_per_jad <= 5:
            raise ValueError(
                f"healers_per_jad must be 0-5, got {self.healers_per_jad}"
            )


def get_action_count(config: JadConfig) -> int:
    """
    Get action count for given Jad configuration.
    Actions: DO_NOTHING + N*AGGRO_JAD + N*3*AGGRO_HEALER + 7 prayers/potions
    """
    return 1 + config.jad_count + config.jad_count * config.healers_per_jad + 7


def parse_config_from_env() -> JadConfig:
    """Parse configuration from environment variables."""
    jad_count = int(os.environ.get("JAD_COUNT", "1"))
    healers_per_jad = int(os.environ.get("HEALERS_PER_JAD", "3"))
    return JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)


# Default configuration
DEFAULT_CONFIG = JadConfig()
