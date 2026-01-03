import os
from jad.types import JadConfig


def parse_config_from_env() -> JadConfig:
    jad_count = int(os.environ.get("JAD_COUNT", "1"))
    healers_per_jad = int(os.environ.get("HEALERS_PER_JAD", "3"))
    return JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)
