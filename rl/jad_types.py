from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass
class JadConfig:
    jad_count: int = 1
    healers_per_jad: int = 3

    def __post_init__(self):
        if not 1 <= self.jad_count <= 6:
            raise ValueError(f"jad_count must be 1-6, got {self.jad_count}")
        if not 0 <= self.healers_per_jad <= 5:
            raise ValueError(
                f"healers_per_jad must be 0-5, got {self.healers_per_jad}"
            )


DEFAULT_CONFIG = JadConfig()


class TerminationState(Enum):
    ONGOING = auto()       # Episode still running
    PLAYER_DIED = auto()   # Player HP reached 0
    JAD_KILLED = auto()    # All Jads HP reached 0
    TRUNCATED = auto()     # Hit max episode length


@dataclass
class JadState:
    hp: int
    attack: int  # 0=none, 1=mage, 2=range, 3=melee
    x: int
    y: int
    alive: bool


@dataclass
class HealerState:
    hp: int
    x: int
    y: int
    target: int  # 0=not_present, 1=jad, 2=player


@dataclass
class Observation:
    # Player state
    player_hp: int
    player_prayer: int
    player_ranged: int      # Current ranged stat (1-118)
    player_defence: int     # Current defence stat (1-118)
    player_location_x: int  # Player x position (0-26)
    player_location_y: int  # Player y position (0-26)
    player_target: int      # 0=none, 1..N=jad, N+1..=healer

    # Prayer state
    active_prayer: int      # 0=none, 1=mage, 2=range, 3=melee
    rigour_active: bool

    # Inventory
    bastion_doses: int
    sara_brew_doses: int
    super_restore_doses: int

    # Dynamic Jad state
    jads: list[JadState] = field(default_factory=list)

    # Dynamic healer state (flattened)
    healers: list[HealerState] = field(default_factory=list)

    # Whether any healers have spawned
    healers_spawned: bool = False

    # Starting doses for normalization
    starting_bastion_doses: int = 4
    starting_sara_brew_doses: int = 4
    starting_super_restore_doses: int = 4


@dataclass
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
