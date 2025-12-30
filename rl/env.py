"""
Jad environment wrapper.
Communicates with the Node.js headless simulation via JSON over stdio.
"""

import subprocess
import json
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass


class TerminationState(Enum):
    """How an episode ended (or didn't)."""
    ONGOING = auto()       # Episode still running
    PLAYER_DIED = auto()   # Player HP reached 0
    JAD_KILLED = auto()    # Jad HP reached 0
    TRUNCATED = auto()     # Hit max episode length


@dataclass
class Observation:
    # Player state
    player_hp: int
    player_prayer: int
    player_attack: int      # Current attack stat (1-118)
    player_strength: int    # Current strength stat (1-118)
    player_defence: int     # Current defence stat (1-118)
    player_x: int           # Player x position (0-19)
    player_y: int           # Player y position (0-19)
    player_aggro: int       # 0=none, 1=jad, 2=healer1, 3=healer2, 4=healer3

    # Prayer state
    active_prayer: int      # 0=none, 1=mage, 2=range, 3=melee
    piety_active: bool

    # Inventory
    super_combat_doses: int
    sara_brew_doses: int
    super_restore_doses: int

    # Jad state
    jad_hp: int
    jad_attack: int         # 0=none, 1=mage, 2=range, 3=melee
    jad_x: int              # Jad x position (0-19)
    jad_y: int              # Jad y position (0-19)

    # Healer state
    healers_spawned: bool
    healer_1_hp: int        # 0 if not present
    healer_1_x: int
    healer_1_y: int
    healer_1_aggro: int     # 0=not_present, 1=jad, 2=player
    healer_2_hp: int
    healer_2_x: int
    healer_2_y: int
    healer_2_aggro: int
    healer_3_hp: int
    healer_3_x: int
    healer_3_y: int
    healer_3_aggro: int

    # Starting doses for normalization
    starting_super_combat_doses: int
    starting_sara_brew_doses: int
    starting_super_restore_doses: int


@dataclass
class StepResult:
    observation: Observation
    terminated: bool


class JadEnv:
    """Environment for Jad prayer switching simulation."""

    # Action constants (12 discrete actions)
    DO_NOTHING = 0
    AGGRO_JAD = 1
    AGGRO_HEALER_1 = 2
    AGGRO_HEALER_2 = 3
    AGGRO_HEALER_3 = 4
    TOGGLE_PROTECT_MELEE = 5
    TOGGLE_PROTECT_MISSILES = 6
    TOGGLE_PROTECT_MAGIC = 7
    TOGGLE_PIETY = 8
    DRINK_SUPER_COMBAT = 9
    DRINK_SUPER_RESTORE = 10
    DRINK_SARA_BREW = 11
    NUM_ACTIONS = 12

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._script_dir = Path(__file__).parent

    def _start_process(self) -> None:
        """Start the Node.js headless environment process."""
        if self._proc is not None:
            return

        bootstrap_path = self._script_dir / "../dist-headless/headless/bootstrap.js"
        self._proc = subprocess.Popen(
            ["node", str(bootstrap_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _send(self, command: dict) -> dict:
        """Send a command and receive the response."""
        if self._proc is None:
            raise RuntimeError("Environment not started")

        self._proc.stdin.write(json.dumps(command) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Environment closed unexpectedly")

        return json.loads(line)

    def _parse_observation(self, obs: dict) -> Observation:
        """Parse observation dict into Observation dataclass."""
        return Observation(
            # Player state
            player_hp=obs["player_hp"],
            player_prayer=obs["player_prayer"],
            player_attack=obs.get("player_attack", 99),
            player_strength=obs.get("player_strength", 99),
            player_defence=obs.get("player_defence", 99),
            player_x=obs.get("player_x", 0),
            player_y=obs.get("player_y", 0),
            player_aggro=obs.get("player_aggro", 0),

            # Prayer state
            active_prayer=obs["active_prayer"],
            piety_active=obs.get("piety_active", False),

            # Inventory
            super_combat_doses=obs.get("super_combat_doses", 0),
            sara_brew_doses=obs.get("sara_brew_doses", 0),
            super_restore_doses=obs.get("super_restore_doses", 0),

            # Jad state
            jad_hp=obs["jad_hp"],
            jad_attack=obs["jad_attack"],
            jad_x=obs.get("jad_x", 0),
            jad_y=obs.get("jad_y", 0),

            # Healer state
            healers_spawned=obs.get("healers_spawned", False),
            healer_1_hp=obs.get("healer_1_hp", 0),
            healer_1_x=obs.get("healer_1_x", 0),
            healer_1_y=obs.get("healer_1_y", 0),
            healer_1_aggro=obs.get("healer_1_aggro", 0),
            healer_2_hp=obs.get("healer_2_hp", 0),
            healer_2_x=obs.get("healer_2_x", 0),
            healer_2_y=obs.get("healer_2_y", 0),
            healer_2_aggro=obs.get("healer_2_aggro", 0),
            healer_3_hp=obs.get("healer_3_hp", 0),
            healer_3_x=obs.get("healer_3_x", 0),
            healer_3_y=obs.get("healer_3_y", 0),
            healer_3_aggro=obs.get("healer_3_aggro", 0),

            # Starting doses for normalization
            starting_super_combat_doses=obs.get("starting_super_combat_doses", 4),
            starting_sara_brew_doses=obs.get("starting_sara_brew_doses", 4),
            starting_super_restore_doses=obs.get("starting_super_restore_doses", 4),
        )

    def reset(self) -> Observation:
        """Reset the environment and return initial observation."""
        self._start_process()
        result = self._send({"command": "reset"})
        return self._parse_observation(result["observation"])

    def step(self, action: int) -> StepResult:
        """Take an action and return the result."""
        if self._proc is None:
            raise RuntimeError("Must call reset() before step()")

        result = self._send({"command": "step", "action": action})
        return StepResult(
            observation=self._parse_observation(result["observation"]),
            terminated=result["terminated"],
        )

    def close(self) -> None:
        """Close the environment."""
        if self._proc is not None:
            try:
                self._send({"command": "close"})
            except RuntimeError:
                pass  # Process may have already exited
            self._proc.wait()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
