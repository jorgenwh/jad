"""
Jad environment wrapper.
Communicates with the Node.js headless simulation via JSON over stdio.
Supports 1-6 Jads with per-Jad healers.
"""

import subprocess
import json
import os
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, field

from config import JadConfig, get_action_count


class TerminationState(Enum):
    """How an episode ended (or didn't)."""
    ONGOING = auto()       # Episode still running
    PLAYER_DIED = auto()   # Player HP reached 0
    JAD_KILLED = auto()    # All Jads HP reached 0
    TRUNCATED = auto()     # Hit max episode length


@dataclass
class JadState:
    """State of a single Jad."""
    hp: int
    attack: int  # 0=none, 1=mage, 2=range, 3=melee
    x: int
    y: int
    alive: bool


@dataclass
class HealerState:
    """State of a single healer."""
    hp: int
    x: int
    y: int
    aggro: int  # 0=not_present, 1=jad, 2=player


@dataclass
class Observation:
    """Observation from the Jad environment."""
    # Config
    jad_count: int
    healers_per_jad: int

    # Player state
    player_hp: int
    player_prayer: int
    player_ranged: int      # Current ranged stat (1-118)
    player_defence: int     # Current defence stat (1-118)
    player_x: int           # Player x position (0-19)
    player_y: int           # Player y position (0-19)
    player_aggro: int       # 0=none, 1..N=jad, N+1..=healer

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
    terminated: bool


class JadEnv:
    """Environment for Jad prayer switching simulation."""

    def __init__(self, config: JadConfig | None = None):
        self._proc: subprocess.Popen | None = None
        self._script_dir = Path(__file__).parent
        self._config = config or JadConfig()

    @property
    def config(self) -> JadConfig:
        return self._config

    @property
    def num_actions(self) -> int:
        return get_action_count(self._config)

    def _start_process(self) -> None:
        """Start the Node.js headless environment process."""
        if self._proc is not None:
            return

        bootstrap_path = self._script_dir / "../dist-headless/headless/bootstrap.js"

        # Set environment variables for config
        env = os.environ.copy()
        env["JAD_COUNT"] = str(self._config.jad_count)
        env["HEALERS_PER_JAD"] = str(self._config.healers_per_jad)

        self._proc = subprocess.Popen(
            ["node", str(bootstrap_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr to see errors
            text=True,
            env=env,
        )

    def _send(self, command: dict) -> dict:
        """Send a command and receive the response."""
        if self._proc is None:
            raise RuntimeError("Environment not started")

        try:
            self._proc.stdin.write(json.dumps(command) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            # Process crashed - try to get stderr for debugging
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Environment process crashed. Stderr: {stderr}")

        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Environment closed unexpectedly. Stderr: {stderr}")

        return json.loads(line)

    def _parse_observation(self, obs: dict) -> Observation:
        """Parse observation dict into Observation dataclass."""
        jad_count = obs.get("jad_count", 1)
        healers_per_jad = obs.get("healers_per_jad", 3)

        # Parse Jad states from array
        jads = []
        jads_data = obs.get("jads", [])
        for jad_data in jads_data:
            jads.append(JadState(
                hp=jad_data.get("hp", 0),
                attack=jad_data.get("attack", 0),
                x=jad_data.get("x", 0),
                y=jad_data.get("y", 0),
                alive=jad_data.get("alive", False),
            ))

        # Parse healer states from array
        healers = []
        healers_data = obs.get("healers", [])
        for healer_data in healers_data:
            healers.append(HealerState(
                hp=healer_data.get("hp", 0),
                x=healer_data.get("x", 0),
                y=healer_data.get("y", 0),
                aggro=healer_data.get("aggro", 0),
            ))

        return Observation(
            # Config
            jad_count=jad_count,
            healers_per_jad=healers_per_jad,

            # Player state
            player_hp=obs["player_hp"],
            player_prayer=obs["player_prayer"],
            player_ranged=obs.get("player_ranged", 99),
            player_defence=obs.get("player_defence", 99),
            player_x=obs.get("player_x", 0),
            player_y=obs.get("player_y", 0),
            player_aggro=obs.get("player_aggro", 0),

            # Prayer state
            active_prayer=obs["active_prayer"],
            rigour_active=obs.get("rigour_active", False),

            # Inventory
            bastion_doses=obs.get("bastion_doses", 0),
            sara_brew_doses=obs.get("sara_brew_doses", 0),
            super_restore_doses=obs.get("super_restore_doses", 0),

            # Jad state
            jads=jads,

            # Healer state
            healers=healers,
            healers_spawned=obs.get("healers_spawned", False),

            # Starting doses for normalization
            starting_bastion_doses=obs.get("starting_bastion_doses", 4),
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
