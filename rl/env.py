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
    player_hp: int
    player_prayer: int
    active_prayer: int  # 0=none, 1=mage, 2=range, 3=melee
    jad_hp: int
    jad_attack: int  # 0=none, 1=mage, 2=range, 3=melee
    restore_doses: int
    super_combat_doses: int
    sara_brew_doses: int
    piety_active: bool
    player_aggro: bool  # Whether player is attacking Jad
    healer_count: int  # Number of alive healers (0-3)


@dataclass
class StepResult:
    observation: Observation
    terminated: bool


class JadEnv:
    """Environment for Jad prayer switching simulation."""

    # Action constants
    WAIT = 0
    PRAY_MAGE = 1         # Toggle protect from magic
    PRAY_RANGE = 2        # Toggle protect from range
    DRINK_RESTORE = 3
    ATTACK = 4
    PRAY_MELEE = 5        # Toggle protect from melee
    DRINK_SUPER_COMBAT = 6
    TOGGLE_PIETY = 7
    DRINK_SARA_BREW = 8

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
            player_hp=obs["player_hp"],
            player_prayer=obs["player_prayer"],
            active_prayer=obs["active_prayer"],
            jad_hp=obs["jad_hp"],
            jad_attack=obs["jad_attack"],
            restore_doses=obs["restore_doses"],
            super_combat_doses=obs.get("super_combat_doses", 0),
            sara_brew_doses=obs.get("sara_brew_doses", 0),
            piety_active=obs.get("piety_active", False),
            player_aggro=obs.get("player_aggro", False),
            healer_count=obs.get("healer_count", 0),
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
