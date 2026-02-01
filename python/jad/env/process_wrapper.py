import subprocess
import json
import os
from pathlib import Path

from jad.types import JadConfig, Observation, StepResult, JadState, HealerState


class EnvProcessWrapper:
    def __init__(self, config: JadConfig | None = None, reward_func: str = "default"):
        self._proc: subprocess.Popen | None = None
        self._script_dir = Path(__file__).parent
        self._config = config or JadConfig()
        self._reward_func = reward_func

    @property
    def config(self) -> JadConfig:
        return self._config

    def reset(self) -> StepResult:
        self._start_process()
        result = self._send({"command": "reset"})
        return StepResult(
            observation=self._parse_observation(result["observation"]),
            reward=0.0,
            terminated=False,
            valid_action_mask=result.get("valid_action_mask", []),
        )

    def step(self, action: int) -> StepResult:
        if self._proc is None:
            raise RuntimeError("Must call reset() before step()")

        result = self._send({"command": "step", "action": action})
        return StepResult(
            observation=self._parse_observation(result["observation"]),
            reward=result["reward"],
            terminated=result["terminated"],
            valid_action_mask=result.get("valid_action_mask", []),
        )

    def close(self) -> None:
        if self._proc is None:
            return

        try:
            self._send({"command": "close"})
        except RuntimeError:
            pass  # Process may have already exited
        self._proc.wait()
        self._proc = None

    def _start_process(self) -> None:
        if self._proc is not None:
            return

        # Navigate from python/jad/env/ to dist-headless/
        bootstrap_path = self._script_dir / "../../../dist-headless/headless/bootstrap.js"

        # Set environment variables for config
        env = os.environ.copy()
        env["JAD_COUNT"] = str(self._config.jad_count)
        env["HEALERS_PER_JAD"] = str(self._config.healers_per_jad)
        env["REWARD_FUNC"] = self._reward_func

        self._proc = subprocess.Popen(
            ["node", str(bootstrap_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr to see errors
            text=True,
            env=env,
        )

    def _send(self, command: dict) -> dict:
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
        # Parse Jad states
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

        # Parse healer states
        healers = []
        healers_data = obs.get("healers", [])
        for healer_data in healers_data:
            healers.append(HealerState(
                hp=healer_data.get("hp", 0),
                x=healer_data.get("x", 0),
                y=healer_data.get("y", 0),
                target=healer_data.get("target", 0),
            ))

        return Observation(
            player_hp=obs["player_hp"],
            player_prayer=obs["player_prayer"],
            player_ranged=obs.get("player_ranged", 99),
            player_defence=obs.get("player_defence", 99),
            player_location_x=obs.get("player_location_x", 0),
            player_location_y=obs.get("player_location_y", 0),
            player_target=obs.get("player_target", 0),

            active_prayer=obs["active_prayer"],
            rigour_active=obs.get("rigour_active", False),

            jads=jads,
            healers=healers,
            healers_spawned=obs.get("healers_spawned", False),

            bastion_doses=obs.get("bastion_doses", 0),
            sara_brew_doses=obs.get("sara_brew_doses", 0),
            super_restore_doses=obs.get("super_restore_doses", 0),
            starting_bastion_doses=obs.get("starting_bastion_doses", 4),
            starting_sara_brew_doses=obs.get("starting_sara_brew_doses", 4),
            starting_super_restore_doses=obs.get("starting_super_restore_doses", 4),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
