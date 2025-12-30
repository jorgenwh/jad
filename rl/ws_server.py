"""
WebSocket server for browser visualization of the trained agent.

Run this server, then open the browser simulation to see the agent play.

Usage:
    python ws_server.py                                        # Use default checkpoint
    python ws_server.py --checkpoint checkpoints/best_sb3.zip  # Specific checkpoint
"""

import asyncio
import json
import numpy as np
import torch
from websockets.asyncio.server import serve
from sb3_contrib import RecurrentPPO
from pathlib import Path

from observations import obs_to_array, get_normalize_mask
from vec_normalize import RunningNormalizer
from env import Observation, JadState, HealerState, TerminationState
from config import JadConfig
from rewards import compute_reward, list_reward_functions


def get_action_names(config: JadConfig) -> dict[int, str]:
    """Generate action names based on config."""
    actions = {0: "DO_NOTHING"}
    idx = 1

    # Aggro Jad actions
    for i in range(config.jad_count):
        actions[idx] = f"AGGRO_JAD_{i+1}"
        idx += 1

    # Aggro healer actions
    for jad_idx in range(config.jad_count):
        for healer_idx in range(config.healers_per_jad):
            actions[idx] = f"AGGRO_H{jad_idx+1}.{healer_idx+1}"
            idx += 1

    # Prayer/potion actions
    actions[idx] = "TOGGLE_PROTECT_MELEE"
    actions[idx + 1] = "TOGGLE_PROTECT_MISSILES"
    actions[idx + 2] = "TOGGLE_PROTECT_MAGIC"
    actions[idx + 3] = "TOGGLE_RIGOUR"
    actions[idx + 4] = "DRINK_BASTION"
    actions[idx + 5] = "DRINK_SUPER_RESTORE"
    actions[idx + 6] = "DRINK_SARA_BREW"

    return actions


class AgentServer:
    def __init__(self, checkpoint_path: str, jad_count: int = 1,
                 healers_per_jad: int = 3, reward_type: str = "default"):
        self.model = None
        self.lstm_state = None
        self.config = JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)
        self.action_names = get_action_names(self.config)
        self.reward_type = reward_type

        # Reward tracking
        self.prev_obs: Observation | None = None
        self.episode_length = 0
        self.cumulative_reward = 0.0

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load SB3 .zip checkpoint."""

        self.model = RecurrentPPO.load(checkpoint_path, device="auto")
        self.lstm_state = None  # Will be initialized on first prediction
        print(f"Loaded checkpoint: {checkpoint_path}")

        # Set up observation normalizer (matching SelectiveVecNormalize used in training)
        self.normalize_mask = get_normalize_mask(self.config)
        n_continuous = int(np.sum(self.normalize_mask))
        self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))

        # Load normalizer stats saved by SelectiveVecNormalize during training
        checkpoint_dir = Path(checkpoint_path).parent
        # Try jad-count specific normalizer first, then generic
        jad_count = self.config.jad_count
        normalizer_path = checkpoint_dir / f"normalizer_sb3_{jad_count}jad.npz"
        if not normalizer_path.exists():
            normalizer_path = checkpoint_dir / "normalizer_sb3.npz"

        if normalizer_path.exists():
            data = np.load(normalizer_path)
            self.obs_normalizer.load_state_dict({
                "mean": data["obs_mean"],
                "var": data["obs_var"],
                "count": int(data["obs_count"]),
            })
            print(f"Loaded normalizer stats: {normalizer_path}")
        else:
            print(f"Warning: No normalizer stats found")
            print("  Observations will not be normalized correctly!")

    def _parse_observation(self, obs_dict: dict) -> Observation:
        """Parse observation dict into Observation dataclass."""
        jad_count = obs_dict.get("jad_count", self.config.jad_count)
        healers_per_jad = obs_dict.get("healers_per_jad", self.config.healers_per_jad)

        # Parse jads array
        jads_data = obs_dict.get("jads", [])
        jads = [
            JadState(
                hp=j.get("hp", 0),
                attack=j.get("attack", 0),
                x=j.get("x", 0),
                y=j.get("y", 0),
                alive=j.get("alive", False),
            )
            for j in jads_data
        ]

        # Parse healers array
        healers_data = obs_dict.get("healers", [])
        healers = [
            HealerState(
                hp=h.get("hp", 0),
                x=h.get("x", 0),
                y=h.get("y", 0),
                aggro=h.get("aggro", 0),
            )
            for h in healers_data
        ]

        return Observation(
            # Config
            jad_count=jad_count,
            healers_per_jad=healers_per_jad,

            # Player state
            player_hp=obs_dict.get("player_hp", 99),
            player_prayer=obs_dict.get("player_prayer", 99),
            player_ranged=obs_dict.get("player_ranged", 99),
            player_defence=obs_dict.get("player_defence", 99),
            player_x=obs_dict.get("player_x", 0),
            player_y=obs_dict.get("player_y", 0),
            player_aggro=obs_dict.get("player_aggro", 0),

            # Prayer state
            active_prayer=obs_dict.get("active_prayer", 0),
            rigour_active=obs_dict.get("rigour_active", False),

            # Inventory
            bastion_doses=obs_dict.get("bastion_doses", 0),
            sara_brew_doses=obs_dict.get("sara_brew_doses", 0),
            super_restore_doses=obs_dict.get("super_restore_doses", 0),

            # Dynamic Jad/healer state
            jads=jads,
            healers=healers,
            healers_spawned=obs_dict.get("healers_spawned", False),

            # Starting doses for normalization
            starting_bastion_doses=obs_dict.get("starting_bastion_doses", 4),
            starting_sara_brew_doses=obs_dict.get("starting_sara_brew_doses", 4),
            starting_super_restore_doses=obs_dict.get("starting_super_restore_doses", 4),
        )

    def get_action(self, obs_dict: dict) -> tuple[int, float, list, Observation]:
        """Get action, value, processed observation, and parsed Observation from observation dictionary."""
        # Update config from observation if provided
        jad_count = obs_dict.get("jad_count", self.config.jad_count)
        healers_per_jad = obs_dict.get("healers_per_jad", self.config.healers_per_jad)
        if jad_count != self.config.jad_count or healers_per_jad != self.config.healers_per_jad:
            self.config = JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)
            self.action_names = get_action_names(self.config)
            self.normalize_mask = get_normalize_mask(self.config)
            n_continuous = int(np.sum(self.normalize_mask))
            self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))

        obs = self._parse_observation(obs_dict)

        # Convert to array
        obs_array = obs_to_array(obs)

        # Normalize only continuous features (matching training)
        # Both custom and SB3 models now use external normalization
        result = obs_array.copy()
        continuous = obs_array[self.normalize_mask]
        normalized_continuous = self.obs_normalizer.normalize(continuous)
        result[self.normalize_mask] = normalized_continuous
        obs_array = result

        # Store processed array for debugging
        processed_obs = obs_array.tolist()

        # Get action from model
        action, self.lstm_state = self.model.predict(
            obs_array,
            state=self.lstm_state,
            deterministic=True,
        )

        # Try to get value estimate from the policy
        value = 0.0
        try:
            if self.lstm_state is not None:
                obs_tensor = torch.as_tensor(obs_array).float().unsqueeze(0).to(self.model.device)
                # LSTM states from predict: (hidden_list, cell_list)
                # Each list contains arrays of shape (n_envs, hidden_size)
                # predict_values expects (hidden, cell) each with shape (n_layers, n_envs, hidden_size)
                hidden = torch.as_tensor(self.lstm_state[0][0]).unsqueeze(0).to(self.model.device)
                cell = torch.as_tensor(self.lstm_state[1][0]).unsqueeze(0).to(self.model.device)
                lstm_states = (hidden, cell)
                episode_starts = torch.as_tensor([False]).float().to(self.model.device)
                value = self.model.policy.predict_values(
                    obs_tensor, lstm_states, episode_starts
                )
                value = float(value.item())
        except Exception as e:
            # Value extraction failed, keep default 0.0
            pass

        return int(action), value, processed_obs, obs

    def reset_state(self):
        """Reset LSTM state and reward tracking for new episode."""
        self.lstm_state = None
        self.prev_obs = None
        self.episode_length = 0
        self.cumulative_reward = 0.0

    async def handle_connection(self, websocket):
        """Handle a WebSocket connection from the browser."""
        print("Browser connected!")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "observation":
                        obs_dict = data.get("observation", {})
                        action, value, processed_obs, obs = self.get_action(obs_dict)

                        # Compute reward using same function as training
                        step_reward = compute_reward(
                            obs=obs,
                            prev_obs=self.prev_obs,
                            termination=TerminationState.ONGOING,
                            episode_length=self.episode_length,
                            reward_type=self.reward_type,
                        )
                        self.cumulative_reward += step_reward
                        self.episode_length += 1
                        self.prev_obs = obs

                        # Generate action names dynamically based on observation's config
                        obs_config = JadConfig(
                            jad_count=obs_dict.get("jad_count", 1),
                            healers_per_jad=obs_dict.get("healers_per_jad", 3),
                        )
                        action_names = get_action_names(obs_config)

                        response = {
                            "type": "action",
                            "action": action,
                            "action_name": action_names.get(action, "UNKNOWN"),
                            "value": value,
                            "step_reward": step_reward,
                            "cumulative_reward": self.cumulative_reward,
                            "episode_length": self.episode_length,
                            "observation": obs_dict,  # Original dict for readable display
                            "processed_obs": processed_obs,  # Actual array fed to model
                        }
                        await websocket.send(json.dumps(response))

                    elif data.get("type") == "reset":
                        print("Episode reset")
                        self.reset_state()
                        await websocket.send(json.dumps({"type": "ready"}))

                    elif data.get("type") == "terminated":
                        result = data.get("result", "unknown")
                        obs_dict = data.get("observation", {})

                        # Compute terminal reward
                        terminal_reward = 0.0
                        if obs_dict:
                            obs = self._parse_observation(obs_dict)
                            if result == "player_died":
                                termination = TerminationState.PLAYER_DIED
                            elif result == "jad_killed":
                                termination = TerminationState.JAD_KILLED
                            else:
                                termination = TerminationState.TRUNCATED

                            terminal_reward = compute_reward(
                                obs=obs,
                                prev_obs=self.prev_obs,
                                termination=termination,
                                episode_length=self.episode_length,
                                reward_type=self.reward_type,
                            )
                            self.cumulative_reward += terminal_reward

                        print(f"Episode ended: {result}, total reward: {self.cumulative_reward:.1f}")

                        # Send final reward back to browser
                        await websocket.send(json.dumps({
                            "type": "terminated_ack",
                            "result": result,
                            "terminal_reward": terminal_reward,
                            "cumulative_reward": self.cumulative_reward,
                            "episode_length": self.episode_length,
                        }))

                        self.reset_state()

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {message}")

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            print("Browser disconnected")

    async def start(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server."""
        print(f"Starting agent server on ws://{host}:{port}")
        print("Waiting for browser connection...")

        async with serve(self.handle_connection, host, port):
            await asyncio.get_running_loop().create_future()  # Run forever


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket server for Jad agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_sb3_1jad.zip",
        help="Path to SB3 checkpoint (.zip)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port",
    )
    parser.add_argument(
        "--jad-count",
        type=int,
        default=1,
        help="Number of Jads (1-6, default: 1)",
    )
    parser.add_argument(
        "--healers-per-jad",
        type=int,
        default=3,
        help="Number of healers per Jad (0-5, default: 3)",
    )
    parser.add_argument(
        "--reward-func",
        type=str,
        default="default",
        choices=list_reward_functions(),
        help=f"Reward function to use (default: default). Available: {list_reward_functions()}",
    )
    args = parser.parse_args()

    print(f"Using reward function: {args.reward_func}")
    server = AgentServer(
        args.checkpoint,
        jad_count=args.jad_count,
        healers_per_jad=args.healers_per_jad,
        reward_type=args.reward_func,
    )
    await server.start(port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
