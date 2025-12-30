"""
WebSocket server for browser visualization of the trained agent.

Run this server, then open the browser simulation to see the agent play.
Supports both custom .pt checkpoints and SB3 .zip checkpoints.

Usage:
    python ws_server.py                                    # Use default custom checkpoint
    python ws_server.py --checkpoint checkpoints/best.pt   # Custom checkpoint
    python ws_server.py --checkpoint checkpoints/best_sb3.zip  # SB3 checkpoint
"""

import asyncio
import json
import numpy as np
import torch
from websockets.asyncio.server import serve

from observations import obs_to_array, NORMALIZE_MASK
from vec_normalize import RunningNormalizer
from env import Observation

# Action names for logging
ACTIONS = {
    0: "WAIT",
    1: "PRAY_MAGE",
    2: "PRAY_RANGE",
    3: "DRINK_RESTORE",
    4: "ATTACK",
    5: "PRAY_MELEE",
    6: "DRINK_SUPER_COMBAT",
    7: "TOGGLE_PIETY",
    8: "DRINK_SARA_BREW",
}


class AgentServer:
    def __init__(self, checkpoint_path: str | None = None):
        self.is_sb3 = False
        self.agent = None
        self.model = None  # For SB3
        self.lstm_state = None  # For SB3 LSTM state

        if checkpoint_path and checkpoint_path.endswith(".zip"):
            # SB3 checkpoint
            self._load_sb3(checkpoint_path)
        else:
            # Custom checkpoint
            self._load_custom(checkpoint_path)

    def _load_custom(self, checkpoint_path: str | None):
        """Load custom .pt checkpoint."""
        from agent import PPOAgent
        from pathlib import Path

        self.agent = PPOAgent()
        self.agent.eval_mode()
        self.agent.reset_hidden()

        # Set up observation normalizer (same as SelectiveVecNormalize)
        n_continuous = int(np.sum(NORMALIZE_MASK))
        self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))
        self.normalize_mask = NORMALIZE_MASK

        if checkpoint_path:
            try:
                self.agent.load(checkpoint_path)
                print(f"Loaded custom checkpoint: {checkpoint_path}")

                # Load normalizer stats saved by SelectiveVecNormalize during training
                checkpoint_dir = Path(checkpoint_path).parent
                normalizer_path = checkpoint_dir / "normalizer.npz"
                if normalizer_path.exists():
                    data = np.load(normalizer_path)
                    # Only need obs normalizer for inference (rewards not used)
                    self.obs_normalizer.load_state_dict({
                        "mean": data["obs_mean"],
                        "var": data["obs_var"],
                        "count": int(data["obs_count"]),
                    })
                    print(f"Loaded normalizer stats: {normalizer_path}")
                else:
                    print(f"Warning: No normalizer stats found at {normalizer_path}")
                    print("  Observations will not be normalized correctly!")
            except FileNotFoundError:
                print(f"Checkpoint not found: {checkpoint_path}")
                print("Using untrained agent")
        else:
            print("No checkpoint specified, using untrained agent")

    def _load_sb3(self, checkpoint_path: str):
        """Load SB3 .zip checkpoint."""
        from sb3_contrib import RecurrentPPO
        from pathlib import Path

        self.is_sb3 = True
        try:
            self.model = RecurrentPPO.load(checkpoint_path, device="auto")
            self.lstm_state = None  # Will be initialized on first prediction
            print(f"Loaded SB3 checkpoint: {checkpoint_path}")

            # Set up observation normalizer (matching SelectiveVecNormalize used in training)
            n_continuous = int(np.sum(NORMALIZE_MASK))
            self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))
            self.normalize_mask = NORMALIZE_MASK

            # Load normalizer stats saved by SelectiveVecNormalize during training
            checkpoint_dir = Path(checkpoint_path).parent
            normalizer_path = checkpoint_dir / "normalizer_sb3.npz"
            if normalizer_path.exists():
                data = np.load(normalizer_path)
                # Only need obs normalizer for inference (rewards not used)
                self.obs_normalizer.load_state_dict({
                    "mean": data["obs_mean"],
                    "var": data["obs_var"],
                    "count": int(data["obs_count"]),
                })
                print(f"Loaded normalizer stats: {normalizer_path}")
            else:
                print(f"Warning: No normalizer stats found at {normalizer_path}")
                print("  Observations will not be normalized correctly!")
        except FileNotFoundError:
            print(f"Checkpoint not found: {checkpoint_path}")
            raise

    def get_action(self, obs_dict: dict) -> tuple[int, float, list]:
        """Get action, value, and processed observation from observation dictionary."""
        # Convert dict to Observation dataclass
        obs = Observation(
            player_hp=obs_dict.get("player_hp", 99),
            player_prayer=obs_dict.get("player_prayer", 99),
            active_prayer=obs_dict.get("active_prayer", 0),
            jad_hp=obs_dict.get("jad_hp", 255),
            jad_attack=obs_dict.get("jad_attack", 0),
            restore_doses=obs_dict.get("restore_doses", 0),
            super_combat_doses=obs_dict.get("super_combat_doses", 0),
            sara_brew_doses=obs_dict.get("sara_brew_doses", 0),
            piety_active=obs_dict.get("piety_active", False),
            player_aggro=obs_dict.get("player_aggro", False),
        )

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

        if self.is_sb3:
            # SB3 model - use predict with LSTM state
            action, self.lstm_state = self.model.predict(
                obs_array,
                state=self.lstm_state,
                deterministic=True,
            )
            # Value estimation for RecurrentPPO is complex due to LSTM state format
            # Return 0.0 as placeholder - main debugging info is observation/action
            return int(action), 0.0, processed_obs
        else:
            # Custom model - use deterministic action selection
            with torch.no_grad():
                action, _, value = self.agent.select_action(obs_array, deterministic=True)
            return action, float(value.item()), processed_obs

    def reset_state(self):
        """Reset LSTM state for new episode."""
        if self.is_sb3:
            self.lstm_state = None
        else:
            self.agent.reset_hidden()

    async def handle_connection(self, websocket):
        """Handle a WebSocket connection from the browser."""
        print("Browser connected!")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "observation":
                        obs = data.get("observation", {})
                        action, value, processed_obs = self.get_action(obs)

                        response = {
                            "type": "action",
                            "action": action,
                            "action_name": ACTIONS.get(action, "UNKNOWN"),
                            "value": value,
                            "observation": obs,  # Original dict for readable display
                            "processed_obs": processed_obs,  # Actual array fed to model
                        }
                        await websocket.send(json.dumps(response))

                    elif data.get("type") == "reset":
                        print("Episode reset")
                        self.reset_state()
                        await websocket.send(json.dumps({"type": "ready"}))

                    elif data.get("type") == "terminated":
                        result = data.get("result", "unknown")
                        print(f"Episode ended: {result}")
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
        default="checkpoints/best.pt",
        help="Path to checkpoint (.pt for custom, .zip for SB3)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port",
    )
    args = parser.parse_args()

    server = AgentServer(args.checkpoint)
    await server.start(port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
