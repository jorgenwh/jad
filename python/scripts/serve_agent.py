import asyncio
import json
import numpy as np
import torch
from websockets.asyncio.server import serve
from sb3_contrib import RecurrentPPO
from pathlib import Path

from jad import JadConfig, JadState, HealerState, Observation
from jad.env import obs_to_array, get_normalize_mask, RunningNormalizer


class AgentServer:
    def __init__(self, checkpoint_path: str, jad_count: int = 1,
                 healers_per_jad: int = 3):
        self.model = None
        self.lstm_state = None
        self.config = JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
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
                target=h.get("target", 0),
            )
            for h in healers_data
        ]

        return Observation(
            # Player state
            player_hp=obs_dict.get("player_hp", 99),
            player_prayer=obs_dict.get("player_prayer", 99),
            player_ranged=obs_dict.get("player_ranged", 99),
            player_defence=obs_dict.get("player_defence", 99),
            player_location_x=obs_dict.get("player_location_x", 0),
            player_location_y=obs_dict.get("player_location_y", 0),
            player_target=obs_dict.get("player_target", 0),

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

    def get_action(self, obs_dict: dict) -> tuple[int, float]:
        obs = self._parse_observation(obs_dict)

        # Convert to array
        obs_array = obs_to_array(obs, self.config)

        # Normalize only continuous features (matching training)
        result = obs_array.copy()
        continuous = obs_array[self.normalize_mask]
        normalized_continuous = self.obs_normalizer.normalize(continuous)
        result[self.normalize_mask] = normalized_continuous
        obs_array = result

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
                hidden = torch.as_tensor(self.lstm_state[0][0]).unsqueeze(0).to(self.model.device)
                cell = torch.as_tensor(self.lstm_state[1][0]).unsqueeze(0).to(self.model.device)
                lstm_states = (hidden, cell)
                episode_starts = torch.as_tensor([False]).float().to(self.model.device)
                value = self.model.policy.predict_values(
                    obs_tensor, lstm_states, episode_starts
                )
                value = float(value.item())
        except Exception:
            pass

        return int(action), value

    def reset_state(self):
        self.lstm_state = None

    async def handle_connection(self, websocket):
        print("Browser connected!")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "step":
                        obs_dict = data.get("observation", {})
                        terminated = data.get("terminated", False)

                        if terminated:
                            print("Episode ended")
                            self.reset_state()
                        else:
                            action, value = self.get_action(obs_dict)
                            await websocket.send(json.dumps({"action": action, "value": value}))

                    elif data.get("type") == "reset":
                        print("Episode reset")
                        self.reset_state()

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {message}")

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            print("Browser disconnected")

    async def start(self, host: str = "localhost", port: int = 8765):
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
    args = parser.parse_args()

    server = AgentServer(
        args.checkpoint,
        jad_count=args.jad_count,
        healers_per_jad=args.healers_per_jad,
    )
    await server.start(port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
