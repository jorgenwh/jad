"""
WebSocket server for browser visualization of the trained agent.

Run this server, then open the browser simulation to see the agent play.
"""

import asyncio
import json
import torch
from websockets.asyncio.server import serve

from agent import PPOAgent
from observations import obs_to_array
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
        self.agent = PPOAgent()
        self.agent.training = False  # Don't update normalizer
        self.agent.eval_mode()
        self.agent.reset_hidden()  # Initialize LSTM hidden state

        if checkpoint_path:
            try:
                self.agent.load(checkpoint_path)
                print(f"Loaded checkpoint: {checkpoint_path}")
            except FileNotFoundError:
                print(f"Checkpoint not found: {checkpoint_path}")
                print("Using untrained agent")
        else:
            print("No checkpoint specified, using untrained agent")

    def get_action(self, obs_dict: dict) -> int:
        """Get action from observation dictionary."""
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

        # Get action from agent
        with torch.no_grad():
            action, _, _ = self.agent.select_action(obs_array)

        return action

    async def handle_connection(self, websocket):
        """Handle a WebSocket connection from the browser."""
        print("Browser connected!")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "observation":
                        obs = data.get("observation", {})
                        action = self.get_action(obs)

                        response = {
                            "type": "action",
                            "action": action,
                            "action_name": ACTIONS.get(action, "UNKNOWN"),
                        }
                        await websocket.send(json.dumps(response))

                    elif data.get("type") == "reset":
                        print("Episode reset")
                        self.agent.reset_hidden()  # Reset LSTM state
                        await websocket.send(json.dumps({"type": "ready"}))

                    elif data.get("type") == "terminated":
                        result = data.get("result", "unknown")
                        print(f"Episode ended: {result}")
                        self.agent.reset_hidden()  # Reset LSTM state for next episode

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
        help="Path to agent checkpoint",
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
