"""
Interactive Jad simulator.
Supports human play or agent testing.

Usage:
  python play.py              # Human play
  python play.py --agent      # Agent play (uses latest checkpoint)
  python play.py --agent checkpoint.pt  # Agent play with specific checkpoint
"""

import argparse
import torch

from env import JadEnv, Observation
from agent import PPOAgent
from observations import obs_to_array

ACTIONS = {
    0: "WAIT",
    1: "PRAY_MAGE",
    2: "PRAY_RANGE",
    3: "DRINK_RESTORE",
    4: "ATTACK",
    5: "PRAY_MELEE",
    6: "DRINK_BASTION",
    7: "TOGGLE_RIGOUR",
    8: "DRINK_SARA_BREW",
}


def print_obs(obs: Observation, tick: int | None = None):
    """Print observation state."""
    print("\n" + "=" * 50)
    if tick is not None:
        print(f"Tick: {tick}")
    print(f"Player HP:    {obs.player_hp}")
    print(f"Prayer:       {obs.player_prayer}")
    print(f"Active Prayer: {['None', 'Protect Mage', 'Protect Range', 'Protect Melee'][obs.active_prayer]}")
    print(f"Rigour:       {'ON' if obs.rigour_active else 'OFF'}")
    print(f"Jad HP:       {obs.jad_hp}")
    print(f"Jad Attack:   {['None', 'MAGIC', 'RANGE', 'MELEE'][obs.jad_attack]}")
    print(f"Restores:     {obs.restore_doses}")
    print(f"Bastion:      {obs.bastion_doses}")
    print(f"Sara Brews:   {obs.sara_brew_doses}")
    print("=" * 50)


def prompt_human_action() -> int | None:
    """Prompt human for action input."""
    print("\nActions:")
    for num, name in ACTIONS.items():
        print(f"  {num}: {name}")
    print("  q: Quit")

    while True:
        choice = input("\nChoose action: ").strip().lower()
        if choice == 'q':
            return None
        try:
            action = int(choice)
            if action in ACTIONS:
                return action
            print("Invalid action number")
        except ValueError:
            print("Enter a number or 'q'")


def run_human(env: JadEnv):
    """Run in human play mode."""
    print("Starting Jad simulator (Human Mode)...")
    obs = env.reset()
    print_obs(obs)

    tick = 0
    while True:
        action = prompt_human_action()
        if action is None:
            break

        result = env.step(action)
        tick += 1
        obs = result.observation

        print_obs(obs, tick)

        if result.terminated:
            if obs.player_hp <= 0:
                print("\n*** YOU DIED ***")
            elif obs.jad_hp <= 0:
                print("\n*** JAD DEFEATED ***")

            again = input("\nPlay again? (y/n): ").strip().lower()
            if again == 'y':
                obs = env.reset()
                print_obs(obs)
                tick = 0
            else:
                break


def run_agent(env: JadEnv, agent: PPOAgent):
    """Run in agent testing mode."""
    print("Starting Jad simulator (Agent Mode)...")
    print("Press Enter to step, 'q' to quit\n")

    obs = env.reset()
    agent.reset_hidden()
    print_obs(obs)

    tick = 0
    while True:
        # Get agent's action
        obs_array = obs_to_array(obs)
        with torch.no_grad():
            action, _, _ = agent.select_action(obs_array)

        print(f"\nAgent action: {action} ({ACTIONS[action]})")

        # Wait for user input
        user_input = input("Press Enter to step (q to quit): ").strip().lower()
        if user_input == 'q':
            break

        result = env.step(action)
        tick += 1
        obs = result.observation

        print_obs(obs, tick)

        if result.terminated:
            if obs.player_hp <= 0:
                print("\n*** AGENT DIED ***")
            elif obs.jad_hp <= 0:
                print("\n*** AGENT DEFEATED JAD ***")

            again = input("\nRun again? (y/n): ").strip().lower()
            if again == 'y':
                obs = env.reset()
                agent.reset_hidden()
                print_obs(obs)
                tick = 0
            else:
                break


def main():
    parser = argparse.ArgumentParser(description="Jad simulator - human or agent play")
    parser.add_argument(
        "--agent",
        nargs="?",
        const="checkpoints/final.pt",
        default=None,
        metavar="CHECKPOINT",
        help="Run with agent (default: checkpoints/final.pt)",
    )
    args = parser.parse_args()

    env = JadEnv()

    try:
        if args.agent:
            # Load agent
            agent = PPOAgent()
            try:
                agent.load(args.agent)
                print(f"Loaded checkpoint: {args.agent}")
            except FileNotFoundError:
                print(f"Checkpoint not found: {args.agent}")
                print("Running with untrained agent...")

            # Set to evaluation mode (don't update normalizer)
            agent.eval_mode()
            run_agent(env, agent)
        else:
            run_human(env)
    finally:
        env.close()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
