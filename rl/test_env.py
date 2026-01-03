#!/usr/bin/env python3
"""Interactive environment tester for debugging and exploration."""

import json
import os
import sys
from dataclasses import asdict

from env_process_wrapper import EnvProcessWrapper
from jad_types import JadConfig
from observations import obs_to_array
from utils import get_action_count, get_action_name


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def format_observation(obs, config: JadConfig) -> str:
    """Format observation in a readable format."""
    lines = []
    lines.append("=== Observation ===")
    lines.append(f"Player: hp={obs.player_hp}, prayer={obs.player_prayer}, "
                 f"ranged={obs.player_ranged}, defence={obs.player_defence}")
    lines.append(f"Location: ({obs.player_location_x}, {obs.player_location_y}), target={obs.player_target}")
    lines.append(f"Prayer: active={obs.active_prayer} (0=none,1=mage,2=range,3=melee), rigour={obs.rigour_active}")
    lines.append(f"Potions: bastion={obs.bastion_doses}/{obs.starting_bastion_doses}, "
                 f"brew={obs.sara_brew_doses}/{obs.starting_sara_brew_doses}, "
                 f"restore={obs.super_restore_doses}/{obs.starting_super_restore_doses}")

    lines.append(f"\nJads ({len(obs.jads)}):")
    for i, jad in enumerate(obs.jads):
        attack_name = {0: "none", 1: "mage", 2: "range", 3: "melee"}.get(jad.attack, "?")
        lines.append(f"  [{i}] hp={jad.hp}, attack={attack_name}, pos=({jad.x},{jad.y}), alive={jad.alive}")

    lines.append(f"\nHealers ({len(obs.healers)}, spawned={obs.healers_spawned}):")
    for i, healer in enumerate(obs.healers):
        target_name = {0: "not_present", 1: "jad", 2: "player"}.get(healer.target, "?")
        lines.append(f"  [{i}] hp={healer.hp}, pos=({healer.x},{healer.y}), target={target_name}")

    return "\n".join(lines)


def format_vector(obs, config: JadConfig, cumulative_reward: float = 0.0) -> str:
    """Format the observation as a numpy vector with labels in two columns."""
    vec = obs_to_array(obs, config)

    jad_count = config.jad_count
    healers_per_jad = config.healers_per_jad
    total_healers = jad_count * healers_per_jad

    # Build list of (index, label, value, marker) tuples
    items = []
    idx = 0

    # Cumulative reward (not part of vector, but useful context)
    items.append((None, "[Cumulative Reward]", None, None))
    items.append(("--", "reward", f"{cumulative_reward:.4f}", ""))

    # Player continuous (9)
    items.append((None, "[Player Continuous]", None, None))
    labels = ["hp", "prayer", "ranged", "defence", "bastion", "brew", "restore", "x", "y"]
    for label in labels:
        items.append((idx, label, f"{vec[idx]:.4f}", ""))
        idx += 1

    # Jad continuous (3 * jad_count)
    items.append((None, f"[Jad Continuous]", None, None))
    for j in range(jad_count):
        items.append((idx, f"jad{j+1}_hp", f"{vec[idx]:.4f}", ""))
        idx += 1
        items.append((idx, f"jad{j+1}_x", f"{vec[idx]:.4f}", ""))
        idx += 1
        items.append((idx, f"jad{j+1}_y", f"{vec[idx]:.4f}", ""))
        idx += 1

    # Healer continuous (3 * total_healers)
    items.append((None, f"[Healer Continuous]", None, None))
    for h in range(total_healers):
        jad_idx = h // healers_per_jad + 1
        healer_idx = h % healers_per_jad + 1
        items.append((idx, f"h{jad_idx}.{healer_idx}_hp", f"{vec[idx]:.4f}", ""))
        idx += 1
        items.append((idx, f"h{jad_idx}.{healer_idx}_x", f"{vec[idx]:.4f}", ""))
        idx += 1
        items.append((idx, f"h{jad_idx}.{healer_idx}_y", f"{vec[idx]:.4f}", ""))
        idx += 1

    # Player target one-hot
    items.append((None, f"[Player Target]", None, None))
    target_labels = ["none"] + [f"jad{j+1}" for j in range(jad_count)]
    for h in range(total_healers):
        jad_idx = h // healers_per_jad + 1
        healer_idx = h % healers_per_jad + 1
        target_labels.append(f"h{jad_idx}.{healer_idx}")
    for label in target_labels:
        marker = "<-" if vec[idx] == 1.0 else ""
        items.append((idx, f"tgt_{label}", f"{vec[idx]:.0f}", marker))
        idx += 1

    # Active prayer one-hot (4)
    items.append((None, "[Active Prayer]", None, None))
    prayer_labels = ["none", "mage", "range", "melee"]
    for label in prayer_labels:
        marker = "<-" if vec[idx] == 1.0 else ""
        items.append((idx, f"pray_{label}", f"{vec[idx]:.0f}", marker))
        idx += 1

    # Jad attack one-hot (4 * jad_count)
    items.append((None, f"[Jad Attack]", None, None))
    attack_labels = ["none", "mage", "range", "melee"]
    for j in range(jad_count):
        for label in attack_labels:
            marker = "<-" if vec[idx] == 1.0 else ""
            items.append((idx, f"j{j+1}_{label}", f"{vec[idx]:.0f}", marker))
            idx += 1

    # Healer target one-hot (3 * total_healers)
    items.append((None, f"[Healer Target]", None, None))
    healer_target_labels = ["absent", "jad", "player"]
    for h in range(total_healers):
        jad_idx = h // healers_per_jad + 1
        healer_idx = h % healers_per_jad + 1
        for label in healer_target_labels:
            marker = "<-" if vec[idx] == 1.0 else ""
            items.append((idx, f"h{jad_idx}.{healer_idx}_{label}", f"{vec[idx]:.0f}", marker))
            idx += 1

    # Binary (2)
    items.append((None, "[Binary]", None, None))
    items.append((idx, "rigour", f"{vec[idx]:.0f}", ""))
    idx += 1
    items.append((idx, "healers_spawned", f"{vec[idx]:.0f}", ""))

    # Format into two columns
    lines = [f"=== Vector (shape={vec.shape}) ==="]
    col_width = 32

    i = 0
    while i < len(items):
        left = items[i]
        right = items[i + 1] if i + 1 < len(items) else None

        if left[0] is None:  # Section header
            if i > 0:
                lines.append("")
            lines.append(left[1])
            i += 1
        elif right and right[0] is None:  # Next item is header, don't pair
            idx_str = f"{left[0]:>2}" if isinstance(left[0], int) else f"{left[0]:>2}"
            left_str = f"  {idx_str}: {left[1]:14s} = {left[2]:>6s} {left[3]}"
            lines.append(left_str)
            i += 1
        elif right:  # Pair two items
            left_idx = f"{left[0]:>2}" if isinstance(left[0], int) else f"{left[0]:>2}"
            right_idx = f"{right[0]:>2}" if isinstance(right[0], int) else f"{right[0]:>2}"
            left_str = f"  {left_idx}: {left[1]:14s} = {left[2]:>6s} {left[3]}"
            right_str = f"  {right_idx}: {right[1]:14s} = {right[2]:>6s} {right[3]}"
            lines.append(f"{left_str:<{col_width}}{right_str}")
            i += 2
        else:  # Odd item at end
            idx_str = f"{left[0]:>2}" if isinstance(left[0], int) else f"{left[0]:>2}"
            left_str = f"  {idx_str}: {left[1]:14s} = {left[2]:>6s} {left[3]}"
            lines.append(left_str)
            i += 1

    return "\n".join(lines)


def format_actions(config: JadConfig) -> str:
    """Format all available actions."""
    count = get_action_count(config)
    lines = [f"=== Actions ({count} total) ==="]
    for i in range(count):
        lines.append(f"  {i}: {get_action_name(i, config)}")
    return "\n".join(lines)


def format_json(obs) -> str:
    """Format observation as JSON."""
    return "=== JSON ===\n" + json.dumps(asdict(obs), indent=2)


def render_screen(config: JadConfig, step: int, cumulative_reward: float, output: str):
    """Clear screen and render the current state."""
    clear_screen()

    # Header
    print(f"=== Jad Environment Tester ===")
    print(f"Config: {config.jad_count} Jad(s), {config.healers_per_jad} healers/Jad")
    print(f"Step: {step}  |  Cumulative Reward: {cumulative_reward:.2f}")
    print()

    # Menu
    print("Commands:")
    print("  <num>  Step with action    o  Observation    v  Vector")
    print("  a      Actions list        j  JSON           r  Reset     q  Quit")
    print("  [Enter = DO_NOTHING]")
    print("-" * 70)

    # Output from previous command
    if output:
        print(output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Jad environment tester")
    parser.add_argument("--jad-count", type=int, default=1, help="Number of Jads (1-6)")
    parser.add_argument("--healers-per-jad", type=int, default=3, help="Healers per Jad (0-5)")
    parser.add_argument("--reward-func", type=str, default="default", help="Reward function")
    args = parser.parse_args()

    config = JadConfig(jad_count=args.jad_count, healers_per_jad=args.healers_per_jad)
    env = EnvProcessWrapper(config=config, reward_func=args.reward_func)

    obs = env.reset()
    step = 0
    cumulative_reward = 0.0
    output = format_observation(obs, config)

    try:
        while True:
            render_screen(config, step, cumulative_reward, output)

            try:
                cmd = input("\n> ").strip().lower()
            except EOFError:
                break

            if not cmd:
                cmd = '0'  # Empty input = DO_NOTHING
            if cmd in ('q', 'quit', 'exit'):
                break
            elif cmd in ('o', 'obs'):
                output = format_observation(obs, config)
            elif cmd in ('v', 'vec', 'vector'):
                output = format_vector(obs, config, cumulative_reward)
            elif cmd in ('j', 'json'):
                output = format_json(obs)
            elif cmd in ('a', 'actions'):
                output = format_actions(config)
            elif cmd in ('r', 'reset'):
                obs = env.reset()
                step = 0
                cumulative_reward = 0.0
                output = "Environment reset.\n\n" + format_observation(obs, config)
            elif cmd.isdigit():
                action = int(cmd)
                action_count = get_action_count(config)
                if action < 0 or action >= action_count:
                    output = f"Invalid action {action}. Must be 0-{action_count-1}"
                    continue

                result = env.step(action)
                obs = result.observation
                step += 1
                cumulative_reward += result.reward

                action_name = get_action_name(action, config)
                lines = [
                    f"Action: {action} ({action_name})",
                    f"Reward: {result.reward:.4f}",
                    f"Terminated: {result.terminated}",
                    "",
                    format_observation(obs, config),
                ]

                if result.terminated:
                    all_jads_dead = all(jad.hp <= 0 for jad in obs.jads)
                    outcome = "JAD_KILLED" if all_jads_dead else "PLAYER_DIED"
                    lines.append(f"\n*** Episode ended: {outcome} ***")
                    lines.append(f"Total steps: {step}, Total reward: {cumulative_reward:.2f}")
                    lines.append("Press 'r' to reset or 'q' to quit.")

                output = "\n".join(lines)
            else:
                output = f"Unknown command: {cmd}"
    finally:
        env.close()
        clear_screen()
        print("Environment closed.")


if __name__ == "__main__":
    main()
