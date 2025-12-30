"""
Training script for Jad RL agent (custom implementation).

Usage:
    python train.py                        # Train with default settings
    python train.py --episodes 50000       # Train for 50k episodes
    python train.py --episodes 0           # Train forever (Ctrl+C to stop)
    python train.py --update-interval 4096 # Larger rollouts before update
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv

from jad_gymnasium_env import make_jad_env
from vec_normalize import SelectiveVecNormalize
from agent import PPOAgent


def train(
    num_episodes: int = 1000,
    update_interval: int = 2048,
    log_interval: int = 100,
    checkpoint_dir: str = "checkpoints",
):
    """
    Main training loop.

    Args:
        num_episodes: Number of episodes to train
        update_interval: Steps between PPO updates
        log_interval: Episodes between logging
        checkpoint_dir: Directory for saving checkpoints
    """
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    normalizer_path = checkpoint_path / "normalizer.npz"

    # Create environment with SelectiveVecNormalize wrapper
    # Even with single env, use the wrapper for consistent normalization
    venv = DummyVecEnv([make_jad_env()])
    env = SelectiveVecNormalize(venv, training=True)

    # Initialize agent
    agent = PPOAgent()

    # Tracking
    episode_raw_rewards = []
    episode_norm_rewards = []
    episode_lengths = []
    total_wins = 0
    interval_wins = 0
    best_avg_reward = float('-inf')
    total_steps = 0
    start_time = time.time()

    # Current episode tracking
    current_episode_raw_reward = 0.0
    current_episode_norm_reward = 0.0
    current_episode_length = 0

    # Handle unlimited episodes
    unlimited = num_episodes == 0
    if unlimited:
        num_episodes = 10**9  # Effectively unlimited

    print(f"Training on device: {agent.device}")
    if unlimited:
        print("Starting training (unlimited, Ctrl+C to stop)...")
    else:
        print(f"Starting training for {num_episodes} episodes...")

    # Initialize
    agent.reset_hidden()
    obs = env.reset()  # Shape: (1, obs_dim) - already normalized
    episode_count = 0

    try:
        while episode_count < num_episodes:
            # Store initial hidden state at the start of a new rollout sequence
            if len(agent.buffer) == 0:
                agent.store_initial_hidden()

            # Select action (obs is already normalized by SelectiveVecNormalize)
            obs_flat = obs[0]  # Remove batch dimension for agent
            action, log_prob, value = agent.select_action(obs_flat)

            # Take step (action needs to be array for VecEnv)
            next_obs, reward, done, info = env.step(np.array([action]))

            # Reward is normalized by SelectiveVecNormalize (VecNormalize-style)
            reward_scalar = reward[0]

            # Accumulate both raw and normalized rewards for logging
            current_episode_raw_reward += info[0].get("raw_reward", 0.0)
            current_episode_norm_reward += reward_scalar
            current_episode_length += 1
            total_steps += 1

            # Store transition
            agent.store_transition(
                obs_flat, action, log_prob, value, reward_scalar, done[0]
            )

            # Update if enough steps collected
            if len(agent.buffer) >= update_interval:
                # Get next value for GAE
                with torch.no_grad():
                    next_obs_tensor = torch.FloatTensor(next_obs[0]).to(agent.device)
                    _, next_value, _ = agent.model.forward(
                        next_obs_tensor.unsqueeze(0), agent.hidden
                    )
                    if done[0]:
                        next_value = torch.zeros_like(next_value)

                agent.update(next_value)

            # Update observation
            obs = next_obs

            # Check for episode end
            if done[0]:
                # Track outcome
                if info[0].get("outcome") == "kill":
                    total_wins += 1
                    interval_wins += 1

                # Store episode stats
                episode_raw_rewards.append(current_episode_raw_reward)
                episode_norm_rewards.append(current_episode_norm_reward)
                episode_lengths.append(current_episode_length)
                episode_count += 1

                # Logging
                if episode_count % log_interval == 0:
                    recent_raw = episode_raw_rewards[-log_interval:]
                    recent_norm = episode_norm_rewards[-log_interval:]
                    recent_len = episode_lengths[-log_interval:]

                    avg_raw_reward = np.mean(recent_raw)
                    avg_norm_reward = np.mean(recent_norm)
                    avg_length = np.mean(recent_len)
                    min_raw, max_raw = np.min(recent_raw), np.max(recent_raw)
                    min_len, max_len = int(np.min(recent_len)), int(np.max(recent_len))

                    # Check if this is a new best (using raw reward for interpretability)
                    is_best = avg_raw_reward > best_avg_reward
                    if is_best:
                        best_avg_reward = avg_raw_reward
                        # Save best model and normalizer
                        agent.save(str(checkpoint_path / "best.pt"))
                        env.save_normalizer(str(normalizer_path))

                    elapsed = int(time.time() - start_time)
                    elapsed_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
                    start_ep = episode_count + 1 - log_interval
                    print(
                        f"Episodes {start_ep}-{episode_count} | "
                        f"Raw: {avg_raw_reward:.1f} ({min_raw:.0f}/{max_raw:.0f}) | "
                        f"Norm: {avg_norm_reward:.2f} | "
                        f"Len: {avg_length:.0f} ({min_len}/{max_len}) | "
                        f"Kills: {interval_wins}/{log_interval} | "
                        f"Steps: {total_steps:,} | "
                        f"Time: {elapsed_str}"
                    )
                    if is_best:
                        print(f"  [NEW BEST - saved to {checkpoint_path / 'best.pt'}]")
                    interval_wins = 0

                # Reset for next episode
                current_episode_raw_reward = 0.0
                current_episode_norm_reward = 0.0
                current_episode_length = 0
                agent.reset_hidden()
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        env.close()

    # actual_episodes = len(episode_raw_rewards)
    # win_rate = (total_wins / actual_episodes * 100) if actual_episodes > 0 else 0
    # print(f"\nTraining complete: {total_wins}/{actual_episodes} wins ({win_rate:.1f}%)")
    # print(f"Best avg raw reward: {best_avg_reward:.1f} (saved to {checkpoint_path / 'best.pt'})")
    # return agent, episode_raw_rewards


def main():
    parser = argparse.ArgumentParser(description="Train Jad agent (custom implementation)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of episodes to train (default: 100000, 0 = unlimited)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Steps between PPO updates (default: 2048)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Episodes between logging (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
