"""
Training script for Jad RL agent.
"""

import numpy as np
import torch
from pathlib import Path

from env import JadEnv, Observation
from agent import PPOAgent
from observations import obs_to_array


MAX_EPISODE_LENGTH = 300  # Hard cap - must kill Jad within this many ticks


def compute_reward(
    obs: Observation,
    prev_obs: Observation | None,
    terminated: bool,
    episode_length: int = 0,
) -> float:
    """
    Original reward structure that learns well.
    Episode length is capped externally to prevent stalling exploit.
    """
    reward = 0.0

    # Prayer switching feedback - every tick jad is attacking
    if prev_obs is not None and prev_obs.jad_attack != 0:
        if obs.active_prayer == prev_obs.jad_attack:
            reward += 1
        else:
            reward -= 1

    # Survival bonus
    reward += 0.1

    if not obs.player_aggro:
        reward -= 1  # Penalty for not attacking Jad

    if prev_obs is not None:
        # Damage taken penalty
        damage_taken = prev_obs.player_hp - obs.player_hp
        if damage_taken > 0:
            reward -= damage_taken * 0.1

        # Damage dealt reward
        damage_dealt = prev_obs.jad_hp - obs.jad_hp
        if damage_dealt > 0:
            reward += damage_dealt * 0.1

    # Terminal rewards
    if terminated:
        if obs.player_hp <= 0:
            reward -= 100.0  # Death
        elif obs.jad_hp <= 0:
            reward += 100.0  # Win

            reward -= episode_length * 0.25  # Faster kills are better

    return reward


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

    # Initialize
    env = JadEnv()
    agent = PPOAgent()

    # Tracking
    episode_rewards = []
    episode_lengths = []
    total_wins = 0
    interval_wins = 0
    best_avg_reward = float('-inf')  # Track best model

    print(f"Training on device: {agent.device}")
    print(f"Starting training for {num_episodes} episodes...")

    # Initialize hidden state once at start
    agent.reset_hidden()

    try:
        for episode in range(num_episodes):
            obs = env.reset()
            # Reset hidden at episode start (new episode = fresh context)
            agent.reset_hidden()

            episode_reward = 0
            episode_length = 0

            while True:
                # Store initial hidden state at the start of a new rollout sequence
                if len(agent.buffer) == 0:
                    agent.store_initial_hidden()

                # Convert observation
                obs_array = obs_to_array(obs)

                # Select action
                action, log_prob, value = agent.select_action(obs_array)

                # Take step
                result = env.step(action)

                # Compute reward (obs = state before action, result.observation = state after)
                raw_reward = compute_reward(result.observation, obs, result.terminated, episode_length)
                episode_reward += raw_reward

                # Scale reward to roughly [-1, 1] range
                scaled_reward = raw_reward / 100.0

                # Store transition
                agent.store_transition(
                    obs_array, action, log_prob, value, scaled_reward, result.terminated
                )
                episode_length += 1

                # Update if enough steps collected
                if len(agent.buffer) >= update_interval:
                    # Get next value for GAE (use normalized observation)
                    next_obs_array = obs_to_array(result.observation)
                    next_obs_normalized = agent.normalize_obs(next_obs_array, update=False)
                    with torch.no_grad():
                        next_obs_tensor = torch.FloatTensor(next_obs_normalized).to(agent.device)
                        _, next_value, _ = agent.model.forward(
                            next_obs_tensor.unsqueeze(0), agent.hidden
                        )
                        if result.terminated:
                            next_value = torch.zeros_like(next_value)

                    losses = agent.update(next_value)
                    # Note: Don't reset hidden here - it continues from current state
                    # The next rollout will store the current hidden as its initial state

                # Update state
                obs = result.observation

                # Check for timeout (stalling prevention)
                if episode_length >= MAX_EPISODE_LENGTH and not result.terminated:
                    # Timeout - treat as death
                    episode_reward -= 100.0
                    timeout_reward = -100.0 / 100.0  # Scaled
                    agent.store_transition(
                        obs_to_array(obs), 0, log_prob, value, timeout_reward, True
                    )
                    agent.reset_hidden()
                    break

                if result.terminated:
                    # Track wins
                    if result.observation.jad_hp <= 0:
                        total_wins += 1
                        interval_wins += 1
                    # Reset hidden state for new episode
                    agent.reset_hidden()
                    break

            # Track episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_length = np.mean(episode_lengths[-log_interval:])

                # Check if this is a new best
                is_best = avg_reward > best_avg_reward
                if is_best:
                    best_avg_reward = avg_reward
                    # Save best model (overwrite previous best)
                    save_path = checkpoint_path / "best.pt"
                    agent.save(str(save_path))

                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {avg_reward:.1f} | "
                    f"Length: {avg_length:.0f} | "
                    f"Kills: {interval_wins}/{log_interval}"
                    f"{' [NEW BEST - saved]' if is_best else ''}"
                )
                interval_wins = 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        env.close()

    print(f"\nTraining complete: {total_wins}/{num_episodes} wins ({total_wins/num_episodes*100:.1f}%)")
    print(f"Best avg reward: {best_avg_reward:.1f} (saved to {checkpoint_path / 'best.pt'})")
    return agent, episode_rewards
