"""
Simplified training script using feedforward network (no LSTM).
"""

import numpy as np
import torch
from env import JadEnv
from simple_agent import SimplePPOAgent
from observations import obs_to_array
from train import compute_reward


def train(
    num_episodes: int = 2000,
    update_interval: int = 2048,
    log_interval: int = 10,
):
    env = JadEnv()
    agent = SimplePPOAgent()

    episode_rewards = []
    total_wins = 0
    interval_wins = 0

    print(f"Training on device: {agent.device}")
    print(f"Simplified training (no LSTM) for {num_episodes} episodes...")
    print()

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0

        while True:
            obs_array = obs_to_array(obs)
            action, log_prob, value = agent.select_action(obs_array)

            result = env.step(action)
            raw_reward = compute_reward(result.observation, obs, result.terminated)
            episode_reward += raw_reward

            # Scale reward
            scaled_reward = raw_reward / 100.0

            agent.store(obs_array, action, scaled_reward, value, log_prob, result.terminated)

            # Update if enough steps
            if agent.buffer_size() >= update_interval:
                next_obs = obs_to_array(result.observation)
                next_obs_norm = agent.normalize_obs(next_obs, update=False)
                with torch.no_grad():
                    _, next_value = agent.model(
                        torch.FloatTensor(next_obs_norm).to(agent.device)
                    )
                    if result.terminated:
                        next_value = torch.zeros_like(next_value)

                agent.update(next_value)

            obs = result.observation

            if result.terminated:
                if result.observation.jad_hp <= 0:
                    total_wins += 1
                    interval_wins += 1
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            win_rate = total_wins / (episode + 1) * 100
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Kills: {interval_wins}/{log_interval} | "
                f"Total: {total_wins} ({win_rate:.1f}%)"
            )
            interval_wins = 0

    env.close()
    print(f"\nFinal: {total_wins}/{num_episodes} wins ({total_wins/num_episodes*100:.1f}%)")

    # Save final checkpoint
    from pathlib import Path
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    agent.save(str(checkpoint_dir / "simple_final.pt"))
    print(f"Saved checkpoint to {checkpoint_dir / 'simple_final.pt'}")

    return agent


if __name__ == "__main__":
    train()
