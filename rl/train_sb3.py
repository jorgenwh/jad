"""
Training script using Stable-Baselines3 with parallelized environments.

Usage:
    python train_sb3.py                    # Train with default settings
    python train_sb3.py --num-envs 8       # Use 8 parallel environments
    python train_sb3.py --timesteps 500000 # Train for 500k timesteps
"""

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from jad_gymnasium_env import make_jad_env, OBS_DIM


class BestModelCallback(BaseCallback):
    """
    Callback that saves the model only when average reward improves.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        save_path: str = "checkpoints/best_sb3.pt",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_mean_reward = float("-inf")
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Track episode rewards
        # Note: with VecEnv, infos contains episode info when done
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        # Check if we should evaluate
        if self.n_calls % self.eval_freq == 0 and len(self.episode_rewards) > 0:
            # Use last 100 episodes for evaluation
            recent_rewards = self.episode_rewards[-100:]
            mean_reward = sum(recent_rewards) / len(recent_rewards)

            if self.verbose > 0:
                print(
                    f"Step {self.n_calls} | "
                    f"Episodes: {len(self.episode_rewards)} | "
                    f"Mean reward (last {len(recent_rewards)}): {mean_reward:.1f}"
                )

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"  [NEW BEST - saved to {self.save_path}]")

        return True


def train(
    num_envs: int = 4,
    total_timesteps: int = 200_000,
    n_steps: int = 512,
    batch_size: int = 64,
    n_epochs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    lstm_hidden_size: int = 64,
    checkpoint_dir: str = "checkpoints",
):
    """
    Train using Stable-Baselines3 PPO with LSTM policy and parallelized environments.

    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps across all envs
        n_steps: Steps per env before update (total = n_steps * num_envs)
        batch_size: Minibatch size for updates
        n_epochs: Epochs per PPO update
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        lstm_hidden_size: LSTM hidden layer size
        checkpoint_dir: Directory for saving checkpoints
    """
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # Create vectorized environments
    print(f"Creating {num_envs} parallel environments...")

    if num_envs == 1:
        # Single env - use DummyVecEnv for easier debugging
        env = DummyVecEnv([make_jad_env()])
    else:
        # Multiple envs - use SubprocVecEnv for true parallelism
        env = SubprocVecEnv([make_jad_env() for _ in range(num_envs)])

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # LSTM policy configuration
    policy_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=1,
    )

    # Create RecurrentPPO model (LSTM-based PPO from sb3-contrib)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto",
    )

    print(f"\nModel architecture:")
    print(f"  Policy: MlpLstmPolicy (hidden_size={lstm_hidden_size})")
    print(f"  Steps per update: {n_steps} * {num_envs} = {n_steps * num_envs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Device: {model.device}")

    # Create callback for saving best model
    callback = BestModelCallback(
        eval_freq=n_steps * num_envs * 10,  # Evaluate every ~10 updates
        save_path=str(checkpoint_path / "best_sb3"),
        verbose=1,
    )

    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = checkpoint_path / "final_sb3"
    model.save(str(final_path))
    print(f"\nSaved final model: {final_path}")
    print(f"Best mean reward: {callback.best_mean_reward:.1f}")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Jad agent with SB3 (LSTM)")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Steps per env before update (default: 512)",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=64,
        help="LSTM hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    args = parser.parse_args()

    train(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        lstm_hidden_size=args.lstm_hidden_size,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
