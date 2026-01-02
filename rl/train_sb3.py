import argparse
import time
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from jad_types import JadConfig
from utils import get_action_count
from jad_gymnasium_env import make_jad_env
from observations import get_obs_dim
from vec_normalize import SelectiveVecNormalize

# Available reward functions (defined in TypeScript src/core/rewards.ts)
REWARD_FUNCTIONS = ['default', 'sparse', 'multijad']


class EpisodeStatsCallback(BaseCallback):
    """
    Callback that tracks episode stats and saves the model when average reward improves.
    Prints stats every 100 episodes similar to the original train.py.
    """

    def __init__(
        self,
        vec_normalize_env: SelectiveVecNormalize,
        num_envs: int,
        log_interval: int = 100,
        save_path: str = "checkpoints/best_sb3.pt",
        normalizer_path: str = "checkpoints/normalizer_sb3.npz",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.vec_normalize_env = vec_normalize_env
        self.num_envs = num_envs
        self.log_interval = log_interval
        self.save_path = save_path
        self.normalizer_path = normalizer_path
        self.best_mean_reward = float("-inf")

        # Track stats for current block of episodes
        self.episode_raw_rewards = []
        self.episode_norm_rewards = []
        self.episode_lengths = []
        self.episode_kills = 0
        self.total_episodes = 0

        # Track per-env normalized reward accumulation
        self.current_norm_rewards = [0.0] * num_envs

        # Time tracking
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Accumulate normalized rewards per env
        rewards = self.locals.get("rewards", [])
        for i, r in enumerate(rewards):
            self.current_norm_rewards[i] += r

        # Track episode stats from VecEnv infos
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if "episode" in info:
                # Raw reward from Monitor wrapper
                self.episode_raw_rewards.append(info["episode"]["r"])
                # Normalized reward we accumulated
                self.episode_norm_rewards.append(self.current_norm_rewards[i])
                self.episode_lengths.append(info["episode"]["l"])
                self.total_episodes += 1

                # Track kills from our custom info
                if info.get("outcome") == "kill":
                    self.episode_kills += 1

                # Check if we should log (every log_interval episodes)
                if self.total_episodes % self.log_interval == 0:
                    self._log_stats()

            # Reset normalized reward accumulator when episode ends
            if dones[i] if i < len(dones) else False:
                self.current_norm_rewards[i] = 0.0

        return True

    def _log_stats(self):
        """Log stats for the last block of episodes."""
        if len(self.episode_raw_rewards) == 0:
            return

        # Calculate stats for the last log_interval episodes
        recent_raw = self.episode_raw_rewards[-self.log_interval :]
        recent_norm = self.episode_norm_rewards[-self.log_interval :]
        recent_len = self.episode_lengths[-self.log_interval :]

        mean_raw_reward = sum(recent_raw) / len(recent_raw)
        mean_norm_reward = sum(recent_norm) / len(recent_norm)
        mean_length = sum(recent_len) / len(recent_len)
        min_raw, max_raw = min(recent_raw), max(recent_raw)
        min_len, max_len = int(min(recent_len)), int(max(recent_len))
        kills = self.episode_kills

        # Print stats
        elapsed = int(time.time() - self.start_time)
        elapsed_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        start_ep = self.total_episodes - self.log_interval + 1
        print(
            f"Episodes {start_ep}-{self.total_episodes} | "
            f"Raw: {mean_raw_reward:.1f} ({min_raw:.0f}/{max_raw:.0f}) | "
            f"Norm: {mean_norm_reward:.2f} | "
            f"Len: {mean_length:.0f} ({min_len}/{max_len}) | "
            f"Kills: {kills}/{self.log_interval} | "
            f"Steps: {self.num_timesteps:,} | "
            f"Time: {elapsed_str}"
        )

        # Save if this is the best so far (using raw reward for interpretability)
        if mean_raw_reward > self.best_mean_reward:
            self.best_mean_reward = mean_raw_reward
            self.model.save(self.save_path)
            # Save normalizer stats alongside the model
            self.vec_normalize_env.save_normalizer(self.normalizer_path)
            print(f"  [NEW BEST - saved to {self.save_path}]")

        # Reset kills counter for next block
        self.episode_kills = 0


def train(
    jad_count: int = 1,
    healers_per_jad: int = 3,
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
    resume_path: str | None = None,
    reward_func: str = "default",
):
    """
    Train using Stable-Baselines3 PPO with LSTM policy and parallelized environments.

    Args:
        jad_count: Number of Jads (1-6)
        healers_per_jad: Number of healers per Jad (0-5)
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
        resume_path: Path to .zip checkpoint to resume training from
        reward_func: Reward function to use (default, sparse, prayer_only, aggressive)
    """
    # Create config
    config = JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)
    obs_dim = get_obs_dim(config)
    action_count = get_action_count(config)

    print(f"Configuration: {jad_count} Jad(s), {healers_per_jad} healers per Jad")
    print(f"Reward function: {reward_func}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action count: {action_count}")

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # Checkpoint paths include jad count for separate models
    model_save_path = checkpoint_path / f"best_sb3_{jad_count}jad"
    normalizer_path = checkpoint_path / f"normalizer_sb3_{jad_count}jad.npz"

    # Create vectorized environments
    print(f"Creating {num_envs} parallel environments...")

    # Note: DummyVecEnv is often faster than SubprocVecEnv for fast environments
    # because it avoids inter-process communication overhead.
    venv = DummyVecEnv([make_jad_env(config, reward_type=reward_func) for _ in range(num_envs)])

    # Wrap with SelectiveVecNormalize for shared observation normalization
    # This normalizes only continuous features, leaving one-hot encodings unchanged
    env = SelectiveVecNormalize(venv, training=True, config=config)

    # Load normalizer stats if resuming
    if resume_path and normalizer_path.exists():
        env.set_normalizer_state(SelectiveVecNormalize.load_normalizer(str(normalizer_path)))
        print(f"Loaded normalizer stats: {normalizer_path}")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # LSTM policy configuration
    policy_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=1,
    )

    # Create or load RecurrentPPO model
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        model = RecurrentPPO.load(resume_path, env=env, device="auto")
    else:
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
            verbose=0,
            device="auto",
        )

    print(f"\nModel architecture:")
    print(f"  Policy: MlpLstmPolicy (hidden_size={lstm_hidden_size})")
    print(f"  Steps per update: {n_steps} * {num_envs} = {n_steps * num_envs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Device: {model.device}")

    # Create callback for tracking stats and saving best model
    callback = EpisodeStatsCallback(
        vec_normalize_env=env,
        num_envs=num_envs,
        log_interval=100,  # Print stats every 100 episodes
        save_path=str(model_save_path),
        normalizer_path=str(normalizer_path),
        verbose=1,
    )

    # Handle unlimited timesteps
    if total_timesteps == 0:
        total_timesteps = 10**12  # Effectively unlimited
        print(f"\nStarting training (unlimited, Ctrl+C to stop)...")
    else:
        print(f"\nStarting training for {total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Jad agent with SB3 (LSTM)")
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
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Total training timesteps (default: 0 = unlimited)",
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
        default=128,
        help="LSTM hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Minibatch size for PPO updates (default: 64)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .zip to resume training from",
    )
    parser.add_argument(
        "--reward-func",
        type=str,
        default="default",
        choices=REWARD_FUNCTIONS,
        help=f"Reward function to use (default: default). Available: {REWARD_FUNCTIONS}",
    )
    args = parser.parse_args()

    train(
        jad_count=args.jad_count,
        healers_per_jad=args.healers_per_jad,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lstm_hidden_size=args.lstm_hidden_size,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        reward_func=args.reward_func,
    )


if __name__ == "__main__":
    main()
