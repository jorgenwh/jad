"""
PPO Agent with LSTM support.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field

from model import ActorCritic
from observations import OBS_DIM


@dataclass
class RolloutBuffer:
    """Stores trajectory data for PPO updates."""

    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    # Initial hidden state for the sequence (stored once at start)
    initial_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)

    def set_initial_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store initial hidden state for this rollout sequence."""
        # Detach and clone to avoid graph issues
        self.initial_hidden = (hidden[0].detach().clone(), hidden[1].detach().clone())

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.initial_hidden = None

    def __len__(self) -> int:
        return len(self.observations)


class PPOAgent:
    """
    PPO agent with LSTM policy.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = 9,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.device = device

        # Model and optimizer
        self.model = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Rollout storage
        self.buffer = RolloutBuffer()

        # LSTM hidden state (reset each episode)
        self.hidden = None

    def reset_hidden(self) -> None:
        """Reset LSTM hidden state for new episode."""
        self.hidden = self.model.init_hidden(batch_size=1)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action given observation.

        Args:
            obs: Pre-normalized observation from SelectiveVecNormalize wrapper
            deterministic: If True, return argmax action instead of sampling

        Returns:
            action: Selected action
            log_prob: Log probability
            value: State value estimate
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)

        with torch.no_grad():
            action, log_prob, value, self.hidden = self.model.get_action(
                obs_tensor, self.hidden, deterministic=deterministic
            )

        return action, log_prob, value

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs, action, log_prob, value, reward, done)

    def compute_gae(
        self,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        gae = 0

        # Work backwards through the trajectory
        values_ext = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values_ext[t + 1] * (1 - dones[t])
                - values_ext[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages).to(self.device)
        returns = advantages + torch.stack(values).to(self.device)

        return advantages, returns

    def store_initial_hidden(self) -> None:
        """Store current hidden state as initial state for this rollout."""
        if self.hidden is not None:
            self.buffer.set_initial_hidden(self.hidden)

    def update(self, next_value: torch.Tensor) -> dict:
        """
        Perform PPO update.

        Returns:
            Dictionary of loss values for logging
        """
        # Compute advantages
        advantages, returns = self.compute_gae(next_value)

        # Flatten to 1D
        advantages = advantages.squeeze()
        returns = returns.squeeze()

        # Prepare batch data (observations already normalized by SelectiveVecNormalize)
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        dones = torch.FloatTensor(self.buffer.dones).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device).squeeze()

        # Get stored initial hidden state
        initial_hidden = self.buffer.initial_hidden

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track losses for logging
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Multiple epochs over the data
        for _ in range(self.update_epochs):
            # Evaluate actions with current policy
            # Pass dones to handle episode boundaries, and initial hidden state
            log_probs, values, entropy = self.model.evaluate_actions(
                obs, actions, dones, hidden=initial_hidden
            )

            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / self.update_epochs,
            "value_loss": total_value_loss / self.update_epochs,
            "entropy": total_entropy / self.update_epochs,
        }

    def train_mode(self) -> None:
        """Set agent to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.model.eval()

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
