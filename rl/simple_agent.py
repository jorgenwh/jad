"""
Simple feedforward PPO agent (no LSTM).
"""

import torch
import torch.nn as nn
import numpy as np
from observations import OBS_DIM, NORMALIZE_MASK
from normalizer import RunningNormalizer


class SimpleActorCritic(nn.Module):
    """Simple feedforward actor-critic."""

    def __init__(self, obs_dim=OBS_DIM, action_dim=5, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def get_action(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value.squeeze(-1), dist.entropy()


class SimplePPOAgent:
    """Simple PPO without LSTM."""

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=5,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,  # More epochs
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.device = device

        self.model = SimpleActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Normalizer for continuous features only
        self.normalize_mask = NORMALIZE_MASK
        n_continuous = int(np.sum(self.normalize_mask))
        self.obs_normalizer = RunningNormalizer(shape=(n_continuous,))
        self.training = True

        # Simple buffer
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.done_buf = []

    def normalize_obs(self, obs, update=True):
        obs = np.asarray(obs, dtype=np.float32)
        result = obs.copy()
        continuous = obs[..., self.normalize_mask]
        if update and self.training:
            self.obs_normalizer.update(continuous)
        result[..., self.normalize_mask] = self.obs_normalizer.normalize(continuous)
        return result

    def select_action(self, obs):
        obs_norm = self.normalize_obs(obs, update=self.training)
        obs_t = torch.FloatTensor(obs_norm).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(obs_t)
        return action, log_prob, value

    def store(self, obs, action, reward, value, log_prob, done):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.val_buf.append(value)
        self.logp_buf.append(log_prob)
        self.done_buf.append(done)

    def update(self, last_value):
        # Compute GAE
        rewards = np.array(self.rew_buf)
        values = torch.stack(self.val_buf).cpu().numpy()
        dones = np.array(self.done_buf)

        # Extend values with last_value
        values_ext = np.append(values, last_value.cpu().item())

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Convert to tensors
        obs = torch.FloatTensor(self.normalize_obs(np.array(self.obs_buf), update=False)).to(self.device)
        actions = torch.LongTensor(self.act_buf).to(self.device)
        old_log_probs = torch.stack(self.logp_buf).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update
        for _ in range(self.update_epochs):
            log_probs, values, entropy = self.model.evaluate(obs, actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, returns_t)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Clear buffer
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()
        self.logp_buf.clear()
        self.done_buf.clear()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def buffer_size(self):
        return len(self.obs_buf)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalizer_state_dict": self.obs_normalizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "normalizer_state_dict" in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
