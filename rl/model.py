"""
LSTM Actor-Critic network for PPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    LSTM-based actor-critic network.

    - Actor (policy): outputs action probabilities
    - Critic (value): outputs state value estimate
    """

    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[Categorical, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            obs: Observation tensor of shape (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: Optional LSTM hidden state (h, c)

        Returns:
            policy: Categorical distribution over actions
            value: State value estimate
            hidden: New LSTM hidden state
        """
        # Add sequence dimension if needed
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(obs)
        else:
            lstm_out, hidden = self.lstm(obs, hidden)

        # Use last timestep output
        features = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Actor: action logits -> distribution
        action_logits = self.actor(features)
        policy = Categorical(logits=action_logits)

        # Critic: state value
        value = self.critic(features).squeeze(-1)

        return policy, value, hidden

    def get_action(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample an action from the policy.

        Args:
            obs: Single observation (obs_dim,) or batched (batch, obs_dim)
            hidden: LSTM hidden state
            deterministic: If True, return argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: State value estimate
            hidden: New LSTM hidden state
        """
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        policy, value, hidden = self.forward(obs, hidden)

        if deterministic:
            action = policy.probs.argmax(dim=-1)
        else:
            action = policy.sample()

        log_prob = policy.log_prob(action)

        return action.item(), log_prob, value, hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update with proper hidden state handling.

        Args:
            obs: Observations (seq_len, obs_dim)
            actions: Actions taken (seq_len,)
            dones: Episode termination flags (seq_len,) - used to reset hidden state
            hidden: Initial LSTM hidden state

        Returns:
            log_probs: Log probabilities of actions (seq_len,)
            values: State value estimates (seq_len,)
            entropy: Policy entropy (seq_len,)
        """
        seq_len = obs.shape[0]
        device = obs.device

        # Initialize hidden state
        if hidden is None:
            h = torch.zeros(1, 1, self.hidden_dim, device=device)
            c = torch.zeros(1, 1, self.hidden_dim, device=device)
        else:
            h, c = hidden

        # Process step by step to handle episode boundaries
        features_list = []
        for t in range(seq_len):
            # Single step: (1, 1, obs_dim)
            obs_t = obs[t:t+1].unsqueeze(0)
            lstm_out, (h, c) = self.lstm(obs_t, (h, c))
            features_list.append(lstm_out.squeeze(0).squeeze(0))  # (hidden_dim,)

            # Reset hidden state after episode termination
            if dones[t]:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)

        # Stack features: (seq_len, hidden_dim)
        features = torch.stack(features_list)

        # Actor: action logits for all timesteps
        action_logits = self.actor(features)  # (seq_len, action_dim)
        policy = Categorical(logits=action_logits)

        # Critic: values for all timesteps
        values = self.critic(features).squeeze(-1)  # (seq_len,)

        log_probs = policy.log_prob(actions)
        entropy = policy.entropy()

        return log_probs, values, entropy

    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)
