import torch
import torch.nn as nn
from typing import Tuple


class LSTMPolicy(nn.Module):
    """LSTM policy matching SB3's MlpLstmPolicy architecture"""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        # Feature extractor (matches SB3's default MLP extractor)
        # Default: two layers of 64 units each
        self.features_extractor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        features_dim = 64

        # LSTM for actor
        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
        )

        # Policy head (actor)
        self.policy_head = nn.Linear(lstm_hidden_size, action_dim)

        # Value head (critic) - separate from actor in SB3 by default
        self.value_head = nn.Linear(lstm_hidden_size, 1)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Handle single timestep input
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        batch_size, seq_len, _ = obs.shape

        # Extract features
        features = self.features_extractor(obs)  # (batch, seq_len, 64)

        # Initialize LSTM states if not provided
        if lstm_states is None:
            h0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=obs.device)
            c0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=obs.device)
            lstm_states = (h0, c0)

        # LSTM forward
        lstm_out, lstm_states = self.lstm(features, lstm_states)  # (batch, seq_len, hidden)

        # Policy and value heads
        logits = self.policy_head(lstm_out)  # (batch, seq_len, action_dim)
        values = self.value_head(lstm_out)  # (batch, seq_len, 1)

        if single_step:
            logits = logits.squeeze(1)  # (batch, action_dim)
            values = values.squeeze(1)  # (batch, 1)

        return logits, values, lstm_states

    def get_action(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = True,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            obs = obs.unsqueeze(0)  # (1, obs_dim)
            logits, _, lstm_states = self.forward(obs, lstm_states)
            logits = logits.squeeze(0)  # (action_dim,)

            if deterministic:
                action = logits.argmax().item()
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

        return action, lstm_states

    def init_hidden(self, batch_size: int = 1, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device

        h0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
