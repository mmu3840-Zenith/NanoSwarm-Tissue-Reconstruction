"""Deep Q-Network model."""
from __future__ import annotations

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Feed-forward Q-value network."""

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action."""
        return self.network(state)
