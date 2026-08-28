"""Reinforcement-learning training utilities."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from src.rl.dqn_model import DQN


@dataclass
class Experience:
    """One replay transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity experience replay buffer."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        """Store one experience."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a random batch."""
        if not 0 < batch_size <= len(self.buffer):
            raise ValueError("invalid batch_size")

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return number of stored experiences."""
        return len(self.buffer)


class DQNTrainer:
    """DQN inference and training configuration."""

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        learning_rate: float = 1e-3,
    ) -> None:
        self.action_dim = action_dim

        self.model = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.target = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.target.load_state_dict(
            self.model.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.gamma = 0.99

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.1,
    ) -> int:
        """Select an epsilon-greedy action."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        tensor_state = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).reshape(1, -1)

        with torch.no_grad():
            q_values = self.model(tensor_state)

        return int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )
