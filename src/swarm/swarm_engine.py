"""Pheromone-inspired decentralized swarm coordination."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwarmAgentState:
    """State of one simulated swarm member."""

    agent_id: int
    position: tuple[int, int]


class SwarmSystem:
    """Computational swarm coordination layer."""

    def __init__(
        self,
        agent_count: int = 100,
        diffusion_rate: float = 0.10,
        decay_rate: float = 0.01,
    ) -> None:
        if agent_count < 1:
            raise ValueError(
                "agent_count must be at least 1"
            )

        if diffusion_rate < 0.0:
            raise ValueError(
                "diffusion_rate cannot be negative"
            )

        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError(
                "decay_rate must be between 0 and 1"
            )

        self.agent_count = agent_count
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate

        self.agents = [
            SwarmAgentState(
                agent_id=i,
                position=(0, 0),
            )
            for i in range(agent_count)
        ]

        self.pheromone: dict[
            tuple[int, int],
            float,
        ] = {}

    def deposit(
        self,
        position: tuple[int, int],
        amount: float = 1.0,
    ) -> None:
        """Deposit an abstract coordination signal."""
        if amount < 0:
            raise ValueError(
                "amount cannot be negative"
            )

        self.pheromone[position] = (
            self.pheromone.get(position, 0.0)
            + amount
        )

    def diffuse(self) -> None:
        """Apply signal decay."""
        for position in list(self.pheromone):
            value = (
                self.pheromone[position]
                * (1.0 - self.decay_rate)
            )

            if value <= 1e-12:
                del self.pheromone[position]
            else:
                self.pheromone[position] = value

    def step(self) -> dict[str, int | float]:
        """Advance the coordination state by one timestep."""
        self.diffuse()

        return {
            "agents": self.agent_count,
            "pheromone_cells": len(
                self.pheromone
            ),
        }
