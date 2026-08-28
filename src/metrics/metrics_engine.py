"""Simulation metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationMetrics:
    """Standard metrics for one simulation run."""

    completion_rate: float
    remaining_damage: int
    convergence_steps: int
    failures: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible metrics."""
        return {
            "completion_rate": self.completion_rate,
            "remaining_damage": self.remaining_damage,
            "convergence_steps": self.convergence_steps,
            "failures": self.failures,
        }
