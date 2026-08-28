"""Computational tissue environment.

This is a simplified computational abstraction.
It is not a biological tissue model.
"""
from __future__ import annotations

import random


class TissueEnvironment:
    """Grid-based abstraction of a damaged computational environment."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        damage_probability: float = 0.10,
        seed: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        if not 0.0 <= damage_probability <= 1.0:
            raise ValueError(
                "damage_probability must be between 0 and 1"
            )

        self.width = width
        self.height = height
        self.damage_probability = damage_probability
        self.random = random.Random(seed)

        self.damage: set[tuple[int, int]] = set()
        self.pheromone: dict[tuple[int, int], float] = {}

        self.reset()

    def reset(self) -> None:
        """Generate the computational environment."""
        self.damage.clear()
        self.pheromone.clear()

        for x in range(self.width):
            for y in range(self.height):
                if self.random.random() < self.damage_probability:
                    self.damage.add((x, y))

    def repair(self, position: tuple[int, int]) -> bool:
        """Repair one damaged cell."""
        if position not in self.damage:
            return False

        self.damage.remove(position)

        self.pheromone[position] = (
            self.pheromone.get(position, 0.0) + 1.0
        )

        return True

    def remaining_damage(self) -> int:
        """Return the number of unrepaired cells."""
        return len(self.damage)

    def completion_rate(self, initial_damage: int) -> float:
        """Return normalized repair completion in [0, 1]."""
        if initial_damage <= 0:
            return 1.0

        repaired = initial_damage - len(self.damage)

        return max(
            0.0,
            min(1.0, repaired / initial_damage),
        )
