"""Computational nanobot agent model.

This class represents a simulated agent.
It does not represent a physical nanorobot.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Nanobot:
    """State and movement logic for one simulated agent."""

    agent_id: int
    x: int = 0
    y: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    ACTIONS = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
        4: (0, 0),
    }

    def position(self) -> tuple[int, int]:
        """Return the current grid position."""
        return self.x, self.y

    def move(self, action: int, width: int, height: int) -> None:
        """Apply a bounded discrete movement action."""
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        dx, dy = self.ACTIONS[action]

        self.velocity_x = 0.7 * self.velocity_x + 0.3 * dx
        self.velocity_y = 0.7 * self.velocity_y + 0.3 * dy

        self.x = max(0, min(width - 1, self.x + dx))
        self.y = max(0, min(height - 1, self.y + dy))

    def heuristic_action(self, target: tuple[int, int]) -> int:
        """Select a greedy action toward a target cell."""
        tx, ty = target
        dx = tx - self.x
        dy = ty - self.y

        if abs(dx) >= abs(dy) and dx != 0:
            return 3 if dx > 0 else 2

        if dy != 0:
            return 1 if dy > 0 else 0

        return 4
