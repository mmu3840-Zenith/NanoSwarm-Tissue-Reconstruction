"""Distributed computational repair interface."""
from __future__ import annotations

from src.environment.tissue_environment import TissueEnvironment


def repair_cell(
    environment: TissueEnvironment,
    position: tuple[int, int],
) -> bool:
    """Attempt one simulated repair event."""
    return environment.repair(position)
