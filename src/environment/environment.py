"""Environment state helpers."""


class EnvironmentState:
    """Small state machine for simulation-level status."""

    def __init__(self, state: str = "stable") -> None:
        self.state = state

    def reset(self) -> None:
        """Restore the initial environment state."""
        self.state = "stable"
