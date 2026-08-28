"""Research plotting utilities.

Figures must be generated from recorded experimental data.
This module provides reusable plotting functions and does not
invent or hard-code experimental results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_completion_rate(
    agent_counts: Sequence[int],
    completion_rates: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot completion rate against agent count."""
    if len(agent_counts) != len(completion_rates):
        raise ValueError("agent_counts and completion_rates must have equal length")

    if not agent_counts:
        raise ValueError("At least one data point is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(agent_counts, completion_rates, marker="o")
    axis.set_title("Completion Rate vs Agent Count")
    axis.set_xlabel("Number of Agents")
    axis.set_ylabel("Completion Rate")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)

    return output


def plot_convergence_steps(
    agent_counts: Sequence[int],
    convergence_steps: Sequence[int],
    output_path: str | Path,
) -> Path:
    """Plot convergence steps against agent count."""
    if len(agent_counts) != len(convergence_steps):
        raise ValueError("agent_counts and convergence_steps must have equal length")

    if not agent_counts:
        raise ValueError("At least one data point is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.plot(agent_counts, convergence_steps, marker="o")
    axis.set_title("Convergence Steps vs Agent Count")
    axis.set_xlabel("Number of Agents")
    axis.set_ylabel("Convergence Steps")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)

    return output


def plot_strategy_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot a comparison between experimental strategies."""
    if len(labels) != len(values):
        raise ValueError("labels and values must have equal length")

    if not labels:
        raise ValueError("At least one comparison value is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure()
    axis = figure.add_subplot(111)

    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)

    return output
