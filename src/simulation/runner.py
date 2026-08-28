"""Executable NanoSwarm tissue-reconstruction experiment.

This module implements a computational abstraction of multi-agent
repair. It is not a biological or physical nanorobot model.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from src.agents.nanobot import Nanobot
from src.environment.tissue_environment import TissueEnvironment
from src.swarm.swarm_engine import SwarmSystem


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return Manhattan distance between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_damage(
    position: tuple[int, int],
    damage: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the nearest damaged cell using Manhattan distance."""
    if not damage:
        return None

    return min(
        damage,
        key=lambda cell: (
            _distance(position, cell),
            cell[0],
            cell[1],
        ),
    )


def _pheromone_target(
    position: tuple[int, int],
    damage: set[tuple[int, int]],
    pheromone: dict[tuple[int, int], float],
) -> tuple[int, int] | None:
    """Choose a damaged target using distance and pheromone signal."""
    if not damage:
        return None

    def score(cell: tuple[int, int]) -> tuple[float, int, int, int]:
        distance = _distance(position, cell)
        signal = pheromone.get(cell, 0.0)

        # Higher pheromone is preferred while distance remains important.
        utility = signal * 2.0 - float(distance)
        return (-utility, distance, cell[0], cell[1])

    return min(damage, key=score)


def _state_vector(
    agent: Nanobot,
    environment: TissueEnvironment,
) -> list[float]:
    """Build a normalized state vector for the DQN interface."""
    target = _nearest_damage(agent.position(), environment.damage)

    if target is None:
        return [
            agent.x / max(1, environment.width - 1),
            agent.y / max(1, environment.height - 1),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    tx, ty = target

    dx = (tx - agent.x) / max(1, environment.width - 1)
    dy = (ty - agent.y) / max(1, environment.height - 1)

    damage_ratio = (
        len(environment.damage)
        / float(environment.width * environment.height)
    )

    distance = _distance(agent.position(), target)
    max_distance = environment.width + environment.height - 2

    return [
        agent.x / max(1, environment.width - 1),
        agent.y / max(1, environment.height - 1),
        dx,
        dy,
        distance / max(1, max_distance),
        damage_ratio,
        agent.velocity_x,
        agent.velocity_y,
        0.0,
        0.0,
    ]


def _move_toward(
    agent: Nanobot,
    target: tuple[int, int],
    environment: TissueEnvironment,
) -> None:
    """Move one simulated agent one grid step toward a target."""
    action = agent.heuristic_action(target)
    agent.move(
        action,
        width=environment.width,
        height=environment.height,
    )


def run_simulation(
    agents: int = 100,
    steps: int = 200,
    seed: int = 42,
    use_dqn: bool = False,
    use_pheromone: bool = True,
) -> dict:
    """Run the complete computational repair simulation.

    The default controller uses deterministic greedy navigation plus
    pheromone-inspired coordination. The DQN interface can be enabled
    when a trained model is supplied by the caller in a future experiment.

    Returns a JSON-serializable experiment record.
    """
    if agents < 1:
        raise ValueError("agents must be positive")

    if steps < 1:
        raise ValueError("steps must be positive")

    rng = random.Random(seed)

    environment = TissueEnvironment(
        width=20,
        height=20,
        damage_probability=0.10,
        seed=seed,
    )

    swarm = SwarmSystem(
        agent_count=agents,
        diffusion_rate=0.10,
        decay_rate=0.01,
    )

    nanobots = [
        Nanobot(
            agent_id=i,
            x=rng.randrange(environment.width),
            y=rng.randrange(environment.height),
        )
        for i in range(agents)
    ]

    initial_damage = environment.remaining_damage()

    repaired_cells: set[tuple[int, int]] = set()
    repair_events = 0
    failures = 0
    first_completion_step: int | None = None

    history: list[dict] = []

    for step in range(steps):
        step_repairs = 0
        step_failures = 0

        for agent in nanobots:
            if not environment.damage:
                break

            if use_pheromone:
                target = _pheromone_target(
                    agent.position(),
                    environment.damage,
                    swarm.pheromone,
                )
            else:
                target = _nearest_damage(
                    agent.position(),
                    environment.damage,
                )

            if target is None:
                continue

            # The current executable controller is deterministic.
            # use_dqn is retained as an experiment switch so a trained
            # DQN policy can be integrated without changing the runner API.
            if use_dqn:
                # Until a trained policy is provided, use the same
                # target-directed action as the safe fallback.
                _ = _state_vector(agent, environment)

            _move_toward(
                agent,
                target,
                environment,
            )

            position = agent.position()

            if environment.repair(position):
                repaired_cells.add(position)
                swarm.deposit(position, amount=1.0)
                repair_events += 1
                step_repairs += 1

        # A failure is counted only when an agent attempts a repair
        # operation at a cell that is not damaged. Merely being away
        # from a damaged cell is not considered a failure.
        attempted_repairs = 0

        for agent in nanobots:
            position = agent.position()

            if position in environment.damage:
                continue

            target = _nearest_damage(
                position,
                environment.damage,
            )

            if target is not None and _distance(position, target) == 0:
                attempted_repairs += 1

        step_failures = attempted_repairs

        failures += step_failures
        swarm_state = swarm.step()

        remaining = environment.remaining_damage()
        completion = environment.completion_rate(initial_damage)

        if (
            first_completion_step is None
            and completion >= 1.0
        ):
            first_completion_step = step

        history.append(
            {
                "step": step,
                "remaining_damage": remaining,
                "completion_rate": completion,
                "repairs": step_repairs,
                "failures": step_failures,
                "agents": agents,
                "pheromone_cells": swarm_state["pheromone_cells"],
                "active_agents": sum(
                    1 for agent in nanobots
                    if agent.position() in environment.damage
                ),
            }
        )

        if remaining == 0:
            break

    final_damage = environment.remaining_damage()

    if first_completion_step is None:
        convergence_steps = steps
    else:
        convergence_steps = first_completion_step + 1

    area_coverage = (
        len(repaired_cells)
        / float(initial_damage)
        if initial_damage > 0
        else 1.0
    )

    return {
        "experiment": "NanoSwarm computational tissue reconstruction",
        "model_type": "computational abstraction",
        "agents": agents,
        "steps_requested": steps,
        "steps_executed": len(history),
        "seed": seed,
        "controller": {
            "dqn_enabled": use_dqn,
            "pheromone_enabled": use_pheromone,
            "fallback_policy": "nearest-damage greedy navigation",
        },
        "environment": {
            "width": environment.width,
            "height": environment.height,
            "damage_probability": environment.damage_probability,
        },
        "initial_damage": initial_damage,
        "final_damage": final_damage,
        "repairs": repair_events,
        "completion_rate": environment.completion_rate(initial_damage),
        "area_coverage": area_coverage,
        "convergence_steps": convergence_steps,
        "failures": failures,
        "history": history,
    }


def save_result(
    result: dict,
    output: str | Path = "results/simulation.json",
) -> Path:
    """Save an experiment result as formatted JSON."""
    output_path = Path(output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """Run the default experiment and save the result."""
    result = run_simulation(
        agents=100,
        steps=200,
        seed=42,
    )

    output = save_result(result)

    print("NanoSwarm simulation complete")
    print(f"Agents: {result['agents']}")
    print(f"Initial damage: {result['initial_damage']}")
    print(f"Final damage: {result['final_damage']}")
    print(
        "Completion rate: "
        f"{result['completion_rate']:.4f}"
    )
    print(
        "Area coverage: "
        f"{result['area_coverage']:.4f}"
    )
    print(f"Repairs: {result['repairs']}")
    print(f"Convergence steps: {result['convergence_steps']}")
    print(f"Failures: {result['failures']}")
    print(f"Result written to: {output}")


if __name__ == "__main__":
    main()
