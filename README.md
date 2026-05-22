# Nanomachine: Hybrid Multi-Agent Tissue Repair System

## Overview

Nanomachine is a hybrid multi-agent reinforcement learning system designed for spatial tissue repair.

It combines:
- Deep Q-Network (DQN)
- Swarm intelligence (pheromone communication)
- Greedy spatial optimization

Agents operate in a 2D grid environment where they collaboratively repair damaged cells.

---

## System Behavior

Each agent:
- perceives local environment state
- selects actions using DQN or hybrid logic
- moves in a 2D grid
- repairs damaged tissue upon contact

Swarm coordination is achieved through pheromone diffusion across the grid.

---

## Experimental Results

| System | 10 Agents | 50 Agents | 100 Agents |
|--------|----------|----------|------------|
| Greedy | 14.5% | 55% | 100% |
| Swarm  | 55% | 97% | 100% |
| Hybrid | 100% | 100% | 100% |

---

## Key Findings

- Hybrid system achieves fastest convergence (21 steps)
- Eliminates failure cases across all agent densities
- Outperforms standalone greedy and swarm systems

---

## Project Structure

Modular design includes:
- agents (nanobot logic)
- environment (tissue grid)
- rl (DQN training system)
- swarm (pheromone communication)
- metrics (evaluation system)
- experiments (scaling studies)
- simulation (execution loop)

---

## Purpose

This project explores emergent coordination in hybrid AI systems for distributed spatial optimization.
