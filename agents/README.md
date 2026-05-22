# Agents Module

This module represents autonomous nanobot agents.

From your implementation:
- Each agent has position (x, y)
- Velocity vector for movement smoothing
- Action space of 5 discrete moves:
  [up, down, left, right, stay]

Behavior:
- Agents use hybrid decision system:
  1. DQN-based action selection
  2. Greedy direction toward nearest damage
  3. Pheromone influence from swarm memory

Purpose:
Enables decentralized coordination for tissue repair.
