# Environment Module

This defines the 2D tissue simulation environment.

From your code:
- GRID = 60x60 spatial field
- damage matrix (binary tissue damage map)
- pheromone diffusion field

Physics:
- Damage cells are randomly initialized
- Pheromone spreads using diffusion equation
- Agents interact by repairing damaged cells
