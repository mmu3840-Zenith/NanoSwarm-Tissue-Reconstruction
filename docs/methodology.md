# NanoSwarm 2.0 Methodology

## Research scope
A computational multi-agent tissue-damage repair simulation.

## Policies
- Greedy: local damage-driven baseline.
- Pheromone: decentralized local policy using pheromone information.
- DQN: learned policy using local damage observations.
- Hybrid: learned policy with additional pheromone observations.

The hybrid label describes observation augmentation, not an independently
verified fusion of separate controllers.

## Experimental fairness
Comparisons use matched initial conditions and shared exogenous failure
schedules within each configuration and seed.

## Evaluation
Training and evaluation must be separated. Evaluation must not update
model parameters.

## Scope limitation
This is a computational simulation, not a validated biological or
clinical tissue-repair system.
