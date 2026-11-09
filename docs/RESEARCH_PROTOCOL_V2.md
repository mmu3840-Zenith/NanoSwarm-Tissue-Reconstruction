# NanoSwarm 2.0 research protocol

This repository contains a computational grid-world abstraction.
It is not a physical nanorobot model and is not validated for biological,
vascular, medical, or clinical applications.

Policies: greedy local, pheromone-guided, DQN-only, and hybrid DQN+pheromone.
Agents observe local 3x3 neighborhoods. Pheromone evolves through spatial
diffusion and evaporation. The DQN uses replay, a target network, Bellman
targets, epsilon exploration during training, and gradient updates.

Training and evaluation use separate seed ranges. Evaluation output is
written to results_v2/trials.csv and summary.json.

Limitations: simplified 2D environment; discrete actions; simplified failure
and recovery; shared policy network; normal-approximation confidence intervals;
no physical fluid dynamics, magnetic actuation, molecular interactions,
hardware validation, or clinical validation. Results must not be interpreted
as evidence of real nanorobot performance.
