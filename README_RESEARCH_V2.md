# NanoSwarm 2.0

A computational swarm-repair benchmark with decentralized local observations,
four policy baselines, a NumPy DQN, spatial pheromone diffusion, and agent
failure/recovery experiments.

## Install and run
python -m pip install numpy
python -m unittest discover -s tests -p "test_nanoswarm_v2.py" -v
python -m nanoswarm_v2 --quick
python -m nanoswarm_v2 --seeds 30

See docs/RESEARCH_PROTOCOL_V2.md for assumptions and limitations.
