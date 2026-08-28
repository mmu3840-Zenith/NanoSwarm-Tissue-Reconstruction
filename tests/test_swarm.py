from src.swarm.swarm_engine import SwarmSystem


def test_swarm_initialization():
    swarm = SwarmSystem(agent_count=10)

    assert swarm.agent_count == 10
    assert len(swarm.agents) == 10


def test_pheromone_decay():
    swarm = SwarmSystem(
        agent_count=2,
        decay_rate=0.10,
    )

    swarm.deposit((1, 1), 1.0)
    swarm.diffuse()

    assert 0.89 < swarm.pheromone[(1, 1)] < 0.91
