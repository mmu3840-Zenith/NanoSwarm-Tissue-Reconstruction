import numpy as np

class SwarmSystem:
    def __init__(self, n_agents=50):
        self.n_agents = n_agents
        self.agents = np.random.rand(n_agents, 2) * 60

    def step(self):
        self.agents += np.random.randn(self.n_agents, 2) * 0.5
        self.agents = np.clip(self.agents, 0, 59)
        return self.agents
