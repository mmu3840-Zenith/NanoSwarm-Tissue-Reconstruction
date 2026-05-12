import torch
import random
import numpy as np
from collections import deque
from src.rl.dqn_model import DQN

class Trainer:
    def __init__(self):
        self.model = DQN()
        self.target = DQN()
        self.target.load_state_dict(self.model.state_dict())

        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def store(self, s, a, r, s2):
        self.memory.append((s,a,r,s2))

    def train(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)
        s,a,r,s2 = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        a = torch.tensor(a)
        r = torch.tensor(r, dtype=torch.float32)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        q_next = self.target(s2).max(1)[0].detach()

        loss = torch.nn.MSELoss()(q, r + self.gamma * q_next)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
