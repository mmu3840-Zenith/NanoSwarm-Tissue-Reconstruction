import numpy as np

class Nanobot:
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)

    def step(self, move):
        self.vel = self.vel * 0.6 + move
        self.pos += self.vel
        self.pos = np.clip(self.pos, 0, 59)
        return self.pos
