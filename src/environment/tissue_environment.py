import numpy as np
import random

GRID = 60
damage = np.zeros((GRID, GRID))
pheromone = np.zeros((GRID, GRID))

for _ in range(200):
    x, y = random.randint(0,59), random.randint(0,59)
    damage[x,y] = 1

def diffuse(field):
    new = field.copy()
    for i in range(1, GRID-1):
        for j in range(1, GRID-1):
            new[i,j] += 0.1 * (
                field[i+1,j] + field[i-1,j] +
                field[i,j+1] + field[i,j-1] -
                4*field[i,j]
            )
    return np.clip(new, 0, 1)
