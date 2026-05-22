import matplotlib.pyplot as plt
import numpy as np

agents = np.array([10, 50, 100])

# REAL DATA FROM YOUR EXPERIMENTS
greedy = [14.5, 55, 100]
swarm = [55, 97, 100]
hybrid = [100, 100, 100]

greedy_steps = [999, 500, 47]
swarm_steps = [800, 350, 298]
hybrid_steps = [143, 43, 21]

# =========================
# 1. COMPLETION RATE
# =========================

plt.figure()
plt.plot(agents, greedy, marker='o', label='Greedy')
plt.plot(agents, swarm, marker='o', label='Swarm')
plt.plot(agents, hybrid, marker='o', label='Hybrid')
plt.title('Completion Rate vs Agents')
plt.xlabel('Agents')
plt.ylabel('Completion %')
plt.legend()
plt.grid()
plt.savefig('completion_rate.png')

# =========================
# 2. CONVERGENCE SPEED
# =========================

plt.figure()
plt.plot(agents, greedy_steps, marker='o', label='Greedy')
plt.plot(agents, swarm_steps, marker='o', label='Swarm')
plt.plot(agents, hybrid_steps, marker='o', label='Hybrid')
plt.title('Convergence Speed')
plt.xlabel('Agents')
plt.ylabel('Steps')
plt.legend()
plt.grid()
plt.savefig('convergence_speed.png')

# =========================
# 3. PERFORMANCE GAP
# =========================

plt.figure()
gap = np.array(swarm_steps) - np.array(hybrid_steps)
plt.bar(['10','50','100'], gap)
plt.title('Hybrid Efficiency Gain')
plt.savefig('efficiency_gap.png')

print('Research figures generated')
