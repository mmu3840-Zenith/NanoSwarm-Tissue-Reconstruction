import matplotlib.pyplot as plt
import numpy as np

agents = np.array([10, 50, 100])

greedy_steps = [999, 500, 47]
swarm_steps = [800, 350, 298]
hybrid_steps = [143, 43, 21]

greedy_completion = [14.5, 55, 100]
swarm_completion = [55, 97, 100]
hybrid_completion = [100, 100, 100]

plt.figure()
plt.plot(agents, greedy_steps, marker='o', label='Greedy')
plt.plot(agents, swarm_steps, marker='o', label='Swarm')
plt.plot(agents, hybrid_steps, marker='o', label='Hybrid')
plt.title('Convergence Speed')
plt.legend()
plt.grid()
plt.savefig('convergence_speed.png')

plt.figure()
plt.plot(agents, greedy_completion, marker='o', label='Greedy')
plt.plot(agents, swarm_completion, marker='o', label='Swarm')
plt.plot(agents, hybrid_completion, marker='o', label='Hybrid')
plt.title('Completion Scaling')
plt.legend()
plt.grid()
plt.savefig('completion_scaling.png')

plt.figure()
gap = np.array(hybrid_steps) - np.array(swarm_steps)
plt.bar(['10','50','100'], gap)
plt.title('Efficiency Gain (Hybrid vs Swarm)')
plt.savefig('performance_gap.png')

print('Plots generated')
