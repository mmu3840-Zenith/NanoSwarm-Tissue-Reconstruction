import matplotlib.pyplot as plt
import numpy as np

agents = np.array([10, 50, 100])

greedy = {"completion":[14.5,55,100],"failure":[100,100,0]}
swarm = {"completion":[55,97,100],"failure":[100,80,20]}
hybrid = {"completion":[100,100,100],"failure":[0,0,0]}

plt.figure()
plt.plot(agents, greedy["completion"], marker='o', label='Greedy')
plt.plot(agents, swarm["completion"], marker='o', label='Swarm')
plt.plot(agents, hybrid["completion"], marker='o', label='Hybrid')
plt.title('Completion Rate')
plt.legend()
plt.grid()
plt.savefig('completion_rate.png')

plt.figure()
plt.plot(agents, greedy["failure"], marker='o')
plt.plot(agents, swarm["failure"], marker='o')
plt.plot(agents, hybrid["failure"], marker='o')
plt.title('Failure Rate')
plt.grid()
plt.savefig('failure_rate.png')

plt.figure()
x = np.arange(len(agents))
plt.bar(x-0.25, greedy["completion"], 0.25, label='Greedy')
plt.bar(x, swarm["completion"], 0.25, label='Swarm')
plt.bar(x+0.25, hybrid["completion"], 0.25, label='Hybrid')
plt.xticks(x, agents)
plt.title('System Comparison')
plt.legend()
plt.savefig('comparison.png')

print("Plots generated")
