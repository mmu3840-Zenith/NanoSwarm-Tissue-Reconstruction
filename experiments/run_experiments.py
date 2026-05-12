from src.swarm.swarm_engine import SwarmSystem
import json, os, numpy as np

def run_experiment(n_agents, steps, name):
    system = SwarmSystem(n_agents)
    log = []

    for t in range(steps):
        agents = system.step()

        log.append({
            "step": t,
            "spread": float(np.std(agents)),
            "mean": float(np.mean(agents))
        })

    os.makedirs("results/logs", exist_ok=True)

    with open(f"results/logs/{name}.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    run_experiment(100, 200, "baseline")
