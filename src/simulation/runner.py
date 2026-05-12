from src.swarm.swarm_engine import SwarmSystem
import json
import os

def run_simulation():
    system = SwarmSystem(100)
    results = []

    for i in range(200):
        state = system.step()
        results.append(state.tolist())

    os.makedirs("results", exist_ok=True)

    with open("results/sim.json", "w") as f:
        json.dump(results, f)

    print("Simulation complete")

if __name__ == "__main__":
    run_simulation()
