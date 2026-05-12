from src.swarm.swarm_engine import SwarmSystem

if __name__ == "__main__":
    system = SwarmSystem(100)

    for _ in range(200):
        system.step()

    print("Simulation complete")
