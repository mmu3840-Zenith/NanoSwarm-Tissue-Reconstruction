import json
import matplotlib.pyplot as plt
import os

def plot(file):
    with open(file) as f:
        data = json.load(f)

    steps = [d["step"] for d in data]
    spread = [d["spread"] for d in data]

    plt.plot(steps, spread)
    plt.title("Swarm Stability")
    plt.xlabel("Step")
    plt.ylabel("Spread")

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(file.replace(".json",".png"))

if __name__ == "__main__":
    for f in os.listdir("results/logs"):
        plot(f"results/logs/{f}")
