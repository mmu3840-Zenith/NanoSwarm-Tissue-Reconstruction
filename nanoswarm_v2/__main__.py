"""NanoSwarm 2.0: reproducible computational swarm-repair benchmark.

This is a computational abstraction, not a validated nanorobot or medical system.
Requires NumPy. Run: python -m nanoswarm_v2 --quick
"""
import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))


class Tissue:
    def __init__(self, size=20, density=.10, seed=1):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.damage = (self.rng.random((size, size)) < density).astype(np.float32)
        self.pheromone = np.zeros((size, size), dtype=np.float32)
        self.initial_damage = int(self.damage.sum())

    def repair(self, x, y):
        if self.damage[y, x] > 0:
            self.damage[y, x] = 0
            return True
        return False

    def update_pheromone(self, evaporation=.04, diffusion=.15):
        p = self.pheromone
        padded = np.pad(p, 1, mode="constant")
        neighbors = (
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[:-2, 1:-1] + padded[2:, 1:-1]
        ) / 4.0
        self.pheromone = (
            (1.0 - evaporation - diffusion) * p +
            diffusion * neighbors
        ).astype(np.float32)
        np.maximum(self.pheromone, 0, out=self.pheromone)

    def observe(self, x, y, include_pheromone=False):
        # Vectorized 3x3 observation; out-of-bounds cells remain zero.
        def patch(grid):
            out = np.zeros((3, 3), dtype=np.float32)

            x0 = max(0, x - 1)
            x1 = min(self.size, x + 2)
            y0 = max(0, y - 1)
            y1 = min(self.size, y + 2)

            if x0 < x1 and y0 < y1:
                ox0 = x0 - (x - 1)
                ox1 = ox0 + (x1 - x0)
                oy0 = y0 - (y - 1)
                oy1 = oy0 + (y1 - y0)
                out[oy0:oy1, ox0:ox1] = grid[y0:y1, x0:x1]

            return out.ravel()

        obs = patch(self.damage)
        if include_pheromone:
            obs = np.concatenate((obs, patch(self.pheromone)))
        return obs


class DQN:
    """Small NumPy neural Q-network with replay and target-network updates."""
    def __init__(self, inputs, outputs=5, seed=1, lr=.002):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, .15, (inputs, 32)).astype(np.float32)
        self.b1 = np.zeros(32, np.float32)
        self.w2 = rng.normal(0, .15, (32, outputs)).astype(np.float32)
        self.b2 = np.zeros(outputs, np.float32)
        self.target = self.copy_weights()
        self.memory = deque(maxlen=10000)
        self.lr = lr
        self.rng = np.random.default_rng(seed + 7919)
        self.gamma = .95
        self.updates = 0
        self.target_syncs = 0
        self.losses = []

    def copy_weights(self):
        return [a.copy() for a in (self.w1, self.b1, self.w2, self.b2)]

    def load_weights(self, weights):
        self.w1, self.b1, self.w2, self.b2 = [a.copy() for a in weights]

    def forward(self, x, target=False):
        w1, b1, w2, b2 = self.target if target else (
            self.w1, self.b1, self.w2, self.b2)
        h = np.maximum(0, x @ w1 + b1)
        return h @ w2 + b2, h

    def act(self, state, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(5))
        return int(np.argmax(self.forward(state[None, :])[0][0]))

    def remember(self, s, a, r, ns, done):
        self.memory.append((s.copy(), a, r, ns.copy(), done))

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        ids = self.rng.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in ids]
        states = np.stack([b[0] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], np.float32)
        dones = np.array([b[4] for b in batch], np.float32)

        q, h = self.forward(states)
        nq, _ = self.forward(next_states, target=True)
        target_q = q.copy()
        target_q[np.arange(batch_size), actions] = (
            rewards + self.gamma * (1 - dones) * nq.max(axis=1)
        )
        error = np.clip(q - target_q, -1, 1)
        loss = float(np.mean(error ** 2))

        grad = (2.0 / batch_size) * error
        gw2 = h.T @ grad
        gb2 = grad.sum(axis=0)
        gh = grad @ self.w2.T
        gh[h <= 0] = 0
        gw1 = states.T @ gh
        gb1 = gh.sum(axis=0)

        for param, grad_param in zip(
            (self.w1, self.b1, self.w2, self.b2),
            (gw1, gb1, gw2, gb2)
        ):
            param -= self.lr * np.clip(grad_param, -5, 5)

        self.updates += 1
        self.losses.append(loss)
        if self.updates % 100 == 0:
            self.target = self.copy_weights()
            self.target_syncs += 1
        return loss


@dataclass
class Agent:
    x: int
    y: int
    failed: bool = False
    recovery: int = 0
    failures: int = 0
    recoveries: int = 0
    distance: int = 0



def greedy_action(env, agent, rng):
    """Greedy policy restricted to the same local 3x3 damage observation."""
    patch = env.observe(agent.x, agent.y).reshape(3, 3)
    candidates = []
    for index, (dx, dy) in enumerate(ACTIONS[:4]):
        if patch[dy + 1, dx + 1] > 0:
            candidates.append(index)
    if candidates:
        return candidates[0]
    # Explore when no adjacent damage is visible; no global target lookup.
    return int(rng.integers(0, 4))


def pheromone_action(env, agent, rng):
    """Follow local damage cues and local pheromone gradients."""
    damage = env.observe(agent.x, agent.y).reshape(3, 3)
    pheromone = env.observe(agent.x, agent.y, True)[9:].reshape(3, 3)

    # Signal a locally visible damage site so other agents may discover it.
    visible_y, visible_x = np.where(damage > 0)
    for py, px in zip(visible_y, visible_x):
        env.pheromone[
            int(np.clip(agent.y + py - 1, 0, env.size - 1)),
            int(np.clip(agent.x + px - 1, 0, env.size - 1))
        ] += 0.2

    scores = []
    for dx, dy in ACTIONS[:4]:
        px, py = dx + 1, dy + 1
        # Prefer adjacent damage; otherwise follow the local pheromone gradient.
        score = 4.0 * float(damage[py, px]) + float(pheromone[py, px])
        scores.append(score)

    if max(scores) > 0:
        best = np.flatnonzero(np.isclose(scores, max(scores)))
        return int(rng.choice(best))
    return int(rng.integers(0, 4))


def make_failure_schedule(agents_n, steps, seed):
    # Exogenous random events: independent of policy action randomness.
    schedule_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(agents_n), int(steps), 77123])
    )
    uniforms = schedule_rng.random((steps, agents_n))
    recovery_durations = schedule_rng.integers(2, 7, size=(steps, agents_n))
    import hashlib
    digest = hashlib.sha256(uniforms.tobytes() + recovery_durations.tobytes()).hexdigest()[:16]
    return {
        "uniforms": uniforms,
        "recovery_durations": recovery_durations,
        "schedule_id": digest,
    }


def run_episode(policy, agents_n=25, density=.10, failure_prob=0,
                seed=1, steps=250, model=None, training=False,
                failure_schedule=None):
    if failure_schedule is None:
        failure_schedule = make_failure_schedule(agents_n, steps, seed)

    expected_shape = (steps, agents_n)
    if failure_schedule["uniforms"].shape != expected_shape:
        raise ValueError("Failure schedule has an incompatible shape.")
    if failure_schedule["recovery_durations"].shape != expected_shape:
        raise ValueError("Recovery schedule has an incompatible shape.")

    # Separate deterministic streams for initialization and policy decisions.
    init_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(agents_n), 991])
    )
    import zlib
    policy_code = zlib.crc32(policy.encode("utf-8"))
    rng = np.random.default_rng(
        np.random.SeedSequence([
            int(seed), int(agents_n), int(round(density * 1000000)),
            int(policy_code), 551
        ])
    )

    env = Tissue(20, density, seed)
    agents = [Agent(int(init_rng.integers(20)), int(init_rng.integers(20)))
              for _ in range(agents_n)]
    include_p = policy == "hybrid"
    total_repairs = 0
    downtime = 0
    start = time.perf_counter()
    epsilon = .25 if training else 0.0

    for t in range(steps):
        if env.damage.sum() == 0:
            break
        for agent_index, a in enumerate(agents):
            if a.failed:
                a.recovery -= 1
                downtime += 1
                if a.recovery <= 0:
                    a.failed = False
                    a.recoveries += 1
                continue

            state = env.observe(a.x, a.y, include_p)
            if policy == "greedy":
                action = greedy_action(env, a, rng)
            elif policy == "pheromone":
                action = pheromone_action(env, a, rng)
            elif policy in ("dqn", "hybrid"):
                action = model.act(state, epsilon, rng)
            else:
                raise ValueError(policy)

            old_damage = float(env.damage.sum())
            dx, dy = ACTIONS[action]
            nx = int(np.clip(a.x + dx, 0, env.size - 1))
            ny = int(np.clip(a.y + dy, 0, env.size - 1))
            a.distance += abs(nx-a.x) + abs(ny-a.y)
            a.x, a.y = nx, ny
            repaired = env.repair(nx, ny)
            reward = 2.0 if repaired else -.03
            if repaired:
                total_repairs += 1
                env.pheromone[ny, nx] += 1.0
            if training and model is not None:
                next_state = env.observe(nx, ny, include_p)
                model.remember(state, action, reward, next_state,
                               env.damage.sum() == 0)
                model.train_step()

            # The same pre-generated exogenous event is presented to
            # every policy at this timestep and agent index.
            if failure_schedule["uniforms"][t, agent_index] < failure_prob:
                a.failed = True
                a.recovery = int(
                    failure_schedule["recovery_durations"][t, agent_index]
                )
                a.failures += 1

        env.update_pheromone()

    elapsed = time.perf_counter() - start
    remaining = int(env.damage.sum())
    return {
        "seed": seed, "policy": policy, "agents": agents_n,
        "failure_schedule_id": failure_schedule["schedule_id"],
        "density": density, "failure_probability": failure_prob,
        "initial_damage": env.initial_damage, "repairs": total_repairs,
        "remaining_damage": remaining,
        "completion": (env.initial_damage - remaining) /
                      max(1, env.initial_damage),
        "steps": t + 1,
        "distance": sum(a.distance for a in agents),
        "failures": sum(a.failures for a in agents),
        "recoveries": sum(a.recoveries for a in agents),
        "agents_recovered": sum(a.recoveries > 0 for a in agents),
        "downtime": downtime,
        "active_ratio": sum(not a.failed for a in agents) / agents_n,
        "runtime_seconds": elapsed,
        "dqn_updates": model.updates if model else 0,
        "dqn_target_syncs": model.target_syncs if model else 0,
        "mean_training_loss": float(np.mean(model.losses))
            if model and model.losses else None
    }


def train_model(policy, seed=9000, episodes=120):
    model = DQN(18 if policy == "hybrid" else 9, seed=seed)
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        run_episode(policy, agents_n=12, density=.10,
                    seed=int(rng.integers(1, 2**31-1)),
                    steps=100, model=model, training=True)
    return model



def summarize(rows, bootstrap_samples=4000):
    """Policy summaries with percentile bootstrap confidence intervals."""
    keys = ("completion", "repairs", "remaining_damage", "steps",
            "distance", "failures", "recoveries", "agents_recovered",
            "downtime", "runtime_seconds")
    result = {}
    rng = np.random.default_rng(20261109)

    for policy in sorted(set(r["policy"] for r in rows)):
        subset = [r for r in rows if r["policy"] == policy]
        result[policy] = {}
        for key in keys:
            vals = np.asarray([r[key] for r in subset], dtype=float)
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

            samples = rng.choice(
                vals, size=(bootstrap_samples, len(vals)), replace=True
            ).mean(axis=1)
            lower, upper = np.percentile(samples, [2.5, 97.5])

            result[policy][key] = {
                "mean": mean,
                "sd": sd,
                "ci95_bootstrap": [float(lower), float(upper)],
                "n": int(len(vals))
            }
    return result



def analyze_paired(rows, bootstrap_samples=4000):
    """Summarize each experimental configuration and paired policy differences.

    A pair is matched on swarm size, damage density, failure probability,
    and evaluation seed. This avoids pooling unlike configurations.
    """
    import itertools

    config_keys = ("agents", "density", "failure_probability")
    policies = sorted(set(r["policy"] for r in rows))
    grouped = {}

    for row in rows:
        config = tuple(row[k] for k in config_keys)
        grouped.setdefault(config, []).append(row)

    rng = np.random.default_rng(20261110)
    output = {"configuration_summaries": [], "paired_comparisons": []}

    for config, config_rows in sorted(grouped.items()):
        config_record = {
            "agents": config[0],
            "density": config[1],
            "failure_probability": config[2],
            "n_seeds": len(set(r["seed"] for r in config_rows)),
            "policies": {}
        }

        for policy in policies:
            vals = [r["completion"] for r in config_rows
                    if r["policy"] == policy]
            if vals:
                config_record["policies"][policy] = {
                    "completion_mean": float(np.mean(vals)),
                    "completion_sd": float(np.std(vals, ddof=1))
                        if len(vals) > 1 else 0.0,
                    "n": len(vals)
                }
        output["configuration_summaries"].append(config_record)

        by_seed = {}
        for row in config_rows:
            by_seed.setdefault(row["seed"], {})[row["policy"]] = row

        for a, b in itertools.combinations(policies, 2):
            pairs = [(seed, policies_for_seed[a], policies_for_seed[b])
                     for seed, policies_for_seed in sorted(by_seed.items())
                     if a in policies_for_seed and b in policies_for_seed]
            if not pairs:
                continue

            differences = np.asarray(
                [left["completion"] - right["completion"]
                 for _, left, right in pairs], dtype=float
            )
            boot = rng.choice(
                differences,
                size=(bootstrap_samples, len(differences)),
                replace=True
            ).mean(axis=1)

            output["paired_comparisons"].append({
                "agents": config[0],
                "density": config[1],
                "failure_probability": config[2],
                "policy_a": a,
                "policy_b": b,
                "difference_definition": "completion(policy_a) - completion(policy_b)",
                "mean_paired_difference": float(differences.mean()),
                "ci95_bootstrap": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5))
                ],
                "n_pairs": int(len(differences))
            })

    return output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-trials", type=int, default=0)
    args = parser.parse_args()

    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if args.max_trials < 0:
        raise ValueError("--max-trials cannot be negative")

    os.makedirs("results_v2", exist_ok=True)

    checkpoint_path = "results_v2/trials_checkpoint.csv"
    final_path = "results_v2/trials.csv"
    temp_path = checkpoint_path + ".tmp"

    policies = ("greedy", "pheromone", "dqn", "hybrid")

    configs = (
        [(10, .05, 0), (25, .10, .005)]
        if args.quick else [
            (n, d, f)
            for n in (10, 25, 50, 100, 200)
            for d in (.05, .10, .15, .20)
            for f in (0, .005, .02)
        ]
    )

    seed_count = 2 if args.quick else args.seeds
    episode_steps = 150 if args.quick else 500
    expected_total = len(configs) * seed_count * len(policies)

    def trial_key(row):
        return (
            int(row["agents"]),
            round(float(row["density"]), 8),
            round(float(row["failure_probability"]), 8),
            int(row["seed"]),
            str(row["policy"])
        )

    rows = []
    completed = set()

    if args.resume:
        if not os.path.exists(checkpoint_path):
            raise SystemExit(
                "No checkpoint exists. Run without --resume first."
            )

        with open(checkpoint_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            key = trial_key(row)
            if key in completed:
                raise SystemExit("Duplicate trial found in checkpoint.")
            completed.add(key)

        print("Loaded checkpoint:", len(rows), "completed trials")

    else:
        if os.path.exists(checkpoint_path):
            import shutil
            archived = checkpoint_path.replace(
                ".csv", "_archive_" +
                time.strftime("%Y%m%d_%H%M%S") + ".csv"
            )
            shutil.copy2(checkpoint_path, archived)
            print("Archived previous checkpoint:", archived)

        rows = []
        completed = set()

    if len(rows) > expected_total:
        raise SystemExit("Checkpoint exceeds the expected experiment size.")

    # Model training is deterministic and repeated on resume.
    print("Training DQN and hybrid models...")
    models = {
        "dqn": train_model(
            "dqn", episodes=20 if args.quick else args.episodes
        ),
        "hybrid": train_model(
            "hybrid", episodes=20 if args.quick else args.episodes
        )
    }

    def save_checkpoint():
        if not rows:
            return

        fields = list(rows[0].keys())
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, checkpoint_path)

    newly_completed = 0

    for n, density, failure in configs:
        for seed in range(seed_count):
            episode_seed = 100000 + seed

            shared_schedule = make_failure_schedule(
                n, episode_steps, episode_seed
            )

            for policy in policies:
                key = (
                    int(n),
                    round(float(density), 8),
                    round(float(failure), 8),
                    int(episode_seed),
                    policy
                )

                if key in completed:
                    continue

                result = run_episode(
                    policy,
                    n,
                    density,
                    failure,
                    seed=episode_seed,
                    steps=episode_steps,
                    model=models.get(policy),
                    failure_schedule=shared_schedule
                )

                rows.append(result)
                completed.add(key)
                newly_completed += 1

                # Atomic checkpoint after every completed policy trial.
                save_checkpoint()

                if len(rows) % 10 == 0 or len(rows) == expected_total:
                    print(
                        f"Progress: {len(rows)}/{expected_total} trials"
                    )

                if (
                    args.max_trials > 0
                    and newly_completed >= args.max_trials
                ):
                    print("Controlled stopping point reached.")
                    print("Checkpoint saved:", checkpoint_path)
                    print("Completed:", len(rows), "/", expected_total)
                    return

    if len(rows) != expected_total:
        raise SystemExit(
            f"Incomplete experiment: {len(rows)}/{expected_total}"
        )

    # Validate uniqueness and completeness before publishing final outputs.
    if len(completed) != expected_total:
        raise SystemExit("Trial-key validation failed.")

    expected_keys = set()
    for n, density, failure in configs:
        for seed in range(seed_count):
            for policy in policies:
                expected_keys.add((
                    int(n),
                    round(float(density), 8),
                    round(float(failure), 8),
                    100000 + seed,
                    policy
                ))

    if completed != expected_keys:
        raise SystemExit("Expected trial matrix does not match checkpoint.")

    # Validate paired failure schedules.
    grouped = {}
    for row in rows:
        group_key = (
            int(row["agents"]),
            round(float(row["density"]), 8),
            round(float(row["failure_probability"]), 8),
            int(row["seed"])
        )
        grouped.setdefault(group_key, []).append(row)

    for group_key, group_rows in grouped.items():
        if len(group_rows) != len(policies):
            raise SystemExit(
                f"Incomplete policy group: {group_key}"
            )

        schedule_ids = {
            row["failure_schedule_id"] for row in group_rows
        }
        if len(schedule_ids) != 1:
            raise SystemExit(
                f"Shared failure schedule mismatch: {group_key}"
            )

    # Publish final dataset only after all validations pass.
    fields = list(rows[0].keys())
    final_temp = final_path + ".tmp"

    with open(final_temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    os.replace(final_temp, final_path)

    summary = summarize(rows)
    with open("results_v2/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    paired = analyze_paired(rows)
    with open(
        "results_v2/paired_analysis.json", "w", encoding="utf-8"
    ) as f:
        json.dump(paired, f, indent=2)

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("Trials:", len(rows))
    print("Configurations:", len(configs))
    print("Seeds per configuration:", seed_count)
    print("Policies:", len(policies))
    print("All trial keys unique: YES")
    print("Shared failure schedules validated: YES")
    print("Saved trials.csv, summary.json, paired_analysis.json")


if __name__ == "__main__":
    main()
