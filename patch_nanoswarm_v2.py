from pathlib import Path
import re

path = Path("nanoswarm_v2/__main__.py")
s = path.read_text(encoding="utf-8")

# 1. Give DQN its own reproducible random generator.
s = s.replace(
    "self.lr = lr\n        self.gamma = .95",
    "self.lr = lr\n        self.rng = np.random.default_rng(seed + 7919)\n        self.gamma = .95"
)
s = s.replace(
    "ids = np.random.choice(len(self.memory), batch_size, replace=False)",
    "ids = self.rng.choice(len(self.memory), batch_size, replace=False)"
)

# 2. Replace global-information policies with local-information policies.
start = s.index("def nearest_damage(")
end = s.index("def run_episode(", start)

replacement = r'''
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


'''
s = s[:start] + replacement + s[end:]

# 3. Pass the episode RNG to local policies.
s = s.replace(
    'action = greedy_action(env, a)',
    'action = greedy_action(env, a, rng)'
)
s = s.replace(
    'action = pheromone_action(env, a)',
    'action = pheromone_action(env, a, rng)'
)

# 4. Replace summary with reproducible bootstrap confidence intervals.
start = s.index("def summarize(")
end = s.index("def main(", start)

summary = r'''
def summarize(rows, bootstrap_samples=4000):
    """Policy summaries with percentile bootstrap confidence intervals."""
    keys = ("completion", "repairs", "remaining_damage", "steps",
            "distance", "failures", "downtime", "runtime_seconds")
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


'''
s = s[:start] + summary + s[end:]

# Add a protocol note clarifying the sensing restriction.
path.write_text(s, encoding="utf-8")
print("Patched local policies, DQN RNG, and bootstrap summaries.")

