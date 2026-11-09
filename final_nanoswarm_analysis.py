import csv, json, os, math, random, statistics
from collections import defaultdict

root = "results_v2"
csv_path = os.path.join(root, "trials.csv")

with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("No trial data found.")

required = ["policy", "seed", "agents", "density",
            "failure_probability", "completion", "failure_schedule_id"]

missing = [x for x in required if x not in rows[0]]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

policies = sorted({r["policy"] for r in rows})
groups = defaultdict(dict)
schedules = defaultdict(set)

for r in rows:
    key = (
        r["agents"], r["density"],
        r["failure_probability"], r["seed"]
    )
    groups[key][r["policy"]] = r
    schedules[key].add(r["failure_schedule_id"])

mismatches = [k for k, v in schedules.items() if len(v) != 1]
if mismatches:
    raise SystemExit(f"Shared schedule mismatch in {len(mismatches)} groups.")

expected_policies = {"greedy", "pheromone", "dqn", "hybrid"}
if not expected_policies.issubset(set(policies)):
    raise SystemExit(f"Expected policies missing. Found: {policies}")

complete = {
    k: v for k, v in groups.items()
    if expected_policies.issubset(v.keys())
}

if len(complete) != len(groups):
    raise SystemExit("Some matched groups do not contain all four policies.")

if len(rows) != 7200:
    raise SystemExit(
        f"Expected 7200 trials for the full matrix; found {len(rows)}. "
        "Do not treat this as a completed full experiment."
    )

random.seed(20261109)

def bootstrap_ci(values, iterations=10000):
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [random.choice(values) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(.025 * iterations)], means[int(.975 * iterations)]

def paired_signflip_p(values, iterations=10000):
    observed = abs(statistics.mean(values))
    n = len(values)
    if n == 0:
        return None
    if n <= 16:
        count = 0
        total = 2 ** n
        for mask in range(total):
            test = sum(
                v if (mask >> i) & 1 else -v
                for i, v in enumerate(values)
            ) / n
            if abs(test) >= observed:
                count += 1
        return count / total
    count = 0
    for _ in range(iterations):
        test = abs(sum(v if random.random() < .5 else -v
                       for v in values) / n)
        if test >= observed:
            count += 1
    return (count + 1) / (iterations + 1)

comparisons = []
metrics = ["completion", "distance", "failures",
           "recoveries", "downtime", "runtime_seconds"]

for policy in sorted(expected_policies - {"greedy"}):
    for config in sorted(set((k[0], k[1], k[2]) for k in complete)):
        for metric in metrics:
            differences = []
            for key, policy_rows in complete.items():
                if key[:3] != config:
                    continue
                a = policy_rows[policy].get(metric, "")
                b = policy_rows["greedy"].get(metric, "")
                try:
                    differences.append(float(a) - float(b))
                except (ValueError, TypeError):
                    continue

            if not differences:
                continue

            lo, hi = bootstrap_ci(differences)
            comparisons.append({
                "policy": policy,
                "baseline": "greedy",
                "agents": config[0],
                "density": config[1],
                "failure_probability": config[2],
                "metric": metric,
                "n_paired": len(differences),
                "mean_paired_difference": statistics.mean(differences),
                "bootstrap_95_ci_low": lo,
                "bootstrap_95_ci_high": hi,
                "paired_signflip_p": paired_signflip_p(differences)
            })

os.makedirs(root, exist_ok=True)

with open(os.path.join(root, "paired_policy_comparisons.csv"),
          "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
    writer.writeheader()
    writer.writerows(comparisons)

report = {
    "trial_count": len(rows),
    "matched_groups": len(complete),
    "policies": sorted(expected_policies),
    "schedule_mismatches": len(mismatches),
    "unique_seeds": len({r["seed"] for r in rows}),
    "comparison_rows": len(comparisons),
    "interpretation": (
        "Paired differences compare each policy with greedy under "
        "the same configuration and seed. Confidence intervals are "
        "bootstrap intervals. P-values are exploratory and are not "
        "adjusted for multiple comparisons."
    )
}

with open(os.path.join(root, "final_analysis.json"), "w",
          encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# Create figures from the actual trial data.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["completion", "runtime_seconds", "failures"]:
        grouped = defaultdict(list)
        for r in rows:
            try:
                grouped[(r["policy"], int(r["agents"]))].append(
                    float(r[metric])
                )
            except (ValueError, TypeError, KeyError):
                pass

        fig, ax = plt.subplots()
        for policy in sorted(expected_policies):
            sizes = sorted({k[1] for k in grouped if k[0] == policy})
            if not sizes:
                continue
            means = [
                statistics.mean(grouped[(policy, n)])
                for n in sizes
            ]
            ax.plot(sizes, means, marker="o", label=policy)

        ax.set_xlabel("Number of agents")
        ax.set_ylabel(metric)
        ax.set_title(f"NanoSwarm policy comparison: {metric}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(root, f"{metric}_comparison.png"), dpi=200)
        plt.close(fig)

    report["figures_generated"] = True

except ImportError:
    report["figures_generated"] = False
    report["figure_note"] = "Matplotlib unavailable; install it to generate plots."

with open(os.path.join(root, "final_analysis.json"), "w",
          encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("Paired comparisons saved.")
