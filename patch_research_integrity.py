from pathlib import Path

p = Path("nanoswarm_v2/__main__.py")
s = p.read_text(encoding="utf-8")

def replace_once(old, new, label):
    global s
    if old not in s:
        raise SystemExit("Could not find expected code for: " + label)
    s = s.replace(old, new, 1)

# Track target-network synchronizations.
replace_once(
    "self.updates = 0\n        self.losses = []",
    "self.updates = 0\n        self.target_syncs = 0\n        self.losses = []",
    "DQN synchronization counter"
)
replace_once(
    "self.target = self.copy_weights()\n        return loss",
    "self.target = self.copy_weights()\n            self.target_syncs += 1\n        return loss",
    "DQN target synchronization"
)

# Track recoveries separately from failure events.
replace_once(
    "failures: int = 0\n    distance: int = 0",
    "failures: int = 0\n    recoveries: int = 0\n    distance: int = 0",
    "agent recovery counter"
)
replace_once(
    "if a.recovery <= 0:\n                    a.failed = False\n                continue",
    "if a.recovery <= 0:\n                    a.failed = False\n                    a.recoveries += 1\n                continue",
    "recovery event recording"
)
replace_once(
    '"failures": sum(a.failures for a in agents),\n        "downtime": downtime,',
    '"failures": sum(a.failures for a in agents),\n        "recoveries": sum(a.recoveries for a in agents),\n        "agents_recovered": sum(a.recoveries > 0 for a in agents),\n        "downtime": downtime,',
    "episode recovery metrics"
)
replace_once(
    '"dqn_updates": model.updates if model else 0,',
    '"dqn_updates": model.updates if model else 0,\n        "dqn_target_syncs": model.target_syncs if model else 0,',
    "episode DQN synchronization metric"
)

# Add configuration-specific summaries and paired comparisons.
insert_at = s.index("def main(")
analysis_code = r'''
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


'''
s = s[:insert_at] + analysis_code + s[insert_at:]

# Include new metrics in the ordinary summaries.
replace_once(
    '"distance", "failures", "downtime", "runtime_seconds")',
    '"distance", "failures", "recoveries", "agents_recovered",\n            "downtime", "runtime_seconds")',
    "summary metric list"
)

# Save the stratified and paired analysis after the existing summary.
replace_once(
    'json.dump(summary, f, indent=2)\n    print("Trials:", len(rows))',
    'json.dump(summary, f, indent=2)\n    paired = analyze_paired(rows)\n    with open("results_v2/paired_analysis.json", "w") as f:\n        json.dump(paired, f, indent=2)\n    print("Trials:", len(rows))',
    "paired analysis output"
)
replace_once(
    'print("Saved results_v2/trials.csv and results_v2/summary.json")',
    'print("Saved trials.csv, summary.json, and paired_analysis.json")',
    "output message"
)

p.write_text(s, encoding="utf-8")
print("Added recovery metrics, DQN target-sync tracking, and paired analysis.")
