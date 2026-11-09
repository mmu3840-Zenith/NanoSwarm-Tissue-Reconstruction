from pathlib import Path

path = Path("nanoswarm_v2/__main__.py")
source = path.read_text(encoding="utf-8")

marker = 'def main():'
end_marker = '\n\nif __name__ == "__main__":'

if marker not in source or end_marker not in source:
    raise SystemExit("Could not locate main() boundaries. No changes made.")

new_main = r'''
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
'''

start = source.index(marker)
end = source.index(end_marker, start)

updated = source[:start] + new_main + source[end:]
path.write_text(updated, encoding="utf-8")

print("Checkpoint/resume runner installed.")
