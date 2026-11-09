import csv, json, os
from collections import defaultdict

path = "results_v2/trials.csv"

if not os.path.exists(path):
    raise SystemExit("Missing trials.csv")

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("No trial records found")

required = ["policy", "seed", "agents", "completion",
            "failure_schedule_id"]

missing = [x for x in required if x not in rows[0]]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

groups = defaultdict(set)
for row in rows:
    key = (row["agents"], row["density"],
           row["failure_probability"], row["seed"])
    groups[key].add(row["failure_schedule_id"])

bad = [k for k, v in groups.items() if len(v) != 1]

print("Trial rows:", len(rows))
print("Matched configuration/seed groups:", len(groups))
print("Schedule mismatches:", len(bad))
print("Policies:", sorted(set(r["policy"] for r in rows)))

if bad:
    raise SystemExit("Shared failure schedule validation failed")

for filename in ("summary.json", "paired_analysis.json"):
    if not os.path.exists("results_v2/" + filename):
        raise SystemExit("Missing " + filename)
    with open("results_v2/" + filename, encoding="utf-8") as f:
        json.load(f)

print("JSON validation passed")
print("Result integrity verification passed")
