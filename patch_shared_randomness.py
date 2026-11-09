from pathlib import Path

path = Path("nanoswarm_v2/__main__.py")
source = path.read_text(encoding="utf-8")

start = source.index("def run_episode(")
end = source.index("\ndef train_model(", start)
episode = source[start:end]

old_header = """def run_episode(policy, agents_n=25, density=.10, failure_prob=0,
                seed=1, steps=250, model=None, training=False):
    rng = np.random.default_rng(seed)
    env = Tissue(20, density, seed)
    agents = [Agent(int(rng.integers(20)), int(rng.integers(20)))
              for _ in range(agents_n)]
"""
new_header = """def make_failure_schedule(agents_n, steps, seed):
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
"""
if old_header not in episode:
    raise SystemExit("Could not locate run_episode initialization. No changes made.")
episode = episode.replace(old_header, new_header, 1)

old_loop = """        for a in agents:
"""
new_loop = """        for agent_index, a in enumerate(agents):
"""
if old_loop not in episode:
    raise SystemExit("Could not locate agent loop. No changes made.")
episode = episode.replace(old_loop, new_loop, 1)

old_failure = """            if rng.random() < failure_prob:
                a.failed = True
                a.recovery = int(rng.integers(2, 7))
                a.failures += 1
"""
new_failure = """            # The same pre-generated exogenous event is presented to
            # every policy at this timestep and agent index.
            if failure_schedule["uniforms"][t, agent_index] < failure_prob:
                a.failed = True
                a.recovery = int(
                    failure_schedule["recovery_durations"][t, agent_index]
                )
                a.failures += 1
"""
if old_failure not in episode:
    raise SystemExit("Could not locate failure event block. No changes made.")
episode = episode.replace(old_failure, new_failure, 1)

old_result = '''        "seed": seed, "policy": policy, "agents": agents_n,
'''
new_result = '''        "seed": seed, "policy": policy, "agents": agents_n,
        "failure_schedule_id": failure_schedule["schedule_id"],
'''
if old_result not in episode:
    raise SystemExit("Could not locate result dictionary. No changes made.")
episode = episode.replace(old_result, new_result, 1)

source = source[:start] + episode + source[end:]

# Update the benchmark loop to construct one shared schedule per seed/config.
old_main = """        for seed in range(seed_count):
            for policy in policies:
                rows.append(run_episode(
                    policy, n, density, failure,
                    seed=100000 + seed, steps=150 if args.quick else 500,
                    model=models.get(policy)
                ))
"""
new_main = """        for seed in range(seed_count):
            episode_seed = 100000 + seed
            episode_steps = 150 if args.quick else 500
            shared_schedule = make_failure_schedule(
                n, episode_steps, episode_seed
            )
            for policy in policies:
                rows.append(run_episode(
                    policy, n, density, failure,
                    seed=episode_seed, steps=episode_steps,
                    model=models.get(policy),
                    failure_schedule=shared_schedule
                ))
"""
if old_main not in source:
    raise SystemExit("Could not locate benchmark loop. No changes made.")
source = source.replace(old_main, new_main, 1)

path.write_text(source, encoding="utf-8")
print("Shared failure schedules and independent RNG streams installed.")
