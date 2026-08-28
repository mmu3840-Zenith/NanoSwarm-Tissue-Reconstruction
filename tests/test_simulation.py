from src.simulation.runner import run_simulation


def test_simulation_is_deterministic():
    first = run_simulation(
        agents=20,
        steps=50,
        seed=42,
    )
    second = run_simulation(
        agents=20,
        steps=50,
        seed=42,
    )

    assert first == second


def test_simulation_repairs_damage():
    result = run_simulation(
        agents=100,
        steps=200,
        seed=42,
    )

    assert result["initial_damage"] > 0
    assert result["final_damage"] == 0
    assert result["completion_rate"] == 1.0
    assert result["area_coverage"] == 1.0
    assert result["repairs"] == result["initial_damage"]


def test_simulation_metrics_are_valid():
    result = run_simulation(
        agents=25,
        steps=100,
        seed=7,
    )

    assert 0.0 <= result["completion_rate"] <= 1.0
    assert 0.0 <= result["area_coverage"] <= 1.0
    assert result["final_damage"] >= 0
    assert result["repairs"] >= 0
    assert result["convergence_steps"] <= result["steps_requested"]
    assert result["failures"] >= 0