from src.metrics.metrics_engine import SimulationMetrics


def test_metrics_serialization():
    metrics = SimulationMetrics(
        0.75,
        3,
        20,
        1,
    )

    data = metrics.as_dict()

    assert data["completion_rate"] == 0.75
    assert data["remaining_damage"] == 3
    assert data["convergence_steps"] == 20
    assert data["failures"] == 1
