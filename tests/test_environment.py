from src.environment.tissue_environment import TissueEnvironment


def test_environment_is_deterministic():
    first = TissueEnvironment(seed=42)
    second = TissueEnvironment(seed=42)

    assert first.damage == second.damage


def test_repair_reduces_damage():
    environment = TissueEnvironment(
        damage_probability=1.0,
        seed=1,
    )

    initial = environment.remaining_damage()

    assert environment.repair((0, 0))
    assert environment.remaining_damage() == initial - 1
    assert environment.completion_rate(initial) > 0


def test_invalid_environment():
    try:
        TissueEnvironment(width=0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )
