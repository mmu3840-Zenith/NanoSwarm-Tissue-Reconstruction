from src.agents.nanobot import Nanobot


def test_nanobot_movement_is_bounded():
    agent = Nanobot(1)

    agent.move(
        2,
        width=5,
        height=5,
    )

    assert agent.position() == (0, 0)


def test_heuristic_action():
    agent = Nanobot(
        1,
        x=2,
        y=2,
    )

    assert agent.heuristic_action(
        (4, 2)
    ) == 3
