from src.rl.dqn_model import DQN


def test_dqn_output_shape():
    import torch

    model = DQN(
        state_dim=10,
        action_dim=5,
    )

    state = torch.zeros(
        (2, 10),
        dtype=torch.float32,
    )

    output = model(state)

    assert output.shape == (2, 5)
