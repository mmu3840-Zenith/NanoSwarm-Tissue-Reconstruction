import unittest
import numpy as np
from nanoswarm_v2.__main__ import (
    Tissue, DQN, Agent, run_episode, greedy_action, pheromone_action
)

class NanoSwarmV2ResearchTests(unittest.TestCase):

    def test_dqn_replay_training_updates_weights(self):
        model = DQN(9, seed=11)
        before = model.w1.copy()
        for i in range(80):
            state = np.full(9, i / 80, dtype=np.float32)
            model.remember(state, i % 5, 1.0, state * .5, False)
        loss = model.train_step(batch_size=32)
        self.assertIsNotNone(loss)
        self.assertGreater(model.updates, 0)
        self.assertFalse(np.array_equal(before, model.w1))

    def test_dqn_replay_rng_is_seeded(self):
        a, b = DQN(9, seed=42), DQN(9, seed=42)
        for i in range(100):
            state = np.full(9, i, dtype=np.float32)
            for model in (a, b):
                model.remember(state, i % 5, float(i), state, False)
        a.train_step(32)
        b.train_step(32)
        np.testing.assert_allclose(a.w1, b.w1)

    def test_local_policy_actions_are_valid(self):
        env = Tissue(seed=12)
        agent = Agent(10, 10)
        rng = np.random.default_rng(12)
        self.assertIn(greedy_action(env, agent, rng), range(5))
        self.assertIn(pheromone_action(env, agent, rng), range(5))

    def test_failure_events_are_recorded(self):
        result = run_episode(
            "greedy", agents_n=3, density=.20,
            failure_prob=1.0, seed=23, steps=15
        )
        self.assertGreater(result["failures"], 0)
        self.assertGreater(result["downtime"], 0)

    def test_observation_is_local(self):
        env = Tissue(seed=1)
        self.assertEqual(env.observe(10, 10).shape, (9,))
        self.assertEqual(env.observe(10, 10, True).shape, (18,))

if __name__ == "__main__":
    unittest.main()
