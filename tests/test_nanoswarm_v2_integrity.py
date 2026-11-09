import unittest
import numpy as np
from nanoswarm_v2.__main__ import DQN, run_episode

class NanoSwarmIntegrityTests(unittest.TestCase):

    def test_target_network_synchronizes(self):
        model = DQN(9, seed=31)
        state = np.ones(9, dtype=np.float32)
        for i in range(80):
            model.remember(state, i % 5, 1.0, state * .5, False)

        for _ in range(100):
            model.train_step(batch_size=32)

        self.assertEqual(model.target_syncs, 1)
        for online, target in zip(model.copy_weights(), model.target):
            np.testing.assert_allclose(online, target)

    def test_recovery_is_measured(self):
        result = run_episode(
            "greedy", agents_n=4, density=.20,
            failure_prob=1.0, seed=81, steps=40
        )
        self.assertGreater(result["failures"], 0)
        self.assertGreater(result["recoveries"], 0)
        self.assertGreater(result["agents_recovered"], 0)

    def test_recovery_metrics_are_consistent(self):
        result = run_episode(
            "greedy", agents_n=4, density=.20,
            failure_prob=.5, seed=82, steps=40
        )
        self.assertLessEqual(result["agents_recovered"], 4)
        self.assertLessEqual(result["recoveries"], result["failures"])

if __name__ == "__main__":
    unittest.main()
