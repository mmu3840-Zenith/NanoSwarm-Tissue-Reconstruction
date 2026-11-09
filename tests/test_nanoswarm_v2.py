import unittest
import numpy as np
from nanoswarm_v2.__main__ import Tissue, DQN, run_episode

class NanoSwarmV2Tests(unittest.TestCase):
    def test_environment_reproducible(self):
        a, b = Tissue(seed=42), Tissue(seed=42)
        np.testing.assert_array_equal(a.damage, b.damage)

    def test_repair_reduces_damage(self):
        e = Tissue(density=1.0, seed=2)
        before = e.damage.sum()
        self.assertTrue(e.repair(0, 0))
        self.assertLess(e.damage.sum(), before)

    def test_pheromone_diffuses_and_evaporates(self):
        e = Tissue(seed=1)
        e.pheromone[10, 10] = 1
        e.update_pheromone()
        self.assertGreater(e.pheromone[10, 9], 0)
        self.assertLess(e.pheromone.sum(), 1)

    def test_dqn_output_shape(self):
        model = DQN(9)
        self.assertEqual(model.forward(np.zeros((1, 9)))[0].shape, (1, 5))

    def test_episode_metrics(self):
        r = run_episode("greedy", agents_n=3, seed=3, steps=10)
        self.assertGreaterEqual(r["completion"], 0)
        self.assertLessEqual(r["completion"], 1)

if __name__ == "__main__":
    unittest.main()
