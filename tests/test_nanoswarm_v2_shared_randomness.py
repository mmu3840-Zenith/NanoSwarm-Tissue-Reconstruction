import unittest
import numpy as np
from nanoswarm_v2.__main__ import (
    make_failure_schedule,
    run_episode,
)

class SharedRandomnessTests(unittest.TestCase):

    def test_failure_schedule_is_reproducible(self):
        a = make_failure_schedule(8, 100, 1234)
        b = make_failure_schedule(8, 100, 1234)
        np.testing.assert_array_equal(a["uniforms"], b["uniforms"])
        np.testing.assert_array_equal(
            a["recovery_durations"], b["recovery_durations"]
        )
        self.assertEqual(a["schedule_id"], b["schedule_id"])

    def test_failure_schedule_is_independent_of_policy(self):
        shared = make_failure_schedule(8, 100, 4321)
        greedy = run_episode(
            "greedy", agents_n=8, density=.10, failure_prob=.02,
            seed=4321, steps=100, failure_schedule=shared
        )
        pheromone = run_episode(
            "pheromone", agents_n=8, density=.10, failure_prob=.02,
            seed=4321, steps=100, failure_schedule=shared
        )
        self.assertEqual(
            greedy["failure_schedule_id"],
            pheromone["failure_schedule_id"]
        )

    def test_different_seeds_produce_different_schedules(self):
        a = make_failure_schedule(8, 100, 1)
        b = make_failure_schedule(8, 100, 2)
        self.assertNotEqual(a["schedule_id"], b["schedule_id"])

    def test_schedule_shape_is_validated(self):
        invalid = make_failure_schedule(4, 10, 1)
        with self.assertRaises(ValueError):
            run_episode(
                "greedy", agents_n=4, density=.10, failure_prob=.01,
                seed=1, steps=11, failure_schedule=invalid
            )

if __name__ == "__main__":
    unittest.main()
