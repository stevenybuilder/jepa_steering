import unittest

import torch

from offline_study.planning.planning_env_smoke import observation_digest
from offline_study.planning.planning_goal_bank import restore_goal


class Observation(dict):
    def clone(self):
        return Observation({k: v.clone() for k, v in self.items()})


class GoalBankTests(unittest.TestCase):
    def inputs(self):
        image = torch.full((1, 3, 224, 224), 100, dtype=torch.uint8)
        image.flatten()[0] = 101
        initial = Observation(visual=image.clone(), proprio=torch.zeros(1, 4))
        goal = Observation(visual=image.clone(), proprio=torch.ones(1, 4))
        saved = {"goal": goal.clone()}
        metadata = {"initial_sha256": observation_digest(initial), "goal_sha256": observation_digest(goal),
                    "goal_state": [1., 2., 3.]}
        return initial, goal, saved, metadata

    def test_one_level_render_jitter_is_not_silently_accepted_as_paired(self):
        initial, goal, saved, metadata = self.inputs()
        goal["visual"].flatten()[4:9] += 1
        self.assertNotEqual(observation_digest(goal), metadata["goal_sha256"])
        result = restore_goal(initial, goal, [1., 2., 3.], saved, metadata)
        self.assertEqual(observation_digest(result), metadata["goal_sha256"])
        # The source image and all saved evidence remain unmodified.
        self.assertNotEqual(observation_digest(goal), metadata["goal_sha256"])
        self.assertTrue(torch.equal(result["proprio"], goal["proprio"]))

    def test_physical_goal_or_initial_change_is_not_a_render_repair(self):
        initial, goal, saved, metadata = self.inputs()
        with self.assertRaises(ValueError):
            restore_goal(initial, goal, [1., 2., 3.0001], saved, metadata)
        goal["proprio"][0, 0] += .001
        with self.assertRaises(ValueError):
            restore_goal(initial, goal, [1., 2., 3.], saved, metadata)
        initial, goal, saved, metadata = self.inputs()
        initial["visual"].flatten()[3] += 1
        with self.assertRaises(ValueError):
            restore_goal(initial, goal, [1., 2., 3.], saved, metadata)

    def test_unknown_render_error_and_altered_bank_fail_closed(self):
        initial, goal, saved, metadata = self.inputs()
        goal["visual"].flatten()[3] += 2
        with self.assertRaises(ValueError):
            restore_goal(initial, goal, [1., 2., 3.], saved, metadata)
        initial, goal, saved, metadata = self.inputs()
        goal["visual"].flatten()[3:68] += 1
        with self.assertRaises(ValueError):
            restore_goal(initial, goal, [1., 2., 3.], saved, metadata)
        initial, goal, saved, metadata = self.inputs()
        saved["goal"]["visual"].flatten()[3] += 1
        with self.assertRaises(ValueError):
            restore_goal(initial, goal, [1., 2., 3.], saved, metadata)
