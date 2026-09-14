import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_scenarios import close_expert_environments, validate_schedule


class ScenarioPreparationTests(unittest.TestCase):
    def test_only_paper_scale_metaworld_inputs(self):
        vendor = Path(__file__).resolve().parents[3] / "vendor/jepa-wms"
        if not vendor.is_dir():
            self.skipTest("Pinned source unavailable")
        contract = prepare(vendor, "reach")
        self.assertEqual(len(validate_schedule(contract)), 96)
        changed = copy.deepcopy(contract)
        changed["episodes"][0]["environment_seed"] += 1
        with self.assertRaises(ValueError):
            validate_schedule(changed)
        with self.assertRaises(ValueError):
            validate_schedule(prepare(vendor, "pusht"))

    def test_auxiliary_envs_closed_and_factory_restored_even_on_error(self):
        closed = []
        factory = lambda: SimpleNamespace(close=lambda: closed.append(True))
        module = SimpleNamespace(make_env=factory)
        with self.assertRaises(RuntimeError):
            with close_expert_environments(module):
                module.make_env()
                module.make_env()
                raise RuntimeError("expert failed")
        self.assertIs(module.make_env, factory)
        self.assertEqual(closed, [True, True])
