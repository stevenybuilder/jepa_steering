import copy
import unittest
import tempfile
from pathlib import Path

from offline_study.behavioral_rank import REVIEWED_EQUIVALENT, components, validate_engineering, verify_source_pair
from offline_study.planning_support import TRANSFER_POLICY
from offline_study.behavioral_analysis import shard_paths


class BehavioralRankTests(unittest.TestCase):
    def test_analysis_discovers_single_stream_and_grouped_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("shard-0", "rank-4", "unrelated"):
                (root / name).mkdir()
            self.assertEqual([p.name for p in shard_paths(root)], ["rank-4", "shard-0"])

    def test_source_equivalence_is_exact_and_closed_to_new_changes(self):
        verify_source_pair("unchanged.py", "abc", "abc")
        for name, pair in REVIEWED_EQUIVALENT.items():
            verify_source_pair(name, *pair)
            with self.assertRaises(ValueError):
                verify_source_pair(name, pair[0], "unreviewed")
        with self.assertRaises(ValueError):
            verify_source_pair("support_operator.py", "before", "after")

    def test_only_exact_engineered_members_enabled(self):
        self.assertEqual(components("reach", "combined"), ("joint_equal_standardized_energy", "rank4"))
        self.assertEqual(components("reach-wall", "matched_random_rank4"), ("native", "matched_random_rank4"))
        for task, arm in (("reach", "rank4_only"), ("reach-wall", "combined"),
                          ("reach", "native"), ("pusht", "combined")):
            with self.assertRaises(ValueError):
                components(task, arm)

    def test_engineering_cannot_be_relabelled_or_shortened(self):
        binding = {"protocol_sha256": "p", "bank_sha256": "b"}
        report = {"status": "selected_primary_recipe_full_cem_simulator_smoke_complete",
            "task": "reach-wall", "arm": "rank4", "combined": False,
            "actual_cem_integration_validated": True, "full_simulator_episode_validated": True,
            "parameters_unchanged": True, "scientific_efficacy_measurement": False,
            "protected_outcomes_accessed": False, "result": {"elementary_steps": 100}}
        ep = {"task": "reach-wall", "arm": "rank4", "combined": False,
              "rank_binding": binding, "coupling_binding": None, "transfer_policy": TRANSFER_POLICY}
        validate_engineering(report, ep, "reach-wall", "rank4_only", binding, None)
        for key, value in (("arm", "matched_random_rank4"), ("combined", True),
                           ("parameters_unchanged", False), ("result", {"elementary_steps": 30}),
                           ("scientific_efficacy_measurement", True), ("protected_outcomes_accessed", True)):
            with self.assertRaises(ValueError):
                validate_engineering({**report, key: value}, ep, "reach-wall", "rank4_only", binding, None)
        changed = copy.deepcopy(ep)
        changed["transfer_policy"]["candidate_chunk_size"] = 32
        with self.assertRaises(ValueError):
            validate_engineering(report, changed, "reach-wall", "rank4_only", binding, None)
        with self.assertRaises(ValueError):
            validate_engineering(report, ep, "reach-wall", "rank4_only", {**binding, "bank_sha256": "new"}, None)
