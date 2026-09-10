import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("selection", Path(__file__).parents[1] / "scripts/geometry_map/select_residual_search_v1.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def rows(cid="a", edited=.8, sham=1., block=3):
    return [{"candidate_id": cid, "episode": ep, "candidate": {"family": "residual", "block": block, "dose": .25},
             "baseline_mse": [1.]*6, "edited_mse": [edited]*6, "sham_mse": [sham]*6} for ep in range(8)]


class SelectionTests(unittest.TestCase):
    def test_episode_balanced_not_horizon_n(self):
        report = M.aggregate_candidates(rows())[0]
        self.assertEqual(report["independent_episode_count"], 8)
        self.assertAlmostEqual(report["search_score"], .2)

    def test_beating_sham_but_worse_baseline_not_promoted(self):
        report = M.aggregate_candidates(rows(edited=1.1, sham=1.2))
        self.assertEqual(M.diverse_parents(report), [])

    def test_missing_bad_start_not_dropped(self):
        report = M.aggregate_candidates(rows()[:-1])[0]
        self.assertFalse(report["complete"])
        self.assertFalse(report["eligible_for_physics"])

    def test_duplicates_and_held_rejected(self):
        with self.assertRaises(ValueError): M.aggregate_candidates(rows() + rows()[:1])
        bad = rows(); bad[0]["episode"] = 12
        with self.assertRaises(ValueError): M.aggregate_candidates(bad)

    def test_diversity_and_all_outcomes_retained(self):
        report = M.aggregate_candidates(rows("a", .7) + rows("b", .75) + rows("c", .8, block=0))
        self.assertEqual(len(report), 3)
        self.assertEqual(M.diverse_parents(report, 2), ["a", "c"])

    def test_pairs_budget_not_doubled(self):
        report = M.aggregate_candidates(rows("a") + rows("b", block=0))
        pair = M.pair_proposals(["a", "b"], report)[0]
        self.assertEqual(sum(pair["component_budget_shares"]), 1.)
        self.assertIn("A_at_half_budget", pair["required_comparators"])

    def test_mutation_provenance_and_bounds(self):
        report = M.aggregate_candidates(rows())
        mutations = M.mutate_specs(["a"], report, {"block": [0, 3, 5], "dose": [.25, .5]}, limit=2)
        self.assertEqual(len(mutations), 2)
        self.assertEqual({r["mutation_field"] for r in mutations}, {"block", "dose"})
        self.assertTrue(all(r["parent_ids"] == ["a"] for r in mutations))


if __name__ == "__main__": unittest.main()
