import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parents[1]/"scripts/geometry_map"
sys.path.insert(0, str(ROOT))
import evolve_residual_search_v1 as M


def report(eligible=True):
    rows = []
    for i in range(6):
        c = {"id": f"base-{i}", "family": "pca_gain", "block": i, "site": "residual", "head": None,
             "mask": "all", "rank": 4, "pulse": 1, "dose": .25, "sign": 1}
        rows.append({"candidate_id": c["id"], "candidate": c, "complete": True,
                     "eligible_for_physics": eligible, "search_score": .1-i*.001 if eligible else -.1-i*.001,
                     "worst_terminal_benefit": -.2})
    return {"all_candidates": rows}


class EvolutionTests(unittest.TestCase):
    def test_population_breadth_and_budget(self):
        result = M.evolve([report()], 1)
        self.assertEqual(len(result["candidates"]), 128)
        self.assertEqual(sum(r["origin"] == "restart" for r in result["ancestry"].values()), 32)
        self.assertEqual(result["coverage"]["blocks"], list(range(6)))
        self.assertEqual(len(result["coverage"]["sites"]), 3)
        self.assertTrue(all(M.valid(c) for c in result["candidates"]))

    def test_determinism_and_no_duplicate_identity(self):
        a, b = M.evolve([report()], 1), M.evolve([report()], 1)
        self.assertEqual(a, b)
        self.assertEqual(len({M.spec_key(c) for c in a["candidates"]}), 128)

    def test_negative_parents_not_relabelled_as_success(self):
        result = M.evolve([report(False)], 1)
        self.assertTrue(result["negative_parent_fallback"])
        self.assertTrue(all(not row["parent_eligible_for_physics"] for row in result["ancestry"].values()))

    def test_coordinate_and_head_constraints(self):
        c = report()["all_candidates"][0]["candidate"]
        self.assertFalse(M.valid(dict(c, family="coordinate")))
        self.assertFalse(M.valid(dict(c, head=2)))
        self.assertFalse(M.valid(dict(c, site="attention_preproj", head=16)))
        self.assertTrue(M.valid(dict(c, family="coordinate", block=3, rank=8)))

    def test_prior_population_not_repeated(self):
        first = M.evolve([report()], 1)
        second = M.evolve([report()], 2, [first])
        self.assertFalse({M.spec_key(c) for c in first["candidates"]} & {M.spec_key(c) for c in second["candidates"]})

    def test_no_completed_result_is_not_scientific_failure(self):
        with self.assertRaises(ValueError): M.evolve([{"all_candidates": []}], 1)


if __name__ == "__main__": unittest.main()
