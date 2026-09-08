import ast
from pathlib import Path
import unittest

import torch

from offline_study.routing_hmm import fit_hmm, route_prefix


class RoutingHMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.features = torch.randn(24, 6, 400, generator=torch.Generator().manual_seed(27))
        cls.model = fit_hmm(cls.features)

    def test_independent_archived_math_matches_port(self):
        path = Path(__file__).resolve().parents[1] / "archive/2026-09-07-workspace/scripts/geometry_map/run_reach_hmm_routing.py"
        if not path.exists():
            self.skipTest("Historical source absent from clean checkout")
        tree = ast.parse(path.read_text())
        names = {"emission", "filter_beliefs", "fit_hmm", "route"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes}, names)
        namespace = {"torch": torch}
        # Only the four inspected pure-math functions, never archived workloads.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "independent_archived_math", "exec"), namespace)
        source = namespace["fit_hmm"](self.features)
        for name, value in self.model.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, source[name]), name)
            else:
                self.assertEqual(value, source[name])
        expected = namespace["route"](self.features, source)
        actual = route_prefix(self.features[:, :2], self.model)
        for name in actual:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_future_horizons_rejected_and_no_state_shared_between_candidates(self):
        with self.assertRaisesRegex(ValueError, "exactly native H1/H2"):
            route_prefix(self.features, self.model)
        prefix = self.features[:, :2]
        batch = route_prefix(prefix, self.model)
        for i in range(len(prefix)):
            single = route_prefix(prefix[i:i+1], self.model)
            for key in batch:
                torch.testing.assert_close(batch[key][i:i+1], single[key], atol=1e-12, rtol=1e-12)
        reverse = route_prefix(prefix.flip(0), self.model)
        for key in batch:
            torch.testing.assert_close(batch[key], reverse[key].flip(0), atol=1e-12, rtol=1e-12)

    def test_memoryless_ignores_h1_but_hmm_can_use_it(self):
        a = self.features[:, :2].clone()
        b = a.clone(); b[:, 0] = b[:, 0].flip(0)
        first, second = route_prefix(a, self.model), route_prefix(b, self.model)
        self.assertTrue(torch.equal(first["memoryless"], second["memoryless"]))
        self.assertTrue(torch.equal(first["static"], second["static"]))
        self.assertGreater(float((first["posterior"] - second["posterior"]).abs().max()), 1e-8)

    def test_finite_probability_and_fixed_fit_contract(self):
        result = route_prefix(self.features[:, :2], self.model)
        torch.testing.assert_close(result["posterior"].sum(-1), torch.ones(24, dtype=torch.float64))
        self.assertTrue(((result["hmm"] >= .5) & (result["hmm"] <= 1)).all())
        self.assertEqual(len(self.model["log_likelihood_history"]), 20)
        for bad in (self.features[:, :2], torch.full_like(self.features, float("nan"))):
            with self.assertRaises(ValueError):
                fit_hmm(bad)


if __name__ == "__main__":
    unittest.main()
