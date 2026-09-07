import unittest

import torch

from offline_study.backends import ToyBackend
from offline_study.benchmark import (
    batches,
    filter_reviewed_development,
    score_predictions,
    select_rows,
    toy_rows,
)
from offline_study.protocol import validate_manifest


class BenchmarkTests(unittest.TestCase):
    def test_toy_backend_refuses_unvalidated_precision_modes(self):
        with self.assertRaises(ValueError):
            ToyBackend(precision="bfloat16")
        with self.assertRaises(ValueError):
            ToyBackend(allow_tf32=True)

    def test_temporal_metric_alignment(self):
        target = torch.arange(7.).view(1, 7, 1).expand(2, 7, 3)
        prediction = target.transpose(0, 1).clone()
        result = score_predictions({"visual": prediction, "proprio": prediction},
                                   {"visual": target, "proprio": target})
        self.assertTrue(all(v == 0 for row in result for v in row.values()))
        prediction[3] += 2
        result = score_predictions({"visual": prediction, "proprio": prediction},
                                   {"visual": target, "proprio": target})
        self.assertEqual(result[0]["visual_mse_h3"], 4.)
        self.assertEqual(result[0]["visual_mse_h6"], 0.)

    def test_initial_context_excludes_future_observations(self):
        backend = ToyBackend()
        encoded = {"visual": torch.randn(2, 7, 16), "proprio": torch.randn(2, 7, 4)}
        context = backend.context(encoded)
        actions = torch.randn(6, 2, 20)
        before = backend.predict(context, actions)
        encoded["visual"][:, 1:] = 1e9
        encoded["proprio"][:, 1:] = 1e9
        after = backend.predict(backend.context(encoded), actions)
        self.assertTrue(torch.equal(before["visual"], after["visual"]))
        self.assertTrue(torch.equal(before["proprio"], after["proprio"]))

    def test_identity_and_all_three_tasks(self):
        rows = toy_rows(12)
        validate_manifest(rows)
        selection = select_rows(rows, ["all"], "development", 3, 0, 1)
        self.assertEqual({r["task"] for r in selection}, {"mw-reach", "mw-reach-wall", "pusht"})
        backend = ToyBackend()
        meta, visual, proprio, raw = next(batches(selection, 5, backend, None))
        self.assertEqual(raw.shape, (5, 6, 5, 4))
        encoded = backend.encode(visual, proprio)
        context, actions = backend.context(encoded), backend.normalize_actions(raw)
        native = backend.predict(context, actions)
        patched = backend.predict(context, actions, instrument=True)
        self.assertEqual(native["visual"].shape[:2], (7, 5))
        self.assertTrue(torch.equal(native["visual"], patched["visual"]))
        self.assertEqual(len(meta), 5)

    def test_missing_task_and_invalid_windows_rejected(self):
        with self.assertRaises(ValueError):
            select_rows(toy_rows(3), ["missing"], "development", 3, 0, 1)
        with self.assertRaises(ValueError):
            select_rows(toy_rows(3), ["all"], "development", 1, 0, 1)
        rows = toy_rows(1)
        rows[0]["starts"] = [70]
        with self.assertRaises(ValueError):
            validate_manifest(rows)

    def test_exposure_filter_precedes_selection_and_sharding(self):
        rows = toy_rows(3)
        registry = {
            "manifest_sha256": "manifest",
            "review_basis": "test audit",
            "trajectories": {
                "toy:0": {"use": "development", "evidence": "reviewed"},
                "toy:1": {"use": "protected", "evidence": "legacy holdout"},
            },
        }
        filtered = filter_reviewed_development(rows, registry, "manifest")
        self.assertEqual([row["trajectory_id"] for row in filtered], ["toy:0"])
        with self.assertRaisesRegex(ValueError, "Exposure registry"):
            filter_reviewed_development(rows, registry, "different")


if __name__ == "__main__":
    unittest.main()
