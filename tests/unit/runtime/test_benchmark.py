import unittest
from collections import Counter

import torch

from offline_study.models.backends import ToyBackend
from offline_study.runtime.benchmark import batches, filter_reviewed_development, score_predictions, select_rows, toy_rows
from offline_study.core.protocol import study_split, validate_manifest
from offline_study.data.inventory import _metaworld_trajectory_group, assign_metaworld_splits, assign_study_splits


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
                "toy:0": {"use": "development", "lineage_group": "toy:0",
                          "evidence": "reviewed"},
                "toy:1": {"use": "protected", "lineage_group": "toy:1",
                          "evidence": "legacy holdout"},
            },
        }
        filtered = filter_reviewed_development(rows, registry, "manifest")
        self.assertEqual([row["trajectory_id"] for row in filtered], ["toy:0"])
        with self.assertRaisesRegex(ValueError, "Exposure registry"):
            filter_reviewed_development(rows, registry, "different")

    def test_lineage_groups_never_cross_splits(self):
        rows = [
            {"trajectory_id": f"pusht:train:{group * 3 + variant}", "source_pool": "train",
             "lineage_group": f"group-{group}"}
            for group in range(20) for variant in range(3)
        ]
        rows.append({"trajectory_id": "pusht:val:0", "source_pool": "val",
                     "lineage_group": "external"})
        membership = assign_study_splits(rows, 234)
        self.assertEqual(len(membership), 20)
        for group in range(20):
            self.assertEqual(len({
                row["split"] for row in rows if row["lineage_group"] == f"group-{group}"
            }), 1)
        self.assertEqual(rows[-1]["split"], "external_reserve")

    def test_exposed_holdout_groups_are_reassigned_to_development(self):
        rows = [
            {"trajectory_id": f"pusht:train:{group}", "source_pool": "train",
             "lineage_group": f"group-{group}"}
            for group in range(20)
        ]
        membership = assign_study_splits(rows, 234, holdout_eligible=False)
        self.assertEqual(dict(Counter(membership.values())), {
            "fit": 16, "development": 4,
        })
        self.assertNotIn("holdout", {row["split"] for row in rows})

    def test_metaworld_exact_state_action_duplicates_share_lineage(self):
        states = torch.arange(24, dtype=torch.float32).view(6, 4)
        actions = torch.arange(12, dtype=torch.float32).view(6, 2)
        group = _metaworld_trajectory_group("mw-reach", states, actions)
        self.assertEqual(
            group,
            _metaworld_trajectory_group(
                "mw-reach", states.clone(), actions.clone()
            ),
        )
        self.assertNotEqual(
            group,
            _metaworld_trajectory_group(
                "mw-reach", states, actions + 1
            ),
        )
        self.assertNotEqual(
            group,
            _metaworld_trajectory_group(
                "mw-reach-wall", states, actions
            ),
        )

    def test_metaworld_duplicate_split_preserves_exposure_and_holdout(self):
        groups = [f"group-{i}" for i in range(20)]
        rows = [
            {
                "trajectory_id": f"metaworld:all:{i}",
                "legacy_split_unit": f"metaworld:all:{i}",
                "lineage_group": groups[i],
            }
            for i in range(20)
        ]
        legacy = study_split(
            [row["legacy_split_unit"] for row in rows], 234
        )
        by_split = {
            split: next(
                i for i, row in enumerate(rows)
                if legacy[row["legacy_split_unit"]] == split
            )
            for split in ("fit", "development", "holdout")
        }

        # A development duplicate can no longer be confirmation data.
        dev_i, holdout_i = by_split["development"], by_split["holdout"]
        rows[holdout_i]["lineage_group"] = rows[dev_i]["lineage_group"]
        membership, reconstructed = assign_metaworld_splits(rows, 234)
        self.assertEqual(reconstructed, legacy)
        self.assertEqual(membership[rows[dev_i]["lineage_group"]], "development")
        self.assertEqual(rows[holdout_i]["split"], "development")

        # Without development exposure, holdout wins over an unused fit copy.
        rows = [
            {
                "trajectory_id": f"metaworld:all:{i}",
                "legacy_split_unit": f"metaworld:all:{i}",
                "lineage_group": groups[i],
            }
            for i in range(20)
        ]
        fit_i, holdout_i = by_split["fit"], by_split["holdout"]
        rows[fit_i]["lineage_group"] = rows[holdout_i]["lineage_group"]
        membership, _ = assign_metaworld_splits(rows, 234)
        self.assertEqual(membership[rows[holdout_i]["lineage_group"]], "holdout")
        self.assertEqual(rows[fit_i]["split"], "holdout")

    def test_manifest_rejects_lineage_leakage_across_splits(self):
        rows = toy_rows(2)
        rows[1]["lineage_group"] = rows[0]["lineage_group"]
        rows[1]["split"] = "holdout"
        with self.assertRaisesRegex(ValueError, "Lineage group crosses study splits"):
            validate_manifest(rows)

    def test_exposure_filter_fails_closed_for_partial_development_family(self):
        rows = toy_rows(2)
        rows[1]["lineage_group"] = rows[0]["lineage_group"]
        registry = {
            "manifest_sha256": "manifest",
            "review_basis": "test audit",
            "trajectories": {
                "toy:0": {"use": "development", "lineage_group": rows[0]["lineage_group"],
                          "evidence": "reviewed"},
            },
        }
        self.assertEqual(filter_reviewed_development(rows, registry, "manifest"), [])


if __name__ == "__main__":
    unittest.main()
