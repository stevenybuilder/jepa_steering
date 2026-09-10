import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from offline_study.author_validation import (
    SOURCE_REVISION, approximate_standardized_mde, audit, official_clips,
    official_partition, require_disjoint_fit,
)
from offline_study.vendor import use_vendor
from offline_study.author_runtime import encoded_batches


def rows(n, kind="metaworld", pool="all", length=99):
    return [dict(dataset=kind, source_pool=pool, index=i, length=length,
                 trajectory_id=f"{pool}:{i}", lineage_group=f"{pool}:{i}",
                 task="reach" if i % 2 else "wall") for i in range(n)]


class AuthorValidationTests(unittest.TestCase):
    def test_source_order_not_input_order_controls_membership(self):
        data = rows(100)
        self.assertEqual(official_partition(data, "metaworld"),
                         official_partition(list(reversed(data)), "metaworld"))

    def test_missing_source_rows_and_duplicate_rows_fail(self):
        data = rows(100)
        for invalid in (data[1:], data + [data[0]]):
            with self.assertRaises(ValueError):
                official_partition(invalid, "metaworld")

    def test_pusht_uses_supplied_val_even_when_seed_and_fraction_change(self):
        data = rows(100, "pusht", "train") + rows(21, "pusht", "val")
        train, val = official_partition(data, "pusht", seed=999, train_fraction=.5)
        self.assertEqual(len(train), 100)
        self.assertEqual(len(val), 21)
        self.assertEqual({row["source_pool"] for row in val}, {"val"})

    def test_fit_guard_blocks_related_rows_not_just_identical_ids(self):
        val = rows(2)
        fit = copy.deepcopy(val)
        fit[0]["trajectory_id"] = "different-row-same-family"
        with self.assertRaisesRegex(ValueError, "refit required"):
            require_disjoint_fit(fit, val)
        require_disjoint_fit(rows(1, pool="train"), val)

    def test_sensitivity_is_not_row_count_power(self):
        self.assertAlmostEqual(approximate_standardized_mde(36), .466930868, places=6)
        self.assertGreater(approximate_standardized_mde(21), approximate_standardized_mde(36))
        with self.assertRaises(ValueError):
            approximate_standardized_mde(1)

    def test_clip_boundaries(self):
        clips = official_clips(rows(1), 18, 5)
        self.assertEqual(len(clips), 10)
        self.assertEqual(sorted(start for _, start, _ in clips), list(range(10)))
        self.assertEqual(official_clips(rows(1, length=89), 18, 5), [])

    def test_audit_preserves_protection_and_never_authorizes_execution(self):
        data = rows(12600)
        for row in data:
            row.update(split="fit", source_revision=SOURCE_REVISION)
        _, val = official_partition(data, "metaworld")
        val[0]["split"] = "holdout"
        original = copy.deepcopy(data)
        cfg = {"data": {"seed": 234, "custom": {"split_ratio": .9, "frameskip": 5},
                        "validation": {"num_frames_val": 18, "val_dataset_batch_size": 4,
                                       "val_dataset_drop_last": False}},
               "meta": {"dtype": "bfloat16", "data_traj_rollout_eval": {
                   "data_traj_eval_rollout_steps": 6, "data_traj_eval_ctxt_window": 3}},
               "data_aug": {}}
        report = audit(data, "metaworld", cfg, fits={"bad_fit": [val[1]]})
        self.assertEqual(data, original)
        self.assertFalse(report["authors_evaluation_reproduced"])
        self.assertIn("NOT_execution_authorization", report["status"])
        self.assertEqual(report["validation_rows"], 1260)
        self.assertEqual(report["validation_clips"], 12600)
        self.assertIn(val[0]["trajectory_id"], report["tasks"][val[0]["task"]]["protected_ids"])
        self.assertEqual(report["fit_validation_overlap"]["bad_fit"], [val[1]["lineage_group"]])
        with self.assertRaisesRegex(ValueError, "complete released inventory"):
            audit(data[:300], "metaworld", cfg)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "vendor/jepa-wms/app/plan_common/datasets/traj_dset.py").is_file(),
                         "Pinned upstream checkout required for direct parity test")
    def test_partition_and_clip_order_match_actual_upstream(self):
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        previous_path = sys.path[:]
        try:
            use_vendor(vendor)
            spec = importlib.util.spec_from_file_location("author_traj_parity", vendor / "app/plan_common/datasets/traj_dset.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            class MetadataDataset:
                proprio_dim, action_dim, state_dim = 4, 4, 39

                def __len__(self):
                    return 100

                def get_seq_length(self, index):
                    return 99

                def __getitem__(self, index, subtask=None):
                    obs = {"visual": torch.arange(99 * 3 * 2 * 2).reshape(99, 3, 2, 2).float() + index,
                           "proprio": torch.arange(99 * 4).reshape(99, 4).float() + index}
                    return obs, torch.arange(99 * 4).reshape(99, 4).float() - index, None, None, None

            class IdentityEncoder:
                def encode_clip(self, obs, actions):
                    return {**obs, "action": actions}

            _, actual_val = module.split_traj_datasets(MetadataDataset(), .9, 234)
            _, val = official_partition(rows(100), "metaworld")
            self.assertEqual([row["index"] for row in val], actual_val.indices)
            for frames in (8, 18):
                actual_clips = module.TrajSlicerDataset(actual_val, frames, 5, 1,
                                                       generator=torch.Generator().manual_seed(234))
                expected = [(val[i]["trajectory_id"], start, end) for i, start, end in actual_clips.slices]
                self.assertEqual(official_clips(val, frames, 5), expected)
                # Compare actual upstream tensors, not only our duplicated arithmetic.
                config = {"data": {"validation": {"num_frames_val": frames}, "custom": {"frameskip": 5}}}
                observed = {}
                for meta, tensors in encoded_batches(IdentityEncoder(), MetadataDataset(), val[:1], config):
                    for i, group in enumerate(meta):
                        observed[(group[0]["trajectory_id"], group[0]["clip_start"])] = {
                            key: value[i] for key, value in tensors.items()}
                for i, (row_index, start, _) in enumerate(actual_clips.slices):
                    if row_index != 0:
                        continue
                    obs, actions, _, _ = actual_clips[i]
                    candidate = observed[(val[0]["trajectory_id"], start)]
                    for key in obs:
                        self.assertTrue(torch.equal(obs[key], candidate[key]))
                    self.assertTrue(torch.equal(actions, candidate["action"]))
        finally:
            sys.path[:] = previous_path


if __name__ == "__main__":
    unittest.main()
