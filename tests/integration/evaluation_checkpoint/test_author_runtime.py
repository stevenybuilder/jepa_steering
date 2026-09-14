import contextlib
import unittest

import torch

from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, build_cohort, examples, prefix_batches, validate_cohort
from offline_study.evaluation.checkpoint.author_validation import SOURCE_REVISION


class Core:
    """History-dependent fixture: losing observed context or shifting actions fails."""
    def forward_pred(self, visual, actions, proprio, debug):
        assert visual.shape[1] == actions.shape[1] == proprio.shape[1]
        offset = actions.sum(1).reshape(-1, 1, 1)
        return visual + visual.mean(1, keepdim=True) + offset, None, proprio + offset


class AuthorRuntimeTests(unittest.TestCase):
    def backend(self):
        backend = object.__new__(AuthorBackend)
        backend.core = Core()
        backend.autocast = contextlib.nullcontext
        return backend

    def test_rollout_uses_three_context_and_never_future_targets(self):
        backend = self.backend()
        enc = {"visual": torch.arange(18.).reshape(1, 18, 1),
               "proprio": torch.ones(1, 18, 1), "action": torch.ones(1, 18, 1)}
        ctx = backend.context_at(enc, 5)
        self.assertEqual(ctx["visual"].flatten().tolist(), [3., 4., 5.])
        self.assertEqual(ctx["past_actions"].shape[1], 2)
        prediction = backend.predict(ctx, enc["action"][:, 5:11].transpose(0, 1))
        self.assertEqual(prediction["visual"][1].item(), 12.)
        self.assertEqual(prediction["visual"].shape, (7, 1, 1))
        enc["visual"][:, 6:] = 100000
        again = backend.predict(backend.context_at(enc, 5), enc["action"][:, 5:11].transpose(0, 1))
        self.assertTrue(torch.equal(prediction["visual"], again["visual"]))

    def test_all_clip_prefix_pairs_have_collision_free_ids(self):
        rows = [{"length": 99, "trajectory_id": "a", "lineage_group": "g", "task": "reach"}]
        config = {"data": {"validation": {"num_frames_val": 18}, "custom": {"frameskip": 5}}}
        full = examples(rows, config)
        self.assertEqual(len(full), 120)
        self.assertEqual(len({m["start"] for m in full}), 120)
        fit = examples(rows, config, fitting=True)
        self.assertEqual(len(fit), 4)
        self.assertEqual([m["start"] for m in fit], [0, 40, 79, 119])

    def test_fit_prefix_selection_retains_pairing_in_mixed_clip_batch(self):
        backend = self.backend()
        enc = {"visual": torch.zeros(2, 8, 1), "proprio": torch.ones(2, 8, 1),
               "action": torch.ones(2, 8, 1)}
        groups = [[{"prefix": 0, "id": "a"}], [{"prefix": 1, "id": "b"}]]
        batches = list(prefix_batches(backend, groups, enc))
        self.assertEqual(len(batches), 2)
        self.assertEqual([b[0][0]["id"] for b in batches], ["a", "b"])
        self.assertEqual([b[1]["visual"].shape[1] for b in batches], [1, 2])
        self.assertEqual(batches[0][2].shape, (6, 1, 1))

    def test_protected_pusht_val_can_never_silently_become_development(self):
        rows = []
        for pool, count in (("train", 18685), ("val", 21)):
            for i in range(count):
                rows.append({"dataset": "pusht", "source_pool": pool, "index": i,
                    "length": 80, "trajectory_id": f"{pool}:{i}",
                    "lineage_group": f"{pool}:{i // 101 if pool == 'train' else i}",
                    "task": "pusht", "split": "fit" if pool == "train" else "external_reserve",
                    "source_revision": SOURCE_REVISION})
        config = {"data": {"seed": 234, "custom": {"split_ratio": .9, "frameskip": 5},
                           "validation": {"num_frames_val": 8, "val_dataset_batch_size": 4, "val_dataset_drop_last": False}},
                  "meta": {"dtype": "bfloat16", "data_traj_rollout_eval": {"data_traj_eval_rollout_steps": 6}}, "data_aug": {}}
        registry = {"manifest_sha256": "0" * 64, "trajectories": {
            f"val:{i}": {"use": "protected", "evidence": "historically accessed but reserved"} for i in range(21)}}
        cohort = build_cohort(rows, "pusht", config, registry, "0" * 64)
        validate_cohort(cohort)
        self.assertEqual(len(cohort["fit"]), 128)
        self.assertEqual(cohort["evaluation"], [])
        self.assertEqual(cohort["evaluation_permission"], "none_fit_only")
        cohort["evaluation"].append(cohort["protected_official_validation"][0])
        with self.assertRaisesRegex(ValueError, "Protected family"):
            validate_cohort(cohort)


if __name__ == "__main__":
    unittest.main()
