import copy
import unittest

import torch

from offline_study.fixed_response import ARMS, METHOD, FixedResponseIntervention
from offline_study.fixed_response_offline import PAIRS, ROLE, run_batch, validate_measurements, validate_protocol
from test_fixed_response import Backend, fixture_bank


class FixedResponseOfflineTests(unittest.TestCase):
    def test_freeze_rejects_subset_extra_arms_and_confirmation(self):
        protocol = {"schema_version": 1, "status": "frozen", "method": METHOD, "task": "reach",
            "evaluation_role": ROLE, "fresh_confirmation": False, "newly_opened_untouched_rows": 0,
            "shard_count": 4, "precisions": ["bfloat16", "float32"], "primary_precision": "bfloat16",
            "arms": [{"name": arm} for arm in ARMS],
            "primary_contrasts": [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in PAIRS],
            "offline_significance_required_for_behavioral_admission": False}
        validate_protocol(protocol)
        for key, value in (("arms", protocol["arms"][:-1]), ("fresh_confirmation", True),
                           ("task", "pusht"), ("newly_opened_untouched_rows", 1),
                           ("offline_significance_required_for_behavioral_admission", True)):
            with self.assertRaises(ValueError):
                validate_protocol({**protocol, key: value})

    def test_coverage_binds_lineage_not_only_row_count(self):
        meta = {"trajectory_id": "t", "start": 0, "lineage_group": "g", "task": "mw-reach", "prefix": 0}
        rows = [{**meta, "arm": arm, "metrics": {"proprio_mse_h6": 1.}} for arm in ARMS]
        validate_measurements(rows, [meta])
        for bad in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                validate_measurements(bad, [meta])
        changed = copy.deepcopy(rows)
        changed[1]["lineage_group"] = "another-family"
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_measurements(changed, [meta])
        changed = copy.deepcopy(rows)
        changed[1]["metrics"]["proprio_mse_h6"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_measurements(changed, [meta])

    def test_paired_batch_preserves_all_arms_and_reports_realized_energy(self):
        before = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            backend = Backend()
            backend.metrics = lambda prediction, target: [
                {"visual_mse_h6": float(value.square().mean())} for value in prediction["visual"][-1]]
            adapters = {arm: FixedResponseIntervention(backend, fixture_bank(), arm) for arm in ARMS}
            meta = [{"trajectory_id": "t", "lineage_group": "g", "task": "mw-reach", "start": 0}]
            rows = run_batch(backend, adapters, meta, torch.ones(1, 256, 400), torch.zeros(6, 1, 1), {})
            validate_measurements(rows, meta)
            self.assertEqual(backend.calls, 4)
            self.assertEqual(rows[0]["metrics"], rows[1]["metrics"])
            self.assertEqual(rows[0]["realized_l2"], 0.)
            self.assertAlmostEqual(rows[2]["requested_l2"], 2.)
            self.assertTrue(rows[2]["realized_energy_within_fp32_tolerance"])
        finally:
            torch.set_num_threads(before)


if __name__ == "__main__":
    unittest.main()
