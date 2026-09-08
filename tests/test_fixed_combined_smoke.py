import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from offline_study import fixed_combined_smoke as smoke
from offline_study.fixed_combined import ARM_COMPONENTS, METHOD
from offline_study.fixed_combined_check import COUNTS, HORIZONS, STATUS
from offline_study.protocol import sha256, write_json


def record(arm, horizon, count):
    active = arm in ARM_COMPONENTS and horizon == 6
    value = {"backend_calls": 1, "response_probe_rollouts": 0, "full_native_shadow_rollouts": 0,
        "native_prefix_replays": int(active), "extra_native_predictor_blocks": 4 * int(active),
        "main_predictor_blocks": horizon * 6, "rank_applications": int(active),
        "horizon": horizon, "candidates": count, "coupling": []}
    if active:
        value.update(coefficients=[[0.] * 4 for _ in range(count)], active=[False] * count,
            requested_l2=[0.] * count, realized_l2=[0.] * count,
            coupling=[{"site": site, "block": block, "horizon": 3, "applications": 1}
                      for site, block in (("predictor_visual", None), ("block_condition", 3))])
    return value


class CombinedSmokeTests(unittest.TestCase):
    def test_fixed_four_episode_order_native_repeat_before_edits(self):
        self.assertEqual(smoke.EPISODES, ("native", "native_repeat", "combined_fixed_rank4",
                                        "matched_random_combined_fixed_rank4"))
        self.assertEqual(smoke.MAX_SECONDS, 3600)

    def test_exact_work_short_horizons_native_and_full_combination(self):
        for arm in ("native", *ARM_COMPONENTS):
            for h in range(1, 7):
                for n in (1, 8, 19, 300):
                    smoke.validate_record(record(arm, h, n), arm, h, n)

    def test_reject_extra_rollout_prefix_or_misplaced_coupling(self):
        arm = next(iter(ARM_COMPONENTS))
        for key in ("backend_calls", "response_probe_rollouts", "full_native_shadow_rollouts",
                    "extra_native_predictor_blocks", "native_prefix_replays", "rank_applications"):
            value = record(arm, 6, 8); value[key] += 1
            with self.assertRaises(ValueError): smoke.validate_record(value, arm, 6, 8)
        for transform in (lambda v: v["coupling"].pop(),
                          lambda v: v["coupling"][0].update(horizon=2),
                          lambda v: v.update(coefficients=[[float("nan")] * 4] * 8),
                          lambda v: v.update(active=[True])):
            value = record(arm, 6, 8); transform(value)
            with self.assertRaises(ValueError): smoke.validate_record(value, arm, 6, 8)

    def test_native_observer_single_call_and_persistent_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = SimpleNamespace(calls=0, predictor=torch.nn.Linear(1, 1))
            def predict(context, actions, **kwargs):
                backend.calls += 1
                return {k: torch.zeros(len(actions) + 1, actions.shape[1], 1) for k in ("visual", "proprio")}
            backend.predict = predict
            observed = smoke.ObservedCombined(backend, "native", Path(directory))
            with patch.object(torch.cuda, "synchronize"):
                observed({}, torch.zeros(6, 300, 20))
                observed({}, torch.zeros(2, 1, 20))
            self.assertEqual(backend.calls, 2)
            self.assertEqual([r["horizon"] for r in observed.calls], [6, 2])
            self.assertEqual(observed.calls[0]["record"]["extra_native_predictor_blocks"], 0)

    def fixture(self, root):
        check, source = root / "check", root / "source"
        check.mkdir(); source.mkdir()
        fit = root / "fit.json"; write_json(fit, {"fixture": True})
        path = source / "fixed_combined.py"
        shutil.copyfile(Path(smoke.__file__).parent / path.name, path)
        digest = hashlib.sha256(path.name.encode() + b"\0" + path.read_bytes()).hexdigest()
        contract = {"method": METHOD, "task": "mw-reach", "arms": list(ARM_COMPONENTS),
            "candidate_counts": list(COUNTS), "horizons": list(HORIZONS), "planning_context": 2,
            "precision": "strict_float32_no_autocast_no_tf32", "timing_repeats": 3,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False,
            "source_sha256": digest, "bound_files": {str(fit): sha256(fit)}}
        write_json(check / "contract.json", contract)
        for name in ("model-before.json", "model-after.json"):
            write_json(check / name, {"fixture": True})
        keys = [(a, h, n) for a in ARM_COMPONENTS for h in HORIZONS for n in COUNTS]
        report = {"status": STATUS, "contract_sha256": sha256(check / "contract.json"),
            "parameter_buffer_bytes_modes_unchanged": True, "hooks_clean": True,
            "source_and_bound_inputs_unchanged": True, "scientific_efficacy_measurement": False,
            "fresh_confirmation": False, "development_or_protected_outcomes_accessed": False,
            "device_uuid": "device", "backend": {"checkpoint_sha256": "checkpoint"},
            "model_before_sha256": sha256(check / "model-before.json"),
            "model_after_sha256": sha256(check / "model-after.json"),
            "checks": [{"arm": a, "horizon": h, "candidate_count": n,
                "all_output_horizons_byte_equal": True, "input_shape_stride_bytes_unchanged": True,
                "cpu_cuda_rng_unchanged": True, "adapter_record": record(a, h, n)} for a, h, n in keys],
            "timings": [{"arm": a, "horizon": h, "candidate_count": n, "seconds": [.1, .1, .1],
                         "warmup_forecasts": 1} for a, h, n in keys]}
        return check, source, contract["bound_files"], report

    def save(self, check, report):
        write_json(check / "report.json", report)
        digest = sha256(check / "report.json")
        write_json(check / "DONE.json", {"report_sha256": digest})
        return digest

    def test_receipt_complete_grid_fit_source_model_and_device_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            check, source, files, report = self.fixture(Path(directory))
            digest = self.save(check, report)
            proof = smoke.verify_numerical(check, digest, source, files, "checkpoint", "device")
            self.assertEqual(proof["report_sha256"], digest)
            for cp, device in (("other", "device"), ("checkpoint", "other")):
                with self.assertRaises(ValueError): smoke.verify_numerical(check, digest, source, files, cp, device)
            with self.assertRaises(ValueError):
                smoke.verify_numerical(check, digest, source, {"different": "hash"}, "checkpoint", "device")
            write_json(check / "model-after.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "model byte"):
                smoke.verify_numerical(check, digest, source, files, "checkpoint", "device")

    def test_missing_duplicate_or_nonexact_case_cannot_pass(self):
        for change in (lambda r: r["checks"].pop(), lambda r: r["checks"].append(r["checks"][0]),
                       lambda r: r["checks"][0].update(cpu_cuda_rng_unchanged=False),
                       lambda r: r["timings"][0].update(seconds=[.1]),
                       lambda r: r.update(parameter_buffer_bytes_modes_unchanged=False)):
            with tempfile.TemporaryDirectory() as directory:
                check, source, files, report = self.fixture(Path(directory))
                change(report); digest = self.save(check, report)
                with self.assertRaises(ValueError):
                    smoke.verify_numerical(check, digest, source, files, "checkpoint", "device")

    def test_stale_receipt_failed_check_and_source_edit_fail(self):
        for change in (lambda c, s: write_json(c / "FAILED.json", {}),
                       lambda c, s: write_json(c / "DONE.json", {"report_sha256": "wrong"}),
                       lambda c, s: (s / "fixed_combined.py").write_text("changed")):
            with tempfile.TemporaryDirectory() as directory:
                check, source, files, report = self.fixture(Path(directory))
                digest = self.save(check, report); change(check, source)
                with self.assertRaises(ValueError):
                    smoke.verify_numerical(check, digest, source, files, "checkpoint", "device")


if __name__ == "__main__":
    unittest.main()
