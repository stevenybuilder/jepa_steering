import unittest
import tempfile
from pathlib import Path

import torch

from offline_study.fixed_response import (ARMS, METHOD, ZERO_THRESHOLD,
    FixedResponseHook, FixedResponseIntervention, compose_map, validate_bank)
from offline_study.fixed_response import load_fitted_bank
from offline_study.fixed_response_check import CountedBackend, reference_fields
from offline_study.interventions import PredictorIntervention
from offline_study.support_operator import NativeFieldCapture
from offline_study.fixed_response_fit import check_examples, collect_responses


def fixture_bank():
    basis = torch.zeros(4, 256 * 400)
    basis[torch.arange(4), torch.arange(4)] = 1
    projection = torch.zeros(3, 256 * 400)
    projection[torch.arange(3), torch.arange(3)] = 1
    return {"schema_version": 1, "method": METHOD, "binding": {}, "dose": 2.,
        "zero_threshold": ZERO_THRESHOLD, "mean": torch.zeros(256 * 400),
        "projection": projection, "scale": torch.ones(3),
        "operators": {arm: {"basis": basis.reshape(4, 256, 400).clone(), "map": torch.eye(4)}
                      for arm in ARMS[2:]}}


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    def forward(self, value, actions=None, proprio=None):
        for block in self.predictor_blocks:
            value = block(value)
        return value


class Backend:
    def __init__(self):
        self.predictor, self.device, self.calls = Predictor(), torch.device("cpu"), 0
        self.allow_tf32 = False

    def predict(self, context, actions):
        self.calls += 1
        value = context
        values = [value]
        for _ in actions:
            value = self.predictor(value, None, None)
            values.append(value)
        return {"visual": torch.stack(values), "proprio": torch.stack(values)[..., :1]}

    def expand_context(self, context, repeats):
        return context.repeat_interleave(repeats, 0)


class FixedResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_composed_map_matches_explicit_small_inverse(self):
        generator = torch.Generator().manual_seed(44)
        response = torch.randn(4, 19, generator=generator)
        weight = torch.randn(4, 19, generator=generator)
        phi = torch.randn(7, 4, generator=generator)
        gram = response.double() @ response.double().T
        rhs = response.double() @ (phi.double() @ weight.double()).T
        expected = torch.linalg.solve(gram + .01 * gram.diagonal().mean() * torch.eye(4), rhs).T
        actual = phi @ compose_map(response, weight).T
        self.assertTrue(torch.allclose(actual.double(), expected, atol=1e-6, rtol=1e-5))

    def test_zero_response_and_invalid_input(self):
        self.assertTrue(torch.equal(compose_map(torch.zeros(4, 9), torch.ones(4, 9)), torch.zeros(4, 4)))
        for response in (torch.zeros(3, 9), torch.full((4, 9), float("nan"))):
            with self.assertRaises(ValueError):
                compose_map(response, torch.ones(4, 9))

    def test_one_forecast_no_probes_edit_only_h3_newest_field(self):
        backend, bank = Backend(), fixture_bank()
        original = torch.ones(2, 260, 400)
        original[1] *= 3
        adapter = FixedResponseIntervention(backend, bank, "fixed_rank4")
        output = adapter(original, torch.zeros(6, 2, 1))["visual"]
        self.assertEqual(backend.calls, 1)
        self.assertTrue(torch.equal(output[:3], original.unsqueeze(0).expand(3, -1, -1, -1)))
        self.assertTrue(torch.equal(output[:, :, :4], original[:, :4].unsqueeze(0).expand(7, -1, -1, -1)))
        self.assertTrue(torch.allclose((output[3] - original).flatten(1).norm(dim=1), torch.tensor([2., 2.])))
        self.assertFalse(torch.equal(adapter.last_record["coefficients"][0], adapter.last_record["coefficients"][1]))
        self.assertEqual(adapter.last_record["response_probe_rollouts"], 0)
        self.assertEqual(adapter.last_record["native_shadow_rollouts"], 0)
        self.assertFalse(backend.predictor._forward_pre_hooks)
        self.assertFalse(backend.predictor.predictor_blocks[3]._forward_hooks)

    def test_short_native_and_zero_are_exact(self):
        for arm in ARMS:
            for horizon in (1, 2, 5, 6):
                if horizon == 6 and arm in ARMS[2:]:
                    continue
                backend = Backend()
                adapter = FixedResponseIntervention(backend, fixture_bank(), arm)
                context = torch.randn(2, 256, 400).bfloat16()
                result = adapter(context, torch.zeros(horizon, 2, 1))
                self.assertEqual(backend.calls, 1)
                self.assertTrue(torch.equal(result["visual"][-1], context))

    def test_zero_rule_independent_of_other_arm_and_per_candidate(self):
        bank, backend = fixture_bank(), Backend()
        bank["operators"]["fixed_rank4"]["map"][:, 0] = 0
        bank["operators"]["matched_random_fixed_rank4"]["map"].zero_()
        context = torch.zeros(2, 256, 400)
        context[1] = 1
        adapter = FixedResponseIntervention(backend, bank, "fixed_rank4")
        output = adapter(context, torch.zeros(6, 2, 1))["visual"][-1]
        self.assertTrue(torch.equal(output[0], context[0]))
        self.assertFalse(torch.equal(output[1], context[1]))
        self.assertEqual(adapter.last_record["active"].tolist(), [False, True])

    def test_rejects_combined_hooks_and_cleans_up_failed_forecast(self):
        backend, bank = Backend(), fixture_bank()
        handle = backend.predictor.predictor_blocks[0].register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "coexist"):
            FixedResponseIntervention(backend, bank, "fixed_rank4")(torch.zeros(1, 256, 400), torch.zeros(6, 1, 1))
        handle.remove()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            with FixedResponseHook(backend.predictor, bank, "fixed_rank4"):
                backend.predict(torch.zeros(1, 256, 400), torch.zeros(2, 1, 1))
        self.assertFalse(backend.predictor._forward_pre_hooks)
        self.assertFalse(backend.predictor.predictor_blocks[3]._forward_hooks)

    def test_rejects_invalid_basis_and_unknown_arm(self):
        bank = fixture_bank()
        bank["operators"]["fixed_rank4"]["basis"][1] = bank["operators"]["fixed_rank4"]["basis"][0]
        with self.assertRaises(ValueError):
            validate_bank(bank)
        with self.assertRaises(ValueError):
            FixedResponseIntervention(Backend(), fixture_bank(), "rank8")

    def test_fitting_probes_are_only_four_directions_per_control(self):
        backend, bank = Backend(), fixture_bank()
        fitted = {"bases": {key: {"blocks": [3], "positions": list(range(256)),
            "basis": bank["operators"][arm]["basis"][:, None]}
            for key, arm in (("rank", "fixed_rank4"), ("rank_random", "matched_random_fixed_rank4"))}}
        response = collect_responses(backend, torch.zeros(2, 256, 400), torch.zeros(6, 2, 1), fitted, 2.)
        self.assertEqual(backend.calls, 4)  # 16 probes, chunks of four; offline only.
        self.assertTrue(torch.equal(response["rank"], fitted["bases"]["rank"]["basis"].flatten(1)[None].expand(2, -1, -1)))

    def test_nonfinite_runtime_does_not_leave_hooks_attached(self):
        backend = Backend()
        adapter = FixedResponseIntervention(backend, fixture_bank(), "fixed_rank4")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            adapter(torch.full((1, 256, 400), float("nan")), torch.zeros(6, 1, 1))
        self.assertFalse(backend.predictor._forward_pre_hooks)
        self.assertFalse(backend.predictor.predictor_blocks[3]._forward_hooks)

    def test_bfloat16_reports_realized_not_just_requested_dose(self):
        adapter = FixedResponseIntervention(Backend(), fixture_bank(), "fixed_rank4")
        adapter(torch.full((1, 256, 400), 10000., dtype=torch.bfloat16), torch.zeros(6, 1, 1))
        self.assertAlmostEqual(float(adapter.last_record["requested_l2"][0]), 2., places=5)
        self.assertEqual(float(adapter.last_record["realized_l2"][0]), 0.)

    def test_fit_coverage_rejects_unequal_or_duplicated_examples(self):
        from offline_study.author_runtime import examples
        config = {"data": {"validation": {"num_frames_val": 8}, "custom": {"frameskip": 1}}}
        rows = [{"trajectory_id": "a", "lineage_group": "family", "task": "mw-reach", "length": 12}]
        meta = examples(rows, config, fitting=True)
        check_examples(meta, rows, config)
        for bad in (meta[:-1], meta + meta[:1]):
            with self.assertRaises(ValueError):
                check_examples(bad, rows, config)

    def test_completed_fit_reload_checks_hashes_task_checkpoint_and_failure(self):
        from offline_study.protocol import sha256, write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "cohort.json", {"fixture_only": True})
            bank = fixture_bank()
            bank["binding"] = {"method": METHOD, "task": "mw-reach", "checkpoint_sha256": "fixture-checkpoint",
                               "cohort_sha256": sha256(root / "cohort.json")}
            torch.save(bank, root / "operator_bank.pt")
            torch.save({}, root / "mean_responses.pt")
            write_json(root / "contract.json", {"binding": bank["binding"]})
            write_json(root / "report.json", {"binding": bank["binding"], "development_or_protected_outcomes_accessed": False})
            write_json(root / "PARITY.json", {})
            write_json(root / "audit_metrics.json", [])
            write_json(root / "DONE.json", {"status": "fit_only_complete_not_behavioral_clearance", **{
                p.name: sha256(p) for p in root.iterdir()}})
            loaded = load_fitted_bank(root, task="mw-reach", checkpoint_sha256="fixture-checkpoint")
            self.assertEqual(loaded["method"], METHOD)
            for task, checkpoint in (("mw-reach-wall", "fixture-checkpoint"), ("mw-reach", "wrong-checkpoint")):
                with self.assertRaisesRegex(ValueError, "binding"):
                    load_fitted_bank(root, task=task, checkpoint_sha256=checkpoint)
            write_json(root / "FAILED.json", {"fixture": "interrupted"})
            with self.assertRaisesRegex(ValueError, "failed"):
                load_fitted_bank(root, task="mw-reach", checkpoint_sha256="fixture-checkpoint")
            (root / "FAILED.json").unlink()
            write_json(root / "audit_metrics.json", {"tampered": True})
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_fitted_bank(root, task="mw-reach", checkpoint_sha256="fixture-checkpoint")

    def test_independent_planning_reference_and_backend_counter(self):
        backend, bank = Backend(), fixture_bank()
        context = torch.randn(3, 260, 400)
        actions = torch.zeros(6, 3, 1)
        counted = CountedBackend(backend)
        for arm in ARMS[2:]:
            adapter = FixedResponseIntervention(counted, bank, arm)
            with NativeFieldCapture(backend.predictor) as capture:
                backend.predict(context, actions)
            fields = reference_fields(capture.values[3], adapter.bank, arm)
            with PredictorIntervention(backend.predictor, fields):
                expected = backend.predict(context, actions)
            before = counted.calls
            actual = adapter(context, actions)
            self.assertEqual(counted.calls - before, 1)
            self.assertTrue(torch.equal(actual["visual"], expected["visual"]))


if __name__ == "__main__":
    unittest.main()
