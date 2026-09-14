"""CPU fixtures for the receiving checker; no GPU, data download, or behavior."""
import copy
import json
import signal
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from offline_study.validation import fixed_combined_check as check
from offline_study.interventions.fixed_combined import ARM_COMPONENTS, CombinedFixedResponseIntervention
from offline_study.validation.fixed_response_check import CountedBackend
from offline_study.core.protocol import sha256
from tests.integration.interventions.test_fixed_combined import Backend, Context, coupling_fixture, inputs
from tests.unit.interventions.test_fixed_response import fixture_bank


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


class CheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def adapter(self, arm="combined_fixed_rank4", bank=None):
        backend, bank = Backend(), bank or fixture_bank()
        protocol, coupling = coupling_fixture()
        counted = CountedBackend(backend)
        adapter = CombinedFixedResponseIntervention(counted, bank, protocol, coupling, arm)
        return backend, bank, protocol, coupling, adapter

    def assert_clean(self, backend):
        for module in backend.predictor.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_oracle_and_combined_both_arms_actual_counts_and_candidate_records(self):
        for arm in ARM_COMPONENTS:
            with self.subTest(arm=arm):
                backend, bank, protocol, coupling, adapter = self.adapter(arm)
                context, actions = inputs()
                before = check.model_signature(backend.model)
                record = check.equivalence_case(backend, adapter, protocol, coupling, context, actions)
                self.assertEqual(record["reference_backend_calls"], 2)
                self.assertEqual(record["reference_predictor_blocks"], 72)
                self.assertEqual(backend.calls, 3)
                self.assertEqual(backend.predictor.trace[:72], list(range(6)) * 12)
                self.assertEqual(backend.predictor.trace[72:], list(range(6)) * 2 + list(range(4)) + list(range(6)) * 4)
                self.assertEqual(len(record["adapter_record"]["coefficients"]), 2)
                self.assertEqual(len(record["adapter_record"]["coefficients"][0]), 4)
                self.assertEqual(check.model_signature(backend.model), before)
                self.assertTrue(record["cpu_cuda_rng_unchanged"])
                self.assert_clean(backend)

    def test_all_short_horizons_and_singleton_context_are_native(self):
        for horizon in range(1, 6):
            backend, _, protocol, coupling, adapter = self.adapter()
            context, actions = inputs(context_batch=1, candidates=2, horizon=horizon)
            result = check.equivalence_case(backend, adapter, protocol, coupling, context, actions)
            self.assertEqual(result["reference_predictor_blocks"], horizon * 6)
            self.assertEqual(result["reference_backend_calls"], 1)
            self.assertEqual(result["adapter_record"]["native_prefix_replays"], 0)
            self.assertEqual(result["adapter_record"]["coupling"], [])
            self.assert_clean(backend)

    def test_oracle_autocast_disabled_and_inactive_signed_zero_preserved(self):
        backend, bank, protocol, coupling, adapter = self.adapter()
        bank["operators"]["fixed_rank4"]["map"].zero_()
        backend, _, protocol, coupling, adapter = self.adapter(bank=bank)
        context, actions = inputs()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            record = check.equivalence_case(backend, adapter, protocol, coupling, context, actions)
        self.assertEqual(record["adapter_record"]["active"], [False, False])
        # Explicit -0 regression; torch.equal alone cannot establish this.
        target = torch.tensor([[[-0., -0.]], [[1., 2.]]])
        hook = check.BytePreservingReference(backend.predictor, [], torch.tensor([False, True]))
        changed = hook._replace(target, torch.zeros_like(target), "block_output/3")
        check.assert_bytes(changed[0], target[0], "inactive signed zero")
        self.assertTrue(torch.signbit(changed[0]).all())

    def test_byte_comparator_rejects_signed_zero_dtype_and_shape(self):
        self.assertTrue(torch.equal(torch.tensor([0.]), torch.tensor([-0.])))
        for actual, expected in ((torch.tensor([0.]), torch.tensor([-0.])),
                                 (torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64)),
                                 (torch.tensor([1.]), torch.tensor([[1.]]))):
            with self.assertRaisesRegex(ValueError, "Byte parity"):
                check.assert_bytes(actual, expected, "fixture")
        check.assert_bytes(torch.tensor([True]), torch.tensor([True]), "bool")

    def test_input_model_rng_signatures_detect_nonvalue_and_value_changes(self):
        context, actions = inputs()
        before = check.input_signature(context, actions)
        cloned = {k: v.contiguous() for k, v in context.items()}
        self.assertNotEqual(before, check.input_signature(cloned, actions))
        actions[0, 0, 0].add_(1)
        self.assertNotEqual(before, check.input_signature(context, actions))
        model = Backend().model
        model.register_buffer("sentinel", torch.tensor([-0.]))
        signature = check.model_signature(model)
        model.sentinel.zero_()
        self.assertNotEqual(signature, check.model_signature(model))
        signature = check.model_signature(model)
        model.train()
        self.assertNotEqual(signature, check.model_signature(model))
        rng = torch.get_rng_state()
        try:
            before_rng = check.rng_signature(torch.device("cpu"))
            torch.rand(1)
            self.assertNotEqual(before_rng, check.rng_signature(torch.device("cpu")))
        finally:
            torch.set_rng_state(rng)

    def test_foreign_failure_cleans_independent_instrumentation(self):
        class Foreign(BaseException):
            pass
        backend, bank, protocol, coupling, _ = self.adapter()
        context, actions = inputs()
        backend.predictor.predictor_blocks[2].fail_on_visit = 1
        backend.predictor.predictor_blocks[2].failure = Foreign("preserve me")
        with self.assertRaisesRegex(Foreign, "preserve me"):
            check.independent_reference(backend, bank, protocol, coupling,
                                        "combined_fixed_rank4", context, actions)
        self.assert_clean(backend)

    def test_accounting_or_candidate_record_drift_rejected(self):
        backend, bank, protocol, coupling, adapter = self.adapter()
        context, actions = inputs()
        _, reference = check.independent_reference(backend, adapter.bank, protocol, coupling,
                                                   adapter.arm, context, actions)
        adapter(context, actions)
        for key in ("backend_calls", "native_prefix_replays", "response_probe_rollouts",
                    "extra_native_predictor_blocks", "main_predictor_blocks"):
            corrupt = copy.deepcopy(adapter.last_record)
            corrupt[key] += 1
            with self.assertRaises(ValueError):
                check.compare_records(corrupt, reference, 6, 2)
        corrupt = copy.deepcopy(adapter.last_record)
        corrupt["coefficients"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "coefficients"):
            check.compare_records(corrupt, reference, 6, 2)

    def test_three_repeats_and_warmup_preserve_strides_rng(self):
        backend, _, _, _, adapter = self.adapter()
        context, actions = inputs(horizon=2)
        result = check.measure(lambda: adapter(context, actions), backend.device, context, actions)
        self.assertEqual(backend.calls, 4)
        self.assertEqual(len(result["seconds"]), 3)
        self.assertEqual(result["warmup_forecasts"], 1)
        self.assertTrue(all(s > 0 for s in result["seconds"]))
        self.assertFalse(result["cuda_synchronized"])
        self.assert_clean(backend)

    def test_coverage_rejects_missing_duplicate_repeats_and_extra_cases(self):
        rows = [{"arm": a, "horizon": h, "candidate_count": n, "seconds": [1., 1., 1.],
                 "warmup_forecasts": 1} for a in ARM_COMPONENTS for h in check.HORIZONS for n in check.COUNTS]
        self.assertEqual(len(rows), 18)
        check.validate_coverage(rows, rows)
        for bad in (rows[:-1], rows + [rows[0]], [{**r, "seconds": [1.]} for r in rows],
                    [{**r, "seconds": [1., float("nan"), 1.]} for r in rows],
                    [{**r, "candidate_count": 96} for r in rows]):
            with self.assertRaises(ValueError):
                check.validate_coverage(rows, bad)

    def test_hook_audit_includes_foreign_global_hooks_without_removing_them(self):
        from torch.nn.modules import module as global_state
        backend = Backend()
        self.assertTrue(check.hooks_clean(backend.predictor))
        handle = global_state.register_module_forward_hook(lambda module, args, output: None)
        try:
            self.assertFalse(check.hooks_clean(backend.predictor))
            self.assertTrue(global_state._global_forward_hooks)
        finally:
            handle.remove()
        self.assertTrue(check.hooks_clean(backend.predictor))

    def test_immutable_receipts_and_bound_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "DONE.json"
            check.exclusive_json(path, {"value": 1})
            bound = {str(path): sha256(path)}
            check.verify_files(bound)
            with self.assertRaises(FileExistsError):
                check.exclusive_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            path.write_text("changed")
            with self.assertRaises(ValueError):
                check.verify_files(bound)

    def test_bounded_timeout_restores_timer_handler(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, "cap reached"):
            with check.deadline(.005):
                signal.pause()
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_only_normalized_fixed_shape_fit_payload_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            value = {"observations": {"visual": torch.zeros(1, 18, 3, 224, 224),
                                      "proprio": torch.zeros(1, 18, 4)},
                     "actions": torch.zeros(1, 18, 20)}
            torch.save(value, path / "stimulus.pt")
            self.assertEqual(check.load_stimulus(path)["actions"].shape, (1, 18, 20))
            value["actions"][0, 0, 0] = float("nan")
            torch.save(value, path / "stimulus.pt")
            with self.assertRaisesRegex(ValueError, "Invalid normalized"):
                check.load_stimulus(path)
            value["development_outcomes"] = []
            torch.save(value, path / "stimulus.pt")
            with self.assertRaisesRegex(ValueError, "only normalized"):
                check.load_stimulus(path)


class BindingTests(unittest.TestCase):
    def make_fit(self, root):
        fit, parent = root / "fixed", root / "original"
        coupling = parent / "vision_action_coupling"
        fit.mkdir()
        coupling.mkdir(parents=True)
        rows = [{"task": "mw-reach", "dataset": "metaworld", "split": "fit", "source_pool": "all",
                 "index": 10587 + i, "length": 99, "trajectory_id": f"metaworld:all:{10587+i}",
                 "lineage_group": f"family-{i}"} for i in range(128)]
        cohort = {"task": "mw-reach", "dataset": "metaworld", "fit": rows, "evaluation": [],
            "all_protected_groups": [], "all_author_validation_groups": [], "protected_official_validation": [],
            "coverage": {"evaluated_rows": 0, "official_rows": 0},
            "reference_config": {"data": {"custom": {"frameskip": 5}, "validation": {"num_frames_val": 18}}}}
        for d in (fit, parent):
            dump(d / "cohort.json", cohort)
        protocol, bank = coupling_fixture()
        protocol.update(tasks=["mw-reach"], evaluation_split="development")
        dump(coupling / "protocol.json", protocol)
        bank["protocol_sha256"] = sha256(coupling / "protocol.json")
        torch.save(bank, coupling / "operator_bank.pt")
        receipt = {"status": "author_train_fit_only_complete", "task": "mw-reach", "fit_split": "fit",
            "fit_lineage_group_count": 128, "fit_lineage_groups": [r["lineage_group"] for r in rows],
            "fit_window_count": 512,
            "fit_trajectory_ids": [r["trajectory_id"] for r in rows], "development_outcomes_accessed": False,
            "holdout_outcomes_accessed": False, "model_parameters_and_buffers_unchanged": True}
        dump(coupling / "fit_receipt.json", receipt)
        for path in (fit / "DONE.json", fit / "operator_bank.pt", coupling / "DONE.json", parent / "PARITY.json"):
            path.write_text("{}")
        fixed = fixture_bank()
        fixed["binding"] = {"fit_precision": "bfloat16"}
        return fit, coupling, fixed, cohort, receipt, protocol

    def test_original_same128_family_fit_accepted_and_existing_verifier_invoked(self):
        with tempfile.TemporaryDirectory() as temp:
            fit, coupling, fixed, cohort, receipt, protocol = self.make_fit(Path(temp))
            with patch.object(check, "load_fitted_bank", return_value=fixed), patch.object(
                    check, "verify_fit", return_value=(receipt, protocol)) as original:
                _, _, _, _, selected, bound = check.verify_bindings(fit, coupling, "mw-reach", "a" * 64)
            original.assert_called_once_with(coupling, sha256(fit / "cohort.json"), "a" * 64, "bfloat16")
            self.assertEqual(selected, cohort["fit"][0])
            check.verify_files(bound)

    def test_failed_wrong_population_exposure_protocol_and_precision_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fit, coupling, fixed, _, receipt, protocol = self.make_fit(Path(temp))
            with patch.object(check, "load_fitted_bank", return_value=fixed), patch.object(
                    check, "verify_fit", return_value=(receipt, protocol)):
                for key, corrupt in (("fit_lineage_group_count", 127), ("development_outcomes_accessed", True),
                                     ("fit_window_count", 511),
                                     ("fit_trajectory_ids", []), ("model_parameters_and_buffers_unchanged", False)):
                    previous, receipt[key] = receipt[key], corrupt
                    with self.assertRaises(ValueError):
                        check.verify_bindings(fit, coupling, "mw-reach", "a" * 64)
                    receipt[key] = previous
                fixed["binding"]["fit_precision"] = "float32"
                with self.assertRaises(ValueError):
                    check.verify_bindings(fit, coupling, "mw-reach", "a" * 64)
                fixed["binding"]["fit_precision"] = "bfloat16"
                (coupling / "FAILED.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "failed fitting"):
                    check.verify_bindings(fit, coupling, "mw-reach", "a" * 64)
            with self.assertRaisesRegex(ValueError, "frozen to Reach"):
                check.verify_bindings(fit, coupling, "mw-reach-wall", "a" * 64)

    def make_stimulus(self, root):
        directory, data_root = root / "stimulus", root / "raw"
        directory.mkdir()
        data_root.mkdir()
        selected = {"index": 10587, "trajectory_id": "metaworld:all:10587", "task": "mw-reach"}
        names = [f"train-{i:05d}-of-00126.parquet" for i in range(126)]
        raw = data_root / names[105]
        raw.write_bytes(b"fixture selected original parquet")
        manifest = {name: {"bytes": raw.stat().st_size, "sha256": sha256(raw)} for name in names}
        for name in ("stimulus.pt", "normalization.pt", "official_manifest.json", "raw_input_report.json"):
            (directory / name).write_text("fixture bytes only; must not deserialize during verification")
        dump(directory / "normalization_source.json", {
            "normalization_sha256": sha256(directory / "normalization.pt"),
            "raw_manifest_sha256": check.MANIFEST_HASHES["metaworld"],
            "published_stats_source_sha256": check.STATS_SOURCE_SHA256,
            "native_normalization_reused": True, "normalization_recomputed": False})
        dump(directory / "source_mapping.json", {"raw_manifest_sha256": check.MANIFEST_HASHES["metaworld"],
            "parquet_rows": dict.fromkeys(names, 100), "selected_fit_row": selected,
            "selected_source": {"path": names[105], "local_row": 87}})
        receipt = {"schema_version": 1, "status": "frozen_normalized_fit_stimulus", "task": "mw-reach",
            "checkpoint_sha256": "a" * 64, "cohort_sha256": "b" * 64, "selected_fit_row": selected,
            "normalized_fit_payload_only": True, "new_full_pool_state_scan": False,
            "development_or_protected_outcomes_accessed": False,
            "normalization_scope": check.NORMALIZATION_SCOPE,
            "clip_start": 0, "frames": 18, "frameskip": 5,
            "files": {p.name: sha256(p) for p in directory.iterdir()}}
        dump(directory / "STIMULUS.json", receipt)
        return directory, data_root, selected, manifest, receipt

    def test_stimulus_hashes_mapping_and_reused_normalization_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, data, selected, manifest, receipt = self.make_stimulus(Path(temp))
            with patch.object(check, "bound_manifest", return_value=manifest) as original, patch.object(
                    check, "RAW_PROOF_SHA256", receipt["files"]["raw_input_report.json"]), patch.object(torch, "load") as load:
                actual, bound = check.verify_stimulus(directory, sha256(directory / "STIMULUS.json"),
                    data, "b" * 64, selected, "a" * 64)
                self.assertEqual(actual, receipt)
                check.verify_files(bound)
                load.assert_not_called()
                original.assert_called_once_with("metaworld", directory / "official_manifest.json")
                (data / "train-00105-of-00126.parquet").write_text("changed")
                with self.assertRaisesRegex(ValueError, "source parquet"):
                    check.verify_stimulus(directory, sha256(directory / "STIMULUS.json"), data,
                        "b" * 64, selected, "a" * 64)

    def test_stimulus_rejects_new_scan_wrong_cohort_renumbered_subset_and_unknown_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, data, selected, manifest, receipt = self.make_stimulus(Path(temp))
            with patch.object(check, "bound_manifest", return_value=manifest), patch.object(
                    check, "RAW_PROOF_SHA256", receipt["files"]["raw_input_report.json"]):
                for key, bad in (("new_full_pool_state_scan", True), ("cohort_sha256", "c" * 64),
                                 ("development_or_protected_outcomes_accessed", True)):
                    changed = {**receipt, key: bad}
                    dump(directory / "STIMULUS.json", changed)
                    with self.assertRaisesRegex(ValueError, "contract differs"):
                        check.verify_stimulus(directory, sha256(directory / "STIMULUS.json"), data,
                            "b" * 64, selected, "a" * 64)
                mapping = json.loads((directory / "source_mapping.json").read_text())
                mapping["selected_source"]["local_row"] = 0
                dump(directory / "source_mapping.json", mapping)
                receipt["files"]["source_mapping.json"] = sha256(directory / "source_mapping.json")
                dump(directory / "STIMULUS.json", receipt)
                with self.assertRaisesRegex(ValueError, "source parquet"):
                    check.verify_stimulus(directory, sha256(directory / "STIMULUS.json"), data,
                        "b" * 64, selected, "a" * 64)
            with patch.object(check, "bound_manifest", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "complete raw-input proof"):
                    check.verify_stimulus(directory, sha256(directory / "STIMULUS.json"), data,
                        "b" * 64, selected, "a" * 64)

    def test_main_failure_freezes_no_fake_done_and_does_not_open_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp), Path(temp) / "output"
            argv = ["--task", "mw-reach", "--checkpoint-sha256", "a" * 64, "--stimulus-sha256", "b" * 64]
            for name in ("fit", "coupling-fit", "vendor", "checkpoint", "data-root", "stimulus"):
                argv.extend(["--" + name, str(root / name)])
            argv.extend(["--output", str(output)])
            with patch.object(check, "verify_bindings", side_effect=ValueError("unbound fit")):
                with self.assertRaisesRegex(ValueError, "unbound fit"):
                    check.main(argv)
            self.assertTrue((output / "FAILED.json").is_file())
            self.assertFalse((output / "DONE.json").exists())
            original = (output / "FAILED.json").read_bytes()
            with self.assertRaises(FileExistsError):
                check.main(argv)
            self.assertEqual((output / "FAILED.json").read_bytes(), original)

    def test_positive_cli_lifecycle_freezes_contract_before_payload_and_binds_done(self):
        # Small synthetic registry exercises CLI publication/integrity order;
        # separate tests assert the immutable production 18-case registry.
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp), Path(temp) / "output"
            fit, coupling_fit, bank, cohort, receipt, protocol = self.make_fit(root)
            coupling = torch.load(coupling_fit / "operator_bank.pt", weights_only=True)
            checkpoint = root / "checkpoint"
            checkpoint.write_text("synthetic checkpoint bytes")
            backend = Backend()
            backend.model.ctxt_window = 2
            backend.provenance = {"synthetic": True}
            context, _ = inputs(context_batch=1, candidates=2)
            backend.model.model = types.SimpleNamespace(encode=lambda observations, actions:
                (context["visual"].expand(1, 2, 1, 16, 16, 400), context["proprio"], None))
            payload = {"observations": {"visual": torch.zeros(1), "proprio": torch.zeros(1)},
                       "actions": torch.arange(18 * 400).remainder(17).reshape(1, 18, 400).float() / 103}
            def load(directory):
                self.assertTrue((output / "contract.json").is_file())
                self.assertFalse((output / "DONE.json").exists())
                return payload
            timing = {"seconds": [1., 1., 1.], "mean_seconds": 1., "warmup_forecasts": 1,
                      "repeats": 3, "cuda_synchronized": False, "peak_gpu_memory_bytes": None}
            argv = ["--task", "mw-reach", "--checkpoint-sha256", sha256(checkpoint),
                    "--stimulus-sha256", "b" * 64, "--fit", str(fit), "--coupling-fit", str(coupling_fit),
                    "--checkpoint", str(checkpoint), "--vendor", str(root / "vendor"),
                    "--stimulus", str(root / "stimulus"), "--data-root", str(root / "raw"), "--output", str(output)]
            with ExitStack() as stack:
                for name, value in (("COUNTS", (2,)), ("HORIZONS", (2, 6))):
                    stack.enter_context(patch.object(check, name, value))
                stack.enter_context(patch.object(check, "verify_bindings", return_value=(
                    bank, cohort, protocol, coupling, cohort["fit"][0], {})))
                stack.enter_context(patch.object(check, "verify_stimulus", return_value=({}, {})))
                stack.enter_context(patch.object(check, "use_vendor"))
                stack.enter_context(patch.object(check, "JepaBackend", return_value=backend))
                stack.enter_context(patch.object(check, "load_stimulus", side_effect=load))
                stack.enter_context(patch.dict("sys.modules", {"tensordict": types.SimpleNamespace(
                    TensorDict=lambda value, batch_size: Context(value))}))
                stack.enter_context(patch.object(check, "measure", return_value=timing))
                stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
                stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
                stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="synthetic CPU fixture"))
                stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=types.SimpleNamespace(uuid="fixture")))
                check.main(argv)
            done = json.loads((output / "DONE.json").read_text())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(done["report_sha256"], sha256(output / "report.json"))
            self.assertEqual(report["contract_sha256"], sha256(output / "contract.json"))
            self.assertEqual(len(report["checks"]), 4)
            self.assertTrue(report["parameter_buffer_bytes_modes_unchanged"])
            self.assertEqual(report["model_before_sha256"], sha256(output / "model-before.json"))
            self.assertEqual(report["model_after_sha256"], sha256(output / "model-after.json"))
            for field in ("behavioral_launch_ready", "full_cem_episode_run", "fresh_confirmation",
                          "scientific_efficacy_measurement", "normalization_was_fit_only"):
                self.assertIs(report[field], False)
            self.assertFalse((output / "FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()
