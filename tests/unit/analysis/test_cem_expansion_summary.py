import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("cem_expansion_summary", ROOT / "analysis/mechanism/cem_expansion_summary.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def trace(offset=0., initial_elites=None):
    rows = []
    for i in range(15):
        costs = np.arange(300, dtype=float) + 10
        elites = list(range(10))
        if i == 0 and initial_elites is not None:
            elites = initial_elites
            costs[elites] = np.arange(10, dtype=float)
        std = np.full((6, 1, 20), .1)
        rows.append(dict(iteration=i, proposal_mean=np.full((6, 1, 20), offset if i else 0.).tolist(),
                         proposal_std=std.tolist(), proposal_entropy_nats=float(np.sum(np.log(std) + .5 * math.log(2 * math.pi * math.e))),
                         objective_costs=costs.tolist(), elite_indices=elites,
                         candidate_actions_sha256=("b" if offset and i else "a") * 64,
                         candidate_actions_shape=[6, 300, 20], candidate_actions_dtype="float32"))
    return dict(selected_plan=np.full((3, 20), offset).tolist(), final_mean=np.full((6, 20), 2 * offset).tolist(),
                iterations=rows, source_trace_sha256="d" * 64)


def payload():
    return {"traces": {"native": trace(), m.ARMS[0]: trace(.2), m.ARMS[1]: trace(.1)}}


def frames():
    iterations, prefixes = [], []
    for task in m.TASKS:
        for episode in range(32):
            cohort = "initial8" if episode < 4 else "extension56"
            for arm in m.ARMS:
                # Initial pilot is deliberately unlike the extension to catch leakage.
                effect = (100. if episode < 4 else 1. + episode / 100) if arm == m.ARMS[0] else 0.
                common = dict(cohort=cohort, task=task, episode=episode, arm=arm)
                prefixes.append(dict(**common, **{key: effect for key in m.PREFIX_METRICS}))
                for i in range(15):
                    iterations.append(dict(**common, iteration=i, **{key: effect for key in (*m.ITERATION_METRICS, *m.SHARED_METRICS)}))
    return pd.DataFrame(iterations), pd.DataFrame(prefixes)


def parity():
    return dict(selected_plan_byte_equal=True, local_generator_byte_equal=True, global_rng_unchanged=True,
                native_iteration0_actions_byte_equal=True, untraced_forecast_calls=30, traced_forecast_calls=30)


def bound_fixture(directory):
    value = payload()
    binding = dict(task="reach", episode=4, candidate_seed=123)
    value.update(input_binding=binding, execution_manifest_sha256="e" * 64, protocol_sha256=m.PROTOCOL_SHA,
                 parities={arm: parity() for arm in m.ALL_ARMS})
    files = {arm + "-cem-trace.pt": "d" * 64 for arm in m.ALL_ARMS}
    name = "cem-expansion-summary.json"
    (directory / name).write_text(json.dumps(value))
    files[name] = m.sha(directory / name)
    report = dict(input_binding=binding, files=files, execution_manifest_sha256="e" * 64, protocol_sha256=m.PROTOCOL_SHA,
                  parities=value["parities"], parameters_unchanged=True, inputs_unchanged=True,
                  gpu_uuid="01234567-1234-1234-1234-123456789abc", fit_bank_sha256=m.FIT_SHA["reach"],
                  physical_outcomes_measured=False, fresh_confirmation=False, global_rng_unchanged=True,
                  tf32_matmul=False, tf32_cudnn=False,
                  backend_provenance=dict(checkpoint_sha256=m.CHECKPOINT_SHA, dino_source_sha256=m.DINO_SOURCE_SHA,
                                          dino_weights_sha256=m.DINO_WEIGHTS_SHA, precision="float32", allow_tf32=False,
                                          dino_loader_source="verified_local_cache_no_network_branch_resolution"))
    (directory / "report.json").write_text(json.dumps(report))
    (directory / "DONE.json").write_text(json.dumps(dict(report_sha256=m.sha(directory / "report.json"), files=files)))
    cloud = dict(gcs_download_sha256_verified=True, all_report_files_hash_verified=True, raw_preservation_pending=False,
                 cloud_uri="gs://fixture/object", generation="123", sha256="c" * 64,
                 compact_sha256={key: m.sha(directory / key) for key in (name, "report.json", "DONE.json")},
                 verified_archive_members={f"root/reach/episode-4/{key}": value for key, value in files.items()})
    (directory / "CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
    return report, cloud, value


class CEMExpansionTests(unittest.TestCase):
    def test_pinned_protocol_and_helper(self):
        self.assertEqual(m.protocol()["new_cases"], 56)
        self.assertTrue(callable(m.frozen_helper().compare_case))

    def test_actual_prefix_shape_and_distinct_full_mean(self):
        iterations, prefixes = m.compare_case(payload(), ("reach", 4), "extension56")
        self.assertEqual(len(iterations), 30)
        self.assertAlmostEqual(prefixes[0]["selected_prefix_delta_rms"], .2)
        self.assertAlmostEqual(prefixes[0]["selected_prefix_delta_l2"], .2 * np.sqrt(60))
        self.assertAlmostEqual(prefixes[0]["final_mean_delta_rms"], .4)
        self.assertEqual(prefixes[0]["first_candidate_divergence_iteration"], 1)
        bad = payload()
        bad["traces"][m.ARMS[0]]["selected_plan"] = np.zeros((6, 20)).tolist()
        with self.assertRaises(ValueError):
            m.compare_case(bad, ("reach", 4), "extension56")

    def test_initial_elites_are_measured_not_favorable_gate(self):
        value = payload()
        value["traces"][m.ARMS[0]] = trace(.2, list(range(10, 20)))
        iterations, _ = m.compare_case(value, ("reach", 4), "extension56")
        self.assertEqual(iterations[0]["shared_population_elite_overlap_count"], 0)
        self.assertIsNone(iterations[1]["shared_population_elite_overlap_count"])
        self.assertLess(iterations[0]["native_proposal_entropy_nats"], 0)

    def test_every_iteration_requires_valid_fingerprint(self):
        value = payload()
        value["traces"][m.ARMS[0]]["iterations"][7]["candidate_actions_sha256"] = "bad"
        with self.assertRaises(ValueError):
            m.compare_case(value, ("reach", 4), "extension56")

    def test_partial_or_duplicate_or_wrong_stratum_rejected(self):
        iterations, prefixes = frames()
        for invalid in (prefixes.iloc[:-1], pd.concat([prefixes, prefixes.iloc[:1]])):
            with self.assertRaises(ValueError):
                m.primary_cases(iterations, invalid)
        wrong = iterations.copy()
        wrong.loc[wrong.episode == 0, "cohort"] = "extension56"
        with self.assertRaises(ValueError):
            m.primary_cases(wrong, prefixes)

    def test_primary56_excludes_initial8_and_pool_is_descriptive(self):
        iterations, prefixes = frames()
        outputs = m.summarize(iterations, prefixes, replicates=100)
        summary = outputs["primary_summary"]
        extension = summary[summary.population == "extension56"]
        self.assertEqual(len(extension), 6)
        self.assertTrue((extension.n == 28).all())
        self.assertTrue(np.allclose(extension["mean"], 1.175))
        self.assertTrue((extension.bonferroni_family == 6).all())
        pool = summary[summary.population == "pooled64"]
        self.assertTrue((pool.n == 32).all())
        self.assertTrue(np.allclose(pool["mean"], (4 * 100 + 28 * 1.175) / 32))
        self.assertTrue(pool.bonferroni_95_low.isna().all())
        self.assertTrue((pool.inference_scope == "descriptive_only").all())

    def test_se_and_bonferroni_quantiles_are_scenario_based(self):
        weights = m.bootstrap_weights("reach", 2000)["extension56"]
        values = np.arange(28, dtype=float)[:, None]
        row = m.statistics(values, weights, primary=True)[0]
        self.assertAlmostEqual(row["scenario_se"], np.std(np.arange(28), ddof=1) / np.sqrt(28))
        self.assertAlmostEqual(row["bonferroni_95_low"], np.quantile(weights @ values, .05 / 12))
        self.assertLessEqual(row["bonferroni_95_low"], row["marginal_95_low"])
        self.assertGreaterEqual(row["bonferroni_95_high"], row["marginal_95_high"])
        self.assertTrue(np.allclose(m.bootstrap_weights("reach", 2000)["pooled64"].sum(axis=1), 1))

    def test_same_weights_preserve_pairing_across_columns(self):
        weights = m.bootstrap_weights("reach", 100)["extension56"]
        values = np.arange(28, dtype=float)
        rows = m.statistics(np.column_stack([values, values + 3]), weights)
        self.assertAlmostEqual(rows[1]["marginal_95_low"] - rows[0]["marginal_95_low"], 3)
        self.assertAlmostEqual(rows[1]["scenario_se"], rows[0]["scenario_se"])

    def test_all_three_same_runtime_parities_required(self):
        report = dict(parities={arm: parity() for arm in m.ALL_ARMS})
        m.validate_parities(report, m.ALL_ARMS)
        del report["parities"]["native"]
        with self.assertRaises(ValueError):
            m.validate_parities(report, m.ALL_ARMS)
        report = dict(parities={arm: parity() for arm in m.ALL_ARMS})
        report["parities"]["native"]["traced_forecast_calls"] = 15
        with self.assertRaises(ValueError):
            m.validate_parities(report, m.ALL_ARMS)

    def test_full_archive_and_input_gpu_fit_binding(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            report, cloud, value = bound_fixture(directory)
            m.require_complete_files({("reach", 4): directory}, "cem-expansion-summary.json")
            m.read_bound_case(directory, "cem-expansion-summary.json")
            original_read = m.read
            def metadata_read(path):
                self.assertNotEqual(Path(path).name, "cem-expansion-summary.json")
                return original_read(path)
            with patch.object(m, "read", side_effect=metadata_read):
                self.assertIsNone(m.read_bound_case(directory, "cem-expansion-summary.json", metadata_only=True)[2])
            args = (("reach", 4), {"fit_bank_sha256": m.FIT_SHA}, "e" * 64, report["input_binding"], report["gpu_uuid"])
            m.validate_extension(report, cloud, value, *args)
            for key, altered in (("gpu_uuid", "different"), ("fit_bank_sha256", "f" * 64), ("parameters_unchanged", False)):
                bad = copy.deepcopy(report)
                bad[key] = altered
                with self.assertRaises(ValueError):
                    m.validate_extension(bad, cloud, value, *args)
            bad_cloud = copy.deepcopy(cloud)
            del bad_cloud["verified_archive_members"]["root/reach/episode-4/native-cem-trace.pt"]
            with self.assertRaises(ValueError):
                m.validate_extension(report, bad_cloud, value, *args)
            cloud["raw_preservation_pending"] = True
            (directory / "CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            with self.assertRaises(ValueError):
                m.read_bound_case(directory, "cem-expansion-summary.json")

    def test_no_partial_payload_read(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "no partial aggregate"):
                m.require_complete_files(m.case_paths(name, "extension56"), "cem-expansion-summary.json")

    def test_analysis_freeze_is_create_once_and_source_bound(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "freeze.json"
            m.freeze(path)
            m.validate_freeze(path)
            with self.assertRaises(FileExistsError):
                m.freeze(path)
            value = m.read(path)
            value["analysis_source_sha256"] = "bad"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                m.validate_freeze(path)

    def test_original_native_requires_its_own_full_raw_proof(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            report = dict(input_binding=dict(task="reach", episode=0), files={"native-cem-trace.pt": "a" * 64})
            (directory / "report.json").write_text(json.dumps(report))
            digest = m.sha(directory / "report.json")
            (directory / "DONE.json").write_text(json.dumps(dict(report_sha256=digest, files=report["files"])))
            cloud = dict(gcs_download_sha256_verified=True, all_report_files_hash_verified=True, raw_preservation_pending=False,
                         parent_trace_sha256="a" * 64, cloud_uri="gs://fixture/native", generation="123", sha256="b" * 64,
                         compact_sha256={key: m.sha(directory / key) for key in ("report.json", "DONE.json")})
            (directory / "CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            old = dict(original_native_report_sha256=digest, native_trace_sha256="a" * 64, input_binding=report["input_binding"])
            self.assertEqual(m.validate_original_native(directory, old)["original_native_cloud_generation"], "123")
            cloud["raw_preservation_pending"] = True
            (directory / "CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            with self.assertRaises(ValueError):
                m.validate_original_native(directory, old)

    def test_gpu_registry_requires_all56_once(self):
        manifest = dict(slots={"0": [dict(task=task, episode=e) for task in m.TASKS for e in range(4, 32)]},
                        receivers={"0": dict(gpu_uuid="01234567-1234-1234-1234-123456789abc")})
        self.assertEqual(len(m.assigned_gpus(manifest)), 56)
        manifest["slots"]["0"].append(dict(task="reach", episode=4))
        with self.assertRaises(ValueError):
            m.assigned_gpus(manifest)


if __name__ == "__main__":
    unittest.main()
