"""Source-bound complete56 CEM extension; original8 and pooled64 are descriptive.

No GPU use, fit updates, partial-cohort aggregation, or edits to published8 outputs.
The frozen eight-case module supplies only trace-validation/comparison functions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data"
BASE = ROOT / "artifacts/offline_study/layer-pilot-20260913-v1"
PROTOCOL = DATA / "cem_expansion_protocol.json"
PROTOCOL_SHA = "4766488e4907dff62f588894845f5b5aecf084e2c47a5ab4a1605cc2f07463db"
HELPER = Path(__file__).with_name("cem_steering_summary.py")
HELPER_SHA = "568c9b8203328d3f315d6fc43119a94ce368ac34eaf33c420967f14ae490abfb"
INITIAL_RECEIPT_SHA = "e1c15f276c58d69d9e986ac654ade790ac0ce9bea47f51534fa8aec71de6d7e6"
INITIAL_MANIFEST_SHA = "7130c52473255171a2122c89efb87292afb22630bc4bf9ee15eac0d4fa7763cf"
CHECKPOINT_SHA = "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
INPUT_MANIFEST_SHA = "7eaaec460a9057078a36cb348e859222cdf426842620107bb0d2cdabe7df70ef"
DINO_SOURCE_SHA = "88b35b92ca99c27c3bd9c650d930f43e78c7e6341fb13ecfffdc26272fbf80a5"
DINO_WEIGHTS_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
FIT_SHA = {"reach": "22a0ef202f299a5243a889fe661a1104a59a929a959b4f8411fc53212fb40dbe",
           "reach-wall": "082cd8be3cae2b127b7660d8f541e5a84a1c533c369f99fcf409e710c9733e42"}
TASKS = ("reach", "reach-wall")
ARMS = ("fixed_rank4", "matched_random_fixed_rank4")
ALL_ARMS = ("native", *ARMS)
POPULATIONS = {"extension56": tuple(range(4, 32)), "initial8": tuple(range(4)), "pooled64": tuple(range(32))}
PRIMARY = ("selected_prefix_rms_learned_minus_random",
           "mean_proposal_mean_rms_learned_minus_random",
           "mean_proposal_entropy_delta_learned_minus_random")
ITERATION_METRICS = ("proposal_mean_delta_l2", "proposal_mean_delta_rms", "proposal_std_delta_l2",
                     "proposal_std_delta_rms", "native_proposal_entropy_nats", "arm_proposal_entropy_nats",
                     "proposal_entropy_delta_nats", "native_best_runnerup_margin", "arm_best_runnerup_margin",
                     "best_runnerup_margin_delta", "native_elite_boundary_margin", "arm_elite_boundary_margin",
                     "elite_boundary_margin_delta")
PREFIX_METRICS = ("selected_prefix_delta_l2", "selected_prefix_delta_rms", "final_mean_delta_l2", "final_mean_delta_rms")
SHARED_METRICS = ("shared_population_elite_overlap_count", "shared_population_cost_delta_mean", "shared_population_cost_delta_rms")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_helper():
    require(sha(HELPER) == HELPER_SHA, "Frozen pure-function source changed")
    spec = importlib.util.spec_from_file_location("cem_expansion_frozen_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    require(sha(PROTOCOL) == PROTOCOL_SHA, "Frozen extension protocol changed")
    value = read(PROTOCOL)
    require(value["new_episode_ids"] == list(range(4, 32)) and value["new_cases"] == 56,
            "Extension registry changed")
    require(value["bootstrap_replicates"] == 20000 and value["bootstrap_seed"] == 20260913,
            "Prospective statistical settings changed")
    return value


def freeze(path):
    protocol()
    frozen_helper()
    result = dict(status="analysis_frozen_before_extension_outcomes", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=PROTOCOL_SHA, analysis_source_sha256=sha(__file__),
                  frozen_helper_sha256=HELPER_SHA, initial_receipt_sha256=INITIAL_RECEIPT_SHA,
                  primary_cases=56, n_primary_per_task=28, primary_metrics=list(PRIMARY),
                  bonferroni_family=6, bootstrap_replicates=20000, bootstrap_seed=20260913)
    with Path(path).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def validate_freeze(path):
    value = read(path)
    for name, expected in (("protocol_sha256", PROTOCOL_SHA), ("analysis_source_sha256", sha(__file__)),
                           ("frozen_helper_sha256", HELPER_SHA), ("initial_receipt_sha256", INITIAL_RECEIPT_SHA)):
        require(value.get(name) == expected, "Analysis freeze binding mismatch: " + name)
    require(value["primary_metrics"] == list(PRIMARY) and value["bonferroni_family"] == 6,
            "Primary analysis family changed")
    return value


def compare_case(payload, key, cohort):
    """Extract one case without imposing a favorable initial elite overlap."""
    require(key[0] in TASKS and key[1] in POPULATIONS[cohort], "Wrong scenario/cohort")
    helper = frozen_helper()
    for trace in payload["traces"].values():
        helper.finite(trace["final_mean"], (6, 20))
        require(re.fullmatch(r"[0-9a-f]{64}", trace["source_trace_sha256"]) is not None,
                "Missing source trace fingerprint")
        for row in trace["iterations"]:
            for name in ("proposal_mean", "proposal_std"):
                require(np.asarray(row[name]).shape in ((6, 20), (6, 1, 20)), "Unexpected native proposal layout")
            require(row["candidate_actions_dtype"] == "float32", "Unexpected candidate precision")
            require(np.all(helper.finite(row["objective_costs"], (300,)) >= 0), "Official squared goal costs must be nonnegative")
            # Require valid fingerprints on every iteration, not just the shared first bank.
            helper.same_candidates(row, row)
    iterations, prefixes = helper.compare_case(payload, key)
    native = payload["traces"]["native"]
    for row in iterations + prefixes:
        row["cohort"] = cohort
        row["native_comparator"] = "same_receiving_gpu" if cohort == "extension56" else "archived_initial_native"
    for row in prefixes:
        other = payload["traces"][row["arm"]]
        delta = np.asarray(other["final_mean"]) - np.asarray(native["final_mean"])
        row.update(final_mean_delta_l2=float(np.linalg.norm(delta)), final_mean_delta_rms=float(np.sqrt(np.mean(delta ** 2))),
                   selected_prefix_coordinates=60, proposal_coordinates=120,
                   first_candidate_divergence_iteration=next((i for i, (a, b) in enumerate(zip(native["iterations"], other["iterations"]))
                                                             if not helper.same_candidates(a, b)), None))
    return iterations, prefixes


def require_frame_coverage(frame, iteration=False):
    keys = ["task", "episode", "arm"] + (["iteration"] if iteration else [])
    require(not frame.duplicated(keys).any(), "Duplicate case/arm/iteration")
    expected = {(task, episode, arm, *([i] if iteration else [])) for task in TASKS for episode in range(32)
                for arm in ARMS for i in (range(15) if iteration else [0])}
    require(set(frame[keys].itertuples(index=False, name=None)) == expected, "Require complete64 frames; never partial means")
    for row in frame[["cohort", "episode"]].itertuples(index=False):
        require(row.cohort == ("initial8" if row.episode < 4 else "extension56"), "Initial/extension strata mixed")


def primary_cases(iterations, prefixes):
    require_frame_coverage(iterations, True)
    require_frame_coverage(prefixes)
    rows = []
    for (task, episode), block in prefixes.groupby(["task", "episode"], sort=True):
        values = block.set_index("arm")
        curves = iterations[(iterations.task == task) & (iterations.episode == episode)].groupby("arm")
        learned = [float(values.loc[ARMS[0], "selected_prefix_delta_rms"]),
                   float(curves.get_group(ARMS[0]).proposal_mean_delta_rms.mean()),
                   float(curves.get_group(ARMS[0]).proposal_entropy_delta_nats.mean())]
        random = [float(values.loc[ARMS[1], "selected_prefix_delta_rms"]),
                  float(curves.get_group(ARMS[1]).proposal_mean_delta_rms.mean()),
                  float(curves.get_group(ARMS[1]).proposal_entropy_delta_nats.mean())]
        for metric, a, b in zip(PRIMARY, learned, random):
            rows.append(dict(cohort="initial8" if episode < 4 else "extension56", task=task, episode=episode,
                             metric=metric, learned=a, random=b, effect=a - b))
    return pd.DataFrame(rows)


def bootstrap_weights(task, replicates=20000, seed=20260913):
    """Paired across all arms/iterations; descriptive pool preserves fixed strata."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, TASKS.index(task)]))
    extension = rng.multinomial(28, np.full(28, 1 / 28), size=replicates) / 28
    initial = rng.multinomial(4, np.full(4, .25), size=replicates) / 4
    return {"extension56": extension, "initial8": initial,
            "pooled64": np.concatenate([initial * (4 / 32), extension * (28 / 32)], axis=1)}


def statistics(values, weights, primary=False):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and np.isfinite(values).all() and weights.shape[1] == len(values),
            "Invalid scenario-unit metric matrix")
    draws = weights @ values
    lows, highs = np.quantile(draws, [.025, .975], axis=0)
    corrected = np.quantile(draws, [.05 / 12, 1 - .05 / 12], axis=0) if primary else None
    rows = []
    for i in range(values.shape[1]):
        sd = float(np.std(values[:, i], ddof=1))
        rows.append(dict(n=len(values), mean=float(values[:, i].mean()), scenario_sd=sd, scenario_se=sd / np.sqrt(len(values)),
                         marginal_95_low=float(lows[i]), marginal_95_high=float(highs[i]),
                         bonferroni_family=6 if primary else None,
                         bonferroni_95_low=float(corrected[0, i]) if primary else None,
                         bonferroni_95_high=float(corrected[1, i]) if primary else None,
                         inference_scope="primary_extension_family6" if primary else "descriptive_only"))
    return rows


def aggregate(frame, groups, metrics, weights):
    rows = []
    for population, episodes in POPULATIONS.items():
        selected = frame[frame.episode.isin(episodes)]
        for key, block in selected.groupby(groups, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            identity = dict(zip(groups, key))
            require(set(block.episode) == set(episodes) and len(block) == len(episodes), "Incomplete scenario cell")
            block = block.sort_values("episode")
            stats = statistics(block[list(metrics)].to_numpy(), weights[identity["task"]][population])
            rows.extend(dict(population=population, **identity, metric=metric, **stat) for metric, stat in zip(metrics, stats))
    return pd.DataFrame(rows)


def summarize(iterations, prefixes, replicates=20000, seed=20260913):
    primary = primary_cases(iterations, prefixes)
    weights = {task: bootstrap_weights(task, replicates, seed) for task in TASKS}
    primary_rows = []
    for population, episodes in POPULATIONS.items():
        for task in TASKS:
            block = primary[(primary.task == task) & primary.episode.isin(episodes)].pivot(index="episode", columns="metric", values="effect")
            block = block.loc[list(episodes), list(PRIMARY)]
            values = statistics(block.to_numpy(), weights[task][population], primary=population == "extension56")
            primary_rows.extend(dict(population=population, task=task, metric=metric, **stat) for metric, stat in zip(PRIMARY, values))
    return dict(iteration_cases=iterations, selected_prefix_cases=prefixes, primary_cases=primary,
                iteration_summary=aggregate(iterations, ["task", "arm", "iteration"], ITERATION_METRICS, weights),
                selected_prefix_summary=aggregate(prefixes, ["task", "arm"], PREFIX_METRICS, weights),
                shared_iteration0_summary=aggregate(iterations[iterations.iteration == 0], ["task", "arm"], SHARED_METRICS, weights),
                primary_summary=pd.DataFrame(primary_rows))


def case_paths(compact_root, cohort):
    return {(task, episode): Path(compact_root) / task / f"episode-{episode}"
            for task in TASKS for episode in POPULATIONS[cohort]}


def require_complete_files(cases, compact_name):
    required = ("report.json", "DONE.json", "CLOUD_VERIFIED.json", compact_name)
    missing = [(task, episode, name) for (task, episode), directory in cases.items()
               for name in required if not (directory / name).is_file()]
    require(not missing, f"Incomplete source/DONE/cloud coverage; {len(missing)} files missing; no partial aggregate")


def read_bound_case(directory, compact_name, metadata_only=False):
    report, done, cloud = (read(directory / name) for name in ("report.json", "DONE.json", "CLOUD_VERIFIED.json"))
    require(done["report_sha256"] == sha(directory / "report.json") and done["files"] == report["files"], "DONE/report mismatch")
    require(cloud.get("gcs_download_sha256_verified") is True and cloud.get("all_report_files_hash_verified") is True
            and cloud.get("raw_preservation_pending") is False, "Full raw preservation not verified")
    for name in ("report.json", "DONE.json", compact_name):
        require(cloud.get("compact_sha256", {}).get(name) == sha(directory / name), "Compact archive hash mismatch: " + name)
    require(report["files"][compact_name] == sha(directory / compact_name), "Unbound compact payload")
    require(report.get("physical_outcomes_measured") is False and report.get("fresh_confirmation") is False
            and report.get("global_rng_unchanged") is True, "Experiment scope/global RNG mismatch")
    for name in ("sha256",):
        require(re.fullmatch(r"[0-9a-f]{64}", cloud[name]) is not None, "Malformed full archive digest")
    require(str(cloud["cloud_uri"]).startswith("gs://") and bool(cloud.get("generation")), "Missing generation-pinned cloud archive")
    if metadata_only:
        return report, cloud, None
    payload = read(directory / compact_name)
    require(payload["input_binding"] == report["input_binding"], "Payload/report input mismatch")
    return report, cloud, payload


def validate_parities(report, arms):
    require(set(report["parities"]) == set(arms), "Missing same-runtime arm parity")
    for arm in arms:
        parity = report["parities"][arm]
        for name in ("selected_plan_byte_equal", "local_generator_byte_equal", "global_rng_unchanged", "native_iteration0_actions_byte_equal"):
            require(parity.get(name) is True, "Missing exact CEM parity: " + arm + "/" + name)
        require(parity["untraced_forecast_calls"] == 30 and parity["traced_forecast_calls"] == 30,
                "Unexpected native CEM callback count")


def validate_original_native(directory, old):
    report, done, cloud = (read(directory / name) for name in ("report.json", "DONE.json", "CLOUD_VERIFIED.json"))
    require(sha(directory / "report.json") == old["original_native_report_sha256"] == done["report_sha256"]
            and done["files"] == report["files"], "Original native DONE/report binding changed")
    require(report["input_binding"] == old["input_binding"]
            and report["files"]["native-cem-trace.pt"] == old["native_trace_sha256"], "Original native input/trace changed")
    require(cloud.get("gcs_download_sha256_verified") is True and cloud.get("all_report_files_hash_verified") is True
            and cloud.get("raw_preservation_pending") is False and cloud.get("parent_trace_sha256") == old["native_trace_sha256"],
            "Original native full raw preservation not verified")
    for name in ("report.json", "DONE.json"):
        require(cloud["compact_sha256"].get(name) == sha(directory / name), "Original native receipt metadata changed")
    return dict(original_native_cloud_receipt_sha256=sha(directory / "CLOUD_VERIFIED.json"),
                original_native_cloud_uri=cloud["cloud_uri"], original_native_cloud_generation=cloud["generation"],
                original_native_cloud_archive_sha256=cloud["sha256"])


def validate_extension(report, cloud, payload, key, manifest, manifest_sha, binding, gpu_uuid):
    require(report["execution_manifest_sha256"] == manifest_sha and report["protocol_sha256"] == PROTOCOL_SHA,
            "Extension source/protocol binding mismatch")
    require(report["input_binding"] == binding and (binding["task"], binding["episode"]) == key, "Scenario/seed binding mismatch")
    validate_parities(report, ALL_ARMS)
    require(report.get("parameters_unchanged") is True and report.get("inputs_unchanged") is True,
            "Frozen parameters/input parity not attested")
    require(report.get("gpu_uuid") == gpu_uuid, "Assigned physical GPU changed")
    require(report.get("tf32_matmul") is False and report.get("tf32_cudnn") is False, "TF32 not explicitly disabled")
    backend = report["backend_provenance"]
    for name, expected in (("checkpoint_sha256", CHECKPOINT_SHA), ("dino_source_sha256", DINO_SOURCE_SHA),
                           ("dino_weights_sha256", DINO_WEIGHTS_SHA), ("precision", "float32"), ("allow_tf32", False)):
        require(backend.get(name) == expected, "Backend provenance changed: " + name)
    require(backend.get("dino_loader_source") == "verified_local_cache_no_network_branch_resolution", "Unverified DINO loading")
    require(report.get("fit_bank_sha256") == FIT_SHA[key[0]], "Task fitted operator changed")
    if payload is not None:
        require(payload.get("execution_manifest_sha256") == manifest_sha and payload.get("protocol_sha256") == PROTOCOL_SHA,
                "Compact execution/protocol mismatch")
        require(payload.get("parities") == report["parities"], "Compact parity mismatch")
    members = cloud.get("verified_archive_members", {})
    for name, digest in report["files"].items():
        matches = [value for path, value in members.items() if path == name or path.endswith(f"/{key[0]}/episode-{key[1]}/{name}")]
        require(matches == [digest], "Missing unique full-original archive member: " + name)
    for arm in ALL_ARMS:
        require(arm + "-cem-trace.pt" in report["files"], "Missing concurrent native/arm trace")
        if payload is not None:
            require(payload["traces"][arm]["source_trace_sha256"] == report["files"][arm + "-cem-trace.pt"],
                    "Concurrent native/arm trace parent mismatch")
    require(manifest["fit_bank_sha256"] == FIT_SHA, "Manifest fit mismatch")


def source_receipt(key, cohort, directory, cloud, report, payload, compact_name):
    return dict(task=key[0], episode=key[1], cohort=cohort, report_sha256=sha(directory / "report.json"),
                done_sha256=sha(directory / "DONE.json"), compact_sha256=sha(directory / compact_name),
                cloud_receipt_sha256=sha(directory / "CLOUD_VERIFIED.json"), cloud_uri=cloud["cloud_uri"],
                cloud_generation=cloud["generation"], cloud_archive_sha256=cloud["sha256"],
                input_binding=report["input_binding"], gpu_uuid=report.get("gpu_uuid"),
                trace_sha256={arm: payload["traces"][arm]["source_trace_sha256"] for arm in ALL_ARMS})


def assigned_gpus(manifest):
    assignments = {}
    for slot, cases in manifest["slots"].items():
        gpu = manifest["receivers"][slot]["gpu_uuid"]
        require(isinstance(gpu, str) and re.fullmatch(r"(?:GPU-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu),
                "Missing physical GPU assignment")
        for case in cases:
            key = (case["task"], case["episode"])
            require(key not in assignments, "Duplicate execution assignment")
            assignments[key] = gpu
    expected = {(task, episode) for task in TASKS for episode in range(4, 32)}
    require(set(assignments) == expected, "Incomplete/extra execution assignments")
    return assignments


def run(args):
    protocol()
    freeze_receipt = validate_freeze(args.analysis_freeze)
    # Load and validate the exact prospective manifests before any new case payload.
    require(sha(args.execution_manifest) == args.execution_manifest_sha256, "Execution manifest hash changed")
    manifest = read(args.execution_manifest)
    require(manifest["protocol_sha256"] == PROTOCOL_SHA and manifest["scenarios"] == {task: list(range(4, 32)) for task in TASKS},
            "Prospective extension scenario/protocol registry changed")
    require(manifest["arms"] == list(ALL_ARMS) and manifest["checkpoint_sha256"] == CHECKPOINT_SHA
            and manifest["fit_bank_sha256"] == FIT_SHA, "Frozen arm/checkpoint/fit registry changed")
    require(bool(manifest.get("source_sha256")) and bool(manifest.get("vendor_source_sha256")),
            "Missing pinned execution/vendor source registry")
    require(manifest["input_manifest_sha256"] == INPUT_MANIFEST_SHA == sha(args.input_manifest), "Input manifest hash changed")
    bindings = {(row["task"], row["episode"]): row for row in read(args.input_manifest)["records"]}
    gpus = assigned_gpus(manifest)
    for name, digest in manifest["source_sha256"].items():
        require(Path(name).name == name and sha(ROOT / "src/offline_study" / name) == digest, "Execution source changed: " + name)
    for name, digest in manifest.get("vendor_source_sha256", {}).items():
        require(".." not in Path(name).parts and not Path(name).is_absolute() and sha(ROOT / "vendor/jepa-wms" / name) == digest,
                "Pinned vendor source changed: " + name)
    initial_receipt_path = DATA / "cem_steering_summary.json"
    require(sha(initial_receipt_path) == INITIAL_RECEIPT_SHA, "Published initial8 receipt changed")
    initial_receipt = read(initial_receipt_path)
    require(initial_receipt["analysis_source_sha256"] == HELPER_SHA and initial_receipt["execution_manifest_sha256"] == INITIAL_MANIFEST_SHA,
            "Initial8 source binding changed")
    for name, info in initial_receipt["outputs"].items():
        require(sha(ROOT / name) == info["sha256"], "Published initial8 table changed")
    old_sources = {(row["task"], row["episode"]): row for row in initial_receipt["sources"]}
    extension = case_paths(args.compact_root, "extension56")
    initial = case_paths(args.initial_compact_root, "initial8")
    original_native = case_paths(args.initial_native_compact_root, "initial8")
    require_complete_files(extension, args.compact_name)
    require_complete_files(initial, "steered-cem-summary.json")
    require_complete_files(original_native, "cem_summary.json")
    # Preflight every source/DONE/archive/parity receipt before parsing any new outcomes.
    for key, directory in sorted(extension.items()):
        report, cloud, _ = read_bound_case(directory, args.compact_name, metadata_only=True)
        validate_extension(report, cloud, None, key, manifest, args.execution_manifest_sha256, bindings[key], gpus[key])
    for key, directory in sorted(initial.items()):
        read_bound_case(directory, "steered-cem-summary.json", metadata_only=True)
        validate_original_native(original_native[key], old_sources[key])
    iterations, prefixes, sources = [], [], []
    for cohort, cases, compact_name in (("extension56", extension, args.compact_name), ("initial8", initial, "steered-cem-summary.json")):
        for key, directory in sorted(cases.items()):
            report, cloud, payload = read_bound_case(directory, compact_name)
            extra_source = {}
            if cohort == "extension56":
                validate_extension(report, cloud, payload, key, manifest, args.execution_manifest_sha256, bindings[key], gpus[key])
            else:
                old = old_sources[key]
                for name, expected in (("report.json", old["report_sha256"]), ("CLOUD_VERIFIED.json", old["cloud_receipt_sha256"]),
                                       (compact_name, old["comparison_sha256"])):
                    require(sha(directory / name) == expected, "Initial8 original source changed")
                require(report["execution_manifest_sha256"] == INITIAL_MANIFEST_SHA and report["input_binding"] == old["input_binding"],
                        "Initial8 input/execution changed")
                validate_parities(report, ARMS)
                require(payload["traces"]["native"]["source_trace_sha256"] == old["native_trace_sha256"], "Original native trace changed")
                for arm in ARMS:
                    require(payload["traces"][arm]["source_trace_sha256"] == old["arm_trace_sha256"][arm], "Original steered trace changed")
                extra_source = validate_original_native(original_native[key], old)
            i, p = compare_case(payload, key, cohort)
            iterations.extend(i)
            prefixes.extend(p)
            sources.append(source_receipt(key, cohort, directory, cloud, report, payload, compact_name) | extra_source)
    frames = summarize(pd.DataFrame(iterations), pd.DataFrame(prefixes))
    outputs = {}
    for name, frame in frames.items():
        path = DATA / f"cem_expansion_{name}.csv"
        frame.to_csv(path, index=False)
        outputs[str(path.relative_to(ROOT))] = dict(sha256=sha(path), rows=len(frame))
    result = dict(status="complete56_extension_with_separate_initial8", primary_cases=56, primary_n_per_task=28,
                  initial_cases=8, initial_n_per_task=4, pooled_cases=64, pooled_n_per_task=32, pooled_scope="descriptive_only",
                  analysis_source_sha256=sha(__file__), frozen_helper_sha256=HELPER_SHA, protocol_sha256=PROTOCOL_SHA,
                  analysis_freeze_sha256=sha(args.analysis_freeze), analysis_frozen_at_utc=freeze_receipt["frozen_at_utc"],
                  execution_manifest_sha256=args.execution_manifest_sha256, input_manifest_sha256=INPUT_MANIFEST_SHA,
                  initial_receipt_sha256=INITIAL_RECEIPT_SHA, primary_bonferroni_family=6, sources=sources, outputs=outputs,
                  complete_full_raw_preservation=True, physical_outcomes_measured=False, fresh_confirmation=False,
                  audits=dict(extension_searches=56 * 6, extension_forecast_callbacks=56 * 180,
                              extension_primary_contrast_count=6, initial_iteration_elite_overlap_is_observed_not_a_gate=True,
                              later_elite_index_overlap_not_computed=True, cpu_recomputes_scores_entropy_and_plan_differences=True,
                              gpu_attests_full_trace_actions_and_rng_parity=True),
                  caveats=["Same receiving-GPU native comparator only in extension56; initial8 archived-native comparator is separate",
                           "Pooled64 is descriptive; primary inference excludes the viewed initial8",
                           "More unsigned divergence is not improved physical action quality",
                           "Displayed SE is across scenarios; paired percentile intervals are separately retained",
                           "Later candidate indices are not paired actions; differential entropy is preclip and coordinate dependent"])
    (DATA / "cem_expansion_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, help="Write a new pre-outcome analysis freeze; no source outcomes are read")
    parser.add_argument("--analysis-freeze", type=Path, default=DATA / "cem_expansion_analysis_freeze.json")
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--compact-root", type=Path)
    parser.add_argument("--compact-name", default="steered-cem-summary.json")
    parser.add_argument("--initial-compact-root", type=Path, default=BASE / "compact/steered-cem-v2")
    parser.add_argument("--initial-native-compact-root", type=Path, default=BASE / "compact/development-v1")
    parser.add_argument("--input-manifest", type=Path, default=BASE / "development-inputs-64/INPUT_MANIFEST.json")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(args.freeze), indent=2))
    else:
        if not all((args.execution_manifest, args.execution_manifest_sha256, args.compact_root)):
            parser.error("Execution manifest, its SHA256 and complete compact root are required")
        print(json.dumps({key: value for key, value in run(args).items() if key not in ("sources", "outputs")}, indent=2))
