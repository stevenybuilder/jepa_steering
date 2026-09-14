"""Full fixed Push-T coupling factorial, paired on released source segments.

No offline efficacy admission gate, changed native planner or confirmation access.
Reuse the proven static compiler, but keep Push-T's source-family identity and
actual dataset sampler; navigation initial/goal clusters are not Push-T families.
"""
from offline_study._paths import source_path
import argparse
import copy
import json
import random
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import open_normalized_dataset
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, verified_report
from offline_study.evaluation.fixed_response_behavior import device_uuid
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.data.inventory import _initial_state_group
from offline_study.tasks.navigation.navigation_coupling_behavior import ARMS, action_hashes, one_episode, source_digest
from offline_study.tasks.navigation.navigation_replication import validate_records
from offline_study.tasks.navigation.navigation_smoke import comparison_record, validate_complete
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_env_smoke import SingleFitTrajectory
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED
from offline_study.core.protocol import sha256, write_json
from offline_study.tasks.pusht.pusht_planning_replication import TracedDataset, checked_cohort, input_hashes, trace_segment
from offline_study.models.vendor import use_vendor


ROLE = "full_static_coupling_pusht_development"
FIT_FILES = ("protocol.json", "operator_bank.pt", "fit_receipt.json", "DONE.json",
             "source_protocol.json", "source_fit_receipt.json")
SEGMENT_KEYS = ("trajectory_id", "source_index", "lineage_group", "source_pool",
                "sampled_states_sha256", "sampled_actions_sha256", "sampled_initial_state", "segment_frames")
INTERACTION = {"joint": 1, "visual_only": -1, "action_condition_only": -1, "native": 1}


def paired_inputs(native, candidate):
    for key in ("episode", "logical_rank", "local_seed", "environment_seed"):
        if native[key] != candidate[key]:
            raise ValueError("Push-T stream/episode pairing changed: " + key)
    for key in ("initial_sha256", "goal_sha256"):
        if native["result"][key] != candidate["result"][key]:
            raise ValueError("Push-T initial/goal pairing changed: " + key)
    for key in SEGMENT_KEYS:
        if native["source_segment"][key] != candidate["source_segment"][key]:
            raise ValueError("Push-T source segment/family pairing changed: " + key)


def validate_push_records(records, expected, cohort):
    validate_records(records, expected)
    by_index = {r["index"]: r for r in cohort["evaluation"]}
    if set(by_index) != set(range(21)):
        raise ValueError("Require all21 original released rows")
    for record in records:
        segment = record["source_segment"]
        row = by_index.get(segment["source_index"])
        if (row is None or segment["source_pool"] != "val" or segment["segment_frames"] != 31 or
                any(segment[k] != row[k] for k in ("trajectory_id", "lineage_group"))):
            raise ValueError("Unknown or changed Push-T source family/segment")
        if (np.asarray(record["planned_actions"]).shape != (1, 6, 10) or
                any(len(segment[k]) != 64 or set(segment[k]) - set("0123456789abcdef")
                    for k in ("sampled_states_sha256", "sampled_actions_sha256")) or
                not np.isfinite(np.asarray(segment["sampled_initial_state"], dtype=float)).all()):
            raise ValueError("Malformed Push-T action/segment trace")


def verify_native_reference(reference, old_source, cohort_hash, planning):
    """Verify the immutable older source, not its hash against our additive tree."""
    path = reference / "freeze/protocol.json"
    native = json.loads(path.read_text())
    if (json.loads((reference / "freeze/FROZEN.json").read_text())["protocol_sha256"] != sha256(path) or
            native["role"] != "native_pusht_released_pool_planning_replication" or
            native["episodes"] != schedule() or native["episodes_per_condition_total"] != 96 or
            native["planning_contract"] != planning or native["checkpoint_sha256"] != CHECKPOINTS["pusht"] or
            native["cohort_sha256"] != cohort_hash or native["native_only"] is not True or
            native["fresh_confirmation"] is not False or native["source_families"] != 21 or
            source_digest(old_source) != native["source_sha256"]):
        raise ValueError("Native Push-T reference freeze/source/population changed")
    # Every existing source file must remain byte-identical. Only added modules
    # are permitted; do not rebind old GPU proof to a changed implementation.
    for old in old_source.glob("*.py"):
        current = source_path(old.name)
        if not current.is_file() or sha256(current) != sha256(old):
            raise ValueError("Proven native source file changed: " + old.name)
    root = reference / "engineering"
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    initial_planning = copy.deepcopy(planning)
    initial_planning["config"]["meta"]["seed"] = 1
    if (report["status"] != "full_native_pusht_engineering_passed" or
            report["protocol_sha256"] != sha256(root / "protocol.json") or
            digest != native["engineering_report_sha256"] or
            protocol["planning_contract"] != initial_planning or
            protocol["source_sha256"] != native["source_sha256"] or
            protocol["cohort_sha256"] != cohort_hash or
            protocol["checkpoint_sha256"] != CHECKPOINTS["pusht"] or
            protocol["smoke_seed"] != SMOKE_SEED or protocol["repetitions"] != 2 or
            protocol["precision"] != "float32_strict_no_tf32" or
            report["device_uuid"] != native["receiving_device_uuid"] or
            report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False or
            report["same_seed_actions_and_outcomes_exact"] is not True or len(report["repetition_sha256"]) != 2):
        raise ValueError("Original full native/repeat proof changed")
    rows = []
    for i, expected in enumerate(report["repetition_sha256"]):
        p = root / f"repetition-{i}.json"
        if sha256(p) != expected:
            raise ValueError("Original native engineering trace changed")
        row = json.loads(p.read_text())
        validate_complete(row["result"], [tuple(c) for c in row["unroll_calls"]])
        actions = np.asarray(row["planned_actions"])
        if actions.shape != (1, 6, 10) or not np.isfinite(actions).all():
            raise ValueError("Original native action trace changed")
        rows.append(row)
    if comparison_record(rows[0]["result"], action_hashes(rows[0]["planned_actions"])) != comparison_record(
            rows[1]["result"], action_hashes(rows[1]["planned_actions"])):
        raise ValueError("Original native repetitions differ")
    return native, rows[0], protocol


def load_native(reference, native, cohort, logical_ranks=None):
    """Complete assigned streams at execution; all eight at panel analysis."""
    ranks = list(range(8)) if logical_ranks is None else sorted(logical_ranks)
    expected_panel = assigned_rows(schedule(), ranks)
    rows, bindings = [], {}
    for rank in ranks:
        root = reference / "native" / f"shard-{rank}"
        report, digest = verified_report(root)
        launch = json.loads((root / "protocol.json").read_text())
        expected = assigned_rows(schedule(), [rank])
        names = [f"episode-{r['episode']:03d}.json" for r in expected]
        if (report["status"] != "native_pusht_planning_replication_shard_complete" or
                report["task"] != "pusht" or report["arm"] != "native" or report["episodes"] != 12 or
                report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False or
                report["protocol_sha256"] != sha256(root / "protocol.json") or
                launch != {"freeze_sha256": sha256(reference / "freeze/protocol.json"),
                    "source_sha256": native["source_sha256"], "task": "pusht", "arm": "native",
                    "logical_ranks": [rank], "expected_episodes": expected, "fresh_confirmation": False} or
                set(report["episode_files_sha256"]) != set(names)):
            raise ValueError("Unbound native Push-T shard")
        for name in names:
            if sha256(root / name) != report["episode_files_sha256"][name]:
                raise ValueError("Native Push-T episode changed")
            row = json.loads((root / name).read_text())
            if row["arm"] != "native":
                raise ValueError("Reference contains an intervention")
            rows.append(row)
        bindings[str(root / "report.json")] = digest
    validate_push_records(rows, expected_panel, cohort)
    return rows, bindings


def make_contract(planning, fit, bindings):
    names = [a["name"] for a in fit["arms"]]
    pairs = [c for c in fit["primary_contrasts"] if c["candidate"] != "zero_dose"]
    if (fit["category"] != "vision_action_coupling" or len(names) != 10 or
            set(names) != set(ARMS) | {"zero_dose"} or len(pairs) != 15 or
            len({c["name"] for c in pairs}) != 15 or
            any(c["candidate"] not in ARMS or c["control"] not in ARMS for c in pairs) or
            "geometry" in fit or "support_operator" in fit):
        raise ValueError("Require exactly the original full static coupling registry")
    return {"role": ROLE, "task": "pusht", "arms": list(ARMS), "source_arms": fit["arms"],
        "source_sha256": source_hash(), "planning": planning, "bindings": bindings,
        "episodes": schedule(), "episodes_per_condition_total": 96, "source_families_available": 21,
        "sampling": "unchanged upstream row/offset sampling and persistent12-episode RNG streams",
        "precision": "float32_strict_no_tf32", "fit_precision": "bfloat16",
        "candidate_admission_requires_offline_significance": False, "offline_outcomes_loaded": False,
        "fresh_confirmation": False, "replication_not_fresh_family_confirmation": True,
        "old_freezes_and_results_unchanged": True,
        "native_reference_gate": "complete assigned12-episode streams before candidates; all96 before analysis",
        "engineering_seed": SMOKE_SEED, "pairwise_contrasts": pairs,
        "factorial_success_interaction": INTERACTION,
        "primary_endpoint": "official simulated binary success after all30 elementary steps",
        "analysis": {"complete_panel_required": True, "interval_family_size": 16,
            "interval_method": "paired source-initial-state-family cluster bootstrap, Bonferroni95%",
            "point_estimate": "episode-weighted success percentage; equal-family descriptive sensitivity",
            "bootstrap_draws": 20000, "bootstrap_seed": 2026090801,
            "minimum_useful_success_gain_percentage_points": 5,
            "selection_from_partial_results": False, "automatic_confirmation_authorized": False},
        "zero_dose": "full native/source/episode identity engineering only",
        "secondary_endpoints": ["native reward", "native distance", "runtime", "actions", "edit energy"]}


def checked_inputs(args):
    use_vendor(args.vendor)
    cohort = checked_cohort(args.cohort)
    planning = prepare(args.vendor, "pusht")
    planning["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    native, smoke, smoke_protocol = verify_native_reference(
        args.reference, args.reference_code / "src/offline_study", sha256(args.cohort), planning)
    if input_hashes(args.data_root / "val") != native["input_files_sha256"]:
        raise ValueError("Actual released validation inputs changed")
    receipt, fit = verify_fit(args.fit, sha256(args.cohort), CHECKPOINTS["pusht"], "bfloat16")
    fit_groups = {r["lineage_group"] for r in cohort["fit"]}
    if (receipt["task"] != "pusht" or receipt["development_outcomes_accessed"] is not False or
            set(receipt["fit_lineage_groups"]) != fit_groups or len(fit_groups) != 128 or
            fit_groups & {r["lineage_group"] for r in cohort["evaluation"]} or
            smoke_protocol["fit_trajectory"] != cohort["fit"][0]):
        raise ValueError("Changed fitting-only population or overlap with validation")
    parity = json.loads((args.fit.parent / "PARITY.json").read_text())
    if (parity["fit_only"] is not True or parity["precision"] != "bfloat16" or
            len(parity["checks"]) != 2 or any(c["losses_match"] is not True or
                c["predictions_bitwise_equal"] is not True for c in parity["checks"])):
        raise ValueError("Original fit-only upstream parity is incomplete")
    for name, digest in smoke_protocol["fit_input_files_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(args.data_root / "train" / name) != digest:
            raise ValueError("Original fitting stimulus input changed")
    bindings = {"native_freeze_sha256": sha256(args.reference / "freeze/protocol.json"),
        "native_source_sha256": native["source_sha256"],
        "native_engineering_report_sha256": native["engineering_report_sha256"],
        "cohort_sha256": sha256(args.cohort), "checkpoint_sha256": CHECKPOINTS["pusht"],
        "fit": {name: sha256(args.fit / name) for name in FIT_FILES},
        "fit_parity_sha256": sha256(args.fit.parent / "PARITY.json")}
    return make_contract(planning, fit, bindings), fit, cohort, native, smoke


def read_contract(args):
    expected, fit, cohort, native, smoke = checked_inputs(args)
    p = args.freeze / "protocol.json"
    if (json.loads((args.freeze / "FROZEN.json").read_text())["protocol_sha256"] != sha256(p) or
            json.loads(p.read_text()) != expected):
        raise ValueError("Push-T frozen source/registry/analysis/input bindings changed")
    return expected, fit, cohort, native, smoke


def freeze(args):
    protocol, _, _, _, _ = checked_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
        "candidate_behavioral_outcomes_observed_before_freeze": False})


def validate_engineering(root, freeze_hash, source, reference_smoke):
    report, digest = verified_report(root)
    launch = json.loads((root / "protocol.json").read_text())
    names = ("native", "native", "zero_dose") + ARMS[1:]
    expected_files = [f"engineering-{i:02d}.json" for i in range(len(names))]
    if (report["status"] != "complete_pusht_coupling_engineering" or report["episodes"] != 11 or
            report["source_sha256"] != source or report["freeze_sha256"] != freeze_hash or
            report["protocol_sha256"] != sha256(root / "protocol.json") or
            report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False or
            report["scientific_efficacy_measurement"] is not False or
            launch["freeze_sha256"] != freeze_hash or launch["engineering_only"] is not True or
            launch["device_uuid"] != report["device_uuid"] or
            set(report["episode_files_sha256"]) != set(expected_files)):
        raise ValueError("Missing complete receiving-device factorial engineering")
    row_key = {"episode": 0, "logical_rank": 0, "local_seed": SMOKE_SEED, "environment_seed": SMOKE_SEED}
    for name, arm in zip(expected_files, names, strict=True):
        if sha256(root / name) != report["episode_files_sha256"][name]:
            raise ValueError("Receiving engineering episode changed")
        row = json.loads((root / name).read_text())
        validate_records([row], [row_key])
        if (row["arm"] != arm or row["source_parity_candidate_counts"] != [1, 300] or
                np.asarray(row["planned_actions"]).shape != (1, 6, 10) or any(
                    row["result"][k] != reference_smoke["result"][k] for k in ("initial_sha256", "goal_sha256"))):
            raise ValueError("Missing source parity or changed excluded fitting stimuli")
        if arm in ("native", "zero_dose") and comparison_record(
                row["result"], action_hashes(row["planned_actions"])) != comparison_record(
                    reference_smoke["result"], action_hashes(reference_smoke["planned_actions"])):
            raise ValueError("Receiving native/zero-dose actions or outcomes differ")
    return report, digest


@torch.no_grad()
def execute(args):
    protocol, fit, cohort, native_protocol, reference_smoke = read_contract(args)
    from app.plan_common.datasets.pusht_dset import PushTDataset
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from omegaconf import OmegaConf
    engineering = args.command == "engineer"
    proof_hash, native = None, None
    if not engineering:
        proof, proof_hash = validate_engineering(args.engineering, sha256(args.freeze / "protocol.json"),
                                                protocol["source_sha256"], reference_smoke)
        native_rows, _ = load_native(args.reference, native_protocol, cohort, args.logical_ranks)
        native = {row["episode"]: row for row in native_rows}
    torch.cuda.set_device(0)
    worker = device_uuid()
    if not engineering and worker != proof["device_uuid"]:
        raise ValueError("This GPU has no matching full coupling engineering proof")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        expected = None if engineering else assigned_rows(schedule(), args.logical_ranks)
        write_json(args.output / "protocol.json", {"freeze_sha256": sha256(args.freeze / "protocol.json"),
            "source_sha256": source_hash(), "task": "pusht", "arm": None if engineering else args.arm,
            "engineering_only": engineering, "engineering_report_sha256": proof_hash, "device_uuid": worker,
            "logical_ranks": None if engineering else args.logical_ranks, "expected_episodes": expected,
            "fresh_confirmation": False})
        # Same setup seed as the existing native Push-T evaluator. Each logical
        # stream then gets its own unchanged CPU/CUDA agent generators.
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["pusht"], "pusht", "cuda:0", "float32")
        if backend.model.ctxt_window != 2 or backend.provenance["normalization_dataset"] != "pusht":
            raise ValueError("Incorrect Push-T model/context/normalization")
        versions = _model_versions(backend.model)
        bank = torch.load(args.fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        if engineering:
            full = open_normalized_dataset("pusht", args.data_root, cohort["reference_config"], True)
            row = cohort["fit"][0]
            if (_initial_state_group(full.states[row["index"], 0]) != row["lineage_group"] or
                    full.get_seq_length(row["index"]) != row["length"]):
                raise ValueError("Fitting example differs from original native engineering")
            dataset = SingleFitTrajectory(full, row["index"])
        else:
            full = PushTDataset(data_path=str(args.data_root / "val"), n_rollout=None,
                transform=backend.preprocessor.transform, normalize_action=True, with_velocity=True)
            by_index = {r["index"]: r for r in cohort["evaluation"]}
            if len(full) != 21 or any(full.get_seq_length(i) != r["length"] or
                    _initial_state_group(full.states[i, 0]) != r["lineage_group"] for i, r in by_index.items()):
                raise ValueError("Actual released source population changed")
            dataset = TracedDataset(full, by_index)
        global_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        names = ("native", "native", "zero_dose") + ARMS[1:] if engineering else (args.arm,)
        records = {}
        torch.cuda.reset_peak_memory_stats()
        for index, arm in enumerate(names):
            for rank in ([None] if engineering else sorted(args.logical_ranks)):
                if engineering:
                    random.seed(SMOKE_SEED); np.random.seed(SMOKE_SEED); torch.manual_seed(SMOKE_SEED)
                    rows = [{"episode": 0, "logical_rank": 0, "local_seed": SMOKE_SEED, "environment_seed": SMOKE_SEED}]
                else:
                    random.setstate(global_rng[0]); np.random.set_state(global_rng[1]); torch.set_rng_state(global_rng[2])
                    rows = [r for r in expected if r["logical_rank"] == rank]
                cfg = OmegaConf.create(protocol["planning"]["config"])
                cfg.local_seed = rows[0]["local_seed"]
                if engineering:
                    cfg.meta.seed = SMOKE_SEED
                agent = GC_Agent(cfg, backend.model, dset=dataset, preprocessor=backend.preprocessor)
                env = make_env(cfg)
                try:
                    for row in rows:
                        if engineering:
                            record = one_episode(cfg, backend, agent, env, row, arm, fit, bank, True)
                            if any(record["result"][k] != reference_smoke["result"][k] for k in ("initial_sha256", "goal_sha256")):
                                raise ValueError("Original fit initial/goal inputs differ")
                            if arm in ("native", "zero_dose") and comparison_record(
                                    record["result"], action_hashes(record["planned_actions"])) != comparison_record(
                                        reference_smoke["result"], action_hashes(reference_smoke["planned_actions"])):
                                raise ValueError("Cannot reuse native reference: receiving actions/outcomes differ")
                        else:
                            with trace_segment(dataset) as segments:
                                record = one_episode(cfg, backend, agent, env, row, arm, fit, bank)
                            if len(segments) != 1:
                                raise ValueError("Expected one source replay segment")
                            record["source_segment"] = segments[0]
                            validate_push_records([record], [row], cohort)
                            paired_inputs(native[row["episode"]], record)
                        name = f"engineering-{index:02d}.json" if engineering else f"episode-{row['episode']:03d}.json"
                        write_json(args.output / name, record); records[name] = record
                        progress = {"arm": arm, "completed": len(records), "target": 11 if engineering else len(expected),
                            "engineering_only": engineering, "seconds": time.monotonic() - started}
                        write_json(args.output / "progress.json", progress); print(json.dumps(progress), flush=True)
                finally:
                    env.close()
        if not engineering:
            validate_push_records(list(records.values()), expected, cohort)
        if _model_versions(backend.model) != versions:
            raise ValueError("Frozen parameters changed")
        write_json(args.output / "report.json", {"status": "complete_pusht_coupling_engineering" if engineering else "complete_pusht_coupling_shard",
            "task": "pusht", "arm": None if engineering else args.arm, "source_sha256": source_hash(),
            "freeze_sha256": sha256(args.freeze / "protocol.json"), "device_uuid": worker,
            "protocol_sha256": sha256(args.output / "protocol.json"), "episodes": len(records),
            "episode_files_sha256": {name: sha256(args.output / name) for name in records},
            "parameters_unchanged": True, "scientific_efficacy_measurement": not engineering,
            "fresh_confirmation": False, "comparative_analysis_complete": False,
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
        if engineering:
            validate_engineering(args.output, sha256(args.freeze / "protocol.json"), source_hash(), reference_smoke)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


def load_panel(args):
    protocol, _, cohort, native_protocol, smoke = read_contract(args)
    native, bindings = load_native(args.reference, native_protocol, cohort)
    panels = {"native": native}
    freeze_hash = sha256(args.freeze / "protocol.json")
    bindings[str(args.freeze / "protocol.json")] = freeze_hash
    for arm in ARMS[1:]:
        shards = sorted((args.panel / "conditions" / arm).glob("shard-*"))
        if not shards:
            raise ValueError("Incomplete Push-T arm: " + arm)
        records = []
        for root in shards:
            report, digest = verified_report(root)
            launch = json.loads((root / "protocol.json").read_text())
            expected = assigned_rows(schedule(), launch["logical_ranks"])
            if (report["status"] != "complete_pusht_coupling_shard" or report["arm"] != arm or
                    report["task"] != "pusht" or report["source_sha256"] != protocol["source_sha256"] or
                    report["freeze_sha256"] != freeze_hash or report["parameters_unchanged"] is not True or
                    report["fresh_confirmation"] is not False or report["scientific_efficacy_measurement"] is not True or
                    report["protocol_sha256"] != sha256(root / "protocol.json") or
                    launch["freeze_sha256"] != freeze_hash or launch["arm"] != arm or
                    launch["expected_episodes"] != expected or launch["device_uuid"] != report["device_uuid"] or
                    launch["engineering_only"] is not False or launch["fresh_confirmation"] is not False):
                raise ValueError("Unbound Push-T candidate shard")
            proof, proof_hash = validate_engineering(args.panel / "engineering" / report["device_uuid"],
                                                     freeze_hash, protocol["source_sha256"], smoke)
            if proof_hash != launch["engineering_report_sha256"] or proof["device_uuid"] != report["device_uuid"]:
                raise ValueError("Unbound receiving-device proof")
            names = [f"episode-{r['episode']:03d}.json" for r in expected]
            if set(report["episode_files_sha256"]) != set(names) or report["episodes"] != len(names):
                raise ValueError("Incomplete or extra Push-T candidate episode")
            part = []
            for name in names:
                if sha256(root / name) != report["episode_files_sha256"][name]:
                    raise ValueError("Candidate record changed")
                row = json.loads((root / name).read_text())
                if row["arm"] != arm:
                    raise ValueError("Incorrect treatment label")
                part.append(row)
            validate_push_records(part, expected, cohort); records.extend(part)
            bindings[str(root / "report.json")] = digest
            bindings[str(args.panel / "engineering" / report["device_uuid"] / "report.json")] = proof_hash
        records.sort(key=lambda r: r["episode"])
        validate_push_records(records, schedule(), cohort)
        for baseline, row in zip(native, records, strict=True):
            paired_inputs(baseline, row)
        panels[arm] = records
    return panels, protocol, bindings


def analyze(panels, protocol):
    if tuple(panels) != ARMS or any(len(rows) != 96 for rows in panels.values()):
        raise ValueError("Require the complete nine-arm 96-total Push-T panel")
    if protocol["factorial_success_interaction"] != INTERACTION or len(protocol["pairwise_contrasts"]) != 15:
        raise ValueError("Fixed Push-T contrast family changed")
    groups = {}
    for i, row in enumerate(panels["native"]):
        groups.setdefault(row["source_segment"]["lineage_group"], []).append(i)
    if not 2 <= len(groups) <= 21:
        raise ValueError("Insufficient or unrecognized source initial-state families")
    clusters = list(groups.values())
    sizes = np.array([len(g) for g in clusters])
    rng = np.random.default_rng(2026090801)
    draws = rng.integers(len(groups), size=(20000, len(groups)))
    denominator = sizes[draws].sum(1)
    success, summaries = {}, {}
    for arm, rows in panels.items():
        for native, row in zip(panels["native"], rows, strict=True):
            paired_inputs(native, row)
        if any(not isinstance(r["result"]["native_success"], bool) for r in rows):
            raise ValueError("Nonbinary task-success endpoint")
        values = np.array([r["result"]["native_success"] for r in rows], dtype=float)
        success[arm] = values
        summaries[arm] = {"episodes": 96, "successes": int(values.sum()),
            "success_percent": float(values.mean() * 100), "source_families": len(groups),
            "equal_family_success_percent": float(np.mean([values[g].mean() for g in clusters]) * 100),
            "mean_episode_seconds": float(np.mean([r["seconds"] for r in rows])),
            "runtime_hardware_parity_not_assumed": True}
    contrasts = [(c["name"], success[c["candidate"]] - success[c["control"]])
                 for c in protocol["pairwise_contrasts"]]
    contrasts.append(("factorial_success_interaction", sum(success[a] * w for a, w in INTERACTION.items())))
    results = []
    for name, delta in contrasts:
        totals = np.array([delta[g].sum() for g in clusters])
        sampled = totals[draws].sum(1) / denominator * 100
        lo, hi = np.quantile(sampled, [.05 / 32, 1 - .05 / 32])
        # Cluster influence for the episode-weighted mean; only a development
        # sensitivity, not a power guarantee from post-hoc observed success.
        residual = totals - delta.mean() * sizes
        se = float(np.sqrt(len(groups) / (len(groups) - 1) * np.sum(residual ** 2)) / 96 * 100)
        sensitivity = (NormalDist().inv_cdf(1 - .05 / 32) + NormalDist().inv_cdf(.8)) * se
        results.append({"contrast": name, "difference_percentage_points": float(delta.mean() * 100),
            "simultaneous_95_interval_percentage_points": [float(lo), float(hi)],
            "source_families": len(groups), "paired_positive_episodes": int((delta > 0).sum()),
            "paired_negative_episodes": int((delta < 0).sum()),
            "approximate_80pct_power_detectable_effect_percentage_points": sensitivity if se > 0 else None,
            "detectable_effect_is_normal_approximation_not_power_proof": True,
            "zero_discordance_does_not_establish_equivalence": True})
    return {"status": "complete_pusht_coupling_planning_development_analysis", "task": "pusht",
        "arm_summaries": summaries, "contrasts": results,
        "source_family_draw_counts": {key: len(value) for key, value in groups.items()},
        "sampling_unit": "original released source initial-state family, not sampled segment",
        "bootstrap_draws": 20000, "simultaneous_interval_family_size": 16,
        "small_source_family_count_limits_interval_accuracy": True,
        "fresh_confirmation": False, "training_seed_history_complete": False,
        "offline_scores_used_for_admission": False, "selection_from_partial_results": False,
        "automatic_confirmation_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "engineer", "run", "analyze"))
    for name in ("vendor", "reference", "reference-code", "fit", "cohort", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze", "checkpoint", "engineering", "panel"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--arm", choices=ARMS[1:])
    parser.add_argument("--logical-ranks", nargs="+", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "analyze":
        if args.freeze is None or args.panel is None:
            parser.error("Analysis needs --freeze and --panel")
        panels, protocol, bindings = load_panel(args)
        report = analyze(panels, protocol)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "bindings.json", bindings)
        report["bindings_sha256"] = sha256(args.output / "bindings.json")
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    else:
        if args.freeze is None or args.checkpoint is None:
            parser.error("Execution needs --freeze and --checkpoint")
        if args.command == "run" and (args.engineering is None or args.arm is None or not args.logical_ranks):
            parser.error("Candidates need complete engineering, arm and logical streams")
        execute(args)


if __name__ == "__main__":
    main()
