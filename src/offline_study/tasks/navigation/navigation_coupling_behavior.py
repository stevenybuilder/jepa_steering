"""Complete the existing coupling factorial in actual navigation planning.

This is an additive development panel, not a rewrite of the earlier seven-arm
freeze. Offline outcomes are never loaded or used to admit arms. Existing native
episodes may be reused only with verified provenance, full native engineering on
the receiving worker, and exact initial/goal pairing. No dynamic support operator
or protected confirmation access is implemented here.
"""
from offline_study._paths import source_path, snapshot_source_files
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, verified_report
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.interventions.interventions import PredictorIntervention
from offline_study.tasks.navigation.navigation_replication import validate_records
from offline_study.tasks.navigation.navigation_smoke import CHECKPOINTS, comparison_record, validate_complete
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_intervention import StaticPlanningIntervention
from offline_study.planning.planning_native_smoke import SMOKE_SEED, run_episode
from offline_study.planning.planning_support_check import source_coupling_fields
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


ARMS = ("native", "visual_only", "action_condition_only", "joint",
        "joint_equal_standardized_energy", "permuted_visual", "permuted_joint",
        "matched_random", "matched_random_equal_standardized_energy")
TASKS = ("wall", "pointmaze")
SHARED_SOURCE = ("backends.py", "model_loader.py", "planning_native_smoke.py",
                 "planning_contract.py", "navigation_replication.py", "navigation_smoke.py")


def source_digest(root):
    value = hashlib.sha256()
    for path in snapshot_source_files(root):
        value.update(path.resolve().relative_to(root.resolve()).as_posix().encode() + b"\0" + path.read_bytes())
    return value.hexdigest()


def paired_inputs(native, candidate):
    for key in ("initial_sha256", "goal_sha256"):
        if native["result"][key] != candidate["result"][key]:
            raise ValueError("Native/candidate input mismatch: " + key)


def load_native(root, freeze):
    protocol = json.loads((freeze / "protocol.json").read_text())
    freeze_hash = sha256(freeze / "protocol.json")
    if (json.loads((freeze / "FROZEN.json").read_text())["protocol_sha256"] != freeze_hash or
            protocol["episodes"] != schedule() or protocol["task"] not in TASKS or
            protocol["role"] != "frozen_navigation_native_planning_development_not_confirmation"):
        raise ValueError("Native reference freeze changed")
    report, report_hash = verified_report(root)
    launch = json.loads((root / "protocol.json").read_text())
    if (report["status"] != "native_navigation_development_shard_complete" or
            report["episodes"] != 96 or report["arm"] != "native" or
            report["task"] != protocol["task"] or report["parameters_unchanged"] is not True or
            report["fresh_confirmation"] is not False or
            report["protocol_sha256"] != sha256(root / "protocol.json") or
            launch["freeze_sha256"] != freeze_hash or
            launch["source_sha256"] != protocol["source_sha256"]):
        raise ValueError("Unverified complete native reference")
    records = []
    for row in schedule():
        name = f"episode-{row['episode']:03d}.json"
        if report["episode_files_sha256"].get(name) != sha256(root / name):
            raise ValueError("Native episode checksum mismatch")
        record = json.loads((root / name).read_text())
        if record["arm"] != "native":
            raise ValueError("Non-native reference")
        records.append(record)
    if len(report["episode_files_sha256"]) != 96:
        raise ValueError("Extra native episodes")
    validate_records(records, schedule())
    return protocol, records, report_hash


def make_contract(task, planning, fit_protocol, bindings):
    names = tuple(arm["name"] for arm in fit_protocol["arms"])
    if (task not in TASKS or fit_protocol["category"] != "vision_action_coupling" or
            set(names) != set(ARMS) | {"zero_dose"} or len(names) != len(ARMS) + 1 or
            "support_operator" in fit_protocol or "geometry" in fit_protocol):
        raise ValueError("Only the complete pre-existing static coupling registry is permitted")
    contrasts = [row for row in fit_protocol["primary_contrasts"]
                 if row["candidate"] != "zero_dose"]
    if len(contrasts) != 15 or any(row["candidate"] not in ARMS or row["control"] not in ARMS
                                  for row in contrasts):
        raise ValueError("Unexpected original contrast family")
    return {"role": "full_static_coupling_navigation_development", "task": task,
        "arms": list(ARMS), "source_arms": fit_protocol["arms"], "bindings": bindings,
        "source_sha256": source_hash(), "planning": planning, "episodes": schedule(),
        "episodes_per_task_condition": 96, "physical_gpu_count_changes_sample_size": False,
        "precision": "float32_strict_no_tf32", "fit_precision": "bfloat16",
        "candidate_admission_requires_offline_significance": False,
        "offline_outcomes_loaded": False, "old_native_reference_already_observed": True,
        "old_freezes_and_results_unchanged": True, "fresh_confirmation": False,
        "reserved_confirmation_base_seed": 1, "engineering_seed": SMOKE_SEED,
        "primary_endpoint": "official simulated task success after all 30 elementary steps",
        "pairwise_contrasts": contrasts,
        "factorial_success_interaction": {"joint": 1, "visual_only": -1,
                                           "action_condition_only": -1, "native": 1},
        "analysis": {"complete_panel_required": True, "tasks": list(TASKS),
            "interval_family_size": 32, "interval_method": "paired initial/goal cluster bootstrap, Bonferroni 95%",
            "bootstrap_draws": 20000, "bootstrap_seed": 2026090801,
            "minimum_useful_success_gain_percentage_points": 5,
            "selection_from_partial_results": False,
            "automatic_confirmation_authorized": False,
            "interpretation": "planning development; interaction is not activation-level nonlinearity"},
        "secondary_endpoints": ["native distance", "native reward", "runtime", "planned actions", "delivered edit energy"],
        "zero_dose": "bitwise source and complete-episode engineering, not a redundant scientific arm"}


def reference_paths(args):
    return args.reference / args.task


def freeze(args):
    use_vendor(args.vendor)
    ref = reference_paths(args)
    native, _, native_hash = load_native(ref / "native", ref / "freeze")
    old_code = ref / "baseline-source/src/offline_study"
    if source_digest(old_code) != native["source_sha256"]:
        raise ValueError("Native source snapshot does not match its freeze")
    for name in SHARED_SOURCE:
        if sha256(old_code / name) != sha256(source_path(name)):
            raise ValueError("Native execution path changed: " + name)
    planning = prepare(args.vendor, args.task)
    planning["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    if planning != native["planning_contract"]:
        raise ValueError("Native task/planner settings changed")
    cohort = json.loads(args.cohort.read_text())
    receipt, fit = verify_fit(args.fit, sha256(args.cohort), CHECKPOINTS[args.task], "bfloat16")
    if (cohort["task"] != args.task or receipt["development_outcomes_accessed"] is not False or
            set(r["lineage_group"] for r in cohort["fit"]) &
            set(r["lineage_group"] for r in cohort["evaluation"])):
        raise ValueError("Task/fit exposure mismatch")
    smoke, smoke_hash = verified_report(ref / "smoke")
    sp = json.loads((ref / "smoke/protocol.json").read_text())
    if (smoke["task"] != args.task or smoke["status"] != "full_native_navigation_engineering_passed" or
            smoke["protocol_sha256"] != sha256(ref / "smoke/protocol.json") or
            sp["checkpoint_sha256"] != CHECKPOINTS[args.task] or
            smoke_hash != native["engineering_report_sha256"] or
            not smoke["same_seed_actions_and_outcomes_exact"]):
        raise ValueError("Unbound original native smoke")
    for i, digest in enumerate(smoke["repetition_sha256"]):
        if sha256(ref / f"smoke/repetition-{i}.json") != digest:
            raise ValueError("Original smoke episode changed")
    bindings = {"native_report_sha256": native_hash, "native_freeze_sha256": sha256(ref / "freeze/protocol.json"),
        "native_smoke_report_sha256": smoke_hash, "checkpoint_sha256": CHECKPOINTS[args.task],
        "cohort_sha256": sha256(args.cohort),
        "fit": {name: sha256(args.fit / name) for name in ("protocol.json", "operator_bank.pt", "fit_receipt.json", "DONE.json")}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", make_contract(args.task, planning, fit, bindings))
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
        "candidate_behavioral_outcomes_observed_before_freeze": False})


def read_contract(args):
    protocol = json.loads((args.freeze / "protocol.json").read_text())
    if (sha256(args.freeze / "protocol.json") != json.loads((args.freeze / "FROZEN.json").read_text())["protocol_sha256"] or
            protocol["source_sha256"] != source_hash() or protocol["task"] != args.task or
            protocol["role"] != "full_static_coupling_navigation_development" or
            protocol["candidate_admission_requires_offline_significance"] is not False):
        raise ValueError("Behavioral freeze/source changed")
    _, source = verify_fit(args.fit, sha256(args.cohort), CHECKPOINTS[args.task], "bfloat16")
    for name, digest in protocol["bindings"]["fit"].items():
        if sha256(args.fit / name) != digest:
            raise ValueError("Frozen fit changed")
    planning = prepare(args.vendor, args.task)
    planning["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    if protocol != make_contract(args.task, planning, source, protocol["bindings"]):
        raise ValueError("Behavioral registry/analysis changed")
    _, native, native_hash = load_native(reference_paths(args) / "native", reference_paths(args) / "freeze")
    if native_hash != protocol["bindings"]["native_report_sha256"]:
        raise ValueError("Frozen native reference changed")
    return protocol, source, native


def action_hashes(actions):
    result = []
    for actions_one_plan in actions:
        value = torch.tensor(actions_one_plan, dtype=torch.float32).contiguous()
        result.append(hashlib.sha256(str(value.shape).encode() + value.numpy().tobytes()).hexdigest())
    return result


def validate_engineering(root, protocol, freeze_hash):
    proof, proof_hash = verified_report(root)
    names = ("native", "native", "zero_dose") + ARMS[1:]
    expected_files = [f"engineering-{i:02d}.json" for i in range(len(names))]
    launch = json.loads((root / "protocol.json").read_text())
    if (proof["status"] != "complete_coupling_navigation_engineering" or
            proof["freeze_sha256"] != freeze_hash or proof["source_sha256"] != source_hash() or
            proof["task"] != protocol["task"] or proof["episodes"] != len(names) or
            proof["parameters_unchanged"] is not True or proof["fresh_confirmation"] is not False or
            proof["scientific_efficacy_measurement"] is not False or
            proof["protocol_sha256"] != sha256(root / "protocol.json") or
            launch["freeze_sha256"] != freeze_hash or launch["engineering_only"] is not True or
            set(proof["episode_files_sha256"]) != set(expected_files)):
        raise ValueError("Missing complete factorial native/source/CEM engineering")
    native = None
    for filename, arm in zip(expected_files, names, strict=True):
        if sha256(root / filename) != proof["episode_files_sha256"][filename]:
            raise ValueError("Engineering trace changed")
        record = json.loads((root / filename).read_text())
        if (record["arm"] != arm or record["environment_seed"] != SMOKE_SEED or
                record["source_parity_candidate_counts"] != [1, 300]):
            raise ValueError("Engineering arm/seed/source-parity coverage changed")
        validate_records([record], [{"episode": 0, "logical_rank": 0,
            "local_seed": SMOKE_SEED, "environment_seed": SMOKE_SEED}])
        if native is None:
            native = record
        paired_inputs(native, record)
        if arm in ("native", "zero_dose") and comparison_record(
                record["result"], action_hashes(record["planned_actions"])) != comparison_record(
                    native["result"], action_hashes(native["planned_actions"])):
            raise ValueError("Native/zero-dose full-episode identity failed")
    return proof_hash


@torch.no_grad()
def one_episode(cfg, backend, agent, env, row, arm, fit, bank, check_source=False):
    before = time.monotonic()
    adapter = StaticPlanningIntervention(backend, fit, bank, arm)
    calls, actions, checked_counts = [], [], set()
    original_act = agent.act
    original_unroll = agent.planner.unroll

    def observed(context, act_suffix=None, **kwargs):
        result = adapter(context, act_suffix=act_suffix, **kwargs)
        if any(not torch.isfinite(result[k]).all() for k in ("visual", "proprio")):
            raise ValueError("Nonfinite forecast")
        horizon, count = act_suffix.shape[:2]
        calls.append((horizon, count))
        if check_source and count not in checked_counts:
            if arm in ("native", "zero_dose"):
                expected = backend.predict(context, act_suffix)
            else:
                fields = source_coupling_fields((fit, bank, arm), count, backend.device)
                with PredictorIntervention(backend.predictor, fields):
                    expected = backend.predict(context, act_suffix)
            if any(not torch.equal(result[k], expected[k]) for k in ("visual", "proprio")):
                raise ValueError("Independent original compiler/native parity failed")
            checked_counts.add(count)
        return result

    def act(*a, **kw):
        result = original_act(*a, **kw)
        actions.append(result.detach().cpu().tolist())
        return result

    agent.planner.unroll, agent.act = observed, act
    try:
        result = run_episode(cfg, backend, agent, env, row["environment_seed"])
    finally:
        agent.planner.unroll, agent.act = original_unroll, original_act
    record = {**row, "arm": arm, "result": result, "unroll_calls": calls,
        "planned_actions": actions, "edit_energy": adapter.energy,
        "source_parity_candidate_counts": sorted(checked_counts),
        "seconds": time.monotonic() - before}
    validate_records([record], [row])
    if check_source and checked_counts != {1, 300}:
        raise ValueError("Source parity did not cover candidates and CEM means")
    return record


@torch.no_grad()
def execute(args):
    use_vendor(args.vendor)
    protocol, fit, native = read_contract(args)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    engineering = args.command == "engineer"
    proof_hash = None
    if not engineering:
        proof_hash = validate_engineering(args.engineering, protocol, sha256(args.freeze / "protocol.json"))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"freeze_sha256": sha256(args.freeze / "protocol.json"),
            "task": args.task, "arm": None if engineering else args.arm,
            "engineering_report_sha256": proof_hash, "source_sha256": source_hash(),
            "fresh_confirmation": False, "engineering_only": engineering,
            "logical_ranks": None if engineering else args.logical_ranks,
            "expected_episodes": None if engineering else assigned_rows(schedule(), args.logical_ranks)})
        torch.cuda.set_device(0)
        random.seed(DEVELOPMENT_SEED)
        np.random.seed(DEVELOPMENT_SEED)
        torch.manual_seed(DEVELOPMENT_SEED)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS[args.task], args.task, "cuda:0", "float32")
        versions = _model_versions(backend.model)
        bank = torch.load(args.fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        records = {}
        names = ("native", "native", "zero_dose") + ARMS[1:] if engineering else (args.arm,)
        for index, arm in enumerate(names):
            streams = [None] if engineering else sorted(args.logical_ranks)
            expected = schedule() if engineering else assigned_rows(schedule(), args.logical_ranks)
            for stream in streams:
                if engineering:
                    random.seed(SMOKE_SEED)
                    np.random.seed(SMOKE_SEED)
                    torch.manual_seed(SMOKE_SEED)
                    rows = [{"episode": 0, "logical_rank": 0, "local_seed": SMOKE_SEED,
                             "environment_seed": SMOKE_SEED}]
                else:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                    rows = [r for r in expected if r["logical_rank"] == stream]
                cfg = OmegaConf.create(protocol["planning"]["config"])
                cfg.local_seed = rows[0]["local_seed"]
                if engineering:
                    cfg.meta.seed = SMOKE_SEED
                agent = GC_Agent(cfg, backend.model, dset=None, preprocessor=backend.preprocessor)
                env = make_env(cfg)
                try:
                    for row in rows:
                        record = one_episode(cfg, backend, agent, env, row, arm, fit, bank, engineering)
                        if engineering:
                            original = json.loads((reference_paths(args) / "smoke/repetition-0.json").read_text())
                            paired_inputs(original, record)
                            if arm in ("native", "zero_dose"):
                                if comparison_record(record["result"], action_hashes(record["planned_actions"])) != comparison_record(original["result"], original["action_sha256"]):
                                    raise ValueError("Receiving-worker native actions/outcomes differ; cannot reuse old native reference")
                        else:
                            paired_inputs(native[row["episode"]], record)
                        name = f"engineering-{index:02d}.json" if engineering else f"episode-{row['episode']:03d}.json"
                        records[name] = record
                        write_json(args.output / name, record)
                        progress = {"task": args.task, "arm": arm, "completed": len(records),
                            "target": len(names) if engineering else len(expected),
                            "engineering_only": engineering, "seconds": time.monotonic() - started}
                        write_json(args.output / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                finally:
                    env.close()
        if not engineering:
            validate_records(list(records.values()), expected)
        if _model_versions(backend.model) != versions:
            raise ValueError("Frozen weights changed")
        write_json(args.output / "report.json", {"status": "complete_coupling_navigation_engineering" if engineering else "complete_coupling_navigation_shard",
            "task": args.task, "arm": None if engineering else args.arm,
            "freeze_sha256": sha256(args.freeze / "protocol.json"), "source_sha256": source_hash(),
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "episode_files_sha256": {name: sha256(args.output / name) for name in records},
            "episodes": len(records), "parameters_unchanged": True, "fresh_confirmation": False,
            "scientific_efficacy_measurement": not engineering, "comparative_analysis_complete": False,
            "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda},
            "seconds": time.monotonic() - started})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "engineer", "run"))
    for name in ("vendor", "reference", "fit", "cohort", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze", "checkpoint", "engineering"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--arm", choices=ARMS[1:])
    parser.add_argument("--logical-ranks", nargs="+", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    else:
        if args.freeze is None or args.checkpoint is None:
            parser.error("Execution requires --freeze and --checkpoint")
        if args.command == "run" and (args.engineering is None or args.arm is None or not args.logical_ranks):
            parser.error("Candidate execution requires complete engineering, arm and logical streams")
        execute(args)


if __name__ == "__main__":
    main()
