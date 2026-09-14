"""Separate five-arm behavioral development for the approved fixed-response edit.

Uses the existing 96 canonical development stimuli, never the protected base-1
pool. Old seven-arm freezes/results remain immutable. Offline scores are not gates.
"""
from offline_study._paths import PACKAGE_ROOT, source_files
from offline_study._paths import source_path
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_candidate import immutable_source_hash
from offline_study.evaluation.behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, validate_coverage, verified_report
from offline_study.interventions.fixed_response import METHOD, FixedResponseIntervention, load_fitted_bank
from offline_study.validation.fixed_response_smoke import EPISODES, trace_actor, verify_episode, verify_pair
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_goal_bank import GoalBank, verify_delivery
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.planning.planning_panel_engineering import COUPLING_ARMS, H6StaticPlanningIntervention
from offline_study.planning.planning_scenarios import close_expert_environments
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor

TASKS = ("reach", "reach-wall")
ARMS = ("native", "fixed_rank4", "matched_random_fixed_rank4", "coupling_only", "matched_random_coupling")
CONTRASTS = (("fixed_rank4", "native"), ("fixed_rank4", "matched_random_fixed_rank4"),
             ("coupling_only", "native"), ("coupling_only", "matched_random_coupling"))
ROLE = "fixed_response_five_arm_planning_development_not_confirmation"
ANALYSIS = {"replicates": 20000, "seed": 2026090722, "family": 8,
    "intervals": "paired scenario-cluster percentile bootstrap, Bonferroni simultaneous 95 percent",
    "p_values": "exact paired discordance when cluster-homogeneous; Holm across eight",
    "minimum_useful_success_gain_percentage_points": 5,
    "selection": "positive native and random simultaneous lower bounds, native gain >=5pp; highest native lower bound then runtime then arm name",
    "complete_two_task_five_arm_panel_required": True}


def frozen_protocol(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if sha256(root / "protocol.json") != json.loads((root / "FROZEN.json").read_text())["protocol_sha256"]:
        raise ValueError("Changed scientific freeze")
    return protocol


def checked_source(root, expected):
    if immutable_source_hash(root) != expected:
        raise ValueError("Checked immutable source snapshot changed")
    for path in (source_files(root) if (root / "_paths.py").is_file() else root.glob("*.py")):
        if sha256(path) != sha256(source_path(path.resolve().relative_to(root.resolve()))):
            raise ValueError("Previously checked runtime changed: " + path.name)


def verify_cem(root, fit, source, task):
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if ((root / "FAILED.json").exists() or
            report.get("status") != "fixed_response_full_cem_engineering_complete" or
            report.get("task") != task or report.get("method") != METHOD or
            report.get("protocol_sha256") != sha256(root / "protocol.json") or
            report.get("full_simulator_episodes") != 4 or report.get("scenario_count") != 1 or
            protocol.get("fit_done_sha256") != sha256(fit / "DONE.json") or
            protocol.get("fit_bank_sha256") != sha256(fit / "operator_bank.pt") or
            protocol.get("engineering_seed") != SMOKE_SEED or protocol.get("episode_order") != list(EPISODES) or
            report.get("backend", {}).get("checkpoint_sha256") != CHECKPOINTS["metaworld"] or
            report.get("scientific_efficacy_measurement") is not False or
            not all(report.get(k) is True for k in ("native_repeat_exact", "paired_stimuli_exact",
                "full_cem_schedule_verified", "parameters_unchanged"))):
        raise ValueError("Missing bound complete fixed-map CEM engineering")
    checked_source(source, protocol["source_sha256"])
    reference = None
    for name in EPISODES:
        path = root / name
        if sha256(path / "DONE.json") != report["episodes"][name]:
            raise ValueError("CEM episode receipt changed")
        episode, _ = verified_report(path)
        for filename, key in (("unroll_calls.json", "unroll_calls_sha256"), ("action_trace.json", "action_trace_sha256")):
            if sha256(path / filename) != episode[key]:
                raise ValueError("CEM trace changed")
        calls = json.loads((path / "unroll_calls.json").read_text())
        verify_episode(episode["result"], calls)
        for call in calls:
            audit = call["record"]
            if audit["backend_calls"] != 1 or audit["response_probe_rollouts"] != 0 or audit["native_shadow_rollouts"] != 0:
                raise ValueError("Extra online forecasting in checked method")
        current = {"result": episode["result"], "action_trace": json.loads((path / "action_trace.json").read_text())}
        if reference is None:
            reference = current
        else:
            verify_pair(reference, current, native_repeat=name == "native_repeat")
    return digest


def coupling_binding(original, legacy, task):
    root = original / "fits-v1/bfloat16" / task / "vision_action_coupling"
    spec = legacy["tasks"][task]
    _, protocol = verify_fit(root, spec["cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
    for name, digest in spec["components"]["vision_action_coupling"].items():
        if sha256(root / name) != digest:
            raise ValueError("Legacy coupling fit changed")
    return root, protocol, {name: sha256(root / name) for name in ("DONE.json", "protocol.json", "operator_bank.pt", "fit_receipt.json")}


def stimulus_contract(stimuli, task):
    old = stimuli / "behavioral-development-freeze-20260907-v1"
    legacy = frozen_protocol(old)
    if legacy["episodes"] != schedule() or legacy["development_base_seed"] != DEVELOPMENT_SEED:
        raise ValueError("Only existing development streams are authorized")
    old_hash = sha256(old / "protocol.json")
    goals = GoalBank(stimuli / "metaworld-native-goal-bank-20260907-v1", task, old_hash)
    delivery = verify_delivery(stimuli / "metaworld-goal-delivery-check-20260907-v2", goals.report_hash, old_hash)
    # Read only historical input tensors, never old candidate outcomes.
    for row in schedule():
        goals.load(row)
    return legacy, goals, {"legacy_stimulus_freeze_sha256": old_hash,
        "goal_bank_report_sha256": goals.report_hash, "delivery_report_sha256": delivery}


def freeze(args):
    tasks = {}
    for task in TASKS:
        fit = args.fits / task
        load_fitted_bank(fit, task="mw-" + task, checkpoint_sha256=CHECKPOINTS["metaworld"])
        legacy, _, stimuli = stimulus_contract(args.stimuli, task)
        _, _, coupling = coupling_binding(args.original, legacy, task)
        planning = prepare(args.vendor, task)
        planning["config"]["meta"]["seed"] = DEVELOPMENT_SEED
        if planning["config"] != legacy["tasks"][task]["planning"]["config"]:
            raise ValueError("Author planning configuration changed")
        tasks[task] = {"planning": planning, "stimuli": stimuli, "coupling": coupling,
            "fit_done_sha256": sha256(fit / "DONE.json"), "fit_bank_sha256": sha256(fit / "operator_bank.pt")}
    protocol = {"role": ROLE, "method": METHOD, "source_sha256": source_hash(), "tasks": tasks,
        "checkpoint_sha256": CHECKPOINTS["metaworld"], "arms": list(ARMS),
        "primary_contrasts_per_task": [list(c) for c in CONTRASTS], "analysis": ANALYSIS,
        "episodes": schedule(), "episodes_per_task_condition": 96,
        "development_base_seed": DEVELOPMENT_SEED, "reserved_confirmation_base_seed": 1,
        "requested_tasks": 6, "this_panel_tasks": 2, "training_history_complete": False,
        "offline_significance_required": False, "fresh_confirmation": False,
        "confirmation_outcomes_authorized": False, "legacy_results_reused_as_new_method": False,
        "combined_successor_enabled": False, "precision": "float32_strict_no_tf32",
        "edit_schedule": "H3 only inside full H6; true native for every shortened horizon",
        "primary_endpoint": "official simulated task success after all100 elementary steps",
        "legacy_result_keys": "native_success/distance/reward name the executing arm, as identified by arm field",
        "native_baseline": "rerun on the new worker class and exact same canonical stimuli as every arm",
        "launch_gates": "complete fixed-map CEM and receiving-worker native repeat; coupling source/full-CEM receipts; no partial selection"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
        "successor_behavioral_outcomes_observed_before_freeze": False,
        "existing_population_previously_used_for_development": True})


def validate_protocol(root):
    protocol = frozen_protocol(root)
    if (protocol.get("role") != ROLE or protocol.get("method") != METHOD or
            protocol.get("arms") != list(ARMS) or protocol.get("episodes") != schedule() or
            protocol.get("analysis") != ANALYSIS or tuple(protocol.get("tasks", {})) != TASKS or
            protocol.get("primary_contrasts_per_task") != [list(c) for c in CONTRASTS] or
            protocol.get("episodes_per_task_condition") != 96 or
            protocol.get("confirmation_outcomes_authorized") is not False):
        raise ValueError("Unrecognized or changed fixed-response behavioral registry")
    return protocol


class ObservePlanner:
    """Observe both adapters; count actual model.unroll calls independently."""
    def __init__(self, adapter, backend, output):
        self.adapter, self.backend, self.output = adapter, backend, output
        self.calls = []

    def __call__(self, context, act_suffix=None, **kwargs):
        horizon, candidates, _ = act_suffix.shape
        write_json(self.output / "progress.json", {"completed_calls": len(self.calls),
            "running_horizon": horizon, "running_candidates": candidates, "fresh_confirmation": False})
        model, count, started = self.backend.model, 0, time.monotonic()
        original = model.unroll
        def counted(*args, **kw):
            nonlocal count
            count += 1
            return original(*args, **kw)
        model.unroll = counted
        try:
            result = self.adapter(context, act_suffix, **kwargs)
        finally:
            model.unroll = original
        if count != 1 or any(not torch.isfinite(result[k]).all() for k in ("visual", "proprio")):
            raise ValueError("Extra forecast or nonfinite planning output")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(self.adapter, "last_record"):
            energy = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                      for k, v in self.adapter.last_record.items()}
        else:
            energy = dict(self.adapter.energy[-1])
        self.calls.append({"horizon": horizon, "candidates": candidates, "backend_calls": count,
            "seconds": time.monotonic() - started, "energy": energy})
        return result


def verify_coupling_engineering(root, fit, task, arm):
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if ((root / "FAILED.json").exists() or report.get("status") != "common_h6_coupling_panel_full_cem_engineering_passed" or
            report.get("task") != task or report.get("arm") != arm or
            report.get("protocol_sha256") != sha256(root / "protocol.json") or
            protocol.get("fit_protocol_sha256") != sha256(fit / "protocol.json") or
            protocol.get("bank_sha256") != sha256(fit / "operator_bank.pt")):
        raise ValueError("Missing source-bound coupling engineering")
    for name, key in (("fit_parity.json", "fit_parity_sha256"), ("unroll_calls.json", "unroll_calls_sha256")):
        if sha256(root / name) != report[key]:
            raise ValueError("Coupling engineering trace changed")
    for name, digest in protocol["source_sha256"].items():
        if sha256(source_path(name)) != digest:
            raise ValueError("Checked coupling implementation changed")
    return sha256(root / "report.json")


def device_uuid():
    # CUDA_VISIBLE_DEVICES maps logical0 onto a physical index/UUID. Ask CUDA,
    # not nvidia-smi's physical0, so independently sharded workers remain valid.
    return str(torch.cuda.get_device_properties(0).uuid)


def worker_engineering(args):
    """Bind a complete excluded-scenario check to the actual executing device."""
    if args.output.exists():
        raise ValueError("Receiving-worker check must use a new output root")
    before = device_uuid()
    code = source_hash()
    command = [sys.executable, "-u", "-m", "offline_study.validation.fixed_response_smoke",
        "--vendor", str(args.vendor), "--checkpoint", str(args.checkpoint),
        "--fit", str(args.fits / args.task), "--numerical-check", str(args.numerical_check),
        "--checked-source", str(args.checked_source), "--task", args.task, "--output", str(args.output)]
    subprocess.run(command, check=True)
    digest = verify_cem(args.output, args.fits / args.task, PACKAGE_ROOT, args.task)
    if before != device_uuid() or code != source_hash():
        raise ValueError("Receiving device/source changed during engineering")
    write_json(args.output / "WORKER.json", {"cem_report_sha256": digest, "device_uuid": before,
        "same_device_before_after": True, "source_sha256": code,
        "native_repeat_was_full_episode": True, "fresh_confirmation": False})


@torch.no_grad()
def run(args):
    protocol = validate_protocol(args.freeze)
    if protocol["source_sha256"] != source_hash():
        raise ValueError("Behavioral source changed after freezing")
    expected = assigned_rows(protocol["episodes"], args.logical_ranks)
    task = protocol["tasks"][args.task]
    fit = args.fits / args.task
    bank = load_fitted_bank(fit, task="mw-" + args.task, checkpoint_sha256=CHECKPOINTS["metaworld"])
    if sha256(fit / "DONE.json") != task["fit_done_sha256"] or sha256(fit / "operator_bank.pt") != task["fit_bank_sha256"]:
        raise ValueError("Frozen successor fitting bank changed")
    legacy, goals, stimulus = stimulus_contract(args.stimuli, args.task)
    if stimulus != task["stimuli"]:
        raise ValueError("Historical paired stimuli changed")
    coupling_root, coupling_protocol, coupling_hashes = coupling_binding(args.original, legacy, args.task)
    if coupling_hashes != task["coupling"]:
        raise ValueError("Frozen coupling component changed")
    engineering_hash = verify_cem(args.engineering, fit, args.checked_source, args.task)
    worker = json.loads((args.engineering / "WORKER.json").read_text())
    if (worker.get("cem_report_sha256") != engineering_hash or worker.get("device_uuid") != device_uuid() or
            worker.get("same_device_before_after") is not True):
        raise ValueError("Require complete CEM/native-repeat proof on the actual receiving GPU")
    coupling_hash = None
    if args.arm in COUPLING_ARMS:
        coupling_hash = verify_coupling_engineering(args.stimuli / "planning-panel-coupling-engineering-20260907-v1" /
            (args.task + "-" + args.arm), coupling_root, args.task, args.arm)
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    native = prepare(args.vendor, args.task)["config"]
    native["meta"]["seed"] = DEVELOPMENT_SEED
    if native != task["planning"]["config"]:
        raise ValueError("Native planner configuration changed")
    args.output.mkdir(parents=True, exist_ok=False)
    started, records = time.monotonic(), []
    try:
        write_json(args.output / "protocol.json", {"role": ROLE, "freeze_sha256": sha256(args.freeze / "protocol.json"),
            "source_sha256": source_hash(), "task": args.task, "arm": args.arm, "expected_episodes": expected,
            "logical_ranks": args.logical_ranks, "engineering_report_sha256": engineering_hash,
            "receiving_worker_sha256": sha256(args.engineering / "WORKER.json"), "device_uuid": device_uuid(),
            "coupling_engineering_sha256": coupling_hash, "fresh_confirmation": False})
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        if backend.model.ctxt_window != 2:
            raise ValueError("Author planner context changed")
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        coupling_bank = torch.load(coupling_root / "operator_bank.pt", map_location="cpu", weights_only=True)
        torch.cuda.reset_peak_memory_stats()
        for rank in sorted(args.logical_ranks):
            rows = [r for r in expected if r["logical_rank"] == rank]
            random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            cfg = OmegaConf.create(task["planning"]["config"])
            cfg.local_seed = rows[0]["local_seed"]
            agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
            env = make_env(cfg)
            try:
                for row in rows:
                    before = time.monotonic()
                    episode = args.output / f"calls-{row['episode']:03d}"
                    episode.mkdir()
                    adapter = (H6StaticPlanningIntervention(backend, coupling_protocol, coupling_bank, COUPLING_ARMS[args.arm])
                        if args.arm in COUPLING_ARMS else FixedResponseIntervention(backend, bank, args.arm))
                    observed = ObservePlanner(adapter, backend, episode)
                    agent.planner.unroll = observed
                    # Restore the original callback after each episode; never
                    # nest observer wrappers across an RNG stream.
                    original_act = agent.act
                    trace = trace_actor(agent)
                    try:
                        with close_expert_environments(plan_evaluator), goals.deliver(row):
                            result = run_episode(cfg, backend, agent, env, row["environment_seed"])
                    finally:
                        agent.act = original_act
                        write_json(episode / "unroll_calls.json", observed.calls)
                        write_json(episode / "action_trace.json", trace)
                    verify_episode(result, observed.calls)
                    metadata, _ = goals.load(row)
                    initial = np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist()
                    if initial != metadata["rand_vec"] or any(result[k] != metadata[k] for k in ("initial_sha256", "goal_sha256")):
                        raise ValueError("Actual delivered scenario differs from canonical inputs")
                    record = {**row, "arm": args.arm, "result": result, "initial_state_vector": initial,
                        "seconds": time.monotonic() - before, "unroll_calls_sha256": sha256(episode / "unroll_calls.json"),
                        "action_trace_sha256": sha256(episode / "action_trace.json")}
                    validate_coverage([record], [row])
                    records.append(record)
                    write_json(args.output / f"episode-{row['episode']:03d}.json", record)
                    progress = {"task": args.task, "arm": args.arm, "completed": len(records), "target": len(expected),
                        "seconds": time.monotonic() - started, "fresh_confirmation": False}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            finally:
                env.close()
        validate_coverage(records, expected)
        if _model_versions(backend.model) != versions or source_hash() != protocol["source_sha256"]:
            raise ValueError("Frozen parameters/source changed")
        write_json(args.output / "report.json", {"status": "fixed_response_behavioral_shard_complete",
            "task": args.task, "arm": args.arm, "episodes": len(records), "device_name": torch.cuda.get_device_name(),
            "protocol_sha256": sha256(args.output / "protocol.json"), "parameters_unchanged": True,
            "episode_files_sha256": {f"episode-{r['episode']:03d}.json": sha256(args.output / f"episode-{r['episode']:03d}.json") for r in records},
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "scientific_efficacy_measurement": True, "fresh_confirmation": False, "comparative_analysis_complete": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "partial_not_complete": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "worker-engineering"))
    for name in ("vendor", "fits", "original", "stimuli", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze", "checkpoint", "engineering", "checked-source", "numerical-check"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--logical-ranks", type=int, nargs="+")
    args = parser.parse_args()
    if args.mode == "run" and any(getattr(args, k) is None for k in
        ("freeze", "checkpoint", "engineering", "checked_source", "task", "arm", "logical_ranks")):
        parser.error("Run requires complete frozen, fitted, engineering and logical-stream bindings")
    if args.mode == "worker-engineering" and any(getattr(args, k) is None for k in
        ("checkpoint", "checked_source", "numerical_check", "task")):
        parser.error("Worker engineering requires the original numerical/source bindings")
    {"freeze": freeze, "run": run, "worker-engineering": worker_engineering}[args.mode](args)


if __name__ == "__main__":
    main()
