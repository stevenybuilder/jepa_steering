"""Complete the three missing MetaWorld component arms without altering old runs."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_fit import source_hash
from .backends import JepaBackend
from .behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, validate_coverage, verified_report
from .fixed_response_behavior import ObservePlanner, coupling_binding, device_uuid, stimulus_contract
from .fixed_response_smoke import trace_actor, verify_episode, verify_pair
from .intervention_runner import _model_versions
from .planning_contract import prepare
from .planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from .planning_panel_engineering import H6StaticPlanningIntervention
from .planning_scenarios import close_expert_environments
from .protocol import sha256, write_json
from .vendor import use_vendor

METHOD = "metaworld_component_completion_v1"
TASKS = ("reach", "reach-wall")
ARMS = ("native", "visual_only", "action_condition_only", "joint")
ENGINEERING = ("native", "native_repeat", "zero_dose", *ARMS[1:])
CONTRASTS = {
    "visual_vs_native": {"visual_only": 1, "native": -1},
    "action_vs_native": {"action_condition_only": 1, "native": -1},
    "joint_vs_native": {"joint": 1, "native": -1},
    "joint_vs_visual": {"joint": 1, "visual_only": -1},
    "joint_vs_action": {"joint": 1, "action_condition_only": -1},
    "success_probability_interaction": {"joint": 1, "visual_only": -1, "action_condition_only": -1, "native": 1},
}
ANALYSIS = {"replicates": 20000, "seed": 2026091101, "family": 12,
    "interval": "paired_scenario_cluster_percentile_bootstrap_bonferroni_95",
    "endpoint": "official_binary_success_after_100_elementary_steps",
    "automatic_candidate_selection": False, "confirmation_authorized": False}


def read(path):
    return json.loads(Path(path).read_text())


def task_inputs(args, task):
    legacy, goals, stimuli = stimulus_contract(args.stimuli, task)
    fit, protocol, hashes = coupling_binding(args.original, legacy, task)
    bank = torch.load(fit / "operator_bank.pt", map_location="cpu", weights_only=True)
    if bank["protocol_sha256"] != sha256(fit / "protocol.json"):
        raise ValueError("Coupling bank is not bound to its protocol")
    planning = prepare(args.vendor, task)
    planning["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    if planning["config"] != legacy["tasks"][task]["planning"]["config"]:
        raise ValueError("Historical native planning configuration changed")
    return goals, protocol, bank, {"stimuli": stimuli, "coupling": hashes, "planning": planning}


def freeze(args):
    tasks = {task: task_inputs(args, task)[3] for task in TASKS}
    protocol = {"method": METHOD, "source_sha256": source_hash(), "tasks": tasks,
        "checkpoint_sha256": CHECKPOINTS["metaworld"], "arms": list(ARMS),
        "contrasts": CONTRASTS, "analysis": ANALYSIS, "episodes": schedule(),
        "episodes_per_task_condition": 96, "total_episode_evaluations": 768,
        "fresh_confirmation": False, "protected_access": False,
        "precision": "float32_strict_no_tf32", "engineering_seed": SMOKE_SEED,
        "engineering_order": list(ENGINEERING), "legacy_expensive_operator": False,
        "native_reference": "paired_new_receiving_runtime_reference_compare_to_old_before_table_merge"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
        "new_component_outcomes_observed": False, "historical_development_results_already_observed": True})


def validate_protocol(root):
    p = read(root / "protocol.json")
    if sha256(root / "protocol.json") != read(root / "FROZEN.json")["protocol_sha256"]:
        raise ValueError("Changed component freeze")
    expected = {"method": METHOD, "arms": list(ARMS), "contrasts": CONTRASTS,
        "analysis": ANALYSIS, "episodes": schedule(), "episodes_per_task_condition": 96,
        "total_episode_evaluations": 768, "fresh_confirmation": False,
        "protected_access": False, "legacy_expensive_operator": False,
        "precision": "float32_strict_no_tf32", "engineering_seed": SMOKE_SEED,
        "engineering_order": list(ENGINEERING), "checkpoint_sha256": CHECKPOINTS["metaworld"]}
    if any(p.get(k) != v for k, v in expected.items()) or set(p.get("tasks", {})) != set(TASKS):
        raise ValueError("Unrecognized component protocol or altered sample/arm/scope")
    return p


def expected_energy(protocol, bank, arm):
    edits = next(row["edits"] for row in protocol["arms"] if row["name"] == arm)
    return sum(float(bank["global_tensors"][e["tensor"]].double().square().sum()) * e["scale"] ** 2
               for e in edits)


def verify_energy(calls, protocol, bank, arm):
    for call in calls:
        if call["backend_calls"] != 1:
            raise ValueError("Component used additional model calls")
        expected = expected_energy(protocol, bank, arm) if call["horizon"] == 6 else 0.
        audit = call["energy"]
        requested, delivered = audit["requested_squared_l2_mean"], audit["realized_squared_l2_mean"]
        if not all(math.isfinite(v) for v in (requested, delivered)):
            raise ValueError("Nonfinite component dose")
        if expected == 0:
            if requested != 0 or delivered != 0:
                raise ValueError("Native/shortened forecast received an edit")
        elif (not math.isclose(requested, expected, rel_tol=1e-5, abs_tol=1e-9) or
              not math.isclose(delivered, expected, rel_tol=5e-4, abs_tol=1e-9)):
            raise ValueError("Component dose differs from frozen fitting dose")


def verify_engineering(root, freeze_root, task, executing_device=None):
    report, digest = verified_report(root)
    if ((root / "FAILED.json").exists() or report.get("status") != METHOD + "_engineering_complete" or
            report.get("task") != task or report.get("freeze_sha256") != sha256(freeze_root / "protocol.json") or
            report.get("source_sha256") != source_hash() or report.get("full_episodes") != len(ENGINEERING) or
            report.get("scientific_efficacy_measurement") is not False or
            (executing_device is not None and report.get("device_uuid") != executing_device)):
        raise ValueError("Missing source/input/device-bound component engineering")
    if set(report["episodes"]) != set(ENGINEERING):
        raise ValueError("Incomplete engineering arm coverage")
    baseline = None
    for name in ENGINEERING:
        episode, episode_hash = verified_report(root / name)
        if report["episodes"][name] != episode_hash:
            raise ValueError("Engineering episode receipt changed")
        for file, key in (("unroll_calls.json", "unroll_calls_sha256"), ("action_trace.json", "action_trace_sha256")):
            if sha256(root / name / file) != episode[key]:
                raise ValueError("Engineering trace changed")
        calls = read(root / name / "unroll_calls.json")
        verify_episode(episode["result"], calls)
        current = {"result": episode["result"], "action_trace": read(root / name / "action_trace.json")}
        if baseline is None:
            baseline = current
        else:
            verify_pair(baseline, current, native_repeat=name in ("native_repeat", "zero_dose"))
    return digest


def runtime(args):
    p = validate_protocol(args.freeze)
    if p["source_sha256"] != source_hash():
        raise ValueError("Runtime changed after freeze")
    goals, cp, bank, binding = task_inputs(args, args.task)
    if binding != p["tasks"][args.task]:
        raise ValueError("Task inputs changed after freeze")
    use_vendor(args.vendor)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
    if backend.model.ctxt_window != 2:
        raise ValueError("Wrong native planning context")
    return p, goals, cp, bank, backend


@torch.no_grad()
def engineering(args):
    from omegaconf import OmegaConf
    p, goals, cp, bank, backend = runtime(args)
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    args.output.mkdir(parents=True, exist_ok=False)
    versions, uuid, started = _model_versions(backend.model), device_uuid(), time.monotonic()
    canonical = {goals.load(row)[0]["initial_sha256"] for row in schedule()}
    completed, hashes = {}, {}
    try:
        for name in ENGINEERING:
            arm = "native" if name == "native_repeat" else name
            out = args.output / name
            out.mkdir()
            random.seed(SMOKE_SEED); np.random.seed(SMOKE_SEED); torch.manual_seed(SMOKE_SEED)
            cfg = OmegaConf.create(p["tasks"][args.task]["planning"]["config"])
            cfg.meta.seed = cfg.local_seed = SMOKE_SEED
            agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
            observed = ObservePlanner(H6StaticPlanningIntervention(backend, cp, bank, arm), backend, out)
            agent.planner.unroll = observed
            trace, env = trace_actor(agent), make_env(cfg)
            before = time.monotonic()
            try:
                with close_expert_environments(plan_evaluator):
                    result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
            finally:
                env.close()
                write_json(out / "unroll_calls.json", observed.calls)
                write_json(out / "action_trace.json", trace)
            verify_episode(result, observed.calls)
            verify_energy(observed.calls, cp, bank, arm)
            if result["initial_sha256"] in canonical:
                raise ValueError("Engineering scenario overlaps evaluation inputs")
            current = {"result": result, "action_trace": trace}
            if completed:
                verify_pair(completed["native"], current, native_repeat=name in ("native_repeat", "zero_dose"))
            completed[name] = current
            report = {"arm": arm, "result": result, "seconds": time.monotonic() - before,
                "unroll_calls_sha256": sha256(out / "unroll_calls.json"),
                "action_trace_sha256": sha256(out / "action_trace.json"), "scientific_efficacy_measurement": False}
            write_json(out / "report.json", report)
            hashes[name] = sha256(out / "report.json")
            write_json(out / "DONE.json", {"report_sha256": hashes[name]})
            print(json.dumps({"task": args.task, "engineering_completed": name, "seconds": report["seconds"]}), flush=True)
        if _model_versions(backend.model) != versions or device_uuid() != uuid or source_hash() != p["source_sha256"]:
            raise ValueError("Engineering source/model/device changed")
        write_json(args.output / "report.json", {"status": METHOD + "_engineering_complete", "task": args.task,
            "freeze_sha256": sha256(args.freeze / "protocol.json"), "source_sha256": source_hash(),
            "device_uuid": uuid, "episodes": hashes, "full_episodes": len(ENGINEERING),
            "backend": backend.provenance, "seconds": time.monotonic() - started,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


@torch.no_grad()
def run(args):
    from omegaconf import OmegaConf
    p, goals, cp, bank, backend = runtime(args)
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    engineering_hash = verify_engineering(args.engineering, args.freeze, args.task, device_uuid())
    expected = assigned_rows(p["episodes"], args.logical_ranks)
    args.output.mkdir(parents=True, exist_ok=False)
    started, records, versions = time.monotonic(), [], _model_versions(backend.model)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    write_json(args.output / "protocol.json", {"method": METHOD, "task": args.task, "arm": args.arm,
        "freeze_sha256": sha256(args.freeze / "protocol.json"), "source_sha256": source_hash(),
        "logical_ranks": args.logical_ranks, "expected_episodes": expected,
        "engineering_report_sha256": engineering_hash, "device_uuid": device_uuid(), "fresh_confirmation": False})
    try:
        for rank in sorted(args.logical_ranks):
            rows = [r for r in expected if r["logical_rank"] == rank]
            random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
            cfg = OmegaConf.create(p["tasks"][args.task]["planning"]["config"])
            cfg.local_seed = rows[0]["local_seed"]
            agent, env = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor), make_env(cfg)
            try:
                for row in rows:
                    out = args.output / f"calls-{row['episode']:03d}"
                    out.mkdir()
                    observed = ObservePlanner(H6StaticPlanningIntervention(backend, cp, bank, args.arm), backend, out)
                    agent.planner.unroll = observed
                    original_act, before = agent.act, time.monotonic()
                    trace = trace_actor(agent)
                    try:
                        with close_expert_environments(plan_evaluator), goals.deliver(row):
                            result = run_episode(cfg, backend, agent, env, row["environment_seed"])
                    finally:
                        agent.act = original_act
                        write_json(out / "unroll_calls.json", observed.calls)
                        write_json(out / "action_trace.json", trace)
                    verify_episode(result, observed.calls)
                    verify_energy(observed.calls, cp, bank, args.arm)
                    metadata, _ = goals.load(row)
                    initial = np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist()
                    if initial != metadata["rand_vec"] or any(result[k] != metadata[k] for k in ("initial_sha256", "goal_sha256")):
                        raise ValueError("Actual component scenario differs from canonical input")
                    record = {**row, "arm": args.arm, "result": result, "initial_state_vector": initial,
                        "seconds": time.monotonic() - before, "unroll_calls_sha256": sha256(out / "unroll_calls.json"),
                        "action_trace_sha256": sha256(out / "action_trace.json")}
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
        if _model_versions(backend.model) != versions or source_hash() != p["source_sha256"]:
            raise ValueError("Frozen component model/source changed")
        write_json(args.output / "report.json", {"status": METHOD + "_shard_complete", "task": args.task,
            "arm": args.arm, "episodes": len(records), "protocol_sha256": sha256(args.output / "protocol.json"),
            "parameters_unchanged": True, "episode_files_sha256": {
                f"episode-{r['episode']:03d}.json": sha256(args.output / f"episode-{r['episode']:03d}.json") for r in records},
            "seconds": time.monotonic() - started, "backend": backend.provenance,
            "scientific_efficacy_measurement": True, "fresh_confirmation": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "partial_not_complete": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "engineering", "run"))
    for name in ("vendor", "original", "stimuli", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze", "checkpoint", "engineering"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--logical-ranks", type=int, nargs="+")
    args = parser.parse_args()
    if args.mode != "freeze" and any(getattr(args, k) is None for k in ("freeze", "checkpoint", "task")):
        parser.error("Execution requires freeze, checkpoint and task")
    if args.mode == "run" and any(getattr(args, k) is None for k in ("engineering", "arm", "logical_ranks")):
        parser.error("Scientific execution requires engineering, arm and logical ranks")
    {"freeze": freeze, "engineering": engineering, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
