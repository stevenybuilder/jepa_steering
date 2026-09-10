"""Execute fixed coupling conditions in the existing MetaWorld behavioral panel.

The scientific freeze is immutable. This separate launch receipt binds newly
completed engineering to its existing banks, scenario streams and analysis plan.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_evaluate import verify_fit
from .author_fit import source_hash
from .backends import JepaBackend
from .behavioral_development import ARMS, DEVELOPMENT_SEED, assigned_rows, validate_coverage, verified_report
from .intervention_runner import _model_versions
from .planning_contract import prepare
from .planning_native_smoke import CHECKPOINTS, run_episode
from .planning_panel_engineering import COUPLING_ARMS, H6StaticPlanningIntervention
from .planning_scenarios import close_expert_environments
from .planning_support_smoke import ObservedSupport, verify_call_schedule
from .protocol import sha256, write_json
from .vendor import use_vendor


def immutable_source_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def launch_contract(args):
    frozen = json.loads((args.freeze / "FROZEN.json").read_text())
    if sha256(args.freeze / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Scientific freeze changed")
    protocol = json.loads((args.freeze / "protocol.json").read_text())
    if (protocol["role"] != "frozen_planning_development_not_confirmation" or
            protocol["arms"] != [list(arm) for arm in ARMS] or
            immutable_source_hash(args.baseline_code / "src/offline_study") != protocol["source_sha256"]):
        raise ValueError("Wrong panel or unverified immutable baseline source")
    # New engineering modules need not exist in the older snapshot. Native model,
    # simulator, planner call and RNG/episode loop must still be identical.
    shared = ("backends.py", "model_loader.py", "planning_native_smoke.py", "planning_scenarios.py",
              "planning_contract.py", "behavioral_development.py", "planning_env_smoke.py")
    for name in shared:
        if sha256(args.baseline_code / "src/offline_study" / name) != sha256(Path(__file__).with_name(name)):
            raise ValueError("Native comparison path changed: " + name)
    expected = assigned_rows(protocol["episodes"], args.logical_ranks)
    task = protocol["tasks"][args.task]
    native_config = prepare(args.vendor, args.task)["config"]
    native_config["meta"]["seed"] = DEVELOPMENT_SEED
    if task["planning"]["config"] != native_config:
        raise ValueError("Native planning config changed")
    root = args.original_root / "fits-v1/bfloat16" / args.task / "vision_action_coupling"
    for name, digest in task["components"]["vision_action_coupling"].items():
        if sha256(root / name) != digest:
            raise ValueError("Frozen coupling fit changed")
    _, fit_protocol = verify_fit(root, task["cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
    engineering, engineering_hash = verified_report(args.engineering)
    ep = json.loads((args.engineering / "protocol.json").read_text())
    if (engineering["status"] != "common_h6_coupling_panel_full_cem_engineering_passed" or
            engineering["task"] != args.task or engineering["arm"] != args.arm or
            engineering["protocol_sha256"] != sha256(args.engineering / "protocol.json") or
            engineering["fit_parity_sha256"] != sha256(args.engineering / "fit_parity.json") or
            engineering["unroll_calls_sha256"] != sha256(args.engineering / "unroll_calls.json") or
            ep["fit_protocol_sha256"] != sha256(root / "protocol.json") or
            ep["bank_sha256"] != sha256(root / "operator_bank.pt")):
        raise ValueError("Missing bound full-CEM and source-parity engineering")
    for name, digest in ep["source_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("Engineering adapter changed")
    return protocol, expected, fit_protocol, root, engineering_hash


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "freeze", "baseline-code", "original-root", "checkpoint", "engineering", "output", "goal-bank", "goal-delivery-proof"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    parser.add_argument("--arm", choices=tuple(COUPLING_ARMS), required=True)
    parser.add_argument("--logical-ranks", nargs="+", type=int, required=True)
    parser.add_argument("--repair-plan", type=Path,
                        help="Optional input-only whole-stream replacement manifest, frozen before reruns")
    args = parser.parse_args()
    use_vendor(args.vendor)
    protocol, expected, fit_protocol, root, engineering_hash = launch_contract(args)
    bank = torch.load(root / "operator_bank.pt", map_location="cpu", weights_only=True)
    run_shard(args, protocol, expected, engineering_hash,
        lambda backend: H6StaticPlanningIntervention(backend, fit_protocol, bank, COUPLING_ARMS[args.arm]),
        {"source_arm": COUPLING_ARMS[args.arm]})


@torch.no_grad()
def run_shard(args, protocol, expected, engineering_hash, make_adapter, launch_details):
    """Common frozen episode/RNG loop for independently gated fixed components."""
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator
    from .planning_goal_bank import GoalBank, verify_delivery
    goals = GoalBank(args.goal_bank, args.task, sha256(args.freeze / "protocol.json"))
    delivery_hash = verify_delivery(args.goal_delivery_proof, goals.report_hash, sha256(args.freeze / "protocol.json"))
    repair_root = getattr(args, "repair_plan", None)
    repair_hash = None
    if repair_root is not None:
        repair_hash = sha256(repair_root / "plan.json")
        repair = json.loads((repair_root / "plan.json").read_text())
        if (json.loads((repair_root / "FROZEN.json").read_text())["plan_sha256"] != repair_hash or
                repair["task"] != args.task or repair["arm"] != args.arm or
                repair["freeze_sha256"] != sha256(args.freeze / "protocol.json") or
                repair["goal_bank_report_sha256"] != goals.report_hash or
                repair["goal_delivery_report_sha256"] != delivery_hash or
                repair["replacement_episodes"] != expected or repair["logical_ranks"] != args.logical_ranks or
                repair["outcomes_used_for_repair_selection"] is not False or repair["sample_size_reduced"] is not False):
            raise ValueError("Unbound or changed whole-stream repair plan")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"role": "fixed_planning_development_candidate_shard",
            "freeze_sha256": sha256(args.freeze / "protocol.json"), "task": args.task, "arm": args.arm,
            **launch_details, "expected_episodes": expected,
            "logical_ranks": args.logical_ranks, "source_sha256": source_hash(),
            "native_baseline_source_sha256": protocol["source_sha256"],
            "engineering_report_sha256": engineering_hash, "precision": "float32_strict_no_tf32",
            "canonical_native_goal_bank_report_sha256": goals.report_hash,
            "canonical_native_goal_delivery_report_sha256": delivery_hash,
            "input_only_whole_stream_repair_plan_sha256": repair_hash,
            "actual_goal_pixels_bitwise_match_original_native": True,
            "all_native_below_full_h6": True, "fresh_confirmation": False,
            "unchanged_fit_and_scientific_panel": True, "no_selection_from_partial_outcomes": True})
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.set_device(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"],
                              "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        rng_python, rng_numpy, rng_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        records = []
        torch.cuda.reset_peak_memory_stats()
        for rank in sorted(args.logical_ranks):
            rows = [row for row in expected if row["logical_rank"] == rank]
            random.setstate(rng_python)
            np.random.set_state(rng_numpy)
            torch.set_rng_state(rng_torch)
            cfg = OmegaConf.create(protocol["tasks"][args.task]["planning"]["config"])
            cfg.local_seed = rows[0]["local_seed"]
            agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
            env = make_env(cfg)
            try:
                for row in rows:
                    before = time.monotonic()
                    episode_root = args.output / f"calls-{row['episode']:03d}"
                    episode_root.mkdir()
                    adapter = make_adapter(backend)
                    observed = ObservedSupport(adapter, episode_root)
                    agent.planner.unroll = observed
                    with close_expert_environments(plan_evaluator), goals.deliver(row):
                        result = run_episode(cfg, backend, agent, env, row["environment_seed"])
                    verify_call_schedule(result["planning_calls"], observed.calls)
                    record = {**row, "arm": args.arm, "result": result,
                        "initial_state_vector": np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist(),
                        "unroll_calls_sha256": sha256(episode_root / "unroll_calls.json"),
                        "seconds": time.monotonic() - before}
                    validate_coverage([record], [row])
                    records.append(record)
                    write_json(args.output / f"episode-{row['episode']:03d}.json", record)
                    progress = {"task": args.task, "arm": args.arm, "completed": len(records),
                                "target": len(expected), "seconds": time.monotonic() - started}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            finally:
                env.close()
        validate_coverage(records, expected)
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen parameters changed")
        write_json(args.output / "report.json", {"status": "fixed_planning_development_candidate_shard_complete",
            "task": args.task, "arm": args.arm, "episodes": len(records),
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "episode_files_sha256": {f"episode-{r['episode']:03d}.json":
                sha256(args.output / f"episode-{r['episode']:03d}.json") for r in records},
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "scientific_efficacy_measurement": True, "comparative_analysis_complete": False,
            "fresh_confirmation": False, "parameters_unchanged": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


if __name__ == "__main__":
    main()
