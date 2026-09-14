"""Frozen native navigation development, retaining complete upstream RNG streams.

This opens neither the reserved base-1 confirmation stream nor intervention results.
It supplies the paired unsteered reference for the already named seven-condition
panel; candidate fitting and engineering are separately required before execution.
"""
from offline_study._paths import source_path
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_development import ARMS, DEVELOPMENT_SEED, assigned_rows, schedule, verified_report
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.tasks.navigation.navigation_smoke import CHECKPOINTS, validate_complete
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_native_smoke import run_episode
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


def validate_records(records, expected):
    keys = ("episode", "logical_rank", "local_seed", "environment_seed")
    if [tuple(row[k] for k in keys) for row in records] != [tuple(row[k] for k in keys) for row in expected]:
        raise ValueError("Missing, duplicated, reordered or changed navigation episodes")
    for row in records:
        validate_complete(row["result"], [tuple(v) for v in row["unroll_calls"]])
        if not isinstance(row["result"]["native_success"], bool) or len(row["planned_actions"]) != 1:
            raise ValueError("Invalid navigation outcome or incomplete action trace")
        action = torch.tensor(row["planned_actions"][0], dtype=torch.float32)
        if action.ndim != 2 or action.shape[0] != 6 or not torch.isfinite(action).all():
            raise ValueError("Changed native six-action output or nonfinite actions")


def freeze(vendor, engineering, engineering_code, task, output):
    use_vendor(vendor)
    report, report_hash = verified_report(engineering)
    ep = json.loads((engineering / "protocol.json").read_text())
    contract = prepare(vendor, task)
    if (report["status"] != "full_native_navigation_engineering_passed" or report["task"] != task or
            report["protocol_sha256"] != sha256(engineering / "protocol.json") or
            ep["planning_contract"] != contract or ep["checkpoint_sha256"] != CHECKPOINTS[task] or
            not report["same_seed_actions_and_outcomes_exact"]):
        raise ValueError("Missing matching full native navigation engineering")
    for i, digest in enumerate(report["repetition_sha256"]):
        if sha256(engineering / f"repetition-{i}.json") != digest:
            raise ValueError("Engineering episode changed")
    # Bind the proven model/simulator path; the new scheduling/logger is additive.
    prior_source = engineering_code / "src/offline_study"
    digest = hashlib.sha256()
    for path in sorted(prior_source.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    if digest.hexdigest() != ep["source_sha256"]:
        raise ValueError("Engineering source tree is not the recorded immutable version")
    for name in ("backends.py", "model_loader.py", "planning_native_smoke.py", "planning_contract.py"):
        if sha256(prior_source / name) != sha256(source_path(name)):
            raise ValueError("Proven navigation execution source changed: " + name)
    cfg = contract["config"]
    cfg["meta"]["seed"] = DEVELOPMENT_SEED
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "role": "frozen_navigation_native_planning_development_not_confirmation", "task": task,
        "planning_contract": contract, "episodes": schedule(), "source_sha256": source_hash(),
        "engineering_report_sha256": report_hash, "checkpoint_sha256": CHECKPOINTS[task],
        "named_future_panel": [list(arm) for arm in ARMS],
        "candidate_fit_bindings_and_engineering_complete": False,
        "candidate_selection_from_native_outcomes": False,
        "native_reference_fixed_before_candidate_outcomes": True,
        "primary_endpoint": "native simulator success after all 30 elementary steps",
        "secondary_endpoints": ["native state distance", "native reward", "planned actions and norms", "runtime"],
        "episodes_per_condition": 96, "reserved_confirmation_base_seed": 1,
        "fresh_confirmation": False, "training_seed_history_complete": False,
        "precision": "float32_strict_no_tf32"})
    write_json(output / "FROZEN.json", {"protocol_sha256": sha256(output / "protocol.json")})


@torch.no_grad()
def run(args):
    use_vendor(args.vendor)
    frozen = json.loads((args.freeze / "FROZEN.json").read_text())
    protocol = json.loads((args.freeze / "protocol.json").read_text())
    if (frozen["protocol_sha256"] != sha256(args.freeze / "protocol.json") or
            protocol["source_sha256"] != source_hash() or
            protocol["role"] != "frozen_navigation_native_planning_development_not_confirmation"):
        raise ValueError("Native navigation freeze or source changed")
    task = protocol["task"]
    expected = assigned_rows(protocol["episodes"], args.logical_ranks)
    contract = prepare(args.vendor, task)
    contract["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    if contract != protocol["planning_contract"]:
        raise ValueError("Native navigation config changed")
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"role": "native_navigation_development_shard",
            "task": task, "arm": "native", "freeze_sha256": frozen["protocol_sha256"],
            "source_sha256": source_hash(), "expected_episodes": expected, "fresh_confirmation": False})
        torch.cuda.set_device(0)
        random.seed(DEVELOPMENT_SEED)
        np.random.seed(DEVELOPMENT_SEED)
        torch.manual_seed(DEVELOPMENT_SEED)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS[task], task, "cuda:0", "float32")
        versions = _model_versions(backend.model)
        native_unroll = backend.model.unroll
        global_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for rank in sorted(args.logical_ranks):
            random.setstate(global_rng[0])
            np.random.set_state(global_rng[1])
            torch.set_rng_state(global_rng[2])
            rows = [r for r in expected if r["logical_rank"] == rank]
            cfg = OmegaConf.create(contract["config"])
            cfg.local_seed = rows[0]["local_seed"]
            agent = GC_Agent(cfg, backend.model, dset=None, preprocessor=backend.preprocessor)
            env = make_env(cfg)
            try:
                for row in rows:
                    before = time.monotonic()
                    calls, actions = [], []

                    def observed(context, act_suffix=None, **kwargs):
                        result = native_unroll(context, act_suffix=act_suffix, **kwargs)
                        if any(not torch.isfinite(result[k]).all() for k in ("visual", "proprio")):
                            raise ValueError("Nonfinite native forecast")
                        calls.append(tuple(act_suffix.shape[:2]))
                        return result

                    native_act = agent.act

                    def act(*a, **kw):
                        result = native_act(*a, **kw)
                        actions.append(result.detach().cpu().tolist())
                        return result

                    backend.model.unroll = observed
                    # CEM stores the callable at construction, so explicitly bind it.
                    agent.planner.unroll = observed
                    agent.act = act
                    try:
                        result = run_episode(cfg, backend, agent, env, row["environment_seed"])
                    finally:
                        agent.act = native_act
                        backend.model.unroll = native_unroll
                        agent.planner.unroll = native_unroll
                    record = {**row, "arm": "native", "result": result,
                              "unroll_calls": calls, "planned_actions": actions,
                              "seconds": time.monotonic() - before}
                    validate_records([record], [row])
                    records.append(record)
                    write_json(args.output / f"episode-{row['episode']:03d}.json", record)
                    progress = {"task": task, "completed": len(records), "target": len(expected),
                                "seconds": time.monotonic() - started}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            finally:
                env.close()
        validate_records(records, expected)
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen weights changed")
        write_json(args.output / "report.json", {"status": "native_navigation_development_shard_complete",
            "task": task, "arm": "native", "episodes": len(records),
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "episode_files_sha256": {f"episode-{r['episode']:03d}.json":
                sha256(args.output / f"episode-{r['episode']:03d}.json") for r in records},
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "parameters_unchanged": True, "fresh_confirmation": False,
            "comparative_analysis_complete": False, "training_seed_history_complete": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("freeze")
    for name in ("vendor", "engineering", "engineering-code", "output"):
        prep.add_argument("--" + name, required=True, type=Path)
    prep.add_argument("--task", choices=tuple(CHECKPOINTS), required=True)
    execute = sub.add_parser("native")
    for name in ("vendor", "freeze", "checkpoint", "output"):
        execute.add_argument("--" + name, required=True, type=Path)
    execute.add_argument("--logical-ranks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.vendor, args.engineering, args.engineering_code, args.task, args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
