"""Frozen MetaWorld planning-development panel, distinct from untouched confirmation.

Only native execution is enabled initially. Candidate jobs need their own completed
integration receipts; a baseline does not depend on unfinished candidate engineering.
Independent logical RNG streams can move across GPUs without changing episode IDs.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.planning.planning_contract import prepare as planning_contract, seed_schedule
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.planning.planning_scenarios import close_expert_environments
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


DEVELOPMENT_SEED = 2026090721
ARMS = (
    ("native", "native", "native"),
    ("coupling_only", "joint_equal_standardized_energy", "native"),
    ("rank4_only", "native", "rank4"),
    ("combined", "joint_equal_standardized_energy", "rank4"),
    ("matched_random_coupling", "matched_random_equal_standardized_energy", "native"),
    ("matched_random_rank4", "native", "matched_random_rank4"),
    ("matched_random_combined", "matched_random_equal_standardized_energy", "matched_random_rank4"),
)
CONTRASTS = (("coupling_only", "native"), ("coupling_only", "matched_random_coupling"),
             ("rank4_only", "native"), ("rank4_only", "matched_random_rank4"),
             ("combined", "native"), ("combined", "matched_random_combined"))


def schedule():
    rows = seed_schedule(DEVELOPMENT_SEED)
    protected = {row["environment_seed"] for row in seed_schedule(1)} | {SMOKE_SEED}
    if protected & {row["environment_seed"] for row in rows}:
        raise ValueError("Planning-development seeds intersect reserved/smoke seeds")
    return rows


def assigned_rows(rows, logical_ranks):
    if not logical_ranks or len(set(logical_ranks)) != len(logical_ranks) or any(r not in range(8) for r in logical_ranks):
        raise ValueError("Require unique logical ranks in 0..7")
    if rows != schedule():
        raise ValueError("Frozen episode identity or stream order changed")
    return [row for row in rows if row["logical_rank"] in logical_ranks]


def verified_report(root):
    done = json.loads((root / "DONE.json").read_text())
    if sha256(root / "report.json") != done["report_sha256"]:
        raise ValueError("Required completed report checksum mismatch")
    return json.loads((root / "report.json").read_text()), done["report_sha256"]


def freeze(vendor, original_root, native_smoke, output):
    use_vendor(vendor)
    tasks = {}
    for task in ("reach", "reach-wall"):
        native, native_hash = verified_report(native_smoke / task)
        if (native["task"] != task or native["checkpoint_sha256"] != CHECKPOINTS["metaworld"] or
                native["precision"] != "float32_strict_no_tf32" or
                native["result"]["elementary_steps"] != 100 or
                native["fresh_confirmation"] is not False):
            raise ValueError("Missing complete native engineering evidence")
        cohort = original_root / "cohorts" / task / "cohort.json"
        bindings = {}
        for category in ("vision_action_coupling", "operator_rank"):
            root = original_root / "fits-v1/bfloat16" / task / category
            _, protocol = verify_fit(root, sha256(cohort), CHECKPOINTS["metaworld"], "bfloat16")
            required = {arm[1 if category == "vision_action_coupling" else 2] for arm in ARMS}
            if not required <= {arm["name"] for arm in protocol["arms"]}:
                raise ValueError("Missing pre-existing candidate component")
            bindings[category] = {name: sha256(root / name) for name in
                                 ("protocol.json", "operator_bank.pt", "fit_receipt.json")}
        contract = planning_contract(vendor, task)
        contract["config"]["meta"]["seed"] = DEVELOPMENT_SEED
        tasks[task] = {"planning": contract, "native_smoke_report_sha256": native_hash,
                       "cohort_sha256": sha256(cohort), "components": bindings}
    protocol = {"schema_version": 1, "role": "frozen_planning_development_not_confirmation",
        "tasks": tasks, "checkpoint_sha256": CHECKPOINTS["metaworld"],
        "source_sha256": source_hash(), "arms": [list(arm) for arm in ARMS],
        "primary_contrasts_per_task": [list(pair) for pair in CONTRASTS],
        "scope": "first MetaWorld behavioral panel; six-task study and HMM remain separate required work",
        "episodes": schedule(), "episodes_per_task_condition": 96,
        "development_base_seed": DEVELOPMENT_SEED, "reserved_confirmation_base_seed": 1,
        "seed_value_deviation": "new predeclared development stream; preserve the official formula and eight logical streams",
        "candidate_admission_requires_offline_significance": False,
        "candidate_source": "unchanged existing equal-budget coupling/rank4 components, refitted before validation exposure",
        "common_edit_schedule": "H3 edits only in full H6 rollouts; all components native below H6, including drop-one arms",
        "primary_endpoint": "official simulated binary task success; all 100 elementary steps retained",
        "secondary": ["native task distance", "reward", "runtime", "action norms", "cross-task regression"],
        "analysis": {"unit": "paired initial/goal scenario; verify initial-vector uniqueness and cluster any duplicates",
            "family": "12 primary contrasts: three candidates versus native and their own random controls, across two tasks",
            "intervals": "two-sided paired scenario-cluster bootstrap with Bonferroni simultaneous 95% coverage; 20000 draws, seed 2026090722",
            "p_values": "two-sided exact paired discordance test, Holm across 12; report uncertainty regardless of significance",
            "minimum_useful_success_gain_percentage_points": 5,
            "selection": "positive simultaneous native and matched-control lower bounds and >=5pp native point gain; highest native lower bound, then lower runtime, then fewer components",
            "no_eligible_candidate": "retain native; report negative development evidence, no automatic claim of steering improvement",
            "runtime": "report total planner overhead; no outcome-driven reduction of CEM budget",
            "zero_dose": "required bitwise engineering identity, not a redundant 96-episode condition"},
        "confirmation": {"outcomes_authorized_by_this_freeze": False,
            "next_step": "bind selected unchanged candidates, native and controls to audited untouched scenarios before reveal",
            "prepared_base1_scenarios_must_remain_unopened": True},
        "enabled_initial_conditions": ["native"],
        "candidate_launch_gate": "separate exact component/transfer/full-CEM engineering receipts; cannot bypass with this registry",
        "no_post_outcome_method_or_population_changes": True}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "FROZEN.json", {"protocol_sha256": sha256(output / "protocol.json"),
                                        "outcomes_observed_before_freeze": False})


def validate_coverage(records, expected):
    keys = [(row["episode"], row["logical_rank"], row["environment_seed"]) for row in records]
    expected_keys = [(row["episode"], row["logical_rank"], row["environment_seed"]) for row in expected]
    if keys != expected_keys or len(set(keys)) != len(keys):
        raise ValueError("Missing, duplicated or reordered planning episodes")
    for row in records:
        if row["result"]["elementary_steps"] != 100 or not isinstance(row["result"]["native_success"], bool):
            raise ValueError("Incomplete episode or invalid task-success endpoint")


@torch.no_grad()
def run_native(args):
    protocol_path = args.freeze / "protocol.json"
    frozen = json.loads((args.freeze / "FROZEN.json").read_text())
    if sha256(protocol_path) != frozen["protocol_sha256"]:
        raise ValueError("Behavioral freeze changed")
    protocol = json.loads(protocol_path.read_text())
    if (protocol["role"] != "frozen_planning_development_not_confirmation" or
            protocol["arms"] != [list(arm) for arm in ARMS] or
            protocol["source_sha256"] != source_hash()):
        raise ValueError("Wrong behavioral panel or changed immutable source")
    expected = assigned_rows(protocol["episodes"], args.logical_ranks)
    task = protocol["tasks"][args.task]
    report, native_hash = verified_report(args.native_smoke / args.task)
    if native_hash != task["native_smoke_report_sha256"]:
        raise ValueError("Native runtime evidence changed")
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning import plan_evaluator

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"freeze_sha256": frozen["protocol_sha256"],
            "task": args.task, "arm": "native", "logical_ranks": args.logical_ranks,
            "expected_episodes": expected, "precision": "float32_strict_no_tf32",
            "source_sha256": source_hash(), "fresh_confirmation": False})
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.set_device(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"],
                              "metaworld", "cuda:0", "float32")
        model_versions = _model_versions(backend.model)
        rng_python, rng_numpy, rng_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        records = []
        torch.cuda.reset_peak_memory_stats()
        for rank in sorted(args.logical_ranks):
            rows = [row for row in expected if row["logical_rank"] == rank]
            random.setstate(rng_python)
            np.random.set_state(rng_numpy)
            torch.set_rng_state(rng_torch)
            cfg = OmegaConf.create(task["planning"]["config"])
            cfg.local_seed = rows[0]["local_seed"]
            agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
            env = make_env(cfg)
            try:
                for row in rows:
                    before = time.monotonic()
                    with close_expert_environments(plan_evaluator):
                        result = run_episode(cfg, backend, agent, env, row["environment_seed"])
                    record = {**row, "arm": "native", "result": result,
                        "initial_state_vector": np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist(),
                        "seconds": time.monotonic() - before}
                    validate_coverage([record], [row])
                    records.append(record)
                    write_json(args.output / f"episode-{row['episode']:03d}.json", record)
                    write_json(args.output / "progress.json", {"completed_episodes": len(records),
                        "target": len(expected), "seconds": time.monotonic() - started,
                        "fresh_confirmation": False})
                    print(json.dumps({"task": args.task, "completed": len(records),
                                      "target": len(expected), "seconds": time.monotonic() - started}), flush=True)
            finally:
                env.close()
        validate_coverage(records, expected)
        if model_versions != _model_versions(backend.model):
            raise ValueError("Frozen model parameters changed")
        write_json(args.output / "report.json", {"status": "native_planning_development_shard_complete",
            "task": args.task, "arm": "native", "episodes": len(records),
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "episode_files_sha256": {f"episode-{row['episode']:03d}.json": sha256(args.output / f"episode-{row['episode']:03d}.json") for row in records},
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "scientific_efficacy_measurement": True, "comparative_analysis_complete": False,
            "fresh_confirmation": False, "parameters_unchanged": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("freeze")
    for name in ("vendor", "original-root", "native-smoke", "output"):
        prep.add_argument("--" + name, required=True, type=Path)
    run = sub.add_parser("native")
    for name in ("vendor", "freeze", "native-smoke", "checkpoint", "output"):
        run.add_argument("--" + name, required=True, type=Path)
    run.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    run.add_argument("--logical-ranks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.vendor, args.original_root, args.native_smoke, args.output)
    else:
        run_native(args)


if __name__ == "__main__":
    main()
