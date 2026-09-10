"""Recover exact already-recorded native goals for common-pixel paired evaluation.

Only a byte image matching the pre-existing baseline hash is accepted. Repeated
rendering cannot choose a physical goal, candidate or policy outcome. No baseline
hash, episode membership, intervention, seed or pixel-equality gate is relaxed.
"""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .behavioral_development import DEVELOPMENT_SEED, schedule, validate_coverage, verified_report
from .planning_contract import prepare
from .planning_env_smoke import observation_digest
from .planning_scenarios import prepare_episode
from .protocol import sha256, write_json
from .vendor import use_vendor


def native_records(root, task, freeze_hash):
    records, bindings = [], {}
    for shard in sorted((root / task / "native").glob("shard-*")):
        if not shard.is_dir():
            continue
        report, digest = verified_report(shard)
        launch = json.loads((shard / "protocol.json").read_text())
        if (report["status"] != "native_planning_development_shard_complete" or
                report["task"] != task or report["arm"] != "native" or
                launch["freeze_sha256"] != freeze_hash or
                report["protocol_sha256"] != sha256(shard / "protocol.json")):
            raise ValueError("Unbound original native reference")
        bindings[str(shard / "report.json")] = digest
        for name, expected in report["episode_files_sha256"].items():
            if Path(name).name != name or sha256(shard / name) != expected:
                raise ValueError("Original native episode changed")
            row = json.loads((shard / name).read_text())
            records.append(row)
    records.sort(key=lambda r: r["episode"])
    validate_coverage(records, schedule())
    return records, bindings


def restore_goal(initial, goal, goal_state, saved, metadata):
    """Fix delivered pixels, not the pairing checker; retain native physics."""
    if (observation_digest(initial) != metadata["initial_sha256"] or
            not np.array_equal(np.asarray(goal_state), np.asarray(metadata["goal_state"])) or
            not torch.equal(goal["proprio"].cpu(), saved["goal"]["proprio"]) or
            observation_digest(saved["goal"]) != metadata["goal_sha256"]):
        raise ValueError("Goal/initial physical input differs from the frozen native stimulus")
    # Guard against an unrelated renderer/camera failure. Accepted delivered
    # observations below are still BITWISE equal, never merely within tolerance.
    diff = (goal["visual"].cpu().to(torch.int16) - saved["goal"]["visual"].to(torch.int16)).abs()
    if int(diff.max()) > 1 or int(diff.ne(0).sum()) > 64:
        raise ValueError("Renderer difference exceeds the engineering-only one-level/64-value bound")
    result = goal.clone()
    for key in ("visual", "proprio"):
        result[key] = saved["goal"][key].to(goal[key]).clone()
    if observation_digest(result) != metadata["goal_sha256"]:
        raise ValueError("Canonical goal delivery is not bitwise exact")
    return result


class GoalBank:
    def __init__(self, root, task, freeze_hash):
        self.root, self.task = root, task
        report, self.report_hash = verified_report(root)
        protocol = json.loads((root / "protocol.json").read_text())
        if (report["status"] != "all_192_existing_native_goal_images_recovered_exactly" or
                report["protocol_sha256"] != sha256(root / "protocol.json") or
                protocol["freeze_sha256"] != freeze_hash or report["episodes"] != 192 or
                report["pixel_pairing_gate_relaxed"] is not False):
            raise ValueError("Unbound/incomplete canonical native goal bank")
        self.bindings = report["record_files_sha256"]
        self.cache = {}

    def load(self, row):
        episode = row["episode"]
        if episode not in self.cache:
            name = f"{self.task}/episode-{episode:03d}.json"
            path = self.root / name
            if sha256(path) != self.bindings[name]:
                raise ValueError("Canonical goal metadata changed")
            metadata = json.loads(path.read_text())
            tensor_path = path.with_suffix(".pt")
            if sha256(tensor_path) != metadata["tensor_sha256"]:
                raise ValueError("Canonical raw goal changed")
            tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
            self.cache[episode] = metadata, tensors
        metadata, tensors = self.cache[episode]
        if any(metadata[key] != row[key] for key in ("episode", "logical_rank", "local_seed", "environment_seed")):
            raise ValueError("Wrong logical stream for canonical goal")
        return metadata, tensors

    @contextlib.contextmanager
    def deliver(self, row):
        from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
        metadata, tensors = self.load(row)
        original = PlanEvaluator.set_episode

        def bound(evaluator, cfg, agent, env, seed, task_idx=-1):
            if seed != row["environment_seed"]:
                raise ValueError("Unexpected environment seed")
            initial, goal, expert, success = original(evaluator, cfg, agent, env, seed, task_idx)
            if np.asarray(env.proprio_env.unwrapped._last_rand_vec).tolist() != metadata["rand_vec"]:
                raise ValueError("Physical task differs from canonical goal")
            if not torch.equal(evaluator.expert_actions.detach().cpu(), tensors["expert_actions"]):
                raise ValueError("Original expert action sequence changed")
            return initial, restore_goal(initial, goal, evaluator.state_g, tensors, metadata), expert, success

        PlanEvaluator.set_episode = bound
        try:
            yield
        finally:
            PlanEvaluator.set_episode = original


def verify_delivery(root, bank_hash, freeze_hash):
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if (report["status"] != "canonical_native_goal_delivery_verified" or
            report["protocol_sha256"] != sha256(root / "protocol.json") or
            report["records_sha256"] != sha256(root / "records.json") or
            protocol["goal_bank_report_sha256"] != bank_hash or
            protocol["freeze_sha256"] != freeze_hash or report["repetitions"] != 24 or
            not all(report[k] is True for k in ("all_192_original_stimuli_verified",
                "all_source_deliveries_bitwise_exact", "expert_physics_and_actions_unchanged", "planner_rng_unchanged")) or
            report["pixel_pairing_gate_relaxed"] is not False):
        raise ValueError("Missing completed source-native goal delivery verification")
    import inspect
    import hashlib
    source = inspect.getsource(restore_goal) + inspect.getsource(GoalBank)
    if protocol["goal_delivery_implementation_sha256"] != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Verified goal-delivery implementation changed")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "freeze", "reference-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    freeze_hash = sha256(args.freeze / "protocol.json")
    if json.loads((args.freeze / "FROZEN.json").read_text())["protocol_sha256"] != freeze_hash:
        raise ValueError("Scientific freeze changed")
    references, bindings = {}, {}
    for task in ("reach", "reach-wall"):
        references[task], task_bindings = native_records(args.reference_root, task, freeze_hash)
        bindings.update(task_bindings)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"role": "exact_existing_native_stimulus_recovery_not_new_cohort",
        "freeze_sha256": freeze_hash, "native_reference_reports_sha256": bindings,
        "source_sha256": sha256(Path(__file__)), "tasks": ["reach", "reach-wall"],
        "episodes_per_task": 96, "maximum_render_attempts_per_existing_hash": 16,
        "acceptance": "same existing initial vector, initial and goal bytes/hash; no outcome-dependent choice",
        "candidate_outcomes_used": False, "native_outcomes_used_for_selection": False,
        "new_goal_or_episode_substitution": False, "confirmation_authorized": False})
    started, records = time.monotonic(), {}
    try:
        for task, rows in references.items():
            for original in rows:
                episode = original["episode"]
                attempts, recovered = [], False
                for attempt in range(16):
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    cfg = OmegaConf.create(prepare(args.vendor, task)["config"])
                    cfg.meta.seed = DEVELOPMENT_SEED
                    cfg.local_seed = original["local_seed"]
                    agent = SimpleNamespace(local_generator=torch.Generator().manual_seed(cfg.local_seed))
                    env = make_env(cfg)
                    try:
                        record, tensors = prepare_episode(cfg, agent, env, schedule()[episode])
                    finally:
                        env.close()
                    attempts.append({key: record[key] for key in ("initial_sha256", "goal_sha256", "rand_vec")})
                    if record["rand_vec"] != original["initial_state_vector"]:
                        raise ValueError("Regeneration changed the original physical task")
                    if any(record[key] != original["result"][key] for key in ("initial_sha256", "goal_sha256")):
                        continue
                    folder = args.output / task
                    folder.mkdir(exist_ok=True)
                    stem = f"episode-{episode:03d}"
                    torch.save(tensors, folder / (stem + ".pt"))
                    write_json(folder / (stem + ".json"), {**record, "attempts": attempts,
                        "tensor_sha256": sha256(folder / (stem + ".pt")),
                        "exact_original_baseline_observations_recovered": True})
                    records[f"{task}/{stem}.json"] = sha256(folder / (stem + ".json"))
                    recovered = True
                    break
                if not recovered:
                    write_json(args.output / "unrecovered.json", {"task": task, "episode": episode, "attempts": attempts})
                    raise ValueError("Original goal bytes not recovered in fixed bound; cannot substitute")
                write_json(args.output / "progress.json", {"recovered": len(records), "target": 192,
                    "seconds": time.monotonic() - started, "learned_policy_calls": 0})
                print(json.dumps({"task": task, "episode": episode, "attempts": len(attempts), "recovered": len(records)}), flush=True)
        write_json(args.output / "report.json", {"status": "all_192_existing_native_goal_images_recovered_exactly",
            "protocol_sha256": sha256(args.output / "protocol.json"), "record_files_sha256": records,
            "episodes": 192, "seconds": time.monotonic() - started, "learned_policy_calls": 0,
            "pixel_pairing_gate_relaxed": False, "intervention_or_analysis_changed": False,
            "protected_confirmation_accessed": False, "production_delivery_check_pending": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "recovered": len(records), "replacement_allowed": False})
        raise


if __name__ == "__main__":
    main()
