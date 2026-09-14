"""Full native navigation engineering episodes, never behavioral selection/confirmation.

Use each released task's actual simulator and CEM budget twice on one excluded
smoke seed. No new policy, altered objective, truncated episode or fitted edit.
"""
import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_native_smoke import SMOKE_SEED, run_episode
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


CHECKPOINTS = {
    "pointmaze": "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc",
    "wall": "8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e",
}


def validate_complete(result, calls):
    if (result["elementary_steps"] != 30 or result["published_candidate_count"] != 300 or
            len(result["planning_calls"]) != 1 or result["planning_calls"][0]["iterations"] != 30 or
            result["planning_calls"][0]["returned_model_actions"] != 6):
        raise ValueError("Incomplete or changed official navigation episode")
    if calls != [(6, 300), (6, 1)] * 30:
        raise ValueError("Missing/reordered CEM candidate or mean forecasts")


def comparison_record(result, action_hashes):
    return {key: result[key] for key in ("initial_sha256", "goal_sha256", "native_success",
        "native_state_distance", "native_reward", "elementary_steps", "observed_frames")} | {
        "action_sha256": action_hashes}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=tuple(CHECKPOINTS), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        use_vendor(args.vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent

        contract = prepare(args.vendor, args.task)
        write_json(args.output / "protocol.json", {
            "task": args.task, "role": "native_navigation_engineering_not_efficacy",
            "planning_contract": contract, "source_sha256": source_hash(),
            "checkpoint_sha256": CHECKPOINTS[args.task], "repetitions": 2,
            "smoke_seed_excluded_from_confirmation": SMOKE_SEED,
            "fresh_confirmation": False, "selection_from_outcomes": False,
            "precision": "float32_strict_no_tf32", "protected_outcomes_accessed": False})
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS[args.task], args.task,
                              "cuda:0", "float32")
        if backend.model.ctxt_window != 2 or backend.provenance["normalization_dataset"] != args.task:
            raise ValueError("Wrong task model, normalization or planning context")
        versions = _model_versions(backend.model)
        original_unroll = backend.model.unroll
        records, comparisons = [], []
        for repetition in range(2):
            random.seed(SMOKE_SEED)
            np.random.seed(SMOKE_SEED)
            torch.manual_seed(SMOKE_SEED)
            cfg = OmegaConf.create(contract["config"])
            cfg.meta.seed = cfg.local_seed = SMOKE_SEED
            calls, action_hashes = [], []

            def observed(context, act_suffix=None, **kwargs):
                result = original_unroll(context, act_suffix=act_suffix, **kwargs)
                if any(not torch.isfinite(result[key]).all() for key in ("visual", "proprio")):
                    raise ValueError("Nonfinite native navigation forecast")
                calls.append(tuple(act_suffix.shape[:2]))
                write_json(args.output / "progress.json", {"repetition": repetition,
                    "completed_unroll_calls": len(calls), "fresh_confirmation": False})
                return result

            backend.model.unroll = observed
            agent = GC_Agent(cfg, backend.model, dset=None, preprocessor=backend.preprocessor)
            original_act = agent.act

            def actor(*args, **kwargs):
                action = original_act(*args, **kwargs)
                value = action.detach().cpu().contiguous()
                action_hashes.append(hashlib.sha256(str(value.shape).encode() + value.numpy().tobytes()).hexdigest())
                return action

            agent.act = actor
            env = make_env(cfg)
            try:
                result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
                if env.max_steps() != 30:
                    raise ValueError("Official navigation environment cap changed")
            finally:
                env.close()
                backend.model.unroll = original_unroll
            validate_complete(result, calls)
            comparisons.append(comparison_record(result, action_hashes))
            records.append({"result": result, "unroll_calls": calls, "action_sha256": action_hashes})
            write_json(args.output / f"repetition-{repetition}.json", records[-1])
        if comparisons[0] != comparisons[1] or _model_versions(backend.model) != versions:
            raise ValueError("Repeated native actions/outcomes differ or frozen weights changed")
        installed = {}
        for package in ("torch", "gym", "mujoco", "mujoco-py", "d4rl", "numpy"):
            try:
                installed[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                installed[package] = None
        write_json(args.output / "report.json", {
            "status": "full_native_navigation_engineering_passed", "task": args.task,
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "repetition_sha256": [sha256(args.output / f"repetition-{i}.json") for i in range(2)],
            "same_seed_actions_and_outcomes_exact": True, "parameters_unchanged": True,
            "full_native_cem_schedule": True, "fresh_confirmation": False,
            "scientific_efficacy_measurement": False, "protected_outcomes_accessed": False,
            "model_provenance": backend.provenance, "installed_versions": installed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "seconds": time.monotonic() - started})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
