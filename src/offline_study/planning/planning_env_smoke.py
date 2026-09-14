"""Check official simulator reset, expert/replay goal setup, and seed pairing.

Uses only fit-only Push-T data and explicitly non-confirmatory smoke seeds. No CEM
efficacy trial or confirmation outcome is produced by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_runtime import open_normalized_dataset, validate_cohort
from offline_study.planning.planning_contract import prepare
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


class SingleFitTrajectory:
    def __init__(self, dataset, index):
        self.dataset, self.index = dataset, index

    def __len__(self):
        return 1

    def get_seq_length(self, index):
        if index != 0:
            raise IndexError(index)
        return self.dataset.get_seq_length(self.index)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.dataset[self.index]


def observation_digest(observation):
    digest = hashlib.sha256()
    for key in ("visual", "proprio"):
        value = observation[key].detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite simulator observation")
        digest.update(key.encode() + str(value.shape).encode() + value.numpy().tobytes())
    if observation["visual"].shape != (1, 3, 224, 224) or observation["visual"].max() == observation["visual"].min():
        raise ValueError("Incorrect or empty simulator image")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "original-root", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=["reach", "reach-wall", "pusht"],
                        default=["reach", "reach-wall", "pusht"])
    args = parser.parse_args()
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        results = {}
        for task in args.tasks:
            reference = prepare(args.vendor, task)
            records = []
            for repetition in range(2):
                seed = 2026090719
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                cfg = OmegaConf.create(reference["config"])
                cfg.meta.seed = seed
                env = make_env(cfg)
                try:
                    agent = SimpleNamespace(local_generator=torch.Generator().manual_seed(seed))
                    fit_id = None
                    if task == "pusht":
                        cohort = json.loads((args.original_root / "cohorts/pusht/cohort.json").read_text())
                        validate_cohort(cohort)
                        dataset = open_normalized_dataset("pusht", args.data_root / "pusht_noise", cohort["reference_config"], True)
                        agent.dset = SingleFitTrajectory(dataset, cohort["fit"][0]["index"])
                        fit_id = cohort["fit"][0]["trajectory_id"]
                        stats = get_data_stats("pusht")
                        agent.preprocessor = Preprocessor(**{key: torch.tensor(stats[key]) for key in
                            ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")},
                            transform=None)
                    env.reset(seed=seed, task_idx=0)
                    env.proprio_env.unwrapped._freeze_rand_vec = False
                    env.proprio_env.unwrapped.seeded_rand_vec = True
                    env.seed(seed)
                    evaluator = PlanEvaluator(cfg, agent)
                    init, goal, expert, expert_success = evaluator.set_episode(cfg, agent, env, seed, task_idx=0)
                    # Check the authors' endpoint function, independently of a learned policy.
                    self_success = env.eval_state(evaluator.state_g, evaluator.state_g)["success"]
                    if not self_success:
                        raise ValueError("Goal state does not satisfy its own success criterion")
                    records.append({"initial_sha256": observation_digest(init), "goal_sha256": observation_digest(goal),
                        "goal_state": np.asarray(evaluator.state_g).tolist(), "expert_frames": len(expert),
                        "expert_success": float(expert_success), "goal_self_success": bool(self_success),
                        "max_steps": env.max_steps(), "fit_trajectory_id": fit_id})
                finally:
                    env.close()
            if records[0] != records[1]:
                raise ValueError("Identical smoke seeds do not reproduce the paired initial/goal setup: " + task)
            results[task] = {"same_seed_pairing": True, "smoke_seed": seed, **records[0]}
        versions = {name: importlib.metadata.version(name) for name in
                    ("metaworld", "mujoco", "gym", "gymnasium", "pymunk", "pygame", "torch", "numpy")}
        write_json(args.output / "report.json", {"status": "official_simulator_goal_setup_smoke_passed",
            "tasks": results, "versions": versions, "cem_planning_executed": False,
            "fresh_confirmation": False, "protected_outcomes_accessed": False,
            "smoke_seed_excluded_from_confirmation": 2026090719})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
