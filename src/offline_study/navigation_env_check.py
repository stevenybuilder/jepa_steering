"""CPU-only PointMaze native simulator/renderer check on an excluded seed."""
import argparse
import copy
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .droid_native import array_hash
from .planning_contract import prepare
from .planning_native_smoke import SMOKE_SEED
from .protocol import sha256, write_json
from .vendor import use_vendor


def observation(visual, info):
    # Native PixelWrapper returns pixels separately from info["proprio"].
    visual = torch.as_tensor(visual)
    proprio = torch.as_tensor(info["proprio"])
    if visual.shape[-2:] != (224, 224) or visual.max() == visual.min() or not torch.isfinite(proprio).all():
        raise ValueError("Malformed native PointMaze pixels or proprioception")
    return {"visual_sha256": array_hash(visual.numpy()), "visual_shape": list(visual.shape),
            "visual_dtype": str(visual.dtype), "proprio_sha256": array_hash(proprio.numpy())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise ValueError("Hide CUDA explicitly for this CPU-only setup test")
    use_vendor(args.vendor)
    contract = prepare(args.vendor, "pointmaze")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"role": "cpu_native_pointmaze_setup_not_policy_evidence",
            "source_sha256": sha256(Path(__file__)), "planning": contract,
            "seed": SMOKE_SEED, "repetitions": 2, "elementary_actions": "30 fixed zero actions",
            "native_random_state_generator": True, "policy_model_calls": 0,
            "fresh_confirmation": False, "scientific_efficacy_measurement": False})
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        records = []
        for repetition in range(2):
            random.seed(SMOKE_SEED)
            np.random.seed(SMOKE_SEED)
            torch.manual_seed(SMOKE_SEED)
            cfg = OmegaConf.create(copy.deepcopy(contract["config"]))
            cfg.device = "cpu"
            cfg.local_seed = cfg.meta.seed = SMOKE_SEED
            env = make_env(cfg)
            try:
                initial_state, goal_state = env.sample_random_init_goal_states(SMOKE_SEED)
                goal, goal_info = env.prepare(SMOKE_SEED, goal_state)
                initial, initial_info = env.prepare(SMOKE_SEED, initial_state)
                if env.max_steps() != 30:
                    raise ValueError("Source episode length changed")
                obs, rewards, dones, infos = env.step_multiple(torch.zeros(30, 2))
                if len(obs) != 30 or not dones[-1]:
                    raise ValueError("Incomplete native simulator episode")
                record = {"initial": observation(initial, initial_info), "goal": observation(goal, goal_info),
                    "final": observation(obs[-1], infos[-1]), "initial_state": initial_state.tolist(),
                    "goal_state": goal_state.tolist(), "final_state": np.asarray(infos[-1]["state"]).tolist(),
                    "elementary_steps": len(obs)}
                records.append(record)
                write_json(args.output / f"repetition-{repetition}.json", record)
            finally:
                env.close()
        if records[0] != records[1]:
            raise ValueError("Source state sampling, renderer or physics not repeatable")
        write_json(args.output / "report.json", {"status": "cpu_native_pointmaze_simulator_passed",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "files_sha256": {f"repetition-{i}.json": sha256(args.output / f"repetition-{i}.json") for i in range(2)},
            "versions": {name: importlib.metadata.version(name) for name in ("gym", "mujoco-py", "D4RL", "numpy")},
            "same_seed_pixels_and_states_exact": True, "seconds": time.monotonic() - started,
            "policy_model_calls": 0, "fresh_confirmation": False, "scientific_efficacy_measurement": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "policy_model_calls": 0})
        raise


if __name__ == "__main__":
    main()
