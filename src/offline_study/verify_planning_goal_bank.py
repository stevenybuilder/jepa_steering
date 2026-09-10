"""Test exact native goal delivery on source environments, without learned policy calls."""
import argparse
import hashlib
import inspect
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .behavioral_development import DEVELOPMENT_SEED, schedule
from .planning_contract import prepare
from .planning_goal_bank import GoalBank, native_records, restore_goal
from .planning_scenarios import prepare_episode
from .protocol import sha256, write_json
from .vendor import use_vendor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "freeze", "reference-root", "goal-bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    freeze_hash = sha256(args.freeze / "protocol.json")
    if json.loads((args.freeze / "FROZEN.json").read_text())["protocol_sha256"] != freeze_hash:
        raise ValueError("Scientific freeze changed")
    banks = {task: GoalBank(args.goal_bank, task, freeze_hash) for task in ("reach", "reach-wall")}
    for task, bank in banks.items():
        originals, _ = native_records(args.reference_root, task, freeze_hash)
        for row in originals:
            metadata, _ = bank.load(row)
            if any(metadata[key] != row["result"][key] for key in ("initial_sha256", "goal_sha256")):
                raise ValueError("Cached stimulus does not match actual original baseline")
    args.output.mkdir(parents=True, exist_ok=False)
    targets = [("reach", 20), ("reach", 35), ("reach-wall", 25), ("reach", 0)]
    write_json(args.output / "protocol.json", {"role": "native_goal_delivery_engineering_no_learned_policy",
        "goal_bank_report_sha256": banks["reach"].report_hash, "freeze_sha256": freeze_hash,
        "all_original_192_stimuli_verified": True, "targets": targets, "repetitions": 6,
        "goal_delivery_implementation_sha256": hashlib.sha256(
            (inspect.getsource(restore_goal) + inspect.getsource(GoalBank)).encode()).hexdigest(),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            ("planning_goal_bank.py", "verify_planning_goal_bank.py", "planning_scenarios.py")},
        "selection_uses_candidate_outcomes": False, "confirmation_authorized": False})
    records, started = [], time.monotonic()
    try:
        for task, episode in targets:
            for repetition in range(6):
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                cfg = OmegaConf.create(prepare(args.vendor, task)["config"])
                cfg.meta.seed = DEVELOPMENT_SEED
                row = schedule()[episode]
                cfg.local_seed = row["local_seed"]
                agent = SimpleNamespace(local_generator=torch.Generator().manual_seed(cfg.local_seed))
                rng_before = agent.local_generator.get_state().clone()
                env = make_env(cfg)
                try:
                    with banks[task].deliver(row):
                        record, _ = prepare_episode(cfg, agent, env, row)
                finally:
                    env.close()
                expected, _ = banks[task].load(row)
                if any(record[key] != expected[key] for key in ("initial_sha256", "goal_sha256", "rand_vec", "goal_state")):
                    raise ValueError("Source delivery failed exact pairing")
                if not torch.equal(rng_before, agent.local_generator.get_state()):
                    raise ValueError("Goal delivery consumed planner candidate RNG")
                record.update(task=task, repetition=repetition, exact_pairing=True, planner_rng_unchanged=True)
                records.append(record)
                print(json.dumps({"task": task, "episode": episode, "repeat": repetition, "exact_pairing": True}), flush=True)
        write_json(args.output / "records.json", records)
        write_json(args.output / "report.json", {"status": "canonical_native_goal_delivery_verified",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "records_sha256": sha256(args.output / "records.json"), "repetitions": len(records),
            "all_192_original_stimuli_verified": True, "all_source_deliveries_bitwise_exact": True,
            "expert_physics_and_actions_unchanged": True, "planner_rng_unchanged": True,
            "learned_policy_calls": 0, "seconds": time.monotonic() - started,
            "pixel_pairing_gate_relaxed": False, "scientific_reruns_still_required": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "passed_repetitions": len(records)})
        raise


if __name__ == "__main__":
    main()
