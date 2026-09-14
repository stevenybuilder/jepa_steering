"""Outcome-blind repeat audit of native MetaWorld expert goals; no learned model."""
import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from offline_study.evaluation.behavioral_development import DEVELOPMENT_SEED, schedule
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_scenarios import prepare_episode
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


def tensor_digest(value):
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(str(data.dtype).encode() + str(data.shape).encode() + data.numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "output", "reference-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    logging.getLogger().setLevel(logging.WARNING)
    args.output.mkdir(parents=True, exist_ok=False)
    targets = [("reach", 20), ("reach", 35), ("reach-wall", 25), ("reach", 0)]
    write_json(args.output / "protocol.json", {"role": "native_expert_goal_repeat_diagnostic_no_model_outcomes",
        "targets": targets, "repetitions": 6, "source_sha256": sha256(Path(__file__)),
        "fresh_env_each_repeat": True, "no_method_or_episode_selection": True,
        "changes_to_production": False, "confirmation_access": False})
    started, records = time.monotonic(), []
    for task, episode in targets:
        first = None
        for repetition in range(6):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            cfg = OmegaConf.create(prepare(args.vendor, task)["config"])
            cfg.meta.seed = DEVELOPMENT_SEED
            row = schedule()[episode]
            cfg.local_seed = row["local_seed"]
            agent = SimpleNamespace(local_generator=torch.Generator().manual_seed(cfg.local_seed))
            env = make_env(cfg)
            try:
                record, tensors = prepare_episode(cfg, agent, env, row)
            finally:
                env.close()
            stem = f"{task}-{episode:03d}-{repetition}"
            torch.save(tensors, args.output / (stem + ".pt"))
            if first is None:
                first = (record, tensors)
            ref_record, ref_tensors = first
            record.update(task=task, repetition=repetition,
                tensor_file_sha256=sha256(args.output / (stem + ".pt")),
                initial_equals_first={k: torch.equal(v, ref_tensors["initial"][k]) for k, v in tensors["initial"].items()},
                goal_equals_first={k: torch.equal(v, ref_tensors["goal"][k]) for k, v in tensors["goal"].items()},
                goal_modality_sha256={k: tensor_digest(v) for k, v in tensors["goal"].items()},
                goal_state_equals_first=record["goal_state"] == ref_record["goal_state"],
                expert_actions_equal_first=torch.equal(tensors["expert_actions"], ref_tensors["expert_actions"]),
                goal_visual_differing_values=int((tensors["goal"]["visual"] != ref_tensors["goal"]["visual"]).sum()),
                goal_visual_max_abs_difference=float((tensors["goal"]["visual"].float()-ref_tensors["goal"]["visual"].float()).abs().max()))
            references = []
            for path in sorted((args.reference_root / task).glob(f"*/*/episode-{episode:03d}.json")):
                previous = json.loads(path.read_text())
                references.append({"path": str(path), "arm": previous["arm"],
                    "initial_sha256": previous["result"]["initial_sha256"],
                    "goal_sha256": previous["result"]["goal_sha256"]})
            record["historical_stimulus_hashes_only"] = references
            write_json(args.output / (stem + ".json"), record)
            records.append(record)
            print(json.dumps({k: record[k] for k in ("task", "episode", "repetition", "goal_equals_first",
                "goal_state_equals_first", "expert_actions_equal_first", "goal_sha256",
                "goal_visual_differing_values", "goal_visual_max_abs_difference")}), flush=True)
    write_json(args.output / "report.json", {"status": "native_goal_repeat_diagnostic_complete",
        "protocol_sha256": sha256(args.output / "protocol.json"), "repetitions": len(records),
        "records_sha256": {f"{r['task']}-{r['episode']:03d}-{r['repetition']}.json":
            sha256(args.output / f"{r['task']}-{r['episode']:03d}-{r['repetition']}.json") for r in records},
        "all_physical_goals_repeat_exactly": all(r["goal_state_equals_first"] for r in records),
        "all_goal_pixels_repeat_exactly": all(r["goal_equals_first"]["visual"] for r in records),
        "all_goal_proprio_repeat_exactly": all(r["goal_equals_first"]["proprio"] for r in records),
        "seconds": time.monotonic() - started, "learned_model_calls": 0,
        "production_fix_or_equivalence_established": False})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
