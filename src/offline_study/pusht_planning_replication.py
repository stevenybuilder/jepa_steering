"""Native released-Push-T planning replication; never fresh-family confirmation.

Keep the actual source trajectory sampler, action replay, CEM and simulator loop.
The 96 episodes are divided into eight persistent twelve-episode RNG streams.
Record source families and sampled segments; repeated draws do not increase n.
"""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_access import verify_runtime_access
from .author_fit import source_hash
from .author_runtime import validate_cohort
from .backends import JepaBackend
from .behavioral_development import DEVELOPMENT_SEED, assigned_rows, schedule, verified_report
from .droid_native import array_hash
from .intervention_runner import _model_versions
from .inventory import _initial_state_group
from .navigation_replication import validate_records
from .planning_contract import prepare
from .planning_native_smoke import CHECKPOINTS, run_episode
from .protocol import sha256, write_json
from .vendor import use_vendor


class TracedDataset:
    """Forward source data unchanged, only remember the last source row read."""
    def __init__(self, dataset, rows):
        self.dataset, self.rows, self.last_index = dataset, rows, None

    def __len__(self):
        return len(self.dataset)

    def get_seq_length(self, index):
        return self.dataset.get_seq_length(index)

    def __getitem__(self, index):
        result = self.dataset[index]
        self.last_index = index
        return result


@contextlib.contextmanager
def trace_segment(dataset):
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    original = PlanEvaluator.sample_traj_segment_from_dset
    records = []

    def observed(evaluator, cfg, agent, *args, **kwargs):
        if agent.dset is not dataset:
            raise ValueError("Unexpected source dataset")
        result = original(evaluator, cfg, agent, *args, **kwargs)
        obs, states, actions, info = result
        row = dataset.rows[dataset.last_index]
        records.append({"trajectory_id": row["trajectory_id"], "source_index": dataset.last_index,
            "lineage_group": row["lineage_group"], "source_pool": "val",
            "sampled_states_sha256": array_hash(states),
            "sampled_actions_sha256": array_hash(actions.detach().cpu().numpy()),
            "sampled_initial_state": np.asarray(states[0]).tolist(), "segment_frames": len(states)})
        return result

    PlanEvaluator.sample_traj_segment_from_dset = observed
    try:
        yield records
    finally:
        PlanEvaluator.sample_traj_segment_from_dset = original


def checked_cohort(path):
    cohort = json.loads(path.read_text())
    validate_cohort(cohort)
    if not verify_runtime_access(path, cohort) or len(cohort["evaluation"]) != 21:
        raise ValueError("Require the complete already-exposed released replication pool")
    return cohort


def input_hashes(root):
    required = ["states.pth", "rel_actions.pth", "velocities.pth", "seq_lengths.pkl"]
    required += [f"obses/episode_{i:03d}.mp4" for i in range(21)]
    if (root / "shapes.pkl").exists():
        required.append("shapes.pkl")
    return {name: sha256(root / name) for name in required}


def freeze(args):
    from .pusht_planning_check import verify_engineering
    use_vendor(args.vendor)
    cohort = checked_cohort(args.cohort)
    contract = prepare(args.vendor, "pusht")
    engineering, engineering_hash = verify_engineering(args.engineering, contract, sha256(args.cohort))
    for name in ("backends.py", "model_loader.py", "planning_native_smoke.py", "planning_contract.py"):
        if sha256(args.baseline_code / "src/offline_study" / name) != sha256(Path(__file__).with_name(name)):
            raise ValueError("Previously validated native execution changed: " + name)
    contract["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"role": "native_pusht_released_pool_planning_replication",
        "planning_contract": contract, "episodes": schedule(), "episodes_per_condition_total": 96,
        "source_sha256": source_hash(), "checkpoint_sha256": CHECKPOINTS["pusht"],
        "cohort_sha256": sha256(args.cohort), "input_files_sha256": input_hashes(args.data_root / "val"),
        "engineering_report_sha256": engineering_hash, "native_only": True,
        "receiving_device_uuid": engineering["device_uuid"],
        "authorized_basis": "user subsequent six-task end-to-end planning replication instruction; prior validation exposure preserved",
        "source_families": len({r["lineage_group"] for r in cohort["evaluation"]}),
        "sampling": "actual upstream sample_traj_segment_from_dset; uniform row then valid offset, native replay",
        "replication_not_fresh_family_confirmation": True, "fresh_confirmation": False,
        "training_seed_history_complete": False, "candidate_outcomes_observed_before_freeze": False,
        "no_success_dependent_stopping_or_exclusion": True})
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json")})


@torch.no_grad()
def run(args):
    use_vendor(args.vendor)
    from app.plan_common.datasets.pusht_dset import PushTDataset
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    p = json.loads((args.freeze / "protocol.json").read_text())
    frozen_hash = sha256(args.freeze / "protocol.json")
    if (json.loads((args.freeze / "FROZEN.json").read_text())["protocol_sha256"] != frozen_hash or
            p["role"] != "native_pusht_released_pool_planning_replication" or
            p["source_sha256"] != source_hash() or p["cohort_sha256"] != sha256(args.cohort) or
            p["episodes"] != schedule() or not p["native_only"] or p["fresh_confirmation"]):
        raise ValueError("Changed source, sample size or native replication freeze")
    cohort = checked_cohort(args.cohort)
    if input_hashes(args.data_root / "val") != p["input_files_sha256"]:
        raise ValueError("Released validation inputs changed")
    contract = prepare(args.vendor, "pusht")
    contract["config"]["meta"]["seed"] = DEVELOPMENT_SEED
    if contract != p["planning_contract"]:
        raise ValueError("Native planning settings changed")
    expected = assigned_rows(p["episodes"], args.logical_ranks)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"freeze_sha256": frozen_hash,
            "source_sha256": source_hash(), "task": "pusht", "arm": "native",
            "logical_ranks": args.logical_ranks, "expected_episodes": expected, "fresh_confirmation": False})
        torch.cuda.set_device(0)
        from .fixed_response_behavior import device_uuid
        if device_uuid() != p["receiving_device_uuid"]:
            raise ValueError("This GPU has no matching full native/repeat engineering")
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["pusht"], "pusht", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        dataset = PushTDataset(data_path=str(args.data_root / "val"), n_rollout=None,
            transform=backend.preprocessor.transform, normalize_action=True, with_velocity=True)
        by_index = {r["index"]: r for r in cohort["evaluation"]}
        if len(dataset) != 21 or set(by_index) != set(range(21)):
            raise ValueError("Changed released source-row membership")
        for i, row in by_index.items():
            if dataset.get_seq_length(i) != row["length"] or _initial_state_group(dataset.states[i, 0]) != row["lineage_group"]:
                raise ValueError("Actual source family or length differs from exposure manifest")
        dataset = TracedDataset(dataset, by_index)
        original_unroll = backend.model.unroll
        global_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        records = []
        torch.cuda.reset_peak_memory_stats()
        for rank in sorted(args.logical_ranks):
            random.setstate(global_rng[0]); np.random.set_state(global_rng[1]); torch.set_rng_state(global_rng[2])
            rows = [r for r in expected if r["logical_rank"] == rank]
            cfg = OmegaConf.create(contract["config"]); cfg.local_seed = rows[0]["local_seed"]
            agent = GC_Agent(cfg, backend.model, dset=dataset, preprocessor=backend.preprocessor)
            env = make_env(cfg)
            try:
                for row in rows:
                    before = time.monotonic()
                    calls, actions = [], []
                    def observed(context, act_suffix=None, **kwargs):
                        result = original_unroll(context, act_suffix=act_suffix, **kwargs)
                        if any(not torch.isfinite(result[k]).all() for k in ("visual", "proprio")):
                            raise ValueError("Nonfinite native forecast")
                        calls.append(tuple(act_suffix.shape[:2]))
                        return result
                    original_act = agent.act
                    def act(*a, **kw):
                        result = original_act(*a, **kw)
                        actions.append(result.detach().cpu().tolist())
                        return result
                    backend.model.unroll = agent.planner.unroll = observed
                    agent.act = act
                    try:
                        with trace_segment(dataset) as sampled:
                            result = run_episode(cfg, backend, agent, env, row["environment_seed"])
                    finally:
                        backend.model.unroll = agent.planner.unroll = original_unroll
                        agent.act = original_act
                    if len(sampled) != 1 or sampled[0]["segment_frames"] != 31:
                        raise ValueError("Expected exactly one native H6 replay segment")
                    record = {**row, "arm": "native", "result": result, "source_segment": sampled[0],
                        "unroll_calls": calls, "planned_actions": actions, "seconds": time.monotonic() - before}
                    validate_records([record], [row]); records.append(record)
                    write_json(args.output / f"episode-{row['episode']:03d}.json", record)
                    progress = {"completed": len(records), "target": len(expected), "seconds": time.monotonic() - started}
                    write_json(args.output / "progress.json", progress); print(json.dumps(progress), flush=True)
            finally:
                env.close()
        validate_records(records, expected)
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen weights changed")
        write_json(args.output / "report.json", {"status": "native_pusht_planning_replication_shard_complete",
            "task": "pusht", "arm": "native", "episodes": len(records),
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "episode_files_sha256": {f"episode-{r['episode']:03d}.json": sha256(args.output / f"episode-{r['episode']:03d}.json") for r in records},
            "source_families_sampled": len({r["source_segment"]["lineage_group"] for r in records}),
            "seconds": time.monotonic() - started, "parameters_unchanged": True,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "fresh_confirmation": False,
            "comparative_analysis_complete": False, "training_seed_history_complete": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "native"):
        p = sub.add_parser(command)
        for name in ("vendor", "cohort", "data-root", "output"):
            p.add_argument("--" + name, type=Path, required=True)
        if command == "freeze":
            for name in ("engineering", "baseline-code"):
                p.add_argument("--" + name, type=Path, required=True)
        else:
            for name in ("freeze", "checkpoint"):
                p.add_argument("--" + name, type=Path, required=True)
            p.add_argument("--logical-ranks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    freeze(args) if args.command == "freeze" else run(args)


if __name__ == "__main__":
    main()
