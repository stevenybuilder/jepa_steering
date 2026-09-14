"""Receiving-device Push-T native/repeat check on one excluded fitting scenario."""
import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import open_normalized_dataset
from offline_study.models.backends import JepaBackend
from offline_study.evaluation.behavioral_development import verified_report
from offline_study.tasks.droid.droid_native import array_hash
from offline_study.evaluation.fixed_response_behavior import device_uuid
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.data.inventory import _initial_state_group
from offline_study.tasks.navigation.navigation_smoke import comparison_record, validate_complete
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_env_smoke import SingleFitTrajectory
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.core.protocol import sha256, write_json
from offline_study.tasks.pusht.pusht_planning_replication import checked_cohort
from offline_study.models.vendor import use_vendor


def verify_engineering(root, contract, cohort_hash):
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if (report["status"] != "full_native_pusht_engineering_passed" or
            report["protocol_sha256"] != sha256(root / "protocol.json") or
            protocol["planning_contract"] != contract or protocol["cohort_sha256"] != cohort_hash or
            protocol["source_sha256"] != source_hash() or
            protocol["checkpoint_sha256"] != CHECKPOINTS["pusht"] or
            protocol["smoke_seed"] != SMOKE_SEED or protocol["repetitions"] != 2 or
            protocol["precision"] != "float32_strict_no_tf32" or report["fresh_confirmation"] or
            not report["same_seed_actions_and_outcomes_exact"] or not report["parameters_unchanged"]):
        raise ValueError("Missing full source-bound receiving Push-T engineering")
    comparisons = []
    if len(report["repetition_sha256"]) != 2:
        raise ValueError("Require two full native repetitions")
    for i, expected in enumerate(report["repetition_sha256"]):
        path = root / f"repetition-{i}.json"
        if sha256(path) != expected:
            raise ValueError("Engineering episode changed")
        row = json.loads(path.read_text())
        validate_complete(row["result"], [tuple(c) for c in row["unroll_calls"]])
        actions = np.asarray(row["planned_actions"], dtype=np.float32)
        if actions.shape != (1, 6, 10) or not np.isfinite(actions).all():
            raise ValueError("Wrong native Push-T action trace")
        comparisons.append(comparison_record(row["result"], [array_hash(actions[0])]))
    if comparisons[0] != comparisons[1]:
        raise ValueError("Native repeated actions/outcomes differ")
    return report, digest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "data-root", "cohort", "simulator-smoke", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    cohort = checked_cohort(args.cohort)
    fit = cohort["fit"][0]
    simulator, simulator_hash = verified_report(args.simulator_smoke)
    expected = simulator["tasks"]["pusht"]
    if (simulator["status"] != "official_simulator_goal_setup_smoke_passed" or
            not expected["same_seed_pairing"] or expected["fit_trajectory_id"] != fit["trajectory_id"] or
            expected["smoke_seed"] != SMOKE_SEED or simulator["fresh_confirmation"]):
        raise ValueError("Missing fitting-only simulator/goal setup proof")
    contract = prepare(args.vendor, "pusht")
    full = open_normalized_dataset("pusht", args.data_root, cohort["reference_config"], True)
    if (_initial_state_group(full.states[fit["index"], 0]) != fit["lineage_group"] or
            full.get_seq_length(fit["index"]) != fit["length"]):
        raise ValueError("Actual fitting trajectory differs from preserved cohort")
    dataset = SingleFitTrajectory(full, fit["index"])
    files = ["states.pth", "rel_actions.pth", "velocities.pth", "seq_lengths.pkl",
             f"obses/episode_{fit['index']:03d}.mp4"]
    if (args.data_root / "train/shapes.pkl").exists():
        files.append("shapes.pkl")
    bindings = {name: sha256(args.data_root / "train" / name) for name in files}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        worker = device_uuid()
        write_json(args.output / "protocol.json", {"role": "pusht_native_receiving_engineering_not_efficacy",
            "planning_contract": contract, "source_sha256": source_hash(), "cohort_sha256": sha256(args.cohort),
            "checkpoint_sha256": CHECKPOINTS["pusht"], "fit_trajectory": fit, "fit_input_files_sha256": bindings,
            "simulator_receipt_sha256": simulator_hash, "device_uuid": worker, "repetitions": 2,
            "smoke_seed": SMOKE_SEED, "precision": "float32_strict_no_tf32", "fresh_confirmation": False})
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["pusht"], "pusht", "cuda:0", "float32")
        if backend.model.ctxt_window != 2 or backend.provenance["normalization_dataset"] != "pusht":
            raise ValueError("Incorrect Push-T model or planning context")
        versions, original = _model_versions(backend.model), backend.model.unroll
        torch.cuda.reset_peak_memory_stats()
        records = []
        for repetition in range(2):
            random.seed(SMOKE_SEED); np.random.seed(SMOKE_SEED); torch.manual_seed(SMOKE_SEED)
            cfg = OmegaConf.create(contract["config"])
            cfg.meta.seed = cfg.local_seed = SMOKE_SEED
            calls, actions = [], []
            def observed(context, act_suffix=None, **kwargs):
                output = original(context, act_suffix=act_suffix, **kwargs)
                if any(not torch.isfinite(output[k]).all() for k in ("visual", "proprio")):
                    raise ValueError("Nonfinite native Push-T forecast")
                calls.append(tuple(act_suffix.shape[:2]))
                write_json(args.output / "progress.json", {"repetition": repetition, "completed_calls": len(calls)})
                return output
            backend.model.unroll = observed
            agent = GC_Agent(cfg, backend.model, dset=dataset, preprocessor=backend.preprocessor)
            original_act = agent.act
            def actor(*a, **kw):
                action = original_act(*a, **kw)
                actions.append(action.detach().cpu().tolist())
                return action
            agent.act = actor
            env = make_env(cfg)
            try:
                result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
                if env.max_steps() != 30 or any(result[k] != expected[k] for k in ("initial_sha256", "goal_sha256")):
                    raise ValueError("Receiving native setup differs from simulator proof")
            finally:
                env.close(); backend.model.unroll = original
            validate_complete(result, calls)
            records.append({"result": result, "unroll_calls": calls, "planned_actions": actions})
            write_json(args.output / f"repetition-{repetition}.json", records[-1])
        comparisons = [comparison_record(r["result"], [array_hash(np.asarray(r["planned_actions"][0], dtype=np.float32))]) for r in records]
        if comparisons[0] != comparisons[1] or _model_versions(backend.model) != versions:
            raise ValueError("Repeated native actions/outcomes differ or frozen weights changed")
        write_json(args.output / "report.json", {"status": "full_native_pusht_engineering_passed",
            "protocol_sha256": sha256(args.output / "protocol.json"), "device_uuid": worker,
            "repetition_sha256": [sha256(args.output / f"repetition-{i}.json") for i in range(2)],
            "same_seed_actions_and_outcomes_exact": True, "parameters_unchanged": True,
            "full_native_cem_schedule": True, "fresh_confirmation": False, "scientific_efficacy_measurement": False,
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "pymunk", "pygame", "numpy")},
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic() - started})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
        verify_engineering(args.output, contract, sha256(args.cohort))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
