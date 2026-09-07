"""Released-checkpoint DROID baseline: all 64 episodes, eight persistent streams.

Recorded-plan replication, not robot task success or fresh-family confirmation.
No intervention fitting or outcome-based candidate selection is performed here.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .droid_contract import checkpoint_score, prepare
from .droid_native import full_episode, load_model, make_dataset, verified_report
from .planning_contract import seed_schedule
from .protocol import sha256, write_json
from .vendor import use_vendor


def assigned_rows(rows, ranks):
    if (not ranks or len(set(ranks)) != len(ranks) or
            any(rank not in range(8) for rank in ranks) or rows != seed_schedule(1, 64, 8, 3)):
        raise ValueError("Require complete ordered native logical streams, without duplicate ranks")
    return [row for row in rows if row["logical_rank"] in ranks]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "assets", "manifest", "encoder-source", "encoder-root", "engineering", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--logical-ranks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    contract = prepare(args.vendor, json.loads(args.manifest.read_text()))
    rows = assigned_rows(contract["episodes"], args.logical_ranks)
    assets, assets_hash = verified_report(args.assets)
    engineering, engineering_hash = verified_report(args.engineering)
    engineering_protocol = json.loads((args.engineering / "protocol.json").read_text())
    if (assets["status"] != "official_droid_inputs_staged_and_verified" or
            engineering["status"] != "native_droid_full_cem_engineering_passed" or
            not engineering["same_seed_actions_and_metrics_exact"] or
            engineering_protocol["source_sha256"] != sha256(Path(__file__).with_name("droid_native.py")) or
            engineering_protocol["contract"] != contract):
        raise ValueError("Require current-source exact native engineering before opening baseline")
    for name, digest in engineering["files_sha256"].items():
        if sha256(args.engineering / name) != digest:
            raise ValueError("Incomplete engineering evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"role": "released_droid_native_checkpoint_replication",
            "contract": contract, "assigned_episodes": rows, "logical_ranks": sorted(args.logical_ranks),
            "assets_report_sha256": assets_hash, "engineering_report_sha256": engineering_hash,
            "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                ("droid_replication.py", "droid_native.py", "droid_contract.py")},
            "population": "15 source-config recordings; published extra file unreadable, not substituted",
            "claim": "single released checkpoint/code protocol, not three training seeds or exact paper population",
            "native_baseline_only": True, "no_intervention_selection": True,
            "fresh_confirmation": False, "robot_executions": 0,
            "statistical_unit": "recording families; 64 sampled segments are not 64 independent recordings",
            "aggregate": "mean all 64 XYZ errors, then max(0,800*(0.1-mean)); no partial score"})
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        _, preprocessor = make_dataset(args.assets, contract)
        model, provenance = load_model(args.vendor, args.assets, args.encoder_source,
                                      args.encoder_root, contract, preprocessor)
        versions = [(name, p._version) for name, p in model.named_parameters()]
        from omegaconf import OmegaConf
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        records = []
        torch.cuda.reset_peak_memory_stats()
        for rank in sorted(args.logical_ranks):
            # Native eval globally seeds with base 1. Dataset retains its own
            # RandomState(234); GC_Agent retains separate local CPU/CUDA streams.
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            dset, preprocessor = make_dataset(args.assets, contract)
            cfg = OmegaConf.create(copy.deepcopy(contract["config"]))
            rank_rows = [row for row in rows if row["logical_rank"] == rank]
            cfg.local_seed = rank_rows[0]["local_seed"]
            agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
            for row in rank_rows:
                before = time.monotonic()
                result = full_episode(cfg, model, dset, preprocessor, args.output, row["episode"],
                    seed=row["environment_seed"], agent=agent, role="released_checkpoint_replication")
                record = {**row, "arm": "native", "result": result, "seconds": time.monotonic() - before}
                records.append(record)
                write_json(args.output / f"episode-{row['episode']:03d}.json", record)
                progress = {"completed": len(records), "target": len(rows),
                            "seconds": time.monotonic() - started}
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if ([(r["episode"], r["logical_rank"], r["environment_seed"]) for r in records] !=
                [(r["episode"], r["logical_rank"], r["environment_seed"]) for r in rows] or
                versions != [(name, p._version) for name, p in model.named_parameters()]):
            raise ValueError("Changed coverage or frozen model")
        score = None
        if len(records) == 64:
            score = float(checkpoint_score(torch.tensor(
                [r["result"]["metrics"]["action_error_xyz"] for r in records], dtype=torch.float64)))
        write_json(args.output / "report.json", {"status": "released_droid_native_replication_shard_complete",
            "protocol_sha256": sha256(args.output / "protocol.json"), "episodes": len(records),
            "episode_files_sha256": {f"episode-{r['episode']:03d}.json":
                sha256(args.output / f"episode-{r['episode']:03d}.json") for r in records},
            "official_checkpoint_score_if_complete": score, "model_provenance": provenance,
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "parameters_unchanged": True, "comparative_analysis_complete": False,
            "fresh_confirmation": False, "robot_executions": 0, "dummy_success_reported": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


if __name__ == "__main__":
    main()
