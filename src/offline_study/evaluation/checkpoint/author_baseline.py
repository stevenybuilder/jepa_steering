"""Broad native-only baseline on unprotected official MetaWorld validation rows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, encoded_batches, open_normalized_dataset, prefix_batches, validate_cohort
from offline_study.evaluation.checkpoint.author_validation import CONFIGS, audit, official_partition
from offline_study.core.protocol import sha256, summarize_metrics, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vendor", "checkpoint", "manifest", "registry", "parity-cohort", "data-root", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", required=True, choices=["bfloat16", "float32"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
        registry = json.loads(args.registry.read_text())
        if registry["manifest_sha256"] != sha256(args.manifest):
            raise ValueError("Exposure registry mismatch")
        config = yaml.safe_load((args.vendor / CONFIGS["metaworld"]).read_text())
        preflight = audit(rows, "metaworld", config, registry)
        _, val = official_partition(rows, "metaworld")
        protected_ids = {tid for task in preflight["tasks"].values() for tid in task["protected_ids"]}
        selected = [row for row in val if row["trajectory_id"] not in protected_ids]
        if len(selected) != 1120 or len({row["task"] for row in selected}) != 42:
            raise ValueError("Broad coverage differs from preflight: expected 1,120 rows across 42 tasks")
        backend = AuthorBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                                "metaworld", args.device, args.precision)
        dataset = open_normalized_dataset("metaworld", args.data_root, config, False)
        fit_cohort = json.loads(args.parity_cohort.read_text())
        validate_cohort(fit_cohort)
        if fit_cohort["source_manifest_sha256"] != sha256(args.manifest):
            raise ValueError("Fit-only parity source mismatch")
        _, encoded = next(encoded_batches(backend, dataset, fit_cohort["fit"][:1], config))
        write_json(args.output / "PARITY.json", {"fit_only": True, "precision": args.precision,
                                                "checks": backend.verify_reference(encoded)})
        del encoded
        measurements = []
        for clip_meta, encoded in encoded_batches(backend, dataset, selected, config):
            for meta, context, actions, target in prefix_batches(backend, clip_meta, encoded):
                if time.monotonic() - started > 7000:
                    raise TimeoutError("Bounded broad baseline exceeded runtime")
                pred = backend.predict(context, actions)
                scores = backend.metrics(pred, target)
                measurements.extend({**m, "metrics": value} for m, value in zip(meta, scores, strict=True))
            if len(measurements) % 480 == 0:
                print(json.dumps({"stage": "broad_baseline", "precision": args.precision,
                    "prefixes": len(measurements), "total": 134400, "seconds": time.monotonic() - started}), flush=True)
        if len(measurements) != 134400 or len({(m["trajectory_id"], m["start"]) for m in measurements}) != 134400:
            raise ValueError("Broad baseline prefix coverage mismatch")
        write_json(args.output / "window_metrics.json", measurements)
        write_json(args.output / "selection.json", selected)
        report = {"status": "protected_excluded_author_broad_baseline_complete", "precision": args.precision,
                  "task_count": 42, "rollout_trajectories": len(selected), "official_validation_rows": len(val),
                  "protected_excluded_rows": len(protected_ids), "protected_outcomes_accessed": False,
                  "prefix_rollouts": len(measurements), "fresh_confirmation": False,
                  "source_manifest_sha256": sha256(args.manifest), "registry_sha256": sha256(args.registry),
                  "checkpoint_sha256": args.checkpoint_sha256, "source_sha256": source_hash(),
                  "wall_seconds": time.monotonic() - started, "aggregation": summarize_metrics(measurements),
                  "metrics": "official embedding L1/L2 H1-H6, not decoded physical states or planning success"}
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {"status": report["status"], **{
            name + "_sha256": sha256(args.output / (name + ".json")) for name in ("report", "window_metrics", "selection", "PARITY")}})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
