"""Complete author-split Wall/Maze native forecasts, or unchanged five-sweep fits."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .author_fit import source_hash
from .author_runtime import AuthorBackend, encoded_batches, examples, open_normalized_dataset, prefix_batches
from .intervention_runner import _model_versions
from .navigation_cohort import COUNTS, verify_inputs
from .navigation_input_check import CONFIGS
from .navigation_smoke import CHECKPOINTS
from .protocol import sha256, summarize_metrics, write_json
from .vendor import use_vendor


@torch.no_grad()
def baseline(args, cohort):
    task, cfg = cohort["task"], cohort["reference_config"]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"role": "full_author_navigation_validation_native_reference",
            "task": task, "precision": args.precision, "cohort_sha256": sha256(args.cohort),
            "source_sha256": source_hash(), "checkpoint_sha256": CHECKPOINTS[task],
            "selection_from_native_outcomes": False, "fresh_confirmation": False,
            "context": 3, "horizon": 6, "validation_clips": COUNTS[task][3],
            "expected_prefixes": COUNTS[task][3] * 2, "batch_size": 4,
            "endpoint": "official visual/proprioceptive embedding L1/L2 at H1-H6, not task success"})
        torch.cuda.set_device(0)
        backend = AuthorBackend(args.vendor, args.checkpoint, CHECKPOINTS[task], task, "cuda:0", args.precision)
        versions = _model_versions(backend.model)
        dset = open_normalized_dataset(task, args.data_root, cfg, False)
        _, encoded = next(encoded_batches(backend, dset, cohort["fit"][:1], cfg))
        write_json(args.output / "PARITY.json", {"fit_only": True, "precision": args.precision,
            "checks": backend.verify_reference(encoded)})
        del encoded
        rows = cohort["evaluation"]
        expected = examples(rows, cfg)
        measurements = []
        torch.cuda.reset_peak_memory_stats()
        for clip_meta, encoded in encoded_batches(backend, dset, rows, cfg):
            for meta, context, actions, target in prefix_batches(backend, clip_meta, encoded):
                prediction = backend.predict(context, actions)
                metrics = backend.metrics(prediction, target)
                measurements.extend({**m, "arm": "native", "metrics": values}
                                    for m, values in zip(meta, metrics, strict=True))
            if len(measurements) % 80 == 0:
                progress = {"task": task, "precision": args.precision, "prefixes": len(measurements),
                    "total": len(expected), "seconds": time.monotonic() - started}
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        wanted = {(r["trajectory_id"], r["start"]) for r in expected}
        actual = [(r["trajectory_id"], r["start"]) for r in measurements]
        if len(actual) != len(wanted) or set(actual) != wanted or versions != _model_versions(backend.model):
            raise ValueError("Incomplete/duplicated navigation coverage or frozen weights changed")
        write_json(args.output / "window_metrics.json", measurements)
        write_json(args.output / "report.json", {"status": "full_author_navigation_native_forecast_complete",
            "task": task, "precision": args.precision, "protocol_sha256": sha256(args.output / "protocol.json"),
            "parity_sha256": sha256(args.output / "PARITY.json"),
            "window_metrics_sha256": sha256(args.output / "window_metrics.json"),
            "rollout_trajectories": len(rows), "validation_clips": COUNTS[task][3],
            "prefixes": len(actual), "aggregation": summarize_metrics(measurements),
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "parameters_unchanged": True, "fresh_confirmation": False,
            "intervention_comparisons_complete": False, "training_history_complete": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "fit"))
    for name in ("vendor", "checkpoint", "cohort", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    cohort = verify_inputs(args.cohort, args.data_root)
    task = cohort["task"]
    if task not in COUNTS or cohort["upstream_config_sha256"] != sha256(args.vendor / CONFIGS[task]):
        raise ValueError("Wrong navigation task/config")
    if args.stage == "baseline":
        baseline(args, cohort)
    else:
        # Reuse the established algorithm, all arms, fit count and dose rules.
        from .author_fit import main as fit_main
        sys.argv = ["author_fit", "--vendor", str(args.vendor), "--checkpoint", str(args.checkpoint),
            "--cohort", str(args.cohort), "--data-root", str(args.data_root), "--output", str(args.output),
            "--checkpoint-sha256", CHECKPOINTS[task], "--precision", args.precision, "--device", "cuda:0"]
        fit_main()


if __name__ == "__main__":
    main()
