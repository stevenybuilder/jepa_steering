"""Capture native fit-only imagined history for the predeclared routing question.

No HMM is fitted, no intervention is applied, and no evaluation outcome is read.
This is preparation for eligibility assessment, not evidence that routing works.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .author_evaluate import verify_fit
from .author_fit import source_hash
from .author_runtime import (AuthorBackend, encoded_batches, examples,
                            open_normalized_dataset, prefix_batches, validate_cohort)
from .intervention_runner import _model_versions
from .planning_native_smoke import CHECKPOINTS
from .protocol import sha256, write_json


PREDICTOR_FEATURE_DIMENSION = 400  # Pinned MW hidden field includes proprio features.


def history_tensor(values, batch_size):
    history = torch.stack([values[h] for h in range(1, 7)], 1)
    expected = (batch_size, 6, PREDICTOR_FEATURE_DIMENSION)
    if history.shape != expected or not torch.isfinite(history).all():
        raise ValueError(f"Invalid native history: expected {expected}, got {tuple(history.shape)}; finite={bool(torch.isfinite(history).all())}")
    return history


class NativeHistoryCapture:
    """Read only the B3 output, pooled over the newest 256 visual tokens."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.horizon, self.values, self.handles = 0, {}, []

    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 6:
            raise ValueError("Expected the pinned six-block predictor")
        self.handles = [self.predictor.register_forward_pre_hook(self._input),
                        self.predictor.predictor_blocks[3].register_forward_hook(self._output)]
        return self

    def _input(self, module, args):
        self.horizon += 1

    def _output(self, module, args, output):
        if (not 1 <= self.horizon <= 6 or self.horizon in self.values or
                output.ndim != 3 or output.shape[1] < 256):
            raise ValueError("Unexpected native imagined-history capture")
        self.values[self.horizon] = output[:, -256:].detach().float().mean(1).cpu()
        # Returning None leaves the actual predictor tensor untouched.

    def __exit__(self, kind, exc, traceback):
        for handle in self.handles:
            handle.remove()
        if kind is None and set(self.values) != set(range(1, 7)):
            raise ValueError("Missing or duplicated native imagined horizon")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "cohort", "fit", "data-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.shard_index not in range(4):
        parser.error("Exactly four disjoint trajectory shards are frozen")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        cohort = json.loads(args.cohort.read_text())
        validate_cohort(cohort)
        if cohort["task"] not in ("mw-reach", "mw-reach-wall"):
            raise ValueError("No qualifying non-routed Push-T intervention")
        receipt, protocol = verify_fit(args.fit, sha256(args.cohort), CHECKPOINTS["metaworld"], "bfloat16")
        if (protocol["category"] != "operator_rank" or
                receipt["fit_trajectory_ids"] != [r["trajectory_id"] for r in cohort["fit"]] or
                receipt["fit_lineage_groups"] != [r["lineage_group"] for r in cohort["fit"]]):
            raise ValueError("Do not change the already frozen fit population")
        rows = cohort["fit"][args.shard_index::4]
        expected = examples(rows, cohort["reference_config"], fitting=True)
        if len(rows) != 32 or len(expected) != 128:
            raise ValueError("Expected 32 fit families x four prefixes per shard")
        write_json(args.output / "protocol.json", {
            "task": cohort["task"], "role": "fit_only_native_routing_history_preparation",
            "cohort_sha256": sha256(args.cohort), "source_rank_protocol_sha256": sha256(args.fit / "protocol.json"),
            "source_fit_receipt_sha256": sha256(args.fit / "fit_receipt.json"),
            "checkpoint_sha256": CHECKPOINTS["metaworld"], "precision": "bfloat16",
            "site": "B3_block_output_newest_256_visual_tokens_float32_mean",
            "hidden_features": PREDICTOR_FEATURE_DIMENSION,
            "hidden_feature_scope": "all 400 predictor features, including proprioceptive feature conditioning; no projection/truncation",
            "horizons": [1, 2, 3, 4, 5, 6], "routing_decision_may_use_only": [1, 2],
            "later_horizons": "fit-only emission-model training if subsequently frozen; never future input to an H3 routing decision",
            "shard_index": args.shard_index, "shard_count": 4,
            "fit_families_in_full_task": 128, "fit_prefixes_in_full_task": 512,
            "selection": rows, "examples": expected, "source_sha256": source_hash(),
            "new_operator_or_hmm_fit": False, "eligibility_decision": False,
            "evaluation_or_confirmation_access_authorized": False})
        backend = AuthorBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"],
                                "metaworld", "cuda:0", "bfloat16")
        if tuple(backend.predictor.predictor_norm.normalized_shape) != (PREDICTOR_FEATURE_DIMENSION,):
            raise ValueError("Pinned MW predictor hidden dimension changed")
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset("metaworld", args.data_root, cohort["reference_config"], True)
        metadata, histories, identity_checks = [], [], 0
        for clip_metadata, encoded in encoded_batches(backend, dataset, rows, cohort["reference_config"], fitting=True):
            for meta, context, actions, _ in prefix_batches(backend, clip_metadata, encoded):
                # No target scores are computed or retained. The second forward
                # checks that passive capture leaves actual outputs unchanged.
                native = backend.predict(context, actions)
                with NativeHistoryCapture(backend.predictor) as capture:
                    captured = backend.predict(context, actions)
                if any(not torch.equal(native[k], captured[k]) for k in ("visual", "proprio")):
                    raise ValueError("Passive history capture changes the native forecast")
                history = history_tensor(capture.values, len(meta))
                histories.append(history)
                metadata.extend(meta)
                identity_checks += 1
            write_json(args.output / "progress.json", {"fit_prefixes_captured": len(metadata),
                "seconds": time.monotonic() - started, "evaluation_outcomes_accessed": False})
        keys = [(r["trajectory_id"], r["start"]) for r in metadata]
        if len(keys) != len(set(keys)) or set(keys) != {(r["trajectory_id"], r["start"]) for r in expected}:
            raise ValueError("Incomplete or duplicated fit history coverage")
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen parameters changed")
        torch.save({"history": torch.cat(histories), "metadata": metadata}, args.output / "history.pt")
        write_json(args.output / "report.json", {
            "status": "native_fit_history_captured_not_hmm_eligibility_or_efficacy",
            "task": cohort["task"], "shard_index": args.shard_index, "shard_count": 4,
            "protocol_sha256": sha256(args.output / "protocol.json"), "history_sha256": sha256(args.output / "history.pt"),
            "fit_families": 32, "fit_prefixes": 128, "identity_checks": identity_checks,
            "model_parameters_unchanged": True, "passive_capture_bitwise_identity": True,
            "evaluation_outcomes_accessed": False, "eligibility_decision": False,
            "scientific_efficacy_measurement": False, "seconds": time.monotonic() - started})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
