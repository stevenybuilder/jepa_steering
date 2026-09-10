#!/usr/bin/env python3
"""Verify and summarize fixed-observation JEPA-WM action-patching shards.

The independent unit is a saved trajectory snapshot.  Action-pair rows are first
averaged within a snapshot; uncertainty is then estimated by a paired, collection-
seed-stratified episode bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PRIMARY_METRICS = ("transfer_fraction", "effect_cosine")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_bootstrap_ci(
    values: Iterable[tuple[int, float]], rng: random.Random, draws: int
) -> tuple[float, float]:
    groups: dict[int, list[float]] = {}
    materialized = list(values)
    for seed in sorted({seed for seed, _value in materialized}):
        group = [
            value for row_seed, value in materialized if row_seed == seed and math.isfinite(value)
        ]
        if not len(group):
            raise ValueError(f"No finite values for seed {seed}")
        groups[seed] = group
    boot = []
    denominator = sum(len(group) for group in groups.values())
    for _index in range(draws):
        total = sum(rng.choice(group) for group in groups.values() for _ in range(len(group)))
        boot.append(total / denominator)
    boot.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(boot) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return boot[lower]
        weight = position - lower
        return boot[lower] * (1 - weight) + boot[upper] * weight

    return percentile(0.025), percentile(0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_905)
    args = parser.parse_args()

    snapshots: list[dict] = []
    receipts = []
    seen_ids: set[tuple[int, int]] = set()
    for seed_dir in sorted(args.input_root.glob("seed*")):
        receipt_path = seed_dir / "DONE.json"
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("complete") or receipt.get("snapshot_count") != len(receipt["outputs"]):
            raise RuntimeError(f"Incomplete receipt: {receipt_path}")
        if receipt["snapshot_manifest_sha256"] != "08b320729f8b3c5bcfd62d1f5c596e428819723342538d23a06eb5293abbbd95":
            raise RuntimeError(f"Unexpected snapshot manifest: {receipt_path}")
        for output in receipt["outputs"]:
            path = seed_dir / output["path"]
            observed_hash = sha256(path)
            if observed_hash != output["sha256"]:
                raise RuntimeError(f"Hash mismatch: {path}")
            snapshot = json.loads(path.read_text())
            identity = (int(snapshot["snapshot"]["seed"]), int(snapshot["snapshot"]["episode"]))
            if identity in seen_ids:
                raise RuntimeError(f"Duplicate snapshot: {identity}")
            seen_ids.add(identity)
            snapshots.append(snapshot)
        receipts.append(
            {
                "path": str(receipt_path),
                "sha256": sha256(receipt_path),
                "seeds": receipt["seeds"],
                "snapshot_count": receipt["snapshot_count"],
                "duration_seconds": receipt["duration_seconds"],
            }
        )

    if len(snapshots) != 48 or {seed for seed, _episode in seen_ids} != {1, 2, 3}:
        raise RuntimeError(f"Expected 48 snapshots across seeds 1,2,3; observed {len(snapshots)}")

    # Average the 30 ordered action-pair rows within each independent snapshot.
    episode_means: dict[tuple[int, float, str, str], list[tuple[int, int, float]]] = defaultdict(list)
    nuisance: dict[int, list[float]] = defaultdict(list)
    for snapshot in snapshots:
        seed = int(snapshot["snapshot"]["seed"])
        episode = int(snapshot["snapshot"]["episode"])
        buckets: dict[tuple[int, float, str, str], list[float]] = defaultdict(list)
        for row in snapshot["rows"]:
            nuisance[int(row["block"])].append(float(row["sham_cosine"]))
            for metric in (*PRIMARY_METRICS, "goal_cost_transfer"):
                value = float(row[metric])
                if math.isfinite(value):
                    buckets[(int(row["block"]), float(row["beta"]), row["arm"], metric)].append(value)
        for key, values in buckets.items():
            episode_means[key].append((seed, episode, statistics.fmean(values)))

    rng = random.Random(args.bootstrap_seed)
    summaries = []
    paired: dict[tuple[int, float, str], list[tuple[int, int, float]]] = {}
    for block in (2, 3, 5):
        for beta in (0.25, 0.5, 1.0):
            for metric in (*PRIMARY_METRICS, "goal_cost_transfer"):
                patch_rows = episode_means[(block, beta, "patch", metric)]
                sham_rows = episode_means[(block, beta, "orientation_sham", metric)]
                patch_map = {(seed, episode): value for seed, episode, value in patch_rows}
                sham_map = {(seed, episode): value for seed, episode, value in sham_rows}
                common = sorted(set(patch_map) & set(sham_map))
                differences = [
                    (seed, episode, patch_map[(seed, episode)] - sham_map[(seed, episode)])
                    for seed, episode in common
                ]
                paired[(block, beta, metric)] = differences
                diff_values = [(seed, value) for seed, _episode, value in differences]
                low, high = stratified_bootstrap_ci(diff_values, rng, args.bootstrap_draws)
                summaries.append(
                    {
                        "block": block,
                        "beta": beta,
                        "metric": metric,
                        "snapshot_count": len(common),
                        "patch_mean": statistics.fmean(patch_map.values()),
                        "sham_mean": statistics.fmean(sham_map.values()),
                        "paired_difference_mean": statistics.fmean(value for _s, _e, value in differences),
                        "paired_difference_ci95": [low, high],
                    }
                )

    monotonicity = []
    for block in (2, 3, 5):
        by_beta = {
            beta: {(seed, episode): value for seed, episode, value in episode_means[(block, beta, "patch", "transfer_fraction")]}
            for beta in (0.25, 0.5, 1.0)
        }
        common = sorted(set.intersection(*(set(rows) for rows in by_beta.values())))
        indicators = [
            float(by_beta[0.25][identity] <= by_beta[0.5][identity] + 1e-9 <= by_beta[1.0][identity] + 2e-9)
            for identity in common
        ]
        tracking_error = [
            abs(by_beta[beta][identity] - beta) for identity in common for beta in (0.25, 0.5, 1.0)
        ]
        monotonicity.append(
            {
                "block": block,
                "snapshot_count": len(common),
                "strict_dose_monotonic_fraction": statistics.fmean(indicators),
                "mean_absolute_beta_tracking_error": statistics.fmean(tracking_error),
            }
        )

    gate_blocks = []
    for block in (2, 3):
        rows = {
            row["metric"]: row
            for row in summaries
            if row["block"] == block and row["beta"] == 0.5 and row["metric"] in PRIMARY_METRICS
        }
        monotonic = next(row for row in monotonicity if row["block"] == block)
        passed = (
            all(rows[metric]["paired_difference_ci95"][0] > 0 for metric in PRIMARY_METRICS)
            and monotonic["strict_dose_monotonic_fraction"] > 0.5
        )
        gate_blocks.append({"block": block, "passed": passed})

    result = {
        "schema_version": 1,
        "scope": "discovery-only fixed-observation causal action mediation",
        "independent_unit": "trajectory snapshot; action pairs averaged within snapshot",
        "snapshot_count": len(snapshots),
        "collection_seeds": [1, 2, 3],
        "bootstrap": {
            "type": "paired, collection-seed-stratified episode bootstrap",
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
        },
        "receipts": receipts,
        "matched_sham": {
            "description": "feature-permuted and sign-flipped activation delta with equal L2 norm",
            "mean_delta_cosine_by_block": {
                str(block): statistics.fmean(values) for block, values in nuisance.items()
            },
        },
        "summaries": summaries,
        "dose_monotonicity": monotonicity,
        "causal_gate": {
            "rule": "At beta=0.5, both transfer-fraction and effect-cosine paired-difference CI lower bounds exceed zero, and transfer is dose-monotonic in >50% of snapshots.",
            "eligible_intermediate_blocks": gate_blocks,
            "passed": any(row["passed"] for row in gate_blocks),
            "block_5_note": "Final predictor block is reported but excluded because direct output transfer there is near-tautological.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# JEPA-WM causal action-mediation result",
        "",
        f"Verified **{len(snapshots)} independent trajectory snapshots** across collection seeds 1–3. Action-pair rows were averaged within snapshots before inference.",
        "",
        "| Block | Dose | Transfer: patch | Transfer: sham | Paired difference (95% CI) | Cosine paired difference (95% CI) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for block in (2, 3, 5):
        for beta in (0.25, 0.5, 1.0):
            transfer = next(row for row in summaries if row["block"] == block and row["beta"] == beta and row["metric"] == "transfer_fraction")
            cosine = next(row for row in summaries if row["block"] == block and row["beta"] == beta and row["metric"] == "effect_cosine")
            lines.append(
                f"| {block} | {beta:.2f} | {transfer['patch_mean']:.3f} | {transfer['sham_mean']:.3f} | "
                f"{transfer['paired_difference_mean']:.3f} [{transfer['paired_difference_ci95'][0]:.3f}, {transfer['paired_difference_ci95'][1]:.3f}] | "
                f"{cosine['paired_difference_mean']:.3f} [{cosine['paired_difference_ci95'][0]:.3f}, {cosine['paired_difference_ci95'][1]:.3f}] |"
            )
    lines.extend(
        [
            "",
            f"**Causal gate: {'PASS' if result['causal_gate']['passed'] else 'FAIL'}.**",
            "",
            "Block 5 is a near-final-block transfer sanity check, not mechanism-selective evidence. Passing this gate establishes that an intermediate predictor representation causally mediates action-conditioned latent predictions; it does not yet establish closed-loop task improvement.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines))
    print(json.dumps(result["causal_gate"], indent=2))


if __name__ == "__main__":
    main()
