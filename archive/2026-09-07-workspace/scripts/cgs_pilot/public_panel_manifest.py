#!/usr/bin/env python3
"""Create and validate immutable paired-stimulus manifests for Panel P.

The released JEPA-WM evaluator derives an episode seed from one ``local_seed`` and
also initializes the planner from that same seed.  That is convenient for a stock
evaluation but is not a safe experimental split: nominal seed 1 and nominal seed
2 overlap.  This module instead preregisters an independent ``env_seed`` and
``planner_seed`` for every pair.  Evaluation arms consume the same pair manifest;
the arm is deliberately *not* part of a stimulus row.

The preregistration contains development, fit, and protected-evaluation seeds plus
fixed software/model identities.  ``--development-count 0`` retains the original
two-way fit/evaluation bundle for older experiments.  Hashes of
the realized initial state, expert goal and initial image are written to a separate
append-only realization log by the runner, then checked across paired arms.  The
preregistered JSONL is never rewritten after collection starts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "panel-p-paired-stimulus-v1"
DEFAULT_ARMS = (
    "unsteered",
    "identity",
    "equal_compute_cem",
    "outcome_rerank",
    "planner_cost",
    "factor_sonar",
    "matched_sham",
)
MAX_SEED = 2**31 - 1


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rows_sha256(rows: Sequence[dict]) -> str:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def parse_int_set(spec: str | None) -> set[int]:
    """Parse ``1,4-8,11`` into a set, inclusively."""
    out: set[int] = set()
    if not spec:
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"descending integer range: {token}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return out


def legacy_episode_seeds(local_seed: int, episodes: int) -> list[int]:
    modulus = 2**32 - 2
    return [(local_seed * local_seed + ep * local_seed) % modulus for ep in range(episodes)]


class SeedAllocator:
    def __init__(self, namespace: str, excluded: Iterable[int] = ()) -> None:
        self.namespace = namespace
        self.used = {int(seed) for seed in excluded}

    def get(self, split: str, ordinal: int, kind: str) -> int:
        nonce = 0
        while True:
            key = f"{self.namespace}\0{split}\0{ordinal}\0{kind}\0{nonce}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % MAX_SEED + 1
            if seed not in self.used:
                self.used.add(seed)
                return seed
            nonce += 1


def generate_rows(
    *,
    split: str,
    count: int,
    task: str,
    namespace: str,
    checkpoint_sha256: str,
    config_sha256: str,
    repo_commit: str,
    allocator: SeedAllocator,
) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    width = max(3, len(str(count - 1)))
    rows = []
    for ordinal in range(count):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "namespace": namespace,
                "split": split,
                "ordinal": ordinal,
                "pair_id": f"{split}-{ordinal:0{width}d}",
                "task": task,
                "env_seed": allocator.get(split, ordinal, "environment"),
                "planner_seed": allocator.get(split, ordinal, "planner"),
                "checkpoint_sha256": checkpoint_sha256,
                "config_sha256": config_sha256,
                "repo_commit": repo_commit,
            }
        )
    return rows


def validate_rows(groups: Sequence[tuple[str, Sequence[dict]]]) -> dict:
    required = {
        "schema_version",
        "namespace",
        "split",
        "ordinal",
        "pair_id",
        "task",
        "env_seed",
        "planner_seed",
        "checkpoint_sha256",
        "config_sha256",
        "repo_commit",
    }
    errors: list[str] = []
    all_pairs: dict[str, str] = {}
    all_env: dict[int, str] = {}
    all_planner: dict[int, str] = {}
    identities: set[tuple[str, str, str, str, str]] = set()
    total = 0

    for name, rows in groups:
        if not rows:
            errors.append(f"{name}: empty manifest")
            continue
        ordinals: set[int] = set()
        declared_splits: set[str] = set()
        for idx, row in enumerate(rows):
            total += 1
            missing = sorted(required - row.keys())
            if missing:
                errors.append(f"{name}:{idx}: missing {missing}")
                continue
            if row["schema_version"] != SCHEMA_VERSION:
                errors.append(f"{name}:{idx}: unsupported schema {row['schema_version']!r}")
            split = str(row["split"])
            declared_splits.add(split)
            ordinal = int(row["ordinal"])
            if ordinal in ordinals:
                errors.append(f"{name}: duplicate ordinal {ordinal}")
            ordinals.add(ordinal)
            pair_id = str(row["pair_id"])
            if pair_id in all_pairs:
                errors.append(f"pair_id {pair_id!r} reused in {all_pairs[pair_id]} and {name}")
            all_pairs[pair_id] = name
            for key, seen in (("env_seed", all_env), ("planner_seed", all_planner)):
                seed = int(row[key])
                if not 1 <= seed <= MAX_SEED:
                    errors.append(f"{name}:{idx}: {key}={seed} outside [1,{MAX_SEED}]")
                if seed in seen:
                    errors.append(f"{key} {seed} reused in {seen[seed]} and {name}:{idx}")
                seen[seed] = f"{name}:{idx}"
            identities.add(
                (
                    str(row["namespace"]),
                    str(row["task"]),
                    str(row["checkpoint_sha256"]),
                    str(row["config_sha256"]),
                    str(row["repo_commit"]),
                )
            )
        if len(declared_splits) > 1:
            errors.append(f"{name}: contains multiple split labels {sorted(declared_splits)}")
        if ordinals and ordinals != set(range(len(rows))):
            errors.append(f"{name}: ordinals must be exactly 0..{len(rows) - 1}")

    if len(identities) > 1:
        errors.append("manifests disagree on namespace/task/checkpoint/config/repository identity")
    return {
        "passed": not errors,
        "schema_version": SCHEMA_VERSION,
        "n_rows": total,
        "n_groups": len(groups),
        "errors": errors,
        "group_hashes": {name: rows_sha256(list(rows)) for name, rows in groups},
    }


def validate_paired_realizations(groups: Sequence[tuple[str, Sequence[dict]]]) -> dict:
    """Require every arm to realize the same initial observation and expert goal per pair."""
    identity_fields = (
        "initial_observation_sha256",
        "goal_observation_sha256",
        "initial_visual_sha256",
        "goal_visual_sha256",
        "initial_proprio_sha256",
        "goal_proprio_sha256",
        "initial_simulator_state_sha256",
        "goal_state_sha256",
    )
    errors: list[str] = []
    indexed: dict[str, dict[str, dict]] = {}
    for name, rows in groups:
        by_pair: dict[str, dict] = {}
        for idx, row in enumerate(rows):
            pair_id = row.get("pair_id")
            if not pair_id:
                errors.append(f"{name}:{idx}: missing pair_id")
                continue
            if pair_id in by_pair:
                errors.append(f"{name}: duplicate pair_id {pair_id!r}")
            by_pair[str(pair_id)] = row
            missing = [field for field in identity_fields if not row.get(field)]
            if missing:
                errors.append(f"{name}:{idx}: missing realization hashes {missing}")
        indexed[name] = by_pair
    pair_sets = {name: set(rows) for name, rows in indexed.items()}
    if pair_sets:
        reference_name, reference_pairs = next(iter(pair_sets.items()))
        for name, pairs in pair_sets.items():
            if pairs != reference_pairs:
                errors.append(
                    f"{name}: pair IDs differ from {reference_name}; "
                    f"missing={sorted(reference_pairs - pairs)}, extra={sorted(pairs - reference_pairs)}"
                )
        for pair_id in sorted(set.intersection(*pair_sets.values()) if pair_sets else set()):
            reference = indexed[reference_name][pair_id]
            for name, rows in indexed.items():
                for field in identity_fields:
                    if rows[pair_id].get(field) != reference.get(field):
                        errors.append(f"{pair_id}: {field} differs between {reference_name} and {name}")
    return {
        "passed": not errors,
        "n_arms": len(groups),
        "n_common_pairs": len(set.intersection(*pair_sets.values())) if pair_sets else 0,
        "identity_fields": list(identity_fields),
        "errors": errors,
    }


def generate_bundle(args: argparse.Namespace) -> dict:
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty manifest directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    excluded_env = parse_int_set(args.exclude_env_seeds)
    excluded_planner = parse_int_set(args.exclude_planner_seeds)
    # One allocator makes accidental equality across environment/planner namespaces impossible too.
    allocator = SeedAllocator(args.namespace, excluded_env | excluded_planner)
    common = dict(
        task=args.task,
        namespace=args.namespace,
        checkpoint_sha256=args.checkpoint_sha256,
        config_sha256=args.config_sha256,
        repo_commit=args.repo_commit,
        allocator=allocator,
    )
    development = (
        generate_rows(split="development", count=args.development_count, **common)
        if args.development_count
        else []
    )
    fit = generate_rows(split="fit", count=args.fit_count, **common)
    evaluation = generate_rows(split="evaluation", count=args.eval_count, **common)
    groups = []
    if development:
        groups.append(("development", development))
    groups.extend((("fit", fit), ("evaluation", evaluation)))
    result = validate_rows(tuple(groups))
    if not result["passed"]:
        raise RuntimeError(result["errors"])

    if development:
        write_jsonl(out / "development.jsonl", development)
    write_jsonl(out / "fit.jsonl", fit)
    write_jsonl(out / "evaluation.jsonl", evaluation)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "namespace": args.namespace,
        "task": args.task,
        "checkpoint_sha256": args.checkpoint_sha256,
        "config_sha256": args.config_sha256,
        "repo_commit": args.repo_commit,
        "development_count": len(development),
        "fit_count": len(fit),
        "evaluation_count": len(evaluation),
        "frozen_arms": arms,
        "excluded_env_seeds": sorted(excluded_env),
        "excluded_planner_seeds": sorted(excluded_planner),
        "retired_legacy_local_seeds": [1, 2],
        "legacy_seed_overlap_1_vs_2_first_30": sorted(
            set(legacy_episode_seeds(1, 30)) & set(legacy_episode_seeds(2, 30))
        ),
        "realization_contract": (
            "Each run writes a separate append-only record containing hashes of the realized "
            "initial state, expert goal, and initial image. Paired arms must match those hashes."
        ),
        "validation": result,
    }
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", required=True)
    gen.add_argument("--namespace", required=True)
    gen.add_argument("--task", required=True)
    gen.add_argument("--checkpoint-sha256", required=True)
    gen.add_argument("--config-sha256", required=True)
    gen.add_argument("--repo-commit", required=True)
    gen.add_argument("--development-count", type=int, default=0)
    gen.add_argument("--fit-count", type=int, default=15)
    gen.add_argument("--eval-count", type=int, default=30)
    gen.add_argument("--exclude-env-seeds", default="1-62")
    gen.add_argument("--exclude-planner-seeds", default="1")
    gen.add_argument("--arms", default=",".join(DEFAULT_ARMS))

    val = sub.add_parser("validate")
    val.add_argument("manifests", nargs="+")
    realized = sub.add_parser("validate-realizations")
    realized.add_argument("logs", nargs="+")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate":
        result = generate_bundle(args)
    elif args.command == "validate":
        groups = [(Path(path).stem, read_jsonl(Path(path))) for path in args.manifests]
        result = validate_rows(groups)
    else:
        groups = [(Path(path).parent.name, read_jsonl(Path(path))) for path in args.logs]
        result = validate_paired_realizations(groups)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", result.get("validation", {}).get("passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
