#!/usr/bin/env python3
"""Episode-balanced exploratory selection, with complete candidate accounting.

Selection is development-only; neither a winning score nor many candidate rows
are independent confirmation. Ranking uses terminal full-spatial forecast error,
and retains early-horizon errors, both controls, failures and observer errors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def candidate_id(spec):
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _six(values, label):
    import math
    if len(values) != 6 or not all(math.isfinite(float(v)) and float(v) >= 0 for v in values):
        raise ValueError(f"{label} must contain six finite nonnegative errors")
    return [float(v) for v in values]


def aggregate_candidates(rows, expected_episodes=tuple(range(8))):
    groups = {}
    for row in rows:
        cid = row["candidate_id"]
        groups.setdefault(cid, []).append(row)
    aggregate = []
    for cid, values in sorted(groups.items()):
        episodes = [int(r["episode"]) for r in values]
        if len(episodes) != len(set(episodes)):
            raise ValueError("Repeated episode for candidate " + cid)
        if not set(episodes).issubset(expected_episodes):
            raise ValueError("Non-development episode supplied")
        specs = [r.get("candidate", {}) for r in values]
        if any(s != specs[0] for s in specs):
            raise ValueError("Candidate spec changed between episodes")
        failed = [r for r in values if r.get("failed") or r.get("complete") is False]
        result = {"candidate_id": cid, "candidate": specs[0], "episodes": sorted(episodes),
                  "expected_episodes": list(expected_episodes), "independent_episode_count": len(episodes),
                  "failed_rows": failed, "complete": set(episodes) == set(expected_episodes) and not failed}
        if not result["complete"]:
            result.update(eligible_for_physics=False, reason="Missing or failed development episodes; never drop unfavorable starts")
            aggregate.append(result)
            continue
        paired = []
        for row in sorted(values, key=lambda r: r["episode"]):
            b, e, s = [_six(row[key], key) for key in ("baseline_mse", "edited_mse", "sham_mse")]
            denom = [max(x, 1e-12) for x in b]
            benefit = [(x-y)/d for x, y, d in zip(b, e, denom)]
            specificity = [(x-y)/d for x, y, d in zip(s, e, denom)]
            paired.append({"episode": int(row["episode"]), "relative_benefit": benefit,
                           "relative_specificity": specificity, "terminal_benefit": benefit[5],
                           "terminal_specificity": specificity[5], "early_benefit": statistics.mean(benefit[:3]),
                           "observer_error": row.get("observer_error"),
                           "raw_baseline_mse": b, "raw_edited_mse": e, "raw_sham_mse": s})
        mean_b = statistics.mean(r["terminal_benefit"] for r in paired)
        mean_s = statistics.mean(r["terminal_specificity"] for r in paired)
        positive_b = sum(r["terminal_benefit"] > 0 for r in paired)
        positive_s = sum(r["terminal_specificity"] > 0 for r in paired)
        result.update(paired_rows=paired, mean_terminal_benefit=mean_b,
                      mean_terminal_specificity=mean_s,
                      mean_early_benefit=statistics.mean(r["early_benefit"] for r in paired),
                      worst_terminal_benefit=min(r["terminal_benefit"] for r in paired),
                      positive_benefit_episodes=positive_b, positive_specificity_episodes=positive_s,
                      search_score=min(mean_b, mean_s),
                      eligible_for_physics=mean_b > 0 and mean_s > 0 and min(positive_b, positive_s) >= 2,
                      reason="Exploratory heuristic only; real-physics utility and frozen confirmation remain required")
        aggregate.append(result)
    return aggregate


def diverse_parents(aggregate, limit=8):
    """Preserve site/family breadth among *eligible* candidates, never force a win."""
    ranked = sorted((r for r in aggregate if r["eligible_for_physics"]),
                    key=lambda r: (-r["search_score"], -r["worst_terminal_benefit"], r["candidate_id"]))
    selected, deferred, seen = [], [], set()
    for row in ranked:
        spec = row["candidate"]
        key = (spec.get("family", spec.get("kind")), spec.get("block", spec.get("site")))
        if key in seen:
            deferred.append(row)
        else:
            selected.append(row); seen.add(key)
    selected = (selected + deferred)[:limit]
    return [r["candidate_id"] for r in selected]


def pair_proposals(parent_ids, aggregate, limit=8):
    """Propose interactions; execution must enforce the SAME total edit budget.

    A and B must also be tested individually at their pair-assigned component
    doses, then compare AB - A - B + baseline. No synergy claim from twice dose.
    """
    by_id = {r["candidate_id"]: r for r in aggregate}
    pairs = []
    for i, left in enumerate(parent_ids):
        for right in parent_ids[i+1:]:
            a, b = by_id[left]["candidate"], by_id[right]["candidate"]
            pairs.append({"parents": [left, right], "components": [a, b],
                          "pair_total_budget_multiplier": 1.0,
                          "component_budget_shares": [0.5, 0.5],
                          "required_comparators": ["baseline", "A_at_half_budget", "B_at_half_budget", "AB_shared_total_budget", "norm_matched_sham"],
                          "status": "proposed_not_executed", "adaptive_development_only": True})
    return pairs[:limit]


def mutate_specs(parent_ids, aggregate, legal_values, generation=2, limit=64):
    """One-field deterministic mutations; runtime still validates architecture.

    Legal values are supplied by the checkpoint-aware runner, never guessed from
    a different model. No candidate is silently overwritten or retroactively fit.
    """
    by_id = {r["candidate_id"]: r for r in aggregate}
    seen = {candidate_id(r["candidate"]) for r in aggregate}
    output = []
    for field, choices in sorted(legal_values.items()):
        for parent in parent_ids:
            original = by_id[parent]["candidate"]
            if field not in original:
                continue
            for choice in choices:
                spec = dict(original, **{field: choice})
                cid = candidate_id(spec)
                if cid in seen:
                    continue
                seen.add(cid)
                output.append({"candidate_id": cid, "candidate": spec, "parent_ids": [parent],
                               "generation": generation, "mutation_field": field,
                               "status": "proposed_requires_runtime_validation"})
    # Round-robin across mutation fields, so small budgets do not become a
    # lexicographic sweep of a single dimension.
    buckets = {}
    for row in output:
        buckets.setdefault(row["mutation_field"], []).append(row)
    balanced = []
    while any(buckets.values()) and len(balanced) < limit:
        for key in sorted(buckets):
            if buckets[key] and len(balanced) < limit:
                balanced.append(buckets[key].pop(0))
    return balanced


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--parent-limit", type=int, default=8)
    p.add_argument("--legal-values", type=Path)
    args = p.parse_args()
    rows, sources = [], []
    for path in args.inputs:
        raw = path.read_bytes(); value = json.loads(raw)
        rows.extend(value if isinstance(value, list) else value["rows"])
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    aggregate = aggregate_candidates(rows, tuple(args.episodes))
    parents = diverse_parents(aggregate, args.parent_limit)
    result = {"candidate_count": len(aggregate), "sources": sources, "all_candidates": aggregate,
              "parent_ids": parents, "pair_proposals": pair_proposals(parents, aggregate),
              "selection_is_confirmation": False, "physical_promotion_starts": [0, 4, 7],
              "untouched_confirmation_opened": False,
              "interpretation": "Adaptive DEVELOPMENT ranking; no significance claims, no pseudo-replication, no forced winning operator"}
    if args.legal_values:
        result["mutation_proposals"] = mutate_specs(parents, aggregate, json.loads(args.legal_values.read_text()))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"candidate_count": len(aggregate), "parent_ids": parents}))


if __name__ == "__main__":
    main()
