#!/usr/bin/env python3
"""Broad exploratory frontier:96 mutations plus32 genuinely fresh restarts.

This never marks a candidate confirmed or forces a candidate into physics.
Negative parents may motivate another hypothesis when all current parents fail.
Every candidate retains ancestry, the motivating score, and changed fields.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

from select_residual_search_v1 import diverse_parents, pair_proposals

FAMILIES = ("action_gain", "pca_gain", "branch_scale", "coordinate")
SITES = ("residual", "attention_preproj", "mlp_output")
MASKS = ("all", "top", "bottom", "left", "right", "center", "checker")
CHOICES = {"family": FAMILIES, "block": tuple(range(6)), "site": SITES,
           "head": (None, *range(16)), "mask": MASKS, "rank": (1, 4, 8, 16),
           "pulse": tuple(range(7)), "dose": (.125, .25, .5, 1.), "sign": (-1, 1)}


def spec_key(candidate):
    body = {k: v for k, v in candidate.items() if k != "id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid(candidate):
    try:
        if any(candidate[k] not in options for k, options in CHOICES.items()):
            return False
    except KeyError:
        return False
    if candidate["site"] != "attention_preproj" and candidate["head"] is not None:
        return False
    if candidate["family"] == "coordinate":
        return candidate["block"] == 3 and candidate["site"] == "residual" and candidate["rank"] == 8
    if candidate["family"] == "branch_scale":
        return candidate["rank"] == 1
    return True


def fresh_candidate(rng, index):
    # All six blocks recur deterministically; rotations of family/site avoid
    # coupling one family permanently to one block. Coordinate is a small quota.
    family = "coordinate" if index == 31 else FAMILIES[(index//6) % 3]
    site = SITES[(index//2 + index//6) % 3]
    c = {"family": family, "block": index % 6, "site": site,
         "head": rng.randrange(16) if site == "attention_preproj" else None,
         "mask": rng.choice(MASKS), "rank": rng.choice((1, 4, 8, 16)),
         "pulse": rng.randrange(7), "dose": rng.choice(CHOICES["dose"]), "sign": rng.choice((-1, 1))}
    if family == "coordinate": c.update(block=3, site="residual", head=None, rank=8)
    if family == "branch_scale": c["rank"] = 1
    return c


def evolve(reports, generation, prior_populations=(), count=128, seed=2026090609):
    if generation not in (1, 2) or count != 128:
        raise ValueError("Frozen initial campaign permits two128-candidate descendants")
    by_id = {}
    for report in reports:
        for row in report["all_candidates"]:
            cid = row["candidate_id"]
            if cid in by_id and row["candidate"] != by_id[cid]["candidate"]:
                raise ValueError("A candidate ID changed meaning")
            by_id[cid] = row
    complete = [r for r in by_id.values() if r.get("complete") and "search_score" in r]
    if not complete:
        raise ValueError("No completed scientific result; fix execution before mutation")
    eligible = [r for r in complete if r["eligible_for_physics"]]
    negative_parent_fallback = not eligible
    if eligible:
        parent_ids = diverse_parents(eligible, limit=24)
    else:
        ranked = sorted(complete, key=lambda r: (-r["search_score"], r["candidate_id"]))
        # Diversity without relabeling an unsuccessful candidate as eligible.
        seen_groups, parent_ids, leftovers = set(), [], []
        for row in ranked:
            c = row["candidate"]
            group = (c["family"], c["block"], c["site"])
            if group in seen_groups: leftovers.append(row["candidate_id"])
            else: parent_ids.append(row["candidate_id"]); seen_groups.add(group)
        parent_ids = (parent_ids + leftovers)[:24]
    seen = {spec_key(r["candidate"]) for r in by_id.values()}
    prior_ancestry = {}
    for population in prior_populations:
        seen.update(spec_key(c) for c in population["candidates"])
        prior_ancestry.update(population.get("ancestry", {}))
    rng = random.Random(seed + generation)
    candidates, ancestry = [], {}

    def add(spec, parent, kind, changed=()):
        key = spec_key(spec)
        if not valid(spec) or key in seen:
            return False
        seen.add(key)
        cid = f"g{generation}-{kind}-{key[:12]}"
        c = dict(spec, id=cid); candidates.append(c)
        row = by_id.get(parent)
        ancestors = [] if parent is None else list(dict.fromkeys([parent] + prior_ancestry.get(parent, {}).get("ancestor_ids", [])))
        ancestry[cid] = {"parent_ids": [] if parent is None else [parent], "ancestor_ids": ancestors,
                         "generation": generation, "origin": kind, "changed_fields": list(changed),
                         "parent_score": None if row is None else row["search_score"],
                         "parent_eligible_for_physics": False if row is None else row["eligible_for_physics"],
                         "status": "exploratory_not_executed", "motivation": "Test a local alternative" if parent else "Preserve a different search family/site",
                         "falsification": "No improvement in measured consequences beyond unsteered and matched controls; not a confirmation test"}
        return True

    attempts = 0
    while len(candidates) < 96:
        if attempts > 20000: raise RuntimeError("Unable to fill valid unique mutation budget")
        parent = parent_ids[attempts % len(parent_ids)]
        original = {k: v for k, v in by_id[parent]["candidate"].items() if k != "id"}
        c = dict(original)
        fields = rng.sample(tuple(CHOICES), 1 if attempts % 2 == 0 else 2)
        for field in fields: c[field] = rng.choice(CHOICES[field])
        changed = [k for k in CHOICES if c[k] != original[k]]
        if changed: add(c, parent, "mutation", changed)
        attempts += 1
    for index in range(32):
        for _ in range(1000):
            if add(fresh_candidate(rng, index), None, "restart"):
                break
        else: raise RuntimeError("Unable to sample a fresh valid restart")
    ranked_parents = [by_id[p] for p in parent_ids]
    pairs = pair_proposals(parent_ids[:8], ranked_parents, limit=8)
    for p in pairs:
        p["unsuccessful_parents_used_for_exploration"] = negative_parent_fallback
    coverage = {"blocks": sorted({c["block"] for c in candidates}),
                "families": sorted({c["family"] for c in candidates}),
                "sites": sorted({c["site"] for c in candidates}),
                "distinct_block_head_sites": len({(c["block"], c["head"]) for c in candidates if c["head"] is not None}),
                "ranks_requested": sorted({c["rank"] for c in candidates}),
                "pulses": sorted({c["pulse"] for c in candidates})}
    return {"generation": generation, "candidates": candidates, "ancestry": ancestry,
            "pair_proposals": pairs, "coverage": coverage, "mutations": 96, "fresh_restarts": 32,
            "negative_parent_fallback": negative_parent_fallback, "seed": seed + generation,
            "candidate_actual_rank": "Runtime records data-supported rank; requested16 action directions may span at most15",
            "physical_promotion": "Use actual new screen results; generating a child or pair never grants promotion",
            "confirmation_used": False, "candidate_count_is_not_independent_sample_size": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, nargs="+", required=True)
    p.add_argument("--prior-populations", type=Path, nargs="*", default=[])
    p.add_argument("--generation", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = [json.loads(path.read_text()) for path in args.selection]
    populations = [json.loads(path.read_text()) for path in args.prior_populations]
    result = evolve(reports, args.generation, populations)
    result["selection_sources"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.selection]
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"generation": args.generation, "candidate_count": len(result["candidates"]), "coverage": result["coverage"]}))


if __name__ == "__main__": main()
