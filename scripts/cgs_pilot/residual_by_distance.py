#!/usr/bin/env python
"""Within-distance residual cosine / field characterisation for the v0.8 varied-geometry factorial (descriptive).

`relational_transport.py` removes ONE leave-one-scene-out template per field.  On a stimulus with two hazard distances the
field has two clusters, so the residual after a single template is dominated by the distance factor and any same-distance
donor scores well.  This script repeats the residual scoring and field characterisation INSIDE each distance group (the
template and the copy-delta donor are both taken from the same distance), which is the fair version of the C1 test in
`cross model design jepa.md` ("scene-specific consequence beyond the template").  Reuses relational_transport.py helpers.

Usage: residual_by_distance.py --eval-a EV_A --cache-a slim.npz --eval-b EV_B --cache-b slim.npz --out DIR
        [--groups hazard_corridor hazard] [--split-seed 1090 --dist-labels 8m 10m]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relational_transport as rt  # noqa: E402
from token_groups import set_domain, union  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-a", type=Path, required=True); ap.add_argument("--cache-a", type=Path, required=True)
    ap.add_argument("--eval-b", type=Path, required=True); ap.add_argument("--cache-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs="*", default=["hazard_corridor", "hazard"])
    ap.add_argument("--split-seed", type=int, default=1090, help="scene seeds below this are the first distance group")
    ap.add_argument("--dist-labels", nargs=2, default=["8m", "10m"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_domain("driving")
    rng = np.random.default_rng(args.seed)
    arms = {}
    for a, ev, ca in (("A", args.eval_a, args.cache_a), ("B", args.eval_b, args.cache_b)):
        rows_all = rt.load_rows(ev); rows = rt.load_rows(ev, ev / "discovery_seeds.txt")
        arms[a] = {"rows": rows, "latents": rt.restrict(rt.load_latents(ca), rows_all, rows), "eval_dir": ev}
    cells = {a: rt.scene_cells(v["rows"]) for a, v in arms.items()}
    common = sorted(set.intersection(*[set(c) for c in cells.values()]))
    seed_of = {r["pair_id"]: int(r["seed"]) for a in arms for r in arms[a]["rows"]}
    ctx = {a: v["latents"]["context_pooled"].astype(np.float64) for a, v in arms.items()}
    fields = {"ped_did": {(1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1}, "cone_did": {(3, 1): 1, (3, 0): -1, (0, 1): -1, (0, 0): 1}}
    out = {"definitions": {"residual_cosine": "cos(x - g, T - g) with g the LOSO mean of the TRUE field over the OTHER scenes of the SAME distance group",
                           "copy_delta": "true field of the nearest other scene (pooled context L2) WITHIN the same distance group", "groups": args.groups,
                           "distance_groups": {args.dist_labels[0]: f"seed < {args.split_seed}", args.dist_labels[1]: f"seed >= {args.split_seed}"}},
           "interpretation_scope": "descriptive; within-distance version of relational_transport.py residual scoring (fair C1 test on a two-distance stimulus)", "groups": {}}
    for g in args.groups:
        tok = {a: rt.scene_tokens(v["rows"], cells[a], v["eval_dir"], v["latents"]["token_mask_last"], v["latents"]["n_steps"], g)[0] for a, v in arms.items()}
        tokens = {p: union(*[tok[a][p] for a in arms]) for p in common if all(p in tok[a] for a in arms)}
        scenes_all = [p for p in common if p in tokens and len(tokens[p]) > 0]
        gout = {}
        for label, pred in ((args.dist_labels[0], lambda s: s < args.split_seed), (args.dist_labels[1], lambda s: s >= args.split_seed), ("pooled", lambda s: True)):
            scenes = [p for p in scenes_all if pred(seed_of[p])]
            if len(scenes) < 4:
                gout[label] = {"status": "insufficient_scenes", "n": len(scenes)}; continue
            blk = {"n_scenes": len(scenes), "fields": {}}
            for a, v in arms.items():
                # cell index maps for the DiD fields
                M = rt.build_fields(v["latents"]["prediction_last"], cells[a], tokens, scenes)
                T = rt.build_fields(v["latents"]["target_last"], cells[a], tokens, scenes)
                # donors: nearest other scene by pooled context within this subset (use the H1 A1 cell's context as in the main script)
                idx = [cells[a][p][(1, 1)] if (1, 1) in cells[a][p] else cells[a][p][(0, 0)] for p in scenes]
                X = ctx[a][idx]; D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf); donors = D.argmin(1)
                for f in ("ped_did", "cone_did"):
                    if f not in M or f not in T: continue
                    Mf, Tf = np.asarray(M[f], dtype=np.float64), np.asarray(T[f], dtype=np.float64)
                    ok = rt.valid_rows(Mf, Tf)
                    if ok.sum() < 4: continue
                    sc = rt.residual_scoring(Mf[ok], Tf[ok], donors[ok] if len(donors) == len(ok) else np.arange(ok.sum()), [s for s, k in zip(scenes, ok) if k], seed=args.seed)
                    ch_t = rt.characterise_field(Tf[ok], rng, n_mc=20); ch_m = rt.characterise_field(Mf[ok], rng, n_mc=20)
                    blk["fields"][f"{a}.{f}"] = {"n": int(ok.sum()),
                        "model_raw": sc["pooled"]["model"]["raw"], "model_residual": sc["pooled"]["model"]["residual"],
                        "copy_raw": sc["pooled"]["copy_delta"]["raw"], "copy_residual": sc["pooled"]["copy_delta"]["residual"],
                        "model_minus_copy_residual": sc["model_minus_baseline"]["copy_delta"]["residual"],
                        "template_share_of_truth": sc["template_share_of_truth"], "nmse": sc["nmse"],
                        "pc1_true": ch_t.get("pc1_variance_fraction"), "pc1_model": ch_m.get("pc1_variance_fraction"),
                        "spherical_variance_true": ch_t.get("spherical_variance"), "participation_ratio_true": ch_t.get("participation_ratio")}
            gout[label] = blk
        out["groups"][g] = gout
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "residual_by_distance.json").write_text(json.dumps(json.loads(rt.canonical_json(rt.finite(out))), indent=1) + "\n")
    for g, gg in out["groups"].items():
        for label, blk in gg.items():
            if "fields" not in blk: print(g, label, blk); continue
            for f, v in blk["fields"].items():
                print(f"{g} {label} {f} n={v['n']}: model resid {v['model_residual']['mean']:+.3f} [{v['model_residual']['ci_low']:+.2f},{v['model_residual']['ci_high']:+.2f}] | copy resid {v['copy_residual']['mean']:+.3f} | model-copy {v['model_minus_copy_residual']['mean']:+.3f} p {v['model_minus_copy_residual']['sign_flip_p']:.4f} | raw model {v['model_raw']['mean']:.2f} copy {v['copy_raw']['mean']:.2f} | PC1 true {v['pc1_true']:.2f} model {v['pc1_model']:.2f}")


if __name__ == "__main__":
    main()
