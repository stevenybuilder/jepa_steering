#!/usr/bin/env python3
"""Panel P stage 3 — conceptor fitting and inference-time steering on the RELEASED evaluator.

Governed by "Label-first principle, amendment 3" (`cross model design jepa.md`).  The label is the
released evaluator's per-episode success flag (`succ_def: simu`); nothing here changes the planner,
goal sampler, horizon, or success definition.  Two subcommands:

``fit``    Read the stage-2 activation dumps (`public_panel_capture.py`) plus their episode labels and
           fit, per site:
             C_success, C_failure   conceptors  C = R (R + a^-2 I)^-1   on centered pooled states
                                    (Jaeger 2014; COAST arXiv:2605.17144 eq. 1-3)
             C_steer = C_success AND NOT C_failure       (COAST eq. 4; NOT C = I - C,
                                    A AND B = (A^-1 + B^-1 - I)^-1 via pinv)
           Site selection follows COAST's own heuristic — layer by conceptor quota, aperture range
           narrowed by success/failure subspace overlap — computed on FITTING episodes only.
           Also fits the registered control operators: `rank_one` (CAA: mean-difference direction,
           matched Frobenius norm), `random_matched` (matched-spectrum random conceptor) and, when
           `--sonar` is given, the project's own operators through their modules
           (`causal_metric_steer.py`, `modulation_operator.py`) so the same table can carry them.

``steer``  Re-run the released evaluator with a read-only forward hook applying COAST's multiplicative
           gate  h' = h M^T,  M = (1 - beta) I + beta C  at one site, and log the native success flags.
           `--kind sham` (beta = 0) must reproduce the unsteered episodes bit-for-bit; `wrong_site`
           applies the same operator `--unrelated-offset` blocks away.

Outputs
-------
fit:   <out>/operators/<site>.npz            C_steer, C_success, C_failure, mean, quota, overlap
       <out>/fit_summary.json                per-site quota/overlap/rank, selected site, n per class
steer: <out>/episodes.jsonl, DONE.json       same schema as the screen (native success per episode)

Usage
-----
    python public_panel_coast.py fit   --capture <dir> --out <dir> [--pool mean] [--alpha ...]
    python public_panel_coast.py steer --repo ... --env pt --model jepa-wm --checkpoint ... \
        --operators <dir>/operators --site L03.resid_post --beta 0.5 --kind main --out <dir> --episodes 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ conceptor algebra
def conceptor(X: np.ndarray, alpha: float) -> np.ndarray:
    """C = R (R + alpha^-2 I)^-1 on centered rows X [N, D]."""
    Xc = X - X.mean(axis=0, keepdims=True)
    R = (Xc.T @ Xc) / max(len(Xc), 1)
    D = R.shape[0]
    return R @ np.linalg.pinv(R + (alpha ** -2) * np.eye(D))


def c_not(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0]) - C


def c_and(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """A AND B = (A^-1 + B^-1 - I)^-1, computed with a jitter so that rank-deficient conceptors
    (fewer stacked states than dimensions) do not blow the pseudo-inverse up. COAST stacks one
    vector per denoising/autoregressive step, so N >> D there; the jitter keeps small-N cells sane."""
    I = np.eye(A.shape[0])
    Ai = np.linalg.pinv(A + eps * I)
    Bi = np.linalg.pinv(B + eps * I)
    C = np.linalg.pinv(Ai + Bi - I + eps * I)
    # a conceptor's eigenvalues live in [0, 1]; clip numerical excursions
    w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.clip(w, 0.0, 1.0)) @ V.T


def quota(C: np.ndarray) -> float:
    return float(np.trace(C) / C.shape[0])


def overlap(A: np.ndarray, B: np.ndarray) -> float:
    num = float(np.trace(A @ B))
    den = float(np.sqrt(np.trace(A @ A) * np.trace(B @ B))) or 1.0
    return num / den


def matched_random_conceptor(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same eigenvalue spectrum, random orthonormal basis."""
    w = np.linalg.eigvalsh((C + C.T) / 2)
    Q, _ = np.linalg.qr(rng.standard_normal(C.shape))
    return (Q * w) @ Q.T


# ------------------------------------------------------------------ data loading
def load_capture(capture_dir: Path, pool: str, unit: str = "replan"):
    """Stack states from the pre-outcome window.

    ``unit='replan'`` (default, COAST's construction: "one vector h per denoising or autoregressive
    step ... stacking N such vectors yields X") gives one row per replan, labelled by its episode's
    native success flag, with ``groups`` holding the episode index so statistics stay scene-clustered.
    ``unit='episode'`` averages each episode into a single row (rank-deficient for small screens).
    """
    idx = json.loads((capture_dir / "activations" / "index.json").read_text())
    win = idx.get("pre_outcome_window") or [0, 1]
    lo, hi = int(win[0]), max(int(win[1]), 1)
    # `planner` rows are the states the CEM itself computed at the single plan call, i.e. strictly
    # BEFORE any action is executed and before the outcome exists — the cleanest pre-outcome window
    # available in these released configs (they plan once per episode).
    planner_pool = pool == "planner"
    per_site: dict[str, dict[str, list]] = {}
    labels, eps = [], []
    for ei, row in enumerate(idx["episodes"]):
        z = np.load(capture_dir / "activations" / row["file"])
        keys = [k for k in z.files if k.endswith(f"__{pool}")]
        if not keys:
            continue
        y = int(row.get("success", 0))
        for k in keys:
            site = k.split("__")[0]
            v = z[k] if planner_pool else z[k][lo:hi]
            if v.size == 0:
                continue
            d = per_site.setdefault(site, {"X": [], "y": [], "g": []})
            if unit == "episode":
                d["X"].append(v.mean(axis=0)); d["y"].append(y); d["g"].append(ei)
            else:
                for r in v:
                    d["X"].append(r); d["y"].append(y); d["g"].append(ei)
        labels.append(y)
        eps.append(row)
    out = {s: (np.stack(d["X"]), np.array(d["y"]), np.array(d["g"]))
           for s, d in per_site.items() if d["X"]}
    return out, idx, np.array(labels), eps


# ------------------------------------------------------------------ fit
def cmd_fit(args) -> int:
    cap = Path(args.capture)
    out = Path(args.out)
    (out / "operators").mkdir(parents=True, exist_ok=True)
    per_site, idx, labels, eps = load_capture(cap, args.pool, unit=args.unit)
    if not per_site:
        print(json.dumps({"error": "no activations", "capture": str(cap)}))
        return 1
    n_s, n_f = int((labels == 1).sum()), int((labels == 0).sum())
    if n_s < args.min_per_class or n_f < args.min_per_class:
        print(json.dumps({"error": "class floor not met", "n_success": n_s, "n_failure": n_f,
                          "min_per_class": args.min_per_class}))
        return 2

    alphas = [float(a) for a in args.alphas.split(",")]
    rng = np.random.default_rng(args.seed)
    rows = []
    for site, (X, y, g) in sorted(per_site.items()):
        Xs, Xf = X[y == 1], X[y == 0]
        n_ep_s = len(set(g[y == 1].tolist())); n_ep_f = len(set(g[y == 0].tolist()))
        if n_ep_s < args.min_per_class or n_ep_f < args.min_per_class:
            continue
        # Aperture selection: COAST optimises alpha on the fitting rollouts. We use a cheap,
        # label-honest proxy computed by 3-fold cross-validation GROUPED BY EPISODE — fit the
        # contrastive conceptor on the training episodes, then score held-out states by the
        # quadratic form h~^T C h~ and take the success-minus-failure gap in units of its own
        # spread. Quota and success/failure overlap are recorded but do not select.
        groups = np.array(sorted(set(g.tolist())))
        folds = [groups[i::3] for i in range(3)]
        best = None
        for a in alphas:
            gaps = []
            for te in folds:
                tr_mask = ~np.isin(g, te)
                te_mask = np.isin(g, te)
                if tr_mask.sum() < 4 or te_mask.sum() < 2:
                    continue
                ys_tr, yf_tr = X[tr_mask & (y == 1)], X[tr_mask & (y == 0)]
                if len(ys_tr) < 2 or len(yf_tr) < 2:
                    continue
                Ctr = c_and(conceptor(ys_tr, a), c_not(conceptor(yf_tr, a)))
                mu_tr = X[tr_mask].mean(axis=0)
                Z = X[te_mask] - mu_tr
                q = np.einsum("ij,jk,ik->i", Z, Ctr, Z)
                yt = y[te_mask]
                if (yt == 1).sum() < 1 or (yt == 0).sum() < 1:
                    continue
                sd = q.std() or 1.0
                gaps.append(float((q[yt == 1].mean() - q[yt == 0].mean()) / sd))
            if not gaps:
                continue
            Cs, Cf = conceptor(Xs, a), conceptor(Xf, a)
            Cst = c_and(Cs, c_not(Cf))
            cand = dict(alpha=a, quota=quota(Cst), overlap=overlap(Cs, Cf),
                        score=float(np.mean(gaps)), cv_gaps=[round(x, 4) for x in gaps],
                        Cs=Cs, Cf=Cf, Cst=Cst)
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is None:
            continue
        Cst = best["Cst"]
        mu = X.mean(axis=0)
        mu_s, mu_f = Xs.mean(axis=0), Xf.mean(axis=0)
        d = mu_s - mu_f
        rank_one = np.outer(d, d) / (np.linalg.norm(d) ** 2 + 1e-12) * np.linalg.norm(Cst) / (np.sqrt(1) + 1e-12)
        np.savez_compressed(
            out / "operators" / f"{site}.npz",
            C_steer=Cst.astype(np.float32), C_success=best["Cs"].astype(np.float32),
            C_failure=best["Cf"].astype(np.float32), mean=mu.astype(np.float32),
            rank_one=rank_one.astype(np.float32),
            random_matched=matched_random_conceptor(Cst, rng).astype(np.float32),
            alpha=np.float32(best["alpha"]),
        )
        rows.append({"site": site, "alpha": best["alpha"], "cv_score": round(best["score"], 4),
                     "cv_gaps": best["cv_gaps"], "quota": round(best["quota"], 5),
                     "overlap_success_failure": round(best["overlap"], 4),
                     "n_rows_success": int(len(Xs)), "n_rows_failure": int(len(Xf)),
                     "n_episodes_success": int(n_ep_s), "n_episodes_failure": int(n_ep_f),
                     "dim": int(X.shape[1])})

    if not rows:
        print(json.dumps({"error": "no site met the class floor"}))
        return 3
    # COAST: choose the layer by conceptor quota (closest to the target), overlap breaks ties
    sel = sorted(rows, key=lambda r: -r["cv_score"])[0]   # site by cross-validated discrimination
    summary = {
        "capture": str(cap), "pool": args.pool, "pre_outcome_window": idx.get("pre_outcome_window"),
        "unit": args.unit, "n_episodes": len(eps), "n_success": n_s, "n_failure": n_f,
        "alphas": alphas, "selection": "site and aperture by 3-fold episode-grouped CV discrimination",
        "selected_site": sel["site"], "selected_alpha": sel["alpha"], "sites": rows,
        "operator_kinds": ["main", "sham", "rank_one", "random_matched", "wrong_site"],
        "governed_by": "Label-first principle, amendment 3",
    }
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("selected_site", "selected_alpha", "n_success", "n_failure")}
                     | {"cv_score": sel["cv_score"], "quota": sel["quota"]}))
    return 0


# ------------------------------------------------------------------ steer
def install_steering(site: str, op_path: Path, beta: float, kind: str, offset: int, log: dict):
    """Install COAST's multiplicative gate h' = h M^T on the predictor block named `site`."""
    import torch
    from evals.simu_env_planning.planning.gc_agent import GC_Agent

    z = np.load(op_path)
    key = {"main": "C_steer", "rank_one": "rank_one", "random_matched": "random_matched"}.get(kind, "C_steer")
    C = torch.as_tensor(z[key], dtype=torch.float32)
    # Center/covariance decomposition (research thread §33): COAST's gate h' = h M^T is a
    # covariance-subspace operation about the ORIGIN and says nothing about where the success cloud
    # sits. Separating the two makes "does the center matter?" falsifiable with matched arms:
    #   cov_only  h' = h M^T                       (= COAST `main`)
    #   mean_only h' = h + beta (mu_s - mu_f)      (center shift, no subspace)
    #   mean_cov  h' = mu_s + M (h - mu_s)         (affine about the success center)
    mu_s = torch.as_tensor(z["mean_success"], dtype=torch.float32) if "mean_success" in z.files else None
    mu_f = torch.as_tensor(z["mean_failure"], dtype=torch.float32) if "mean_failure" in z.files else None
    if kind in ("mean_only", "mean_cov") and (mu_s is None or mu_f is None):
        raise SystemExit(f"operator {op_path} has no class centers; refit with the current fit stage")
    blk_idx, part = int(site.split(".")[0][1:]), site.split(".")[1]
    if kind == "wrong_site":
        blk_idx = blk_idx + offset
    if kind == "sham":
        beta = 0.0
    state = {"n_calls": 0, "installed": False, "block": blk_idx, "part": part, "beta": beta, "kind": kind}
    orig_plan = GC_Agent.plan

    def plan(self, zc, steps_left=None):
        if not state["installed"]:
            pred = getattr(self.model, "predictor", None) or getattr(getattr(self.model, "model", None), "predictor", None)
            blocks = getattr(pred, "predictor_blocks", None) or getattr(pred, "blocks")
            b = min(state["block"], len(blocks) - 1)
            state["block_used"] = b
            mod = blocks[b] if part == "resid_post" else getattr(blocks[b], {"attn_out": "attn", "mlp_out": "mlp"}[part])
            b = state["beta"]
            M = (1.0 - b) * torch.eye(C.shape[0]) + b * C
            k = state["kind"]

            def hook(_m, _a, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not hasattr(t, "shape") or t.shape[-1] != M.shape[0]:
                    return out
                state["n_calls"] += 1
                Md = M.to(t.device, t.dtype)
                if k == "mean_only":
                    new = t + b * (mu_s - mu_f).to(t.device, t.dtype)
                elif k == "mean_cov":
                    c = mu_s.to(t.device, t.dtype)
                    new = c + (t - c) @ Md.T
                else:                                  # main / sham / rank_one / random_matched / wrong_site
                    new = t @ Md.T
                if isinstance(out, tuple):
                    return (new,) + tuple(out[1:])
                return new

            mod.register_forward_hook(hook)
            state["installed"] = True
            log["steering"] = {k: v for k, v in state.items() if k != "installed"}
        return orig_plan(self, zc, steps_left=steps_left)

    GC_Agent.plan = plan
    return state


def cmd_steer(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import public_panel_eval as ppe

    repo, params, meta, out, jsonl = ppe.resolve_config(
        args,
        extra_deviations=[{
            "field": "runtime", "from": "unsteered",
            "to": f"COAST multiplicative gate at {args.site}, beta={args.beta}, kind={args.kind}",
            "reason": "Panel P stage 3 steering arm; planner/goal/horizon/success untouched"}],
    )
    import logging
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logging.basicConfig(); logging.getLogger().setLevel(logging.INFO)
    from src.utils.distributed import init_distributed
    from evals.scaffold import main as eval_main
    import evals.simu_env_planning.planning.plan_evaluator as pe

    meta = dict(meta); meta.update({"steer_site": args.site, "steer_beta": args.beta, "steer_kind": args.kind})
    ppe.install_logging_hooks(pe, jsonl, meta)
    log: dict = {}
    st = install_steering(args.site, Path(args.operators) / f"{args.site}.npz", args.beta, args.kind,
                          args.unrelated_offset, log)

    init_distributed(rank_and_world_size=(0, 1))   # what evals/main.py --debug does (rank 0, world 1)
    t0 = time.time()
    eval_main(params["eval_name"], args_eval=params)
    rows = [json.loads(l) for l in open(jsonl)] if jsonl.exists() else []
    done = {"n_logged": len(rows), "n_success": int(sum(r.get("success", 0) for r in rows)),
            "success_rate": round(sum(r.get("success", 0) for r in rows) / max(len(rows), 1), 4),
            "steering": log.get("steering"), "hook_calls": st["n_calls"],
            "wall_s": round(time.time() - t0, 1)}
    (out / "DONE.json").write_text(json.dumps(done))
    print(json.dumps(done))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--capture", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--pool", default="planner", choices=["planner", "mean", "mean_all"],
                   help="planner = states computed inside the CEM at the single plan call (default)")
    f.add_argument("--unit", default="replan", choices=["replan", "episode"],
                   help="replan = COAST's construction (one row per planning step)")
    f.add_argument("--alphas", default="0.5,1,2,5,10,20")
    f.add_argument("--target-quota", type=float, default=0.01, help="COAST reports ~1% of hidden dim")
    f.add_argument("--min-per-class", type=int, default=3, help="COAST App. A.8 floor")
    f.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("steer", parents=[], add_help=True)
    for a, kw in [("--repo", {"required": True}), ("--env", {"required": True}), ("--model", {"required": True}),
                  ("--checkpoint", {"required": True}), ("--out", {"required": True}),
                  ("--seed", {"type": int, "default": None}), ("--episodes", {"type": int, "default": None}),
                  ("--task", {"default": None}), ("--label", {"default": "steer"}),
                  ("--operators", {"required": True}), ("--site", {"required": True}),
                  ("--beta", {"type": float, "default": 0.5}),
                  ("--kind", {"default": "main",
                              "choices": ["main", "sham", "rank_one", "random_matched", "wrong_site",
                                          "mean_only", "mean_cov"]}),
                  ("--unrelated-offset", {"type": int, "default": 3})]:
        s.add_argument(a, **kw)

    args = ap.parse_args()
    return cmd_fit(args) if args.cmd == "fit" else cmd_steer(args)


if __name__ == "__main__":
    raise SystemExit(main())
