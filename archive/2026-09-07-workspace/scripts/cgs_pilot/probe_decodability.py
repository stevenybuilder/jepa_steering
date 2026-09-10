#!/usr/bin/env python3
"""Panel P diagnostic: is the native outcome linearly decodable from the captured planner states?

"Decodability is not use" (representational_geometry_paper_concepts.md §14) — but the converse
matters here: if a cross-validated probe cannot separate success from failure at a site, a conceptor
fit at that site has nothing to gate, and a steering null would be uninformative about the method.
Episode-grouped 3-fold CV, ridge probe, AUROC scored per HELD-OUT EPISODE (mean of its rows), plus a
label-permutation null (labels shuffled at the episode level, 200 draws). CPU only.
"""
import json, sys, numpy as np
from pathlib import Path

cap = Path(sys.argv[1]); pool = sys.argv[2] if len(sys.argv) > 2 else "planner"
idx = json.loads((cap / "activations" / "index.json").read_text())
eps = idx["episodes"]

def load(site):
    X, y, g = [], [], []
    for ei, r in enumerate(eps):
        z = np.load(cap / "activations" / r["file"])
        k = f"{site}__{pool}"
        if k not in z.files: continue
        v = z[k]
        X.append(v); y += [int(r.get("success", 0))] * len(v); g += [ei] * len(v)
    return np.concatenate(X), np.array(y), np.array(g)

def auroc(scores, labels):
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    p, n = (labels == 1).sum(), (labels == 0).sum()
    if p == 0 or n == 0: return np.nan
    return (ranks[labels == 1].sum() - p * (p + 1) / 2) / (p * n)

def cv_auroc(X, y, g, lam=1e2, rng=None, perm=False):
    groups = np.array(sorted(set(g.tolist())))
    ep_lab = {e: y[g == e][0] for e in groups}
    if perm:
        vals = rng.permutation([ep_lab[e] for e in groups])
        ep_lab = dict(zip(groups, vals))
        y = np.array([ep_lab[e] for e in g])
    folds = [groups[i::3] for i in range(3)]
    s_all, l_all = [], []
    for te in folds:
        tr = ~np.isin(g, te)
        Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(0); Xc = Xtr - mu
        A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
        w = np.linalg.solve(A, Xc.T @ (ytr - ytr.mean()))
        for e in te:
            m = g == e
            if m.sum() == 0: continue
            s_all.append(float(((X[m] - mu) @ w).mean())); l_all.append(ep_lab[e])
    return auroc(np.array(s_all), np.array(l_all))

sites = sorted({k.split("__")[0] for k in np.load(cap / "activations" / eps[0]["file"]).files if k.endswith(f"__{pool}")})
rng = np.random.default_rng(0)
print(f"{len(eps)} episodes, {sum(e['success'] for e in eps)} success | pool={pool} | sites={len(sites)}")
print(f"{'site':18s} {'AUROC':>7s} {'null_mean':>9s} {'null_p95':>8s} {'p_perm':>7s}")
rows = []
for site in sites:
    X, y, g = load(site)
    obs = cv_auroc(X, y, g)
    null = np.array([cv_auroc(X, y, g, rng=rng, perm=True) for _ in range(60)])
    null = null[~np.isnan(null)]
    p = float((null >= obs).mean()) if len(null) else np.nan
    rows.append((site, obs, null.mean(), np.percentile(null, 95), p))
for site, obs, nm, n95, p in sorted(rows, key=lambda r: -r[1]):
    print(f"{site:18s} {obs:7.3f} {nm:9.3f} {n95:8.3f} {p:7.3f}")
