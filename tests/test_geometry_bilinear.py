"""CPU tests for the bilinear relational probe: a planted bilinear term (route x egg) is recovered by the bilinear
readout and not by linear / kernel readouts on the concatenation; the quartet-shuffled null is at chance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))
sys.path.insert(0, str(ROOT / "tests"))

import geometry_bilinear as gb  # noqa: E402
from geometry_models import BilinearRelational, auroc, bilinear_fit, bilinear_predict  # noqa: E402
from test_geometry_localize import D, build_dump, build_stimulus  # noqa: E402


def planted(n_scenes=80, d=12, rank=2, noise=0.05, seed=0):
    """Cells: z_g, z_e random; label = sign(z_g^T W z_e) with W = U V^T rank 2.  Neither half alone nor their
    concatenation carries a linear signal (W has no rank-1 mean structure); the elementwise product does not
    either because U, V are not axis-aligned."""
    rng = np.random.default_rng(seed)
    n = n_scenes * 4
    Zg = rng.normal(size=(n, d))
    Ze = rng.normal(size=(n, d))
    U = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    V = np.linalg.qr(rng.normal(size=(d, rank)))[0]
    s = np.sum((Zg @ U) * (Ze @ V), axis=1)
    y = (s + noise * rng.normal(size=n) > 0).astype(float)
    pair = np.repeat([f"p{i}" for i in range(n_scenes)], 4)
    hz = np.tile([0, 0, 1, 1], n_scenes)
    ac = np.tile([0, 1, 0, 1], n_scenes)
    return Zg, Ze, y, pair, hz, ac, U, V


def test_bilinear_fit_recovers_planted_term_and_baselines_fail():
    Zg, Ze, y, pair, hz, ac, U, V = planted()
    oof = gb.loso_readouts(Zg, Ze, y, pair, "binary", ranks=(1, 2), seed=0)
    ok = np.isfinite(oof["bilinear_r2"])
    a = {k: auroc(v[ok], y[ok]) for k, v in oof.items()}
    assert a["bilinear_r2"] > 0.8, a
    assert a["bilinear_r2"] > a["bilinear_r1"] - 0.02  # rank 2 needed for a rank-2 term (r1 may partially fit)
    assert a["linear_concat"] < 0.65, a
    assert a["linear_product"] < 0.7, a
    assert a["rbf_concat"] < a["bilinear_r2"] - 0.1, a
    # W recovered: the fitted U V^T subspaces align with the planted ones
    Uf, Vf, b = bilinear_fit(Zg, Ze, y, 2, 1.0)
    from geometry_models import subspace_similarity
    assert subspace_similarity(Uf.T, U.T) > 0.8 and subspace_similarity(Vf.T, V.T) > 0.8
    # quartet-shuffled null is at chance
    null = gb.quartet_shuffled_null(Zg, Ze, y, pair, hz, ac, "binary", 2, n_shuffles=20, seed=0)
    assert abs(np.nanmean(null) - 0.5) < 0.1


def test_entrant_interface_on_concatenated_rows():
    Zg, Ze, y, pair, hz, ac, U, V = planted(n_scenes=80)
    X = np.concatenate([Zg, Ze], axis=1)
    m = BilinearRelational(budget=2, lam=1.0).fit(X, y, {"pair_id": pair, "hazard": hz, "action": ac})
    assert m.geometry_summary()["rank"] == 2
    assert auroc(m.score(X), y) > 0.9
    delta = m.edit_direction(X[0], 1.0, 1.0)
    assert delta.shape == (2 * Zg.shape[1],) and abs(np.linalg.norm(delta) - 1.0) < 1e-6
    assert np.all(delta[Zg.shape[1]:] == 0)


def test_end_to_end_on_fake_dump(tmp_path: Path):
    stim, cells, manifest, bearings = build_stimulus(tmp_path, 12)
    rng = np.random.default_rng(7)
    # activations with a relational interaction: the interaction contrast is carried by a route x egg product
    # structure so that the residualized route/egg means interact bilinearly
    u = np.linalg.qr(rng.normal(size=(D, 2)))[0]
    X = np.zeros((len(cells), D))
    for c in cells:
        s = int(c["seed"]) - 300
        h, a = c["hazard"], c["candidate_action"]
        X[c["row"]] = rng.normal(size=D) * 0.05 + 0.5 * h * u[:, 0] + 0.5 * a * u[:, 1] + (2 * h - 1) * (2 * a - 1) / 4 * (u[:, 0] + u[:, 1])
    dump = build_dump(tmp_path, stim, cells, X, sites=("L07.attn_out",))
    seeds = tmp_path / "disc.txt"
    seeds.write_text("\n".join(str(300 + s) for s in range(12)))
    rep = gb.main(["--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "out"), "--discovery-seeds", str(seeds),
                   "--steps", "0", "--n-boot", "30", "--n-shuffles", "5", "--pca-dims", "8"])
    assert (tmp_path / "out" / "bilinear_map.json").exists()
    e = rep["entries"][0]
    assert e["status"] == "ok"
    t = e["targets"]["interaction"]
    assert t["status"] == "ok"
    for k in ("bilinear_r1", "bilinear_r2", "linear_concat", "linear_product", "linear_concat_product", "rbf_concat", "best_baseline",
              "delta_bilinear_r2_minus_best_baseline", "bilinear_r2_shuffled_null", "bilinear_r2_per_token_pair_max_insample"):
        assert k in t, k
    assert e["targets"]["null_contrast"]["status"] == "unavailable"
    assert "interaction|r2" in rep["ranked_tables"] and rep["ranked_tables"]["interaction|r2"][0]["site_id"] == "L07.attn_out"
