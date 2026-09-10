"""CPU tests (numpy only, no torch) for the COAST-style steered planner ranking: a fake runner with planted
predictions drives the torch-free analysis core.  Checks: the safe-choice indicator equals planner_currency's
``a1_ranks_worse``; sham reproduces the unsteered ranking exactly; a planted strengthen effect on the solid identity
gives a positive DiD with H0 / H0' / hazard-free checks inside the margin (selective) while every control fails; the
frozen calibration rule picks the smallest beta whose DiD CI excludes 0; the sealed run keeps only that beta; the
JSON / table shapes; conceptor loaders (factored vs d x d, transport through W) and the band fallback rule; the
geometry_cross_arm transport export round-trips into the steering loader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import steered_planner_ranking as sp  # noqa: E402
from planner_currency import scene_planner_currency  # noqa: E402
from protocol import canonical_json  # noqa: E402
from stats_utils import finite  # noqa: E402

N_TOK, D, N_SCENES = 16, 6, 12
SITES = ("L01.attn_out", "L03.attn_out")
UNREL = {"L01.attn_out": "L04.attn_out", "L03.attn_out": "L00.attn_out"}


class FakeRunner:
    """Planted scenes around the PROGRESS goal (goal = the scene's h0a1 true future).  Ordinary cells predict their true
    future (+ noise), so throttle is preferred (unsafe) under progress and brake is preferred under the brake goal (the
    degenerate baseline).  The (solid, throttle) prediction sits near the progress goal (the model ignores the collision)
    except in every third scene; the 'main' strengthen edit adds beta * g along a direction orthogonal to the residual
    so the throttle chunk's cost to the goal exceeds the brake chunk's exactly when beta * g > r_i (a per-scene
    threshold); controls change nothing beyond 1e-3 noise; hazard-free cells are never touched."""

    def __init__(self, solid_level: int = 1, gain: float = 1.0, seed: int = 0, n_scenes: int = N_SCENES):
        rng = np.random.default_rng(seed)
        self.scenes = [f"scene{i:02d}" for i in range(n_scenes)]
        self.solid = solid_level
        self.gain = gain
        self.data = {}
        self.region = np.arange(8)
        for i, pid in enumerate(self.scenes):
            ctx = rng.normal(size=(N_TOK, D))
            drive = rng.normal(size=(N_TOK, D)) * 0.5
            true, pred = {}, {}
            for h in (0, 1, 2, 3):
                for a in (0, 1):
                    true[(h, a)] = ctx + a * drive + (0.8 * rng.normal(size=(N_TOK, D)) if (h == solid_level and a == 1) else 0.0)
                    pred[(h, a)] = true[(h, a)] + 0.05 * rng.normal(size=(N_TOK, D))
            goal = true[(0, 1)]  # progress goal
            r_i = 0.15 + 0.6 * i / (n_scenes - 1)  # beta 0.25 flips 1 unsafe scene, 0.5 flips 4, 1.0 flips all
            small = 0.3
            big = np.sqrt(small**2 + r_i**2) if i % 3 != 0 else small * 0.5  # i % 3 == 0 -> already safe unsteered
            e = rng.normal(size=(N_TOK, D)); e /= np.linalg.norm(e[self.region])
            n = rng.normal(size=(N_TOK, D)); n /= np.linalg.norm(n[self.region])
            u = rng.normal(size=(N_TOK, D)); u -= e * float(np.sum(u[self.region] * e[self.region])); u /= np.linalg.norm(u[self.region])
            pred[(solid_level, 1)] = goal + small * e  # near "progress": the collision consequence is missing
            pred[(solid_level, 0)] = goal + big * n
            ctxd = {k: ctx for k in true}
            self.data[pid] = {"true": true, "pred": pred, "ctx": ctxd, "u": u, "rng": np.random.default_rng(1000 + i)}

    def prepare(self, pid):
        pass

    def reference(self, pid):
        d = self.data[pid]
        return d["true"], d["ctx"], self.region

    def predict(self, pid, cfg, zero_action=False):
        d = self.data[pid]
        if zero_action:
            return {k: d["ctx"][k] + 0.01 for k in ((0, 0), (2, 0))}
        pred = {k: v.copy() for k, v in d["pred"].items()}
        if cfg is None or cfg.kind == "unsteered" or cfg.kind == "sham":
            return pred
        if cfg.kind == "main":
            sign = 1.0 if cfg.mode == "strengthen" else -1.0
            pred[(self.solid, 1)] = pred[(self.solid, 1)] + sign * cfg.beta * self.gain * d["u"]
        else:
            pred[(self.solid, 1)] = pred[(self.solid, 1)] + 1e-3 * d["rng"].normal(size=(N_TOK, D))
        return pred

    def edit_stats(self, pid, cfg):
        return cfg.beta * 0.1


def _configs(betas=(0.25, 0.5, 1.0), modes=("strengthen",), sites=(("L01.attn_out", "L03.attn_out"), ("L01.attn_out",))):
    return sp.build_configs([tuple(s) for s in sites], list(betas), list(modes), "hazard_corridor", "own", False, UNREL)


def test_scene_record_matches_planner_currency():
    r = FakeRunner()
    pid = r.scenes[0]
    true, ctx, tokens = r.reference(pid)
    pred = r.predict(pid, None)
    rec = sp.scene_record(pred, true, ctx, tokens, zero=r.predict(pid, None, zero_action=True))
    keys = sorted(pred)
    cur = scene_planner_currency({k: i for i, k in enumerate(keys)}, np.stack([pred[k] for k in keys]).astype(np.float32), np.stack([true[k] for k in keys]).astype(np.float32), tokens)
    assert rec["primary_goal"] == "progress" and set(rec["goals"]) == {"progress", "brake"} and rec["levels"] is rec["goals"]["progress"]
    brake = rec["goals"]["brake"]
    for lvl in ("h0", "h1", "h0prime", "h3"):
        assert brake[lvl]["safe"] == cur["a1_ranks_worse"][lvl]  # brake goal == planner_currency exactly
        assert abs(brake[lvl]["delta"] - cur["delta_cost_a1_minus_a0"][lvl]) < 1e-9
        assert -1.0 <= rec["levels"][lvl]["delta_norm"] <= 1.0
        assert rec["goals"]["progress"][lvl]["goal_cell"] == "h0a1" and brake[lvl]["goal_cell"] == f"h{sp.KEY_LEVEL[lvl]}a0"
    # progress goal by hand: cost to the h0a1 true future on the region tokens
    g = lambda a: np.asarray(a, np.float32)[tokens].astype(np.float64)  # noqa: E731
    c1 = float(np.sum((g(pred[(1, 1)]) - g(true[(0, 1)])) ** 2)); c0 = float(np.sum((g(pred[(1, 0)]) - g(true[(0, 1)])) ** 2))
    assert abs(rec["goals"]["progress"]["h1"]["delta"] - (c1 - c0)) < 1e-6
    assert set(rec["hazard_free"]) == {"arc_h0", "drift_h0", "arc_h0prime", "drift_h0prime"}
    with pytest.raises(ValueError):
        sp.scene_record({k: v for k, v in pred.items() if k[0] != 1}, true, ctx, tokens)


def test_config_builder_families_and_controls():
    cfgs = _configs()
    names = {c.name for c in cfgs}
    assert "unsteered" in names
    fam = "own|strengthen|hazard_corridor|band:L01.attn_out+L03.attn_out"
    assert f"{fam}|sham" in names and f"{fam}|b0.5" in names and f"{fam}|b0.5|wrong_site" in names and f"{fam}|b0.5|wrong_group" in names
    ws = next(c for c in cfgs if c.name == f"{fam}|b0.5|wrong_site")
    assert ws.sites == ("L04.attn_out", "L00.attn_out") and ws.conceptor_sites == ("L01.attn_out", "L03.attn_out") and ws.site_config == "band:L01.attn_out+L03.attn_out"
    wg = next(c for c in cfgs if c.name == f"{fam}|b0.5|wrong_group")
    assert wg.group == "background" and wg.conceptor_group == "hazard_corridor"
    assert sum(1 for c in cfgs if c.kind == "sham") == 2  # one sham per family
    single = next(c for c in cfgs if c.name == "own|strengthen|hazard_corridor|L01.attn_out|b1")
    assert single.site_config == "L01.attn_out"


def test_planted_effect_is_selective_controls_fail_and_calibration_rule():
    runner = FakeRunner()
    cfgs = _configs(sites=(("L01.attn_out", "L03.attn_out"),))
    res = sp.run_analysis(runner, cfgs, "h1", 0.10, n_boot=300, seed=0)
    assert res["primary_goal"] == "progress" and set(res["results"]) == {"progress", "brake"}
    agg = res["results"]["progress"]
    fam = "own|strengthen|hazard_corridor|band:L01.attn_out+L03.attn_out"
    u = agg["unsteered"]
    assert abs(u["safe_choice"]["h1"]["mean"] - 4 / 12) < 1e-9  # 4 scenes safe unsteered by construction
    assert u["safe_choice"]["h0"]["mean"] == 0.0 and u["safe_choice"]["h0prime"]["mean"] == 0.0 and u["safe_choice"]["h3"]["mean"] == 0.0  # unsafe baseline under progress
    ub = res["results"]["brake"]["unsteered"]
    assert ub["safe_choice"]["h0"]["mean"] > 0.9  # brake goal: throttle always costs more when the action effect is predicted (no unsafe baseline)
    sham = agg[f"{fam}|sham"]
    assert sham["reproduces_unsteered"] and sham["did_primary"]["mean"] == 0.0 and not sham["did_ci_excludes_zero_expected"]
    b1 = agg[f"{fam}|b1"]
    assert b1["safe_choice"]["h1"]["mean"] == 1.0  # beta * g = 1 > every r_i
    assert b1["did_ci_excludes_zero_expected"] and b1["did_primary"]["ci_low"] > 0
    assert b1["nontarget_equivalent"]["all"] and b1["hazard_free_checks"]["all_within_margin"]
    assert b1["vs_unsteered"]["h0"]["safe_choice_change"]["mean"] == 0.0 and b1["vs_unsteered"]["h0prime"]["safe_choice_change"]["mean"] == 0.0
    assert b1["did"]["h3_vs_h0"]["safe_choice"]["mean"] == 0.0  # ghost identity untouched
    assert b1["selective"] and b1["all_controls_fail"] and b1["selective_with_controls"]
    assert set(b1["controls_fail"]) == {"sham", "random_matched", "rank_one", "wrong_site", "wrong_group"}
    assert "activation_edit_relative_rms" in b1
    # smallest beta with r_i < beta: 0.25 flips 1 extra scene (CI includes 0), 0.5 flips 4 (CI excludes 0)
    b025 = agg[f"{fam}|b0.25"]
    assert not b025["did_ci_excludes_zero_expected"]
    cal = sp.calibrate(agg, {c.name: c for c in cfgs})
    assert cal["families"][fam]["beta"] == 0.5
    assert [r["beta"] for r in cal["families"][fam]["candidates"]] == [0.25, 0.5, 1.0]
    # sealed run keeps only the calibrated beta (+ sham, unsteered)
    kept, dropped = sp.restrict_to_calibration(cfgs, cal)
    assert {c.beta for c in kept if c.kind == "main"} == {0.5} and any(c.kind == "sham" for c in kept) and kept[0].kind == "unsteered" and not dropped
    kept2, dropped2 = sp.restrict_to_calibration(cfgs, {"families": {fam: {"beta": None}}})
    assert [c.kind for c in kept2] == ["unsteered"] and fam in dropped2
    # suppress on a planted-strengthen world must NOT be selective in the expected (negative) direction beyond noise
    cfg_s = sp.build_configs([("L01.attn_out", "L03.attn_out")], [1.0], ["suppress"], "hazard_corridor", "own", False, UNREL)
    res_s = sp.run_analysis(runner, cfg_s, "h1", 0.10, n_boot=200, seed=0)
    s1 = res_s["results"]["progress"]["own|suppress|hazard_corridor|band:L01.attn_out+L03.attn_out|b1"]
    assert s1["expected_sign"] == -1.0 and not s1["did_ci_excludes_zero_expected"]


def test_coast_rows_markdown_and_json_shape(tmp_path):
    runner = FakeRunner(n_scenes=8)
    cfgs = _configs(betas=(0.5,), sites=(("L01.attn_out",),))
    res = sp.run_analysis(runner, cfgs, "h1", 0.10, n_boot=100, seed=0)
    by = {c.name: c for c in cfgs}
    cal = sp.calibrate(res["results"]["progress"], by)
    rows = sp.coast_rows(res["results"]["progress"], by, "A", 0, "h1", confirmatory=True, calibration=cal)
    rows_b = sp.coast_rows(res["results"]["brake"], by, "A", 0, "h1", confirmatory=True, calibration=cal, goal="brake", primary=False)
    cols = {"arm", "seed", "goal", "setting", "safe_choice_H1_ped", "safe_choice_H1_obj", "safe_choice_H0", "safe_choice_H0prime", "n_scenes", "safe_choice_H0_ci"}
    assert all(cols <= set(r) for r in rows + rows_b)
    assert rows[0]["setting"] == "unsteered" and rows[0]["n_scenes"] == 8 and rows[0]["confirmatory"] is True and rows[0]["goal"] == "progress"
    assert all(r["goal"] == "brake" and r["goal_primary"] is False and r["confirmatory"] is False for r in rows_b)  # secondary rows never confirmatory
    main = [r for r in rows if r["kind"] == "main"]
    assert len(main) == 1 and main[0]["confirmatory"] == (cal["families"][main[0]["family"]]["beta"] == 0.5)
    assert {r["kind"] for r in rows} == {"unsteered", "main", "sham", "random_matched", "rank_one", "wrong_site", "wrong_group"}
    md = sp.coast_markdown(rows + rows_b)
    assert md.startswith("| arm | seed | goal | setting |") and md.count("\n") == len(rows) + len(rows_b) + 2 and "brake (secondary)" in md
    blob = {"results": res["results"], "coast_table": {"rows": rows + rows_b}, "per_scene": res["per_scene"], "interpretation_scope": sp.INTERPRETATION_SCOPE}
    text = canonical_json(finite(blob))  # NaN-free, numpy-free
    back = json.loads(text)
    assert back["coast_table"]["rows"][0]["safe_choice_H1_ped"] == rows[0]["safe_choice_H1_ped"]
    assert "not closed-loop task success" in back["interpretation_scope"]


def test_conceptor_loaders_factored_vs_matrix_and_transport(tmp_path):
    rng = np.random.default_rng(0)
    d, r = 12, 3
    V = np.linalg.qr(rng.normal(size=(d, r)))[0].T
    mu = np.array([0.9, 0.6, 0.3])
    C = sp.factored_to_matrix(V, mu)
    ev = np.linalg.eigvalsh(C)
    assert ev.min() > -1e-12 and ev.max() < 1 + 1e-12 and abs(np.trace(C) / d - mu.sum() / d) < 1e-12
    mean = rng.normal(size=d)
    np.savez(tmp_path / "L01.attn_out__s0__hazard_corridor.npz", safety_eigvecs=V.astype(np.float32), safety_mu=mu.astype(np.float32), mean=mean.astype(np.float32),
             rank_one_direction=(3 * V[0]).astype(np.float32), random_eigvecs=np.linalg.qr(rng.normal(size=(d, r)))[0].T.astype(np.float32), random_mu=mu.astype(np.float32))
    np.savez(tmp_path / "matrix.npz", C_safety=C, mean=mean, mean_diff=V[0])
    a = sp.load_conceptor_file(tmp_path / "L01.attn_out__s0__hazard_corridor.npz", "safety")
    b = sp.load_conceptor_file(tmp_path / "matrix.npz", "C_safety")
    assert np.allclose(a["C"], b["C"], atol=1e-5) and np.allclose(a["rank_one"], V[0], atol=1e-6) and np.allclose(b["rank_one"], V[0])
    assert abs(a["quota"] - mu.sum() / d) < 1e-6 and abs(np.trace(a["C_random"]) - np.trace(a["C"])) < 1e-5 and abs(np.trace(b["C_random"]) - np.trace(b["C"])) < 1e-9
    assert a["mean"].shape == (d,)
    # operators
    M = sp.operator_matrix(C, 0.5, "strengthen")
    assert np.allclose(M, 0.5 * np.eye(d) + 0.5 * C) and np.allclose(sp.operator_matrix(C, 0.5, "suppress"), np.eye(d) - 0.5 * C)
    with pytest.raises(ValueError):
        sp.operator_matrix(C, 0.5, "reverse")
    # transport export: A's conceptor moved into B's coordinates with W (v_B = v_A W)
    W = np.linalg.qr(rng.normal(size=(d, d)))[0]
    meanA, meanB = rng.normal(size=d), rng.normal(size=d)
    np.savez(tmp_path / "transport.npz", meta=json.dumps({}), **{"L01.attn_out__s0__W": W.astype(np.float32),
             "L01.attn_out__s0__hazard_corridor__mean_A": meanA.astype(np.float32), "L01.attn_out__s0__hazard_corridor__mean_B": meanB.astype(np.float32),
             "L01.attn_out__s0__hazard_corridor__A__rel_ped__eigvecs": V.astype(np.float32), "L01.attn_out__s0__hazard_corridor__A__rel_ped__mu": mu.astype(np.float32),
             "L01.attn_out__s0__hazard_corridor__A__rel_ped__rank_one": V[0].astype(np.float32)})
    t = sp.load_transported_conceptor(tmp_path / "transport.npz", "L01.attn_out", 0, "hazard_corridor", "A", "B", "rel_ped")
    assert np.allclose(t["C"], W.T @ C @ W, atol=1e-5) and np.allclose(t["mean"], meanB, atol=1e-6) and np.allclose(t["rank_one"], V[0] @ W, atol=1e-5)
    own = sp.load_transported_conceptor(tmp_path / "transport.npz", "L01.attn_out", 0, "hazard_corridor", "A", "A", "rel_ped")
    assert np.allclose(own["C"], C, atol=1e-5) and np.allclose(own["mean"], meanA, atol=1e-6)
    with pytest.raises(KeyError):
        sp.load_transported_conceptor(tmp_path / "transport.npz", "L02.attn_out", 0, "hazard_corridor", "A", "B", "rel_ped")


def test_top_relational_sites_rule():
    m = {"site_results": [
        {"site_id": "L02.attn_out", "step": 0, "group": "gripper_corridor", "cv_projection": {"relational": {"t": 2.0}, "interaction": {"t": 9.0}}},
        {"site_id": "L05.mlp_out", "step": 0, "group": "gripper_corridor", "cv_projection": {"relational": {"t": 5.0}, "interaction": {"t": 1.0}}},
        {"site_id": "L01.attn_out", "step": 0, "group": "gripper_corridor", "cv_projection": {"relational": {"t": 3.0}, "interaction": {"t": 1.0}}},
        {"site_id": "L03.attn_out", "step": 0, "group": "gripper_corridor", "cv_projection": {"relational": {"t": 1.0}, "interaction": {"t": 1.0}}},
        {"site_id": "L04.attn_out", "step": 1, "group": "gripper_corridor", "cv_projection": {"relational": {"t": 50.0}, "interaction": {"t": 1.0}}},
        {"site_id": "L04.attn_out", "step": 0, "group": "egg", "cv_projection": {"relational": {"t": 50.0}, "interaction": {"t": 1.0}}},
        {"site_id": "L00.adaln", "step": 0, "group": "gripper_corridor", "cv_projection": {"relational": {"t": 99.0}, "interaction": {"t": 1.0}}},
    ]}
    assert sp.top_relational_sites(m, n=3) == ["L05.mlp_out", "L01.attn_out", "L02.attn_out"]
    assert sp.top_relational_sites({"ranked": [{"site_id": "L07.attn_out", "step": 0, "group": "hazard_corridor", "relational_t": 1.0}]}, n=3) == ["L07.attn_out"]


def test_cross_arm_export_round_trips_into_the_steering_loader(tmp_path):
    import geometry_cross_arm as gx

    stim, dA, dB = gx.synthetic_dumps(tmp_path / "syn", null=False, n_scenes=8, d=24, n_steps=1, sites=("L03.attn_out",), seed=0)
    args = gx.build_parser().parse_args(["--out", str(tmp_path / "out"), "--dump-a", str(dA), "--dump-b", str(dB), "--stimulus", str(stim), "--steps", "0",
                                         "--groups", "hazard", "--n-boot", "50", "--n-perm", "50", "--n-perm-refit", "9", "--export-transport", str(tmp_path / "transport.npz")])
    report = gx.run(args)
    assert report["transport_export"] == str(tmp_path / "transport.npz")
    with np.load(tmp_path / "transport.npz") as z:
        meta = json.loads(str(z["meta"]))
        W = z["L03.attn_out__s0__W"]
        assert W.shape == (24, 24) and np.allclose(W @ W.T, np.eye(24), atol=1e-4)
        assert meta["conceptors"][0]["types"] == sorted(gx.TYPES) and meta["n_scenes"] == 8
        VA, muA = z["L03.attn_out__s0__hazard__A__rel_ped__eigvecs"], z["L03.attn_out__s0__hazard__A__rel_ped__mu"]
        meanB = z["L03.attn_out__s0__hazard__mean_B"]
    t = sp.load_transported_conceptor(tmp_path / "transport.npz", "L03.attn_out", 0, "hazard", "A", "B", "rel_ped")
    assert t["C"].shape == (24, 24) and t["mean"].shape == (24,) and 0 < t["quota"] < 1 and t["transport"] == "A->B"
    assert np.allclose(t["C"], sp.factored_to_matrix(VA.astype(np.float64) @ W, muA), atol=1e-4) and np.allclose(t["mean"], meanB, atol=1e-6)
    # the JSON must not carry the in-memory conceptors
    rep = json.loads((tmp_path / "out" / "cross_arm_map.json").read_text())
    assert "_cons" not in rep["entries"][0]["analysis"] and "_rank_one" not in rep["entries"][0]["analysis"]


def test_progress_goal_separates_the_arms_and_brake_goal_does_not():
    """Same planted scenes, arm A (pedestrian solid, level 1) vs arm B (cone solid, level 3): under the progress goal the
    unsteered safe-choice rate is high only at the solid identity in lane and ~0 at H0 / H0' / the ghost identity (the
    unsafe baseline); under the brake goal every level sits near 1 in both arms, so no dissociation is visible."""
    cfgs = [sp.UNSTEERED]
    ra = sp.run_analysis(FakeRunner(solid_level=1), cfgs, "h1", 0.10, n_boot=50, seed=0)["results"]
    rb = sp.run_analysis(FakeRunner(solid_level=3), cfgs, "h3", 0.10, n_boot=50, seed=0)["results"]
    pa, pb = ra["progress"]["unsteered"]["safe_choice"], rb["progress"]["unsteered"]["safe_choice"]
    assert pa["h1"]["mean"] > 0.3 and pa["h3"]["mean"] == 0.0 and pa["h0"]["mean"] == 0.0 and pa["h0prime"]["mean"] == 0.0
    assert pb["h3"]["mean"] > 0.3 and pb["h1"]["mean"] == 0.0 and pb["h0"]["mean"] == 0.0 and pb["h0prime"]["mean"] == 0.0
    # DiD of the unsteered rates: solid-in-lane minus H0 is positive only for the solid identity, reversed across arms
    assert (pa["h1"]["mean"] - pa["h0"]["mean"]) - (pa["h3"]["mean"] - pa["h0"]["mean"]) > 0.3
    assert (pb["h1"]["mean"] - pb["h0"]["mean"]) - (pb["h3"]["mean"] - pb["h0"]["mean"]) < -0.3
    ba, bb = ra["brake"]["unsteered"]["safe_choice"], rb["brake"]["unsteered"]["safe_choice"]
    assert min(ba["h0"]["mean"], ba["h0prime"]["mean"], bb["h0"]["mean"], bb["h0prime"]["mean"]) > 0.9  # no unsafe baseline under the brake goal
    # steering the solid identity under progress: DiD selective for arm B at its own solid level, with the calibration rule intact
    cf = sp.build_configs([("L01.attn_out",)], [0.5, 1.0], ["strengthen"], "hazard_corridor", "own", False, UNREL)
    res = sp.run_analysis(FakeRunner(solid_level=3), cf, "h3", 0.10, n_boot=200, seed=0)
    e = res["results"]["progress"]["own|strengthen|hazard_corridor|L01.attn_out|b1"]
    assert e["did_primary"]["contrast"] == "h3_vs_h0" and e["selective"] and e["did"]["h1_vs_h0"]["safe_choice"]["mean"] == 0.0
    assert sp.calibrate(res["results"]["progress"], {c.name: c for c in cf})["families"]["own|strengthen|hazard_corridor|L01.attn_out"]["beta"] == 0.5
