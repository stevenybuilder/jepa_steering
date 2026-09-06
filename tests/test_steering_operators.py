"""Tests for the two donor-free steering operators (design doc "Registered additions" (a) and (b)) and their integration
into ``steered_planner_ranking.py``: the frozen conceptor path is untouched (config names, families, control set, asdict
schema apart from the additive ``operator`` field); the new modes build their own families and control kinds; the sham of
every operator is bit-identical through the real hook classes on a fake AdaLN predictor; the torch-free analysis core
treats the new control kinds as controls; the module self-tests pass."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import causal_metric_steer as cms  # noqa: E402
import steered_planner_ranking as sp  # noqa: E402
from test_steered_planner_ranking import UNREL, FakeRunner  # noqa: E402

torch = pytest.importorskip("torch")
import modulation_operator as mo  # noqa: E402


def test_conceptor_configs_and_schema_unchanged():
    cfgs = sp.build_configs([("L01.attn_out", "L03.attn_out"), ("L01.attn_out",)], [0.25, 0.5, 1.0], ["strengthen"], "hazard_corridor", "own", False, UNREL)
    names = [c.name for c in cfgs]
    fam = "own|strengthen|hazard_corridor|band:L01.attn_out+L03.attn_out"
    assert names[:8] == ["unsteered", f"{fam}|sham", f"{fam}|b0.25", f"{fam}|b0.25|random_matched", f"{fam}|b0.25|rank_one", f"{fam}|b0.25|wrong_site", f"{fam}|b0.25|wrong_group", f"{fam}|b0.5"]
    assert all(c.operator == "conceptor" for c in cfgs)
    assert {c.kind for c in cfgs} == {"unsteered", "sham", "main", "random_matched", "rank_one", "wrong_site", "wrong_group"}
    assert set(asdict(cfgs[2])) == {"name", "family", "kind", "source", "sites", "conceptor_sites", "beta", "mode", "group", "persistent", "step", "variant", "conceptor_group", "operator"}
    assert sp.OPERATOR_CONTROLS["conceptor"] == sp.CONTROLS


def test_new_operator_families_and_controls():
    md = sp.build_configs([("L03.mlp_out",)], [0.5], ["strengthen", "suppress"], "hazard_corridor", "min_distortion", False, {"L03.mlp_out": "L00.mlp_out"},
                          sp.OPERATOR_CONTROLS["min_distortion"], operator="min_distortion")
    kinds = {c.kind for c in md}
    assert kinds == {"unsteered", "sham", "main", "euclidean_twin", "random_matched", "wrong_site", "wrong_group"}
    assert all(c.operator == "min_distortion" for c in md if c.kind != "unsteered")
    e = next(c for c in md if c.kind == "euclidean_twin")
    assert e.variant == "euclidean_twin" and e.sites == ("L03.mlp_out",) and e.family == "min_distortion|strengthen|hazard_corridor|L03.mlp_out"
    mod = sp.build_configs([("L02.adaln",)], [1.0], ["strengthen"], "hazard", "modulation:all:r2", False, {"L02.adaln": "L05.adaln"}, sp.OPERATOR_CONTROLS["modulation"], operator="modulation")
    assert {c.kind for c in mod} == {"unsteered", "sham", "main", "ungated", "ungated_on", "random_matched", "wrong_block", "wrong_group"}
    wb = next(c for c in mod if c.kind == "wrong_block")
    assert wb.sites == ("L05.adaln",) and wb.conceptor_sites == ("L02.adaln",)
    wg = next(c for c in mod if c.kind == "wrong_group")
    assert wg.group == "corridor" and wg.conceptor_group == "hazard"
    assert sp.TorchRunner.parse_modulation_source("modulation:all:r2") == ("all", 2)
    assert set(sp.CONTROL_KINDS) >= {"euclidean_twin", "ungated", "ungated_on", "wrong_block", "rank_one"}


def test_analysis_core_counts_new_control_kinds():
    class Runner(FakeRunner):
        def predict(self, pid, cfg, zero_action=False):
            d = self.data[pid]
            if zero_action:
                return {k: d["ctx"][k] + 0.01 for k in ((0, 0), (2, 0))}
            pred = {k: v.copy() for k, v in d["pred"].items()}
            if cfg is None or cfg.kind in ("unsteered", "sham"):
                return pred
            if cfg.kind == "main":
                pred[(self.solid, 1)] = pred[(self.solid, 1)] + cfg.beta * self.gain * d["u"]
            else:  # euclidean_twin / random_matched / wrong_site / wrong_group change nothing beyond noise
                pred[(self.solid, 1)] = pred[(self.solid, 1)] + 1e-3 * d["rng"].normal(size=pred[(self.solid, 1)].shape)
            return pred

    cfgs = sp.build_configs([("L03.mlp_out",)], [1.0], ["strengthen"], "hazard_corridor", "min_distortion", False, {"L03.mlp_out": "L00.mlp_out"}, sp.OPERATOR_CONTROLS["min_distortion"], operator="min_distortion")
    res = sp.run_analysis(Runner(), cfgs, "h1", 0.10, n_boot=200, seed=0)
    e = res["results"]["progress"]["min_distortion|strengthen|hazard_corridor|L03.mlp_out|b1"]
    assert set(e["controls_fail"]) == {"sham", "euclidean_twin", "random_matched", "wrong_site", "wrong_group"}
    assert e["selective"] and e["all_controls_fail"] and e["selective_with_controls"]
    assert res["results"]["progress"]["min_distortion|strengthen|hazard_corridor|L03.mlp_out|sham"]["reproduces_unsteered"]
    rows = sp.coast_rows(res["results"]["progress"], {c.name: c for c in cfgs}, "A", 0, "h1", False, None)
    assert {r["operator"] for r in rows} == {"conceptor", "min_distortion"} and rows[0]["operator"] == "conceptor"  # unsteered row keeps the default


def test_min_distortion_self_test_and_hook_sham_bit_identical(tmp_path):
    checks = cms.self_test(tmp_path, seed=0)
    assert checks["passed"] and checks["cos_main_edit_vs_planted"] > 0.9 and checks["cos_euclid_edit_vs_planted"] < 0.6
    assert (tmp_path / "self_test.json").exists()
    # operator round trip through the file format the steering script loads
    syn = cms.synthetic_rows(seed=1)
    op = cms.fit_operator(syn["rel"], syn["groups"], syn["groups"]["action"], syn["receiver"], syn["mean"], rank=2, k_action=1, seed=1, loso=False)
    path = op.save(tmp_path / "L01.attn_out__s0__hazard_corridor.npz")
    back = cms.MinDistortionOperator.load(path)
    H = np.random.default_rng(0).normal(size=(4, op.d))
    for mode in ("strengthen", "suppress"):
        assert np.allclose(back.apply_numpy(H, mode, 0.5), op.apply_numpy(H, mode, 0.5))
    assert np.array_equal(back.apply_numpy(H, "strengthen", 0.0), H)
    # the constraint weights are stored and the D_action bound holds for the exported unit edit
    v = back.meta["variants"]["main"]
    assert v["unit_edit"]["D_action"] <= back.meta["tau2"] * (1 + 1e-6) and v["unit_edit"]["mahal_increment"] <= back.meta["mahal_incr"] * (1 + 1e-6)


def test_modulation_self_test_and_steerer_interface(tmp_path):
    checks = mo.self_test(tmp_path, seed=0)
    assert checks["passed"] and checks["top_slice"] == "gate_mlp" and checks["sham_bit_identical"]
    # file round trip + numpy rule == hook rule on a random modulation vector
    fake = mo._Planted(seed=0)
    rng = np.random.default_rng(0)
    lat, route, hazard = mo._planted_scene(fake, rng)
    rec = mo.fit_scene(fake, fake.model.predictor, lat, route, hazard, 0, 1, [0, 1, 2], ("mlp",), fake.N)
    ops, stats = mo.assemble_operators({"s0": rec, "s1": rec}, [0, 1, 2], ("mlp",), (1, 2, 4), 1, n_boot=20, n_perm=16, seed=0)
    op = ops["L01.adaln__mlp__r2"]
    assert op.rank == 2 and op.a["U"].shape == (2, 6 * fake.d)
    p = op.save(tmp_path / "L01.adaln__mlp__r2.npz")
    back = mo.ModulationOperator.load(p)
    assert back.meta["slices"] == op.meta["slices"] and np.allclose(back.a["gamma"], op.a["gamma"])
    m = rec["mods"][(1, 1)][1][None, :]
    g = back.gate_numpy(rec["xin"][(1, 1)][1][None, :])
    edited = back.apply_numpy(m, g, 0.5, "strengthen")
    assert edited.shape == m.shape and not np.allclose(edited, m)
    assert np.array_equal(back.apply_numpy(m, g, 0.0, "strengthen"), m)
    # ungated twins
    assert np.allclose(back.apply_numpy(m, g, 0.5, "strengthen", "ungated_on"), m + 0.5 * (((m - back.a["m_ref"]) @ back.a["U"].T) * back.a["gamma"]) @ back.a["U"])


def test_torch_runner_operator_paths_without_a_model(tmp_path):
    """Exercise ``TorchRunner._transform_min_distortion`` / ``_modulation_steerer`` on a runner built without the model
    (``__new__`` + the attributes the paths use), on the fake predictors of the two self-tests: the closures must run, the
    sham must be bit-identical, the edit / receiver-Mahalanobis logs must fill, and the modulation steerer must apply."""
    import numpy as np

    # --- min-distortion path
    syn = cms.synthetic_rows(seed=2)
    op = cms.fit_operator(syn["rel"], syn["groups"], syn["groups"]["action"], syn["receiver"], syn["mean"], rank=2, k_action=1, seed=2, loso=False)
    r = sp.TorchRunner.__new__(sp.TorchRunner)
    r.torch, r.device = torch, torch.device("cpu")
    r.operators, r.mod_operators, r.conceptors = {("L03.mlp_out", 0, "hazard_corridor"): op}, {}, {}
    r._edit_log, r._extra_log, r._cur = {}, {}, {"n_steps": 3}
    cfg = sp.SteerConfig("md|b1", "md", "main", "min_distortion", ("L03.mlp_out",), ("L03.mlp_out",), 1.0, "strengthen", "hazard_corridor", False, operator="min_distortion")
    fn = r._transform(cfg, "L03.mlp_out", "L03.mlp_out", 0, np.arange(3))
    h = torch.randn(1, 3, op.d)
    out = fn(h)
    assert out.shape == h.shape and not torch.equal(out, h)
    assert r.edit_stats("p", cfg) > 0 and set(r.extra_stats("p", cfg)) == {"receiver_mahalanobis_before", "receiver_mahalanobis_after", "receiver_mahalanobis_increment"}
    sham = sp.SteerConfig("md|sham", "md", "sham", "min_distortion", ("L03.mlp_out",), ("L03.mlp_out",), 0.0, "strengthen", "hazard_corridor", False, operator="min_distortion")
    assert torch.equal(r._transform(sham, "L03.mlp_out", "L03.mlp_out", 0, np.arange(3))(h), h)
    for variant in ("euclidean_twin", "random_matched"):
        c = sp.SteerConfig(f"md|{variant}", "md", variant, "min_distortion", ("L03.mlp_out",), ("L03.mlp_out",), 0.5, "suppress", "hazard_corridor", False, variant=variant, operator="min_distortion")
        assert r._transform(c, "L03.mlp_out", "L03.mlp_out", 0, np.arange(3))(h).shape == h.shape
    # --- modulation path on the planted fake predictor
    fake = mo._Planted(seed=0)
    rng = np.random.default_rng(0)
    lat, route, hazard = mo._planted_scene(fake, rng)
    rec = mo.fit_scene(fake, fake.model.predictor, lat, route, hazard, 0, 1, [0, 1, 2], ("all",), fake.N)
    ops, _ = mo.assemble_operators({"s0": rec, "s1": rec}, [0, 1, 2], ("all",), (1,), 1, n_boot=20, n_perm=16, seed=0)
    r.predictor, r.n_spatial = fake.model.predictor, fake.N
    r.mod_operators = {("L01.adaln", "all", 1): ops["L01.adaln__all__r1"]}
    r.tokens_for = lambda group, st: hazard if group == "hazard" else np.array([0, 1])
    mcfg = sp.SteerConfig("mod|b1", "mod", "main", "modulation:all:r1", ("L01.adaln",), ("L01.adaln",), 1.0, "strengthen", "hazard", False, operator="modulation")
    st = r._steerer(mcfg)
    from modulation_patch import _predict
    with torch.no_grad():
        base = _predict(fake, lat[(1, 0)]["z_context"], lat[(0, 1)]["actions"], 0)
        with st:
            pred = _predict(fake, lat[(1, 0)]["z_context"], lat[(0, 1)]["actions"], 0)
    assert st.applied and st.expected == 1 and not torch.equal(pred, base) and r.edit_stats("p", mcfg) > 0
    wb = sp.SteerConfig("mod|wb", "mod", "wrong_block", "modulation:all:r1", ("L02.adaln",), ("L01.adaln",), 1.0, "strengthen", "hazard", False, operator="modulation")
    assert r._steerer(wb).edits.keys() == {2}
