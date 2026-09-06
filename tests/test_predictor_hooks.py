"""CPU tests for predictor hooks, token groups, scene-level statistics, interaction
effects and patch bookkeeping.

A tiny fake predictor reproduces the JEPA-WM AdaLN structure (predictor_embed,
predictor_blocks[i].{attn, mlp, adaLN_modulation}, predictor_norm/proj) and the
sliding-window ``unroll`` so the hook/patch mechanics are exercised without any
checkpoint download.  Token groups on the fake 2x2 grid are built by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from cgs_stats import (  # noqa: E402
    benjamini_yekutieli,
    cluster_bootstrap,
    cv_projection,
    loso_projection_scores,
    sign_flip_matrix,
    sign_flip_maxt,
)
from localize_interaction import null_factor_vector, quartet_effects  # noqa: E402
from patch_site import (  # noqa: E402
    PRIMARY_DIRECTION,
    PRIMARY_GROUP,
    aggregate,
    cyclic_assign,
    evaluate_pair,
    is_candidate,
    mahalanobis_score,
    mahalanobis_stats,
    patch_metrics,
)
from predictor_hooks import (  # noqa: E402
    HOOK_POINTS,
    PredictorPatcher,
    PredictorRecorder,
    Site,
    all_sites,
    norm_matched_random,
    site_order_key,
    time_shifted,
    unrelated_site,
    unroll_from_latents,
)
from protocol import CELL_ORDER, paired_did  # noqa: E402
from token_groups import GRID, IMG, CellGroups, groups_from_bbox, groups_from_masks, union  # noqa: E402


G, D, A, DEPTH = 2, 8, 3, 3
N = G * G


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.attn = nn.Linear(D, D)
        self.norm2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))

    def forward(self, x, z, T=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(z).repeat_interleave(N, dim=1).chunk(6, dim=2)
        )
        y = self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa)
        x = x + y * gate_msa
        x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        return x


class FakePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor_embed = nn.Linear(D, D)
        self.action_encoder = nn.Linear(A, D)
        self.predictor_blocks = nn.ModuleList([FakeBlock() for _ in range(DEPTH)])
        self.predictor_norm = nn.LayerNorm(D)
        self.predictor_proj = nn.Linear(D, D)

    def forward(self, x, actions, proprio=None):
        x = self.predictor_embed(x).flatten(2, 4)
        B, T, _, _ = x.shape
        z = self.action_encoder(actions)
        x = x.flatten(1, 2)
        for blk in self.predictor_blocks:
            x = blk(x, z, T=T)
        x = self.predictor_norm(x).view(B, T, N, D)
        return self.predictor_proj(x), None, None


class FakeVideoWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = FakePredictor()


class FakeWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeVideoWM()
        self.grid_size = G
        self.ctxt_window = 2

    def unroll(self, z_ctxt, act_suffix=None):
        T, B, _ = act_suffix.shape
        vid = z_ctxt.expand(B, *z_ctxt.shape[1:])
        acts = act_suffix.permute(1, 0, 2)
        for h in range(T):
            a = acts[:, : h + 1][:, -self.ctxt_window :]
            pred, _, _ = self.model.predictor(vid[:, -self.ctxt_window :], a)
            nxt = pred[:, -1:].view(B, 1, 1, G, G, D)
            vid = torch.cat([vid, nxt], dim=1)
        return vid.permute(1, 0, 2, 3, 4, 5)


@pytest.fixture()
def wm() -> FakeWM:
    torch.manual_seed(0)
    return FakeWM().eval()


def _latents(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "z_context": torch.randn(1, 1, 1, G, G, D, generator=g),
        "z_future": torch.randn(1, 3, 1, G, G, D, generator=g),
        "actions": torch.randn(3, 1, A, generator=g),
    }


def _fake_groups() -> CellGroups:
    frame = {
        "egg": np.array([0]), "gripper": np.array([1]), "corridor": np.array([2]), "background": np.array([3]),
        "gripper_corridor": np.array([1, 2]), "all": np.arange(N),
    }
    return CellGroups(source="masks", frames=[frame] * 4)


# ------------------------------------------------------------------ hooks ---


def test_site_ids_roundtrip_and_order() -> None:
    sites = all_sites(DEPTH)
    assert sites[0].site_id == "embed"
    assert len(sites) == 1 + DEPTH * len(HOOK_POINTS)
    for s in sites:
        assert Site.parse(s.site_id) == s
    with pytest.raises(ValueError):
        Site.parse("L3.resid_post")
    assert site_order_key(Site(0, "resid_post")) == site_order_key(Site(1, "resid_pre"))
    assert site_order_key(Site(0, "attn_out")) < site_order_key(Site(0, "mlp_out")) < site_order_key(Site(0, "resid_post"))
    assert site_order_key(Site(2, "resid_pre"), 0) < site_order_key(Site(0, "resid_pre"), 1)


def test_recorder_counts_steps_and_slices_last_frame(wm: FakeWM) -> None:
    lat = _latents(1)
    rec = PredictorRecorder(wm.model.predictor, all_sites(DEPTH), N)
    pred = unroll_from_latents(wm, lat["z_context"], lat["actions"], recorder=rec)
    assert pred.shape == (3, 1, 1, G, G, D)
    assert rec.n_steps == 3
    for sid, by_step in rec.acts.items():
        assert sorted(by_step) == [0, 1, 2]
        expect = (1, 1, 6 * D) if Site.parse(sid).hook == "adaln" else (1, N, D)
        for t in by_step.values():
            assert t.shape == expect


def test_patch_final_block_reproduces_donor_prediction_at_that_step(wm: FakeWM) -> None:
    base, donor = _latents(3), _latents(4)
    sid = f"L{DEPTH - 1:02d}.resid_post"
    rec_d = PredictorRecorder(wm.model.predictor, [Site.parse(sid)], N)
    pred_donor = unroll_from_latents(wm, donor["z_context"], donor["actions"], recorder=rec_d)
    patcher = PredictorPatcher(wm.model.predictor, {(sid, 0): rec_d.acts[sid][0]}, N)
    pred_patched = unroll_from_latents(wm, base["z_context"], base["actions"], patcher=patcher)
    assert patcher.applied == [(sid, 0)]
    torch.testing.assert_close(pred_patched[0], pred_donor[0])
    pred_base = unroll_from_latents(wm, base["z_context"], base["actions"])
    assert not torch.allclose(pred_patched[0], pred_base[0])


def test_token_subset_patch_changes_only_selected_tokens(wm: FakeWM) -> None:
    base, donor = _latents(3), _latents(4)
    sid = f"L{DEPTH - 1:02d}.resid_post"
    rec_d = PredictorRecorder(wm.model.predictor, [Site.parse(sid)], N)
    pred_donor = unroll_from_latents(wm, donor["z_context"], donor["actions"], recorder=rec_d)
    rec_b = PredictorRecorder(wm.model.predictor, [Site.parse(sid)], N)
    pred_base = unroll_from_latents(wm, base["z_context"], base["actions"], recorder=rec_b)
    idx = torch.tensor([1, 2])
    patcher = PredictorPatcher(wm.model.predictor, {(sid, 0): (idx, rec_d.acts[sid][0][:, idx])}, N)
    pred_patched = unroll_from_latents(wm, base["z_context"], base["actions"], patcher=patcher)
    flat_p, flat_d, flat_b = (p[0].reshape(-1, D) for p in (pred_patched, pred_donor, pred_base))
    # final norm/proj are per token, so patched tokens equal donor and others equal base
    torch.testing.assert_close(flat_p[idx], flat_d[idx])
    torch.testing.assert_close(flat_p[[0, 3]], flat_b[[0, 3]])
    with pytest.raises(ValueError):
        unroll_from_latents(wm, base["z_context"], base["actions"], patcher=PredictorPatcher(wm.model.predictor, {(sid, 0): (torch.tensor([7]), torch.zeros(1, 1, D))}, N))


def test_sham_patch_is_identity(wm: FakeWM) -> None:
    base = _latents(5)
    sid = "L01.mlp_out"
    rec = PredictorRecorder(wm.model.predictor, [Site.parse(sid)], N)
    pred_base = unroll_from_latents(wm, base["z_context"], base["actions"], recorder=rec)
    patcher = PredictorPatcher(wm.model.predictor, {(sid, 1): rec.acts[sid][1].clone()}, N)
    torch.testing.assert_close(unroll_from_latents(wm, base["z_context"], base["actions"], patcher=patcher), pred_base)


def test_control_helpers() -> None:
    g = torch.Generator().manual_seed(0)
    base, donor = torch.randn(2, N, D, generator=g), torch.randn(2, N, D, generator=g)
    rand = norm_matched_random(base, donor, g)
    for b in range(2):
        assert torch.isclose((rand[b] - base[b]).norm(), (donor[b] - base[b]).norm(), rtol=1e-5)
    acts = {0: torch.zeros(1, N, D), 1: torch.ones(1, N, D), 2: torch.full((1, N, D), 2.0)}
    assert sorted(time_shifted(acts, 1)) == [0, 2]
    assert unrelated_site(Site(1, "attn_out"), DEPTH, 6).layer == (1 + 6) % DEPTH
    src = torch.arange(3.0).view(1, 3, 1).expand(1, 3, D)
    assert cyclic_assign(src, 5)[0, :, 0].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


# ----------------------------------------------------------- token groups ---


def test_groups_from_masks_partition_and_geometry() -> None:
    egg = np.zeros((4, IMG, IMG), dtype=bool)
    robot = np.zeros((4, IMG, IMG), dtype=bool)
    egg[:, 100:110, 120:127] = True  # rows 100-110 -> patch row 6, cols 120-127 -> patch col 7
    robot[:, 20:60, 20:60] = True  # patches rows 1-3, cols 1-3
    eef = np.tile(np.array([[40.0, 40.0]]), (4, 1))  # (x, y) -> patch (2, 2)
    eggpx = np.tile(np.array([[125.0, 105.0]]), (4, 1))
    groups = groups_from_masks({"egg_mask": egg, "robot_mask": robot, "eef_px": eef, "egg_px": eggpx})
    f = groups.frames[0]
    assert groups.source == "masks"
    assert 6 * GRID + 7 in f["egg"] and len(f["egg"]) == 9  # dilated 3x3
    assert set(f["gripper"]) == {r * GRID + c for r in (1, 2, 3) for c in (1, 2, 3)}
    assert len(f["corridor"]) > 0 and not (set(f["corridor"]) & set(f["egg"])) and not (set(f["corridor"]) & set(f["gripper"]))
    covered = union(f["egg"], f["gripper"], f["corridor"], f["background"])
    assert len(covered) == GRID * GRID
    assert set(f["gripper_corridor"]) == set(f["gripper"]) | set(f["corridor"])


def test_groups_from_bbox_fallback() -> None:
    groups = groups_from_bbox([112, 107, 118, 115])
    assert groups.source == "bbox_fallback"
    f = groups.frames[0]
    assert 6 * GRID + 7 in f["egg"] and len(f["gripper"]) == 0 and len(f["gripper_corridor"]) == 0
    assert len(f["background"]) == GRID * GRID - len(f["egg"])


# --------------------------------------------------------------- statistics ---


def test_quartet_effects_match_protocol_did_and_pool_tokens() -> None:
    g = torch.Generator().manual_seed(0)
    cells = {key: torch.randn(N, D, generator=g) for key in CELL_ORDER}
    eff = quartet_effects(cells)
    did = paired_did({k: v.numpy() for k, v in cells.items()})
    assert eff["interaction_rms"] == pytest.approx(float(np.sqrt(np.mean(did**2))), rel=1e-5)
    np.testing.assert_allclose(eff["pooled"]["interaction"], did.mean(axis=0), rtol=1e-5)
    h, a = torch.randn(N, D, generator=g), torch.randn(N, D, generator=g)
    additive = {(H, A_): H * h + A_ * a for H, A_ in CELL_ORDER}
    add = quartet_effects(additive)
    assert add["interaction_rms"] < 1e-5
    assert add["hazard_main_rms"] == pytest.approx(float(h.pow(2).mean().sqrt()), rel=1e-5)
    null_cells = {**cells, (2, 0): cells[(0, 0)] + 1.0, (2, 1): cells[(0, 1)] + 1.0}
    np.testing.assert_allclose(null_factor_vector(null_cells), np.zeros(D), atol=1e-6)


def test_loso_projection_has_zero_mean_under_null_and_detects_signal() -> None:
    rng = np.random.default_rng(0)
    null = rng.normal(size=(2000, 6))
    scores = loso_projection_scores(null)
    assert abs(scores.mean()) < 0.1  # unbiased, unlike RMS
    assert np.sqrt(np.mean(null.mean(axis=0) ** 2)) > 0.0
    # explicit check against the definition
    for i in range(3):
        others = np.delete(null, i, axis=0).mean(axis=0)
        assert scores[i] == pytest.approx(null[i] @ others / np.linalg.norm(others), rel=1e-9)
    signal = null[:40] + np.array([3.0, 0, 0, 0, 0, 0])
    proj = cv_projection(signal)
    assert proj["t"] > 4 and proj["n_scenes"] == 40
    assert np.isnan(loso_projection_scores(null[:1])).all()


def test_sign_flip_maxt_is_exact_for_small_n_and_controls_fwer() -> None:
    rng = np.random.default_rng(0)
    assert sign_flip_matrix(3, 100, rng).shape == (8, 3)
    vectors = {f"s{k}": rng.normal(size=(6, 5)) for k in range(10)}
    vectors["sig"] = rng.normal(size=(6, 5)) + np.array([4.0, 0, 0, 0, 0])
    res = sign_flip_maxt(vectors, 500, rng)
    assert res["exact"] and res["n_perm"] == 64 and res["n_sites"] == 11 and res["n_scenes"] == 6
    assert res["sites"]["sig"]["p_raw"] < 0.05 and res["sites"]["sig"]["p_maxt_fwer"] < 0.1
    assert all(res["sites"][s]["p_maxt_fwer"] >= res["sites"][s]["p_raw"] - 1e-12 for s in vectors)
    assert min(res["sites"][s]["p_raw"] for s in vectors) >= 1 / 65
    assert set(res["maxt_null_quantiles"]) == {0.5, 0.9, 0.95, 0.99}


def test_benjamini_yekutieli_and_bootstrap() -> None:
    fdr = benjamini_yekutieli({"a": 0.001, "b": 0.5, "c": 0.02, "d": float("nan")})
    assert fdr["n_tests"] == 3 and "a" in fdr["rejected"] and "b" not in fdr["rejected"]
    assert fdr["q"]["a"] <= fdr["q"]["c"] <= fdr["q"]["b"]
    rng = np.random.default_rng(0)
    out = cluster_bootstrap(rng.normal(1.0, 0.1, size=30), 500, rng)
    assert out["ci_low"] < out["point"] < out["ci_high"] and out["n_clusters"] == 30
    assert cluster_bootstrap(np.asarray([0.3]), 500, rng)["ci_low"] is None


# ------------------------------------------------------------- patching ---


def test_patch_metrics_endpoints() -> None:
    g = torch.Generator().manual_seed(0)
    p_base, p_donor = torch.randn(N, D, generator=g), torch.randn(N, D, generator=g)
    t_base, t_donor, t_unsafe, z = (torch.randn(N, D, generator=g) for _ in range(4))
    region = np.array([0, 1])
    full = patch_metrics(p_donor, p_base, p_donor, t_base, t_donor, t_unsafe, z, region)
    none = patch_metrics(p_base, p_base, p_donor, t_base, t_donor, t_unsafe, z, region)
    assert full["recovery_region"] == pytest.approx(1.0) and full["recovery_all"] == pytest.approx(1.0)
    assert full["corruption"] == pytest.approx(1.0)
    assert none["recovery_region"] == pytest.approx(0.0) and none["cem_l2_cost_delta"] == pytest.approx(0.0)
    assert none["unsafe_future_delta"] == pytest.approx(0.0)
    mu, var = mahalanobis_stats(torch.randn(500, D, generator=g))
    assert 0.7 < mahalanobis_score(torch.randn(50, D, generator=g), mu, var) < 1.3
    assert mahalanobis_score(torch.full((5, D), 20.0), mu, var) > 10


def test_is_candidate_rules() -> None:
    assert is_candidate(Site(3, "mlp_out"), "gripper_corridor", DEPTH) == (True, "candidate")
    assert is_candidate(Site(3, "mlp_out"), "all", DEPTH)[0] is True
    assert is_candidate(Site(1, "resid_post"), "all", DEPTH) == (False, "reference_whole_residual")
    assert is_candidate(Site(DEPTH - 1, "resid_post"), "egg", DEPTH) == (False, "reference_last_layer_resid")
    assert is_candidate(Site(1, "adaln"), "gripper_corridor", DEPTH) == (False, "action_availability_control")


def _pair_latents(base_seed: int, with_null: bool) -> dict:
    keys = list(CELL_ORDER) + ([(2, 0), (2, 1)] if with_null else [])
    return {key: _latents(base_seed + i) for i, key in enumerate(keys)}


def test_evaluate_pair_controls_and_persistence(wm: FakeWM) -> None:
    latents = _pair_latents(10, with_null=True)
    groups = {k: _fake_groups() for k in latents}
    sites = [Site(DEPTH - 1, "resid_post"), Site(0, "mlp_out")]
    g = torch.Generator().manual_seed(0)
    res = evaluate_pair(wm, wm.model.predictor, N, DEPTH, latents, groups, sites, [2], ["gripper_corridor", "all"], 2, 1, g,
                        ["hazard_safe_to_unsafe", "hazard_unsafe_to_safe", "action_gentle_to_aggr"], int_pairs=[(f"L{DEPTH - 1:02d}.resid_post", "L00.mlp_out")])
    last = res[f"L{DEPTH - 1:02d}.resid_post"]["2"]
    full = last["all"]["hazard_safe_to_unsafe"]["fixed_0"]
    assert full["donor"]["recovery_all"] == pytest.approx(1.0, abs=1e-5)
    assert full["sham"]["recovery_region"] == pytest.approx(0.0, abs=1e-5)
    assert len(full["random"]) == 2 and len(full["time_shift"]) == 2
    assert full["unrelated_site"]["site_id"] == "L00.resid_post"
    assert "main_effect_only" in full and "null_factor" in full
    assert "mahalanobis_written" in full["donor"] and "mahalanobis_receiver" in full["donor"]
    # token-group patch at the final block/step: region tokens (egg+gripper = 0,1) partly patched (gripper_corridor = 1,2)
    grp = last["gripper_corridor"]["hazard_safe_to_unsafe"]["fixed_1"]
    assert grp["n_tokens_written"] == 2 and grp["region_size"] == 2
    assert 0.0 < grp["donor"]["recovery_region"] < 1.0 + 1e-6
    assert "wrong_group" in grp and grp["wrong_group"]["edit_rms"] > 0
    # action direction present, hazard-only controls absent there
    act = last["all"]["action_gentle_to_aggr"]["fixed_0"]
    assert "main_effect_only" not in act and "null_factor" not in act
    assert len(res["_int"]) == 1 and {"recovery_x", "recovery_y", "recovery_xy", "int"} <= set(res["_int"][0])
    # persistent mode patches every step
    pers = evaluate_pair(wm, wm.model.predictor, N, DEPTH, latents, groups, [Site(DEPTH - 1, "resid_post")], None, ["all"], 0, 1, g,
                         ["hazard_safe_to_unsafe"], mode="persistent")
    assert pers[f"L{DEPTH - 1:02d}.resid_post"]["persistent"]["all"]["hazard_safe_to_unsafe"]["fixed_0"]["donor"]["recovery_all"] == pytest.approx(1.0, abs=1e-5)


def test_aggregate_liveness_band_and_confirm(wm: FakeWM) -> None:
    sites = [Site(0, "mlp_out"), Site(1, "mlp_out"), Site(2, "mlp_out")]
    g = torch.Generator().manual_seed(0)
    per_pair = {}
    for i in range(3):
        latents = _pair_latents(100 + 10 * i, with_null=True)
        groups = {k: _fake_groups() for k in latents}
        per_pair[f"p{i}"] = evaluate_pair(wm, wm.model.predictor, N, DEPTH, latents, groups, sites, [0], [PRIMARY_GROUP], 1, 1, g, list(PRIMARY_DIRECTION and ["hazard_safe_to_unsafe", "hazard_unsafe_to_safe"]))
    # force a clean, significant band: overwrite recoveries so L01 and L02 are live and above threshold
    for j, (pid, r) in enumerate(per_pair.items()):
        for sid in ("L01.mlp_out", "L02.mlp_out"):
            for direction in ("hazard_safe_to_unsafe", "hazard_unsafe_to_safe"):
                cell = r[sid]["0"][PRIMARY_GROUP][direction]
                cell["fixed_1"]["donor"]["recovery_region"] = 0.8 + 0.01 * j  # consistent, non-degenerate DiD
                cell["fixed_0"]["donor"]["recovery_region"] = 0.4
                for lvl in ("fixed_0", "fixed_1"):
                    if direction == "hazard_safe_to_unsafe":
                        assert "null_factor" in cell[lvl]  # receiver is H0 -> H0' donor exists
                        cell[lvl]["null_factor"]["recovery_region"] = 0.0
                    else:
                        assert "null_factor" not in cell[lvl]  # receiver is H1: no null donor
    agg = aggregate(per_pair, sites, DEPTH, 50, 64, np.random.default_rng(0), 0.3, 0.1, 0.5, allow_missing_null=False)
    by = {r["site_id"]: r for r in agg["site_results"]}
    assert by["L01.mlp_out"]["directions"][PRIMARY_DIRECTION]["donor"]["patch_effect_did"]["point"] == pytest.approx(0.41)
    assert by["L01.mlp_out"]["null_factor_status"] == "pass"
    assert by["L01.mlp_out"]["interaction_live"] and by["L02.mlp_out"]["interaction_live"]
    assert agg["selection"]["selected"]["site_id"] == "L01.mlp_out"
    assert [b["site_id"] for b in agg["selection"]["band"]] == ["L01.mlp_out", "L02.mlp_out"]
    assert agg["selection"]["band"][0]["per_scene_stability"]["fraction_scenes_positive_did"] == 1.0
    conf = aggregate(per_pair, sites, DEPTH, 50, 64, np.random.default_rng(0), 0.3, 0.1, 0.5, allow_missing_null=False, confirm=True)
    assert conf["selection"]["selected"] is None and conf["selection"]["band"] == []
    # without null cells liveness needs the explicit allowance
    for r in per_pair.values():
        for sid in sites:
            for direction in list(r[sid.site_id]["0"][PRIMARY_GROUP]):
                for lvl in ("fixed_0", "fixed_1"):
                    r[sid.site_id]["0"][PRIMARY_GROUP][direction][lvl].pop("null_factor", None)
    strict = aggregate(per_pair, sites, DEPTH, 50, 64, np.random.default_rng(0), 0.3, 0.1, 0.5, allow_missing_null=False)
    assert not any(r["interaction_live"] for r in strict["site_results"])
    assert all(r["null_factor_status"] == "missing" for r in strict["site_results"])
    lenient = aggregate(per_pair, sites, DEPTH, 50, 64, np.random.default_rng(0), 0.3, 0.1, 0.5, allow_missing_null=True)
    assert lenient["selection"]["selected"]["site_id"] == "L01.mlp_out"


# ------------------------------------------------ localization revisions ---


def test_cv_projection_sd_floor_and_dz() -> None:
    const = np.tile(np.array([[1.0, 0.0, 0.0]]), (5, 1))  # identical across scenes (AdaLN-like)
    out = cv_projection(const)
    assert out["t"] is None or np.isnan(out["t"])
    assert np.isnan(out["d_z"])
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(12, 4)) + np.array([2.0, 0, 0, 0])
    out = cv_projection(sig)
    assert out["d_z"] == pytest.approx(out["t"] / np.sqrt(12), rel=1e-9)


def test_null_factor_status_and_magnitude_fields() -> None:
    from localize_interaction import magnitude_fields, null_factor_pairs, null_factor_status

    pairs = {"a": {k: None for k in CELL_ORDER}, "b": {k: None for k in CELL_ORDER + ((2, 0), (2, 1))}}
    assert null_factor_status(pairs) == "partial" and null_factor_pairs(pairs) == ["b"]
    assert null_factor_status({"a": pairs["a"]}) == "none" and null_factor_status({"b": pairs["b"]}) == "full"

    def entry(site, step, group, inter, act, dz=1.0):
        return {"site_id": site, "step": step, "group": group,
                "cv_projection": {"interaction": {"mean": inter, "d_z": dz}, "action": {"mean": act}}}

    entries = [
        entry("L05.attn_out", 1, "gripper_corridor", 0.5, 2.0),
        entry("L05.attn_out", 1, "background", 0.1, 2.0),
        entry("L07.mlp_in", 1, "gripper_corridor", 0.026, 5.0),  # tiny magnitude, consistent direction
        entry("L07.mlp_in", 1, "background", 0.02, 5.0),
        entry("L02.mlp_out", 0, "egg", 0.3, 1.0),  # no background entry -> specificity None
    ]
    magnitude_fields(entries, 0.05, 2.0)
    e0, e1, e2, e3, e4 = entries
    assert e0["ratio_to_action"] == pytest.approx(0.25) and e0["background_specificity"] == pytest.approx(5.0)
    assert e0["magnitude_ok"] and e0["autoregressive"] and e0["d_z"] == 1.0
    assert e1["background_specificity"] == 1.0 and not e1["magnitude_ok"]
    assert e2["ratio_to_action"] == pytest.approx(0.0052) and e2["background_specificity"] == pytest.approx(1.3)
    assert not e2["magnitude_ok"]
    assert e4["background_specificity"] is None and not e4["magnitude_ok"] and not e4["autoregressive"]


def test_partial_null_factor_and_focus_tables(wm: FakeWM) -> None:
    from patch_site import focus_tables

    sites = [Site(1, "mlp_out")]
    g = torch.Generator().manual_seed(0)
    per_pair = {}
    for i in range(4):
        latents = _pair_latents(200 + 10 * i, with_null=(i < 2))  # only 2 of 4 scenes carry h2 cells
        groups = {k: _fake_groups() for k in latents}
        res = evaluate_pair(wm, wm.model.predictor, N, DEPTH, latents, groups, sites, [0], [PRIMARY_GROUP], 1, 1, g,
                            ["hazard_safe_to_unsafe", "hazard_unsafe_to_safe"], int_pairs=[("L01.mlp_out", "L02.attn_out")])
        assert len(res["_int"]) == 1 and res["_int"][0]["y"] == "L02.attn_out"  # partner recorded on demand
        per_pair[f"p{i}"] = {k: v for k, v in res.items() if k != "_int"}
    for j, r in enumerate(per_pair.values()):
        for direction in ("hazard_safe_to_unsafe", "hazard_unsafe_to_safe"):
            cell = r["L01.mlp_out"]["0"][PRIMARY_GROUP][direction]
            cell["fixed_1"]["donor"]["recovery_region"] = 0.8 + 0.01 * j
            cell["fixed_0"]["donor"]["recovery_region"] = 0.4
            for lvl in ("fixed_0", "fixed_1"):
                if "null_factor" in cell[lvl]:
                    cell[lvl]["null_factor"]["recovery_region"] = 0.02
    agg = aggregate(per_pair, sites, DEPTH, 20, 64, np.random.default_rng(0), 0.3, 0.1, 0.5, allow_missing_null=False)
    e = agg["site_results"][0]
    null = e["directions"][PRIMARY_DIRECTION]["null_factor"]
    assert null["n_null_scenes"] == 2 and null["n_scenes"] == 2 and e["n_null_scenes"] == 2
    assert null["per_scene"]["p2"] is None and null["per_scene"]["p0"] == pytest.approx(0.02)
    assert null["donor_minus_null_test"]["exact"] and null["donor_minus_null_test"]["n_scenes"] == 2
    assert e["null_factor_status"] == "pass" and e["interaction_live"]  # subset null control governs liveness
    tables = focus_tables(per_pair, ["L01.mlp_out"])
    rows = tables["L01.mlp_out"]["0"][PRIMARY_GROUP][PRIMARY_DIRECTION]
    assert [r["pair_id"] for r in rows] == ["p0", "p1", "p2", "p3"]
    assert rows[0]["patch_effect_did"] == pytest.approx(0.4) and "fixed_0_null_factor" in rows[0] and "fixed_0_null_factor" not in rows[2]
    assert "fixed_1_sham" in rows[0] and "fixed_1_cem_l2_cost_delta" in rows[0]


def test_dump_keys_include_null_factor_cells_by_default() -> None:
    from localize_interaction import NULL_CELLS, dump_keys

    keys = list(CELL_ORDER) + list(NULL_CELLS)
    assert dump_keys(keys, [0, 1, 2]) == keys  # default: h2 rows are dumped (hazard=2 in index)
    assert dump_keys(keys, [0, 1]) == list(CELL_ORDER)
    assert dump_keys(list(CELL_ORDER), [0, 1, 2]) == list(CELL_ORDER)  # scenes without h2 are unaffected


# ---------------------------------------------------------------- conceptor ---


def test_conceptor_operators() -> None:
    from conceptor_patch import apply_operator, operator, rank_one_edit

    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 5, D, generator=g)
    mean = torch.randn(D, generator=g)
    eye = torch.eye(D)
    # strengthen with C = I is the identity for every beta; any C is identity at beta -> 0
    for beta in (0.0, 0.25, 1.0):
        torch.testing.assert_close(apply_operator(h, operator(eye, beta, "strengthen"), mean), h)
    C = torch.randn(D, D, generator=g)
    C = C @ C.T / D
    torch.testing.assert_close(apply_operator(h, operator(C, 0.0, "strengthen"), mean), h)
    torch.testing.assert_close(apply_operator(h, operator(C, 0.0, "suppress"), mean), h)
    # suppress with a rank-1 projector at beta = 1 removes exactly that direction (about the mean)
    u = torch.randn(D, generator=g)
    u = u / u.norm()
    P = torch.outer(u, u)
    out = apply_operator(h, operator(P, 1.0, "suppress"), mean)
    assert torch.allclose((out - mean) @ u, torch.zeros(2, 5), atol=1e-5)
    perp = (h - mean) - torch.outer(((h - mean) @ u).flatten(), u).view(2, 5, D)
    torch.testing.assert_close(out - mean, perp)
    # strengthen with a projector at beta = 1 keeps only that direction
    keep = apply_operator(h, operator(P, 1.0, "strengthen"), mean)
    torch.testing.assert_close(keep - mean, torch.outer(((h - mean) @ u).flatten(), u).view(2, 5, D))
    # rank-one additive edit has the requested per-batch norm and direction
    edited = rank_one_edit(h, u, torch.tensor([0.3, 0.7]), 1.0)
    delta = edited - h
    assert torch.allclose(delta.flatten(1).norm(dim=1), torch.tensor([0.3, 0.7]), atol=1e-5)
    assert torch.allclose(delta[0, 0] / delta[0, 0].norm(), u, atol=1e-5)


def test_conceptor_selectivity_bookkeeping() -> None:
    from conceptor_patch import aggregate, scene_effects

    def cells(t_h1, t_h0, t_h2=None):
        c = {"h0a0": {"main": {"recovery_unsafe": t_h0, "corruption": 1.0}}, "h0a1": {"main": {"recovery_unsafe": t_h0, "corruption": 1.0}},
             "h1a0": {"main": {"recovery_unsafe": t_h1, "corruption": 1.0}}, "h1a1": {"main": {"recovery_unsafe": t_h1, "corruption": 1.0}}}
        if t_h2 is not None:
            c["h2a0"] = {"main": {"recovery_unsafe": t_h2, "corruption": 1.0}}
            c["h2a1"] = {"main": {"recovery_unsafe": t_h2, "corruption": 1.0}}
        return c

    eff = scene_effects(cells(0.5, 0.1, 0.0), "main")
    assert eff["target_h1"] == 0.5 and eff["selectivity_h0"] == pytest.approx(0.4) and eff["ratio_h0"] == pytest.approx(0.2)
    assert eff["selectivity_h2"] == pytest.approx(0.5) and np.isnan(scene_effects(cells(0.5, 0.1), "main")["nontarget_h2"])
    meta = {"config": "L07.attn_out", "sites": ["L07.attn_out"], "band": False, "step": "0", "group": "gripper_corridor", "mode": "strengthen", "beta": 0.5}
    per_scene = {f"p{i}": {"L07.attn_out|s0|gripper_corridor|strengthen|b0.5": {"cells": cells(0.5 + 0.02 * i, 0.01 * (i % 2), 0.0), "meta": meta}} for i in range(6)}
    agg = aggregate(per_scene, 100, 64, np.random.default_rng(0), margin=0.05, alpha=0.05)
    e = agg["results"]["L07.attn_out|s0|gripper_corridor|strengthen|b0.5"]
    main = e["kinds"]["main"]
    assert main["target_h1"]["point"] == pytest.approx(0.55) and main["equivalent_nontarget_h0"] and main["equivalent_nontarget_h2"]
    # exact enumeration over 6 scenes: the identity flip always ties the observed statistic -> p = 2/65
    assert main["selectivity_h0_test"]["exact"] and main["selectivity_h0_test"]["p_raw"] == pytest.approx(2 / 65)
    assert e["selective"] and agg["band_vs_single"] == []
    # a non-target effect outside the margin fails equivalence and selectivity
    per_scene_bad = {pid: {cid: {"cells": cells(0.5 + 0.02 * i, 0.3, 0.0), "meta": meta}} for i, (pid, d) in enumerate(per_scene.items()) for cid in d}
    bad = aggregate(per_scene_bad, 100, 64, np.random.default_rng(0), margin=0.05, alpha=0.05)["results"]
    assert not next(iter(bad.values()))["kinds"]["main"]["equivalent_nontarget_h0"] and not next(iter(bad.values()))["selective"]


def test_conceptor_evaluate_scene_on_fake_predictor(wm: FakeWM, tmp_path: Path) -> None:
    from conceptor_patch import conceptor_path, evaluate_scene, load_conceptor

    latents = _pair_latents(300, with_null=True)
    groups = {k: _fake_groups() for k in latents}
    sites = [Site(1, "attn_out"), Site(2, "attn_out")]
    g = torch.Generator().manual_seed(0)
    u = torch.randn(D, generator=g)
    u = u / u.norm()
    for s in sites:
        np.savez(conceptor_path(tmp_path, s.site_id, 0, PRIMARY_GROUP), C_safety=np.eye(D), C_int=torch.outer(u, u).numpy(), C_action=np.eye(D),
                 C_random_matched=np.eye(D) * 0.5, mean=np.zeros(D), alpha=10.0, quota=0.5)
    conceptors = {(s.site_id, 0, PRIMARY_GROUP): load_conceptor(conceptor_path(tmp_path, s.site_id, 0, PRIMARY_GROUP), torch.device("cpu")) for s in sites}
    out = evaluate_scene(wm, wm.model.predictor, N, DEPTH, latents, groups, conceptors, sites, sites, [0, 1], [PRIMARY_GROUP], [0.5], "strengthen", "C_safety",
                         persistent=False, wrong_site_offset=1, frame_offset=0)
    cid = "L01.attn_out|s0|gripper_corridor|strengthen|b0.5"
    assert cid in out and "band:L01.attn_out+L02.attn_out|s0|gripper_corridor|strengthen|b0.5" in out
    cells = out[cid]["cells"]
    assert set(cells) == {"h0a0", "h0a1", "h1a0", "h1a1", "h2a0", "h2a1"}
    m = cells["h1a1"]["main"]
    # C_safety = I -> strengthen is the identity: no effect, on-manifold diagnostics equal
    assert m["edit_rms"] == pytest.approx(0.0, abs=1e-6) and m["activation_edit_rms"] == pytest.approx(0.0, abs=1e-6)
    assert m["mahalanobis_written"] == pytest.approx(m["mahalanobis_receiver"]) and m["nn_distance_written"] == pytest.approx(m["nn_distance_receiver"])
    assert {"random_matched", "rank_one", "wrong_site", "wrong_step", "wrong_group"} <= set(cells["h1a1"])
    assert cells["h1a1"]["random_matched"]["edit_rms"] > 0 and cells["h1a1"]["wrong_site"]["site_id"] == "L02.attn_out"
    assert cells["h1a1"]["wrong_step"]["step"] == 1 and cells["h1a1"]["wrong_group"]["group"] == "egg"
    assert out[cid]["meta"]["quota"] == 0.5 and out[cid]["meta"]["n_tokens"] == 2
    # suppress with the rank-1 C_int changes the forecast; persistent mode labels the step
    sup = evaluate_scene(wm, wm.model.predictor, N, DEPTH, latents, groups, conceptors, sites[:1], None, [0, 1], [PRIMARY_GROUP], [1.0], "suppress", "C_int",
                         persistent=True, wrong_site_offset=1, frame_offset=0)
    key = "L01.attn_out|spersistent|gripper_corridor|suppress|b1.0"
    assert key in sup and sup[key]["cells"]["h1a1"]["main"]["edit_rms"] > 0 and "wrong_step" not in sup[key]["cells"]["h1a1"]


# ------------------------------------------------------- action-Jacobian sonar ---


class _SonarBase(nn.Module):
    """Shared unroll for the analytic sonar fakes (same interface as FakeWM)."""

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.model = FakeVideoWM()
        self.model.predictor = predictor
        self.grid_size = G
        self.ctxt_window = 2

    def unroll(self, z_ctxt, act_suffix=None):
        T, B, _ = act_suffix.shape
        vid = z_ctxt.expand(B, *z_ctxt.shape[1:])
        acts = act_suffix.permute(1, 0, 2)
        for h in range(T):
            a = acts[:, : h + 1][:, -self.ctxt_window :]
            pred, _, _ = self.model.predictor(vid[:, -self.ctxt_window :], a)
            vid = torch.cat([vid, pred[:, -1:].view(B, 1, 1, G, G, D)], dim=1)
        return vid.permute(1, 0, 2, 3, 4, 5)


class LinearBlock(nn.Module):
    """x <- x + attn(x) + mlp(x) + U a  (action injected additively, per layer)."""

    def __init__(self, g: torch.Generator) -> None:
        super().__init__()
        self.attn = nn.Linear(D, D, bias=False)
        self.mlp = nn.Linear(D, D, bias=False)
        self.adaLN_modulation = nn.Linear(A, D, bias=False)
        for m in (self.attn, self.mlp):
            m.weight.data = 0.2 * torch.randn(D, D, generator=g)
        self.adaLN_modulation.weight.data = torch.randn(D, A, generator=g)

    def forward(self, x, z):
        return x + self.attn(x) + self.mlp(x) + self.adaLN_modulation(z)[:, -1:, :].expand(-1, x.shape[1], -1)


class LinearPredictor(nn.Module):
    def __init__(self, depth: int, g: torch.Generator) -> None:
        super().__init__()
        self.predictor_embed = nn.Identity()
        self.predictor_blocks = nn.ModuleList([LinearBlock(g) for _ in range(depth)])

    def forward(self, x, actions, proprio=None):
        B, T = x.shape[:2]
        x = x.flatten(2, 4)[:, -1]  # last frame tokens only [B, N, D]
        for blk in self.predictor_blocks:
            x = blk(x, actions)
        return x.view(B, 1, N, D), None, None


class BilinearPredictor(nn.Module):
    """out_t = z_t + (z_t . w) (a . k) u : a planted hazard x action term."""

    def __init__(self, g: torch.Generator) -> None:
        super().__init__()
        self.predictor_embed = nn.Identity()
        self.w = nn.Parameter(torch.randn(D, generator=g))
        self.k = nn.Parameter(torch.randn(A, generator=g))
        self.u = nn.Parameter(torch.randn(D, generator=g))
        self.predictor_blocks = nn.ModuleList([])

    def forward(self, x, actions, proprio=None):
        B = x.shape[0]
        z = x.flatten(2, 4)[:, -1]
        s = (actions[:, -1, :] @ self.k).view(B, 1, 1)
        out = z + (z @ self.w).unsqueeze(-1) * s * self.u.view(1, 1, D)
        return out.view(B, 1, N, D), None, None


def _sonar_latents(g: torch.Generator, w: torch.Tensor | None = None) -> dict:
    z0 = torch.randn(1, 1, 1, G, G, D, generator=g)
    delta = torch.randn(1, 1, 1, G, G, D, generator=g)
    null = torch.randn(1, 1, 1, G, G, D, generator=g)
    if w is not None:  # make the null shift orthogonal to w so the planted term does not see it
        flat = null.view(-1, D)
        flat -= torch.outer(flat @ w, w) / (w @ w)
    acts = {0: torch.randn(3, 1, A, generator=g), 1: torch.randn(3, 1, A, generator=g)}
    fut = torch.randn(1, 3, 1, G, G, D, generator=g)
    lat = {}
    for hz, z in ((0, z0), (1, z0 + delta), (2, z0 + null)):
        for a in (0, 1):
            lat[(hz, a)] = {"z_context": z.clone(), "z_future": fut.clone(), "actions": acts[a].clone()}
    return lat


def test_sonar_bilinear_fake_recovers_planted_interaction() -> None:
    from action_jacobian_sonar import evaluate_scene

    g = torch.Generator().manual_seed(0)
    wm = _SonarBase(BilinearPredictor(g)).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    pred = wm.model.predictor
    lat = _sonar_latents(g, w=pred.w.detach())
    groups = {k: _fake_groups() for k in lat}
    res = evaluate_scene(wm, pred, N, lat, groups, [], ["all"], [0], [], torch.Generator().manual_seed(1), hvp_eps=0.1)
    step = res["steps"]["0"]
    assert step["region_size"] == 3
    rel = step["relational"]["a0"]
    # analytic: dJ_t = ((z1_t - z0_t) . w) u k^T  ->  ||dJ||_F over the region = sqrt(sum_t (delta_t.w)^2) ||u|| ||k||
    delta = (lat[(1, 0)]["z_context"] - lat[(0, 0)]["z_context"]).view(-1, D)[:3]
    expected = float(torch.sqrt(((delta @ pred.w) ** 2).sum()) * pred.u.norm() * pred.k.norm())
    assert rel["dJ_hazard_frobenius_region"] == pytest.approx(expected, rel=1e-4)
    assert rel["dJ_null_frobenius_region"] == pytest.approx(0.0, abs=1e-4)
    assert rel["relational_frobenius_region"] == pytest.approx(expected, rel=1e-4)
    # the planted term is bilinear, so the small-step mixed derivative equals the unit-step difference
    assert rel["mixed_derivative"]["relative_error_hvp_vs_dJ_hazard"] < 1e-4
    assert rel["mixed_derivative"]["cosine_hvp_vs_dJ_hazard"] == pytest.approx(1.0, abs=1e-5)
    # spectrum of a rank-one Jacobian
    assert step["cells"]["h0a0"]["region"]["numerical_rank"] == 1
    assert step["cells"]["h0a0"]["region"]["participation_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert "rand_a0" in step["cells"] and step["cells"]["rand_a0"]["delta_norm"] == pytest.approx(float(delta.new_tensor((lat[(1, 0)]["z_context"] - lat[(0, 0)]["z_context"]).norm())))


def test_sonar_additive_fake_has_no_hazard_dependence_and_mediation_is_a_cut() -> None:
    from action_jacobian_sonar import downstream_response, evaluate_scene, jacobian_forward_ad

    g = torch.Generator().manual_seed(1)
    wm = _SonarBase(LinearPredictor(3, g)).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    pred = wm.model.predictor
    lat = _sonar_latents(g)
    groups = {k: _fake_groups() for k in lat}
    sites = [Site(l, "resid_post") for l in range(3)] + [Site(1, "mlp_out")]
    res = evaluate_scene(wm, pred, N, lat, groups, sites, ["all", "gripper_corridor"], [0], [0], torch.Generator().manual_seed(2))
    step = res["steps"]["0"]
    rel = step["relational"]["a1"]
    assert rel["J_h0_frobenius_region"] > 0.1
    assert rel["dJ_hazard_frobenius_region"] < 1e-5 and rel["relational_frobenius_region"] < 1e-5  # J does not depend on z
    # mediation through the whole-frame residual cut at the last layer carries the entire Jacobian
    last = step["mediation"]["L02.resid_post|all"]["a1"]
    assert last["fraction_of_total_J"] == pytest.approx(1.0, abs=1e-4) and last["cosine_M_vs_J_h0"] == pytest.approx(1.0, abs=1e-5)
    assert "L01.mlp_out|gripper_corridor" in step["mediation"]
    # cut at layer L equals the total Jacobian of the model with injections after L removed (additive decomposition)
    z, a = lat[(0, 1)]["z_context"], lat[(0, 1)]["actions"]
    J, tang = jacobian_forward_ad(wm, pred, z, a, 0, N, sites)
    for L in range(3):
        M = torch.stack([downstream_response(wm, pred, z, a, 0, N, f"L{L:02d}.resid_post", tang[f"L{L:02d}.resid_post"][i]) for i in range(A)], dim=-1)
        saved = [blk.adaLN_modulation.weight.data.clone() for blk in pred.predictor_blocks]
        for blk in pred.predictor_blocks[L + 1 :]:
            blk.adaLN_modulation.weight.data.zero_()
        J_trunc, _ = jacobian_forward_ad(wm, pred, z, a, 0, N, [])
        for blk, wgt in zip(pred.predictor_blocks, saved):
            blk.adaLN_modulation.weight.data = wgt
        torch.testing.assert_close(M, J_trunc, atol=1e-5, rtol=1e-5)
    # increments over layers sum to the total
    Ms = [torch.stack([downstream_response(wm, pred, z, a, 0, N, f"L{L:02d}.resid_post", tang[f"L{L:02d}.resid_post"][i]) for i in range(A)], dim=-1) for L in range(3)]
    total = Ms[0] + sum(Ms[L] - Ms[L - 1] for L in range(1, 3))
    torch.testing.assert_close(total, J, atol=1e-5, rtol=1e-5)


def test_sonar_pooled_statistics_and_site_order() -> None:
    from action_jacobian_sonar import evaluate_scene, pooled_statistics

    g = torch.Generator().manual_seed(3)
    wm = _SonarBase(BilinearPredictor(g)).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    pred = wm.model.predictor
    per_scene = {}
    for i in range(4):
        lat = _sonar_latents(torch.Generator().manual_seed(10 + i), w=pred.w.detach())
        groups = {k: _fake_groups() for k in lat}
        per_scene[f"p{i}"] = evaluate_scene(wm, pred, N, lat, groups, [], ["all"], [0], [], torch.Generator().manual_seed(i))
    pooled = pooled_statistics(per_scene, [0], [], ["all"], 50, 64, np.random.default_rng(0), 0.05)
    s0 = pooled["0"]
    assert s0["n_scenes"] == 4 and not s0["autoregressive"]
    assert s0["a0"]["relational_minus_random_test"]["exact"] and s0["a0"]["relational_frobenius_region"]["n_clusters"] == 4
    assert "loso_dJ_hazard" in s0["a0"] and s0["a0"]["loso_dJ_hazard"]["n_scenes"] == 4
    assert s0["a0"]["mixed_derivative_relative_error"]["point"] < 1e-4
    assert "mediation" not in s0
    assert site_order_key(Site(6, "attn_out")) < site_order_key(Site(6, "mlp_out"))


# ---------------------------------------------- sonar refinements + E2 ---


def test_sonar_refinements_on_bilinear_fake() -> None:
    from action_jacobian_sonar import evaluate_scene, principal_angles, projected_statistic

    g = torch.Generator().manual_seed(5)
    wm = _SonarBase(BilinearPredictor(g)).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    pred = wm.model.predictor
    lat = _sonar_latents(g, w=pred.w.detach())
    groups = {k: _fake_groups() for k in lat}
    res = evaluate_scene(wm, pred, N, lat, groups, [], ["all"], [0], [], torch.Generator().manual_seed(1), fd_eps=(1e-2, 1e-1), value_path=True)
    rel = res["steps"]["0"]["relational"]["a1"]
    # rank-one Jacobians: col J = span of c_z[t, d] = (z_t . w) u_d, so the single principal angle is the angle
    # between c_{H1} and c_{H0}, i.e. between the token weightings (z_t . w) on the route tokens
    route = [1, 2]  # gripper + corridor tokens of the fake groups (the egg token 0 is excluded from the route statistics)
    c0 = (lat[(0, 1)]["z_context"].view(-1, D)[route] @ pred.w).double()
    c1 = (lat[(1, 1)]["z_context"].view(-1, D)[route] @ pred.w).double()
    expected_deg = float(torch.rad2deg(torch.arccos((c0 @ c1 / (c0.norm() * c1.norm())).abs())))
    pa = rel["principal_angles_route"]["hazard"]
    assert pa["rank_1"] == 1 and pa["rank_0"] == 1 and len(pa["angles_deg"]) == 1
    assert pa["top_angle_deg"] == pytest.approx(expected_deg, abs=1e-3)
    assert rel["S_F_route"] > 0 and "S_F_route_sign" in rel and "dJ_sign_frobenius_region" in rel
    # bilinear model: central differences agree with forward AD (fp32 rounding bounds the agreement at ~1e-3)
    assert set(rel["central_difference"]) == {"eps_0.01", "eps_0.1"}
    assert all(v["relative_error"] < 2e-3 and v["cosine"] > 0.999 for v in rel["central_difference"].values())
    # no attention in the fake -> the value path IS the full Jacobian
    assert rel["pattern_value_split_route"]["pattern_fraction"] == pytest.approx(0.0, abs=1e-6)
    # total-level FD confirmation: the planted term is bilinear, so the FD equals the Jacobian prediction
    assert rel["fd_confirmation_route"]["cosine_fd_vs_jacobian"] == pytest.approx(1.0, abs=1e-5)
    assert set(rel["projected_I_true"]) == {"relational", "hazard", "random", "sign"}
    # helpers
    J1 = torch.randn(6, 8, 3, generator=g)
    assert principal_angles(J1, J1)["top_angle_deg"] == pytest.approx(0.0, abs=1e-4)
    Q = torch.linalg.qr(torch.randn(48, 6, generator=g))[0]
    assert principal_angles(Q[:, :3].reshape(6, 8, 3), Q[:, 3:].reshape(6, 8, 3))["top_angle_deg"] == pytest.approx(90.0, abs=1e-3)
    ps = projected_statistic(torch.ones(4, D, 2), torch.tensor([1.0, 0.0]), torch.ones(4, D), np.array([0, 1]))
    assert ps["fraction_of_I_true"] == pytest.approx(1.0) and ps["cosine"] == pytest.approx(1.0)


def test_sonar_site_fd_confirmation_on_linear_fake() -> None:
    from action_jacobian_sonar import evaluate_scene

    g = torch.Generator().manual_seed(6)
    wm = _SonarBase(LinearPredictor(2, g)).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    lat = _sonar_latents(g)
    groups = {k: _fake_groups() for k in lat}
    res = evaluate_scene(wm, wm.model.predictor, N, lat, groups, [Site(1, "resid_post")], ["all"], [0], [0], torch.Generator().manual_seed(2), value_path=False)
    med = res["steps"]["0"]["mediation"]["L01.resid_post|all"]["a1"]
    assert {"S_F_site", "top_angle_deg", "top_angle_sign_deg", "dM_sign_frobenius_region", "fd_confirmation"} <= set(med)
    # linear fake: no hazard dependence anywhere, so both FD and Jacobian relational terms vanish
    assert med["relational_frobenius_region"] < 1e-5 and med["fd_confirmation"]["fd_norm"] < 1e-4


def _one_block_action_fake(active: int, depth: int = 3) -> FakeWM:
    torch.manual_seed(0)
    wm = FakeWM().eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    for b, blk in enumerate(wm.model.predictor.predictor_blocks):
        if b != active:
            blk.adaLN_modulation[1].weight.data.zero_()
            blk.adaLN_modulation[1].bias.data.zero_()
    return wm


def test_modulation_patch_single_block_equals_full_swap_and_sham_is_zero() -> None:
    from modulation_patch import aggregate, capture_modulations, evaluate_scene, predict_with_swaps

    wm = _one_block_action_fake(active=1)
    pred = wm.model.predictor
    lat = _pair_latents(400, with_null=True)
    groups = {k: _fake_groups() for k in lat}
    res = evaluate_scene(wm, pred, lat, groups, 0, ("attn", "mlp", "all"), 2, torch.Generator().manual_seed(0))
    assert res["full_swap_error"] < 1e-5  # swapping every block == running with a1
    b1, b0 = res["blocks"]["b01|all"], res["blocks"]["b00|all"]
    for h in ("h0", "h1", "h2"):
        assert b1["route_fraction_of_full"][h] == pytest.approx(1.0, abs=1e-5) and b1["route_cosine_with_full"][h] == pytest.approx(1.0, abs=1e-5)
        assert b0["route_fraction_of_full"][h] == pytest.approx(0.0, abs=1e-6)
    assert b1["sham_norm"] == pytest.approx(0.0, abs=1e-6) and b0["sham_norm"] == pytest.approx(0.0, abs=1e-6)
    # the single-block interaction equals the full-swap interaction (ceiling) exactly
    assert b1["interaction_route"]["recovery"] == pytest.approx(res["ceiling"]["route"]["recovery"], abs=1e-5)
    assert b1["interaction_route_fraction_of_full_interaction"] == pytest.approx(1.0, abs=1e-5)
    assert "null_interaction_route" in b1 and b1["direct_sensitivity"]["h1"]["n_probes"] == 2 and b1["direct_sensitivity"]["h1"]["dim"] == 6 * D
    assert b1["direct_sensitivity"]["h1"]["frobenius_estimate_route"] > 0
    # capture/swap round-trip: injecting a block's own a0 modulation reproduces the a0 prediction exactly
    P0, mod0 = capture_modulations(wm, pred, lat[(0, 0)]["z_context"], lat[(0, 0)]["actions"], 0)
    torch.testing.assert_close(predict_with_swaps(wm, pred, lat[(0, 0)]["z_context"], lat[(0, 0)]["actions"], 0, {1: ("all", mod0[1])}), P0)
    # aggregation bookkeeping over 3 fake scenes
    per_scene = {f"p{i}": evaluate_scene(wm, pred, _pair_latents(500 + 10 * i, with_null=True), groups, 0, ("all",), 1, torch.Generator().manual_seed(i)) for i in range(3)}
    agg = aggregate(per_scene, ("all",), 20, 64, np.random.default_rng(0), 0.5)
    assert set(agg["table"]) == {"b00|all", "b01|all", "b02|all"}
    assert agg["table"]["b01|all"]["recovery"]["point"] == pytest.approx(agg["ceiling"]["route_recovery"]["point"], abs=1e-5)
    assert agg["table"]["b00|all"]["recovery"]["point"] == pytest.approx(0.0, abs=1e-6)
    assert "recovery_minus_null_test" in agg["table"]["b01|all"] and "distributed_entry" in agg["verdict"]
