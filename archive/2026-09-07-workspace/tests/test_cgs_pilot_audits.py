from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import audit_contamination as ac  # noqa: E402
import counterfactual_validity_gate as cv  # noqa: E402
import droid_reference as dr  # noqa: E402
import latent_cache as lc  # noqa: E402
import merge_heldout as mh  # noqa: E402
import retrieval_baseline as rb  # noqa: E402
import stats_utils as su  # noqa: E402
from protocol import CELL_ORDER, PROTOCOL_VERSION, PROTOCOL_VERSION_V03  # noqa: E402


# --------------------------------------------------------------------------- #
# merge_heldout
# --------------------------------------------------------------------------- #


def _write_seed(root: Path, seed: int, pixel_error: int = 0, force_did: float = 20.0, positioning: bool = True, sextet: bool = False) -> None:
    sdir = root / f"seed_{seed}"
    (sdir / "cells").mkdir(parents=True)
    rows = []
    cells = list(CELL_ORDER) + ([(2, 0), (2, 1)] if sextet else [])
    for h, a in cells:
        cell_id = f"contact_seed{seed:06d}__h{h}a{a}"
        np.savez(sdir / "cells" / f"{cell_id}.npz", x=np.zeros(1))
        err = pixel_error if h == 1 else 0
        rows.append(
            {
                "pair_id": f"contact_seed{seed:06d}", "cell_id": cell_id, "seed": seed, "hazard": h, "candidate_action": a,
                "artifact": f"cells/{cell_id}.npz", "deterministic_replay": err == 0, "replay_frames_bit_exact": err == 0,
                "replay_max_pixel_error": err, "replay_max_state_error": 0.0, "replay_initial_state_error": 0.0,
                "replay_force_error_n": 0.0, "replay_contact_count_equal": True,
            }
        )
    (sdir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "pairs": [{
            "pair_id": f"contact_seed{seed:06d}", "positioning_gate": positioning, "state_contrast_gate": True,
            "contact_pattern_gate": True, "force_interaction_gate": force_did >= 5.0, "contact_did": 1.0,
            "force_did_n": force_did, "control_pose_selection": {"hazard_to_control_centroid_px": 20.0},
        }],
        "phase0_stimulus_gate": {}, "provenance": {},
    }
    (sdir / "summary.json").write_text(json.dumps(summary))


def _jitter_json(path: Path, cell_ids: list[str], differing: int, verdict: str = "render_only") -> None:
    recs = [{"cell_id": c, "verdict": verdict, "replay_vs_replay_frames": [{"differing_pixels": differing, "max_abs_diff": 1}], "saved_vs_replay_frames": [{"differing_pixels": differing, "max_abs_diff": 1}]} for c in cell_ids]
    path.write_text(json.dumps(recs))


def test_merge_v02_strict_and_v03_with_jitter_verification(tmp_path: Path) -> None:
    root = tmp_path / "heldout"
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_seed(root, 201)  # clean
    _write_seed(root, 202, pixel_error=1)  # 1-LSB jitter, verified 1 px
    _write_seed(root, 203, pixel_error=1)  # 1-LSB jitter, unverified
    _write_seed(root, 204, pixel_error=1)  # 1-LSB jitter, verified 3 px -> fail
    _write_seed(root, 205, force_did=2.0)
    _write_seed(root, 206, sextet=True)
    (logs / "heldout_seed_207.log").write_text('RuntimeError("positioning gate failed: hazard target is not visible")\n')
    (logs / "heldout_seed_208.log").write_text("RuntimeError: positioning gate failed for seed 208: gripper-to-approach error 1.3 m\n")
    _jitter_json(tmp_path / "jit_a.json", [f"contact_seed000202__h{h}a{a}" for h, a in CELL_ORDER], 1)
    _jitter_json(tmp_path / "jit_b.json", [f"contact_seed000204__h{h}a{a}" for h, a in CELL_ORDER], 3)

    strict = mh.merge(root, tmp_path / "strict", logs, (201, 209), PROTOCOL_VERSION, [], None, 0, None, (101, 102))
    assert strict["protocol_version"] == PROTOCOL_VERSION
    assert strict["admitted_seeds"] == [201, 206]
    t = strict["tally"]
    assert t["attempted"] == 8 and t["generator_aborted"] == 2
    assert t["exclusion_reason_counts"]["replay:frames_not_bit_exact"] == 3
    assert t["exclusion_reason_counts"]["force_interaction_at_least_5n"] == 1
    assert t["exclusion_reason_counts"]["generator:hazard_not_visible"] == 1
    assert t["physics_exact_frames_not_bit_exact"] == 3
    assert t["admitted_with_null_factor_sextet"] == [206]
    assert strict["per_seed"]["209"]["status"] == "not_run_or_no_log"
    rows = [json.loads(l) for l in (tmp_path / "strict" / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 4 + 6
    assert all(((tmp_path / "strict") / r["artifact"]).exists() for r in rows)

    v03 = mh.merge(root, tmp_path / "v03", logs, (201, 209), PROTOCOL_VERSION_V03, [str(tmp_path / "jit_*.json")], None, 0, None, (101, 102))
    assert v03["protocol_version"] == PROTOCOL_VERSION_V03
    assert v03["admitted_seeds"] == [201, 202, 206]
    r = v03["tally"]["exclusion_reason_counts"]
    assert r["replay:differing_pixel_count_unverified"] == 1  # 203
    assert r["replay:more_than_one_differing_pixel"] == 1  # 204
    assert v03["phase0_stimulus_gate"]["frames_bit_exact_all"] is False
    assert v03["tally"]["jitter_verified_cells"] == 8


def test_merge_label_validation_excludes_failed_seeds(tmp_path: Path) -> None:
    root = tmp_path / "h"
    _write_seed(root, 301)
    _write_seed(root, 302)
    _write_seed(root, 303)
    val = tmp_path / "labels.json"
    val.write_text(json.dumps({"seeds": [{"seed": 301, "passed": True, "failures": []}, {"seed": 302, "passed": False, "failures": ["h1:hazard_not_under_path(0.05m)"]}]}))
    res = mh.merge(root, tmp_path / "o", None, None, PROTOCOL_VERSION, [], val, 0, None, (101, 102))
    assert res["admitted_seeds"] == [301]
    assert res["per_seed"]["302"]["exclusion_reasons"] == ["labels:h1:hazard_not_under_path(0.05m)"]
    assert res["per_seed"]["303"]["exclusion_reasons"] == ["labels:not_validated"]
    # default location <root>/_validation/hazard_label_validation.json is picked up
    (root / "_validation").mkdir()
    (root / "_validation" / "hazard_label_validation.json").write_text(val.read_text())
    res2 = mh.merge(root, tmp_path / "o2", None, None, PROTOCOL_VERSION, [], None, 0, None, (101, 102))
    assert res2["admitted_seeds"] == [301] and res2["tally"]["label_validation"] == "provided"


def test_split_is_deterministic_stable_and_disjoint() -> None:
    seeds = list(range(201, 221))
    d1, c1 = mh.split_seeds(seeds, 7, None)
    assert mh.split_seeds(seeds, 7, None) == (d1, c1)
    assert not set(d1) & set(c1) and sorted(d1 + c1) == seeds
    assert mh.split_seeds(seeds, 8, None)[0] != d1
    # Stability under growth: adding seeds never moves an existing seed between arms.
    d2, c2 = mh.split_seeds(seeds + list(range(221, 261)), 7, None)
    assert set(d1) <= set(d2) and set(c1) <= set(c2)
    # Roughly balanced.
    assert 15 <= len(d2) <= 45


def test_calibration_seeds_are_excluded(tmp_path: Path) -> None:
    root = tmp_path / "h"
    _write_seed(root, 101)
    res = mh.merge(root, tmp_path / "o", None, None, PROTOCOL_VERSION, [], None, 0, None, (101, 102))
    assert res["admitted_seeds"] == [] and res["per_seed"]["101"]["status"] == "excluded_calibration"


# --------------------------------------------------------------------------- #
# droid_reference (protobuf / tfrecord parsing)
# --------------------------------------------------------------------------- #


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _example(features: dict[str, tuple[str, list]]) -> bytes:
    entries = b""
    for key, (kind, vals) in features.items():
        if kind == "bytes":
            lst = b"".join(_ld(1, v) for v in vals)
            feat = _ld(1, lst)
        elif kind == "float":
            import struct

            packed = b"".join(struct.pack("<f", v) for v in vals)
            feat = _ld(2, _ld(1, packed))
        else:
            raise ValueError(kind)
        entries += _ld(1, _ld(1, key.encode()) + _ld(2, feat))
    return _ld(1, entries)


def test_parse_example_roundtrip_and_delta_actions(tmp_path: Path) -> None:
    import struct

    acts = [float(i) for i in range(14)]
    ex = _example({"steps/action": ("float", acts), "steps/observation/exterior_image_2_left": ("bytes", [b"jpg0", b"jpg1"]), "other": ("float", [1.0])})
    parsed = dr.parse_example(ex, {"steps/action", "steps/observation/exterior_image_2_left"})
    assert parsed["steps/action"] == ("float", acts)
    assert parsed["steps/observation/exterior_image_2_left"] == ("bytes", [b"jpg0", b"jpg1"])
    assert "other" not in parsed
    rec = tmp_path / "x.tfrecord"
    with rec.open("wb") as fh:
        for payload in (ex, ex):
            fh.write(struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0")
    assert [len(r) for r in dr.iter_tfrecords(rec)] == [len(ex), len(ex)]
    raw = np.zeros((9, 7))
    raw[:, 3] = np.linspace(3.0, -3.0, 9)  # wraps across +-pi
    raw[:, 0] = np.arange(9) * 0.01
    d = dr.jepa_wm_delta_actions(raw, stride=4)
    assert d.shape == (2, 7)
    assert np.allclose(d[:, 0], 0.04)
    assert np.all(np.abs(d[:, 3]) <= np.pi)


# --------------------------------------------------------------------------- #
# audit_contamination
# --------------------------------------------------------------------------- #


def _synthetic_rows(n_scenes: int):
    rows, scene_ids, hazard, level = [], [], [], []
    for s in range(n_scenes):
        for h, a in CELL_ORDER:
            rows.append({"pair_id": f"p{s}", "hazard": h, "candidate_action": a})
            scene_ids.append(f"p{s}")
            hazard.append(h)
            level.append(a)
    return rows, scene_ids, np.array(hazard), np.array(level)


def test_action_audit_flags_identical_actions_and_hazard_leak() -> None:
    _, scene_ids, hazard, level = _synthetic_rows(10)
    actions = np.zeros((len(scene_ids), 3, 7), np.float32)
    actions[:, :, 0] = level[:, None] * 0.5
    res = ac.action_audit(actions, hazard, level, scene_ids)
    assert res["actions_identical_across_hazard"] and res["all_scenes_share_identical_actions"]
    assert res["unseen_action_control_present"] is False
    assert res["loso_nn_accuracy_action_level_from_actions"] == 1.0
    assert res["hazard_from_actions_at_chance"]
    leak = actions.copy()
    leak[:, 0, 1] = hazard * 10.0
    res2 = ac.action_audit(leak, hazard, level, scene_ids)
    assert res2["actions_identical_across_hazard"] is False and res2["loso_nn_accuracy_hazard_from_actions"] == 1.0
    varied = actions + np.random.RandomState(0).randn(*actions.shape).astype(np.float32) * 0.01
    assert ac.action_audit(varied, hazard, level, scene_ids)["unseen_action_control_present"] is True


def test_domain_shift_ratio_in_and_out_of_distribution() -> None:
    rng = np.random.RandomState(0)
    ref = rng.randn(200, 8)
    groups = np.repeat(np.arange(20), 10)
    inside = ref[:10] + 0.01 * rng.randn(10, 8)
    far = ref[:10] + 50.0
    r_in = ac.domain_shift_ratio(inside, ref, groups)
    r_out = ac.domain_shift_ratio(far, ref, groups)
    assert r_in["ratio_median"] < 0.2 and r_out["ratio_median"] > 10
    assert r_out["fraction_stimuli_beyond_reference_p95"] == 1.0


def test_dtw_and_scaling() -> None:
    a = np.zeros((3, 7))
    assert ac.dtw_distance(a, a) == 0.0
    b = a.copy()
    b[1, 0] = 1.0
    assert ac.dtw_distance(a, b) == pytest.approx(1.0)
    s = ac.scale_actions(np.array([[0.1] * 6 + [0.75]]))
    assert np.allclose(s, 1.0)


def test_action_novelty_runs_on_synthetic_reference() -> None:
    rng = np.random.RandomState(0)
    ref = rng.randn(120, 3, 7) * 0.02
    groups = np.repeat(np.arange(12), 10)
    stim = np.repeat(ref[:1], 8, axis=0)
    res = ac.action_novelty(stim, ref, groups, max_ref=120, seed=0)
    assert res["dtw"]["distinct_stimulus_chunks"] == 1
    assert res["euclidean"]["ratio_median"] == pytest.approx(0.0, abs=1e-6)


def test_duplicate_audit_flags_copied_scene() -> None:
    rng = np.random.RandomState(1)
    frames = rng.randint(0, 255, size=(12, 64, 64, 3)).astype(np.uint8)
    frames[8:12] = frames[0:4]
    scene_ids = [f"p{i // 4}" for i in range(12)]
    res = ac.duplicate_audit(ac.pixel_signature(frames, size=16), scene_ids, None, 0.05, 2.0)
    assert res["n_near_duplicate_context_cells"] == 8


# --------------------------------------------------------------------------- #
# stats + latent cache helpers
# --------------------------------------------------------------------------- #


def test_sign_flip_p_exact_and_monte_carlo() -> None:
    assert su.sign_flip_p(np.array([1.0, 1.0])) == 0.5
    assert su.sign_flip_p(np.ones(10)) == pytest.approx(2 / 1024)
    assert su.sign_flip_p(np.ones(12)) == pytest.approx(2 / 4096)
    assert su.sign_flip_p(np.array([1.0, -1.0, 0.5, -0.5])) == 1.0
    p = su.sign_flip_p(np.ones(20) + 0.01 * np.arange(20), n_perm=2000)
    assert p < 0.01


def test_pixel_mask_to_tokens() -> None:
    m = np.zeros((2, 256, 256), bool)
    m[0, 17, 40] = True  # patch row 1, col 2
    tok = lc.pixel_mask_to_tokens(m)
    assert tok.shape == (2, 16, 16) and tok[0].sum() == 1 and tok[0, 1, 2] and tok[1].sum() == 0


# --------------------------------------------------------------------------- #
# retrieval_baseline + counterfactual gate on synthetic latents
# --------------------------------------------------------------------------- #


def _synthetic_cache(n_scenes: int, good_model: bool = True, seed: int = 0, grid: int = 4, d: int = 6):
    rng = np.random.RandomState(seed)
    rows, ctx, tgt, pred, zero, act, mask = [], [], [], [], [], [], []
    e0 = np.zeros((grid, grid, d))
    e0[..., 0] = 1.0
    e1 = np.zeros((grid, grid, d))
    e1[1:3, 1:3, 1] = 1.0  # interaction lives in a few tokens
    for s in range(n_scenes):
        base = rng.randn(grid, grid, d)
        ctx_by_h = {h: base + 0.02 * h * rng.randn(grid, grid, d) for h in (0, 1)}  # same frame for A0/A1
        for h, a in CELL_ORDER:
            rows.append({"pair_id": f"p{s}", "cell_id": f"p{s}__h{h}a{a}", "seed": s, "hazard": h, "candidate_action": a})
            c = ctx_by_h[h]
            true = c + e0 * (0.5 + a) + e1 * (h * a) * 0.4 + 0.01 * rng.randn(grid, grid, d)
            if good_model:
                p = true + 0.02 * rng.randn(grid, grid, d)
            else:
                p = c + e0 * (0.5 + a) + 0.02 * rng.randn(grid, grid, d)  # ignores hazard entirely
            ctx.append(c)
            tgt.append(true[None])
            pred.append(p[None])
            zero.append((c + 0.001 * rng.randn(grid, grid, d))[None])
            aa = np.zeros((3, 7), np.float32)
            aa[:, 0] = a
            act.append(aa)
            m = np.zeros((2, grid, grid), bool)
            m[:, 0:3, 0:3] = True
            mask.append(m)
    cache = {
        "cell_id": np.array([r["cell_id"] for r in rows]),
        "context": np.stack(ctx).astype(np.float16), "target": np.stack(tgt).astype(np.float16),
        "prediction": np.stack(pred).astype(np.float16), "zero_prediction": np.stack(zero).astype(np.float16),
        "actions": np.stack(act), "token_mask": np.stack(mask), "has_mask": np.ones(len(rows), bool), "meta": {},
    }
    return rows, cache


def test_retrieval_interaction_gate_passes_for_good_model_and_fails_for_hazard_blind() -> None:
    rows, cache = _synthetic_cache(18, good_model=True)
    res = rb.analyze(cache, rows, action_weight=1.0, k=3, lam=1e-2)
    assert res["n_scenes"] == 18
    assert res["model"]["mean_cosine"]["mean"] > 0.8
    for name in ("persistence", "loso_ridge", "copy_delta_scene"):
        assert res["baselines"][name]["sign_flip_p"] < 0.05, name
    assert res["baselines"]["knn_droid"]["status"] == "deferred"
    assert res["interaction_gate"]["passed"] is True and res["interaction_gate"]["deferred_baselines"] == ["knn_droid"]
    rows2, cache2 = _synthetic_cache(18, good_model=False)
    res2 = rb.analyze(cache2, rows2, action_weight=1.0, k=3, lam=1e-2)
    assert res2["interaction_gate"]["passed"] is False


def test_knn_droid_baseline_uses_droid_cache() -> None:
    rows, cache = _synthetic_cache(6)
    cache["droid_pooled_t"] = np.random.RandomState(1).randn(20, 6).astype(np.float32)
    cache["droid_delta"] = np.zeros((20, 4, 4, 6), np.float16)
    cache["droid_actions"] = np.zeros((20, 3, 7), np.float32)
    res = rb.analyze(cache, rows, 1.0, 3, 1e-2)
    assert "baseline_mean_cosine" in res["baselines"]["knn_droid"]
    assert res["baselines"]["knn_droid"]["baseline_mean_cosine"] == 0.0  # zero deltas -> zero interaction


def test_recovered_fraction_math() -> None:
    u = np.zeros(4)
    p = np.array([2.0, 0, 0, 0])
    assert cv.recovered_fraction(p, u, p) == 1.0
    assert cv.recovered_fraction(u, u, p) == 0.0
    assert cv.recovered_fraction(np.array([1.0, 5.0, 0, 0]), u, p) == 0.5


def test_counterfactual_gate_verdicts() -> None:
    rows, cache = _synthetic_cache(18, good_model=True)
    res = cv.evaluate(cache, rows)
    assert res["verdict"] == "PASS"
    assert res["pooled"]["rec_h1"]["median"] > 0.8
    assert res["pooled"]["interaction_cosine"]["sign_flip_p_vs_zero"] < 0.05
    assert all(v["n_tokens_used"] == 9 for v in res["per_scene"].values())
    rows2, cache2 = _synthetic_cache(18, good_model=False)
    res2 = cv.evaluate(cache2, rows2)
    assert res2["verdict"] == "FAIL" and "NOT a result about" in res2["statement"]
    rows3, cache3 = _synthetic_cache(3, good_model=True)
    assert cv.evaluate(cache3, rows3)["verdict"] == "INSUFFICIENT_SCENES"


# --------------------------------------------------------------------------- #
# sonar (model-native contrast subspace) metrics
# --------------------------------------------------------------------------- #

import sonar_metrics as sm  # noqa: E402


def _planted_scenes(n_scenes: int, n_tok: int = 32, d: int = 128, noise: float = 1.2, seed: int = 0):
    """True interaction = shared 2-d contrast (+ small scene-specific part) + big isotropic noise.

    Prediction carries the shared contrast but a different noise draw, so the raw
    cosine is low while the projected cosine is high.
    """

    rng = np.random.RandomState(seed)
    basis = np.linalg.qr(rng.randn(d, 3))[0]  # cols 0-1: interaction subspace, col 2: action direction
    scenes = {}
    for s in range(n_scenes):
        w = rng.randn(2) * np.array([1.0, 0.4]) + np.array([3.0, 0.0])  # shared mean direction along col 0
        shared = np.outer(np.ones(n_tok), basis[:, :2] @ w)
        i_true = shared + noise * rng.randn(n_tok, d)
        i_pred = shared + noise * rng.randn(n_tok, d)
        a_true = np.outer(np.ones(n_tok), basis[:, 2] * 5.0) + 0.5 * rng.randn(n_tok, d)
        ctx = rng.randn(n_tok, d)
        u = ctx + 0.1 * rng.randn(n_tok, d)
        p = u + shared + noise * rng.randn(n_tok, d)
        y = u + shared + noise * rng.randn(n_tok, d)
        q = {(0, 0): ctx, (0, 1): ctx + a_true, (1, 0): u, (1, 1): p}
        scenes[f"p{s}"] = {
            "tokens": np.arange(n_tok), "i_true": i_true, "i_pred": i_pred, "a_true": a_true,
            "rec": {"h1": (y, u, p), "h0": (ctx + a_true + noise * rng.randn(n_tok, d), ctx, ctx + a_true)},
            "quartet_true": q,
        }
    return scenes


def test_sonar_recovers_planted_low_dim_contrast() -> None:
    scenes = _planted_scenes(16)
    res = sm.loso_sonar(scenes, ks=(1, 2, 4), seed=0)
    assert res["status"] == "ok"
    for feature in ("pooled", "per_token"):
        f = res["pooled"][feature]
        raw = f["raw_cosine"]["mean"]
        proj = f["k2"]["interaction"]["cosine"]["mean"]
        rand = f["k2"]["random"]["cosine"]["mean"]
        if feature == "per_token":
            assert raw < 0.35 and abs(rand) < 0.25, (raw, rand)
            assert proj > raw + 0.3, (raw, proj)
        assert proj > (0.6 if feature == "per_token" else 0.7) and proj >= raw - 0.05, (feature, raw, proj)
        assert rand < raw + 0.15, (feature, raw, rand)  # a random subspace never inflates the cosine
        assert proj > rand + 0.25, (feature, proj, rand)
        assert f["k2"]["interaction"]["capture"]["mean"] > 2 * f["k2"]["random"]["capture"]["mean"]
        assert f["k2"]["random"]["capture"]["mean"] < 0.1
        assert f["k2"]["interaction"]["cosine"]["sign_flip_p"] < 0.01
        assert f["k2"]["interaction_minus_random_cosine"]["sign_flip_p"] < 0.01
        # interaction subspace is (nearly) orthogonal to the planted action direction
        assert f["k2"]["principal_angles_interaction_vs_action_deg"]["min_mean"] > 45
        assert f["k2"]["action_main_effect"]["cosine"]["mean"] < proj - 0.3
        assert f["k2"]["interaction"]["rec"]["h1"]["median"] > 0.6


def test_sonar_helpers() -> None:
    rng = np.random.RandomState(0)
    X = rng.randn(10, 6)
    Q = sm.svd_subspace(X, 3)
    assert Q.shape == (6, 3) and np.allclose(Q.T @ Q, np.eye(3))
    assert sm.svd_subspace(X, 50).shape[1] == 6
    v = rng.randn(4, 6)
    pv = sm.project(v, Q)
    assert np.allclose(sm.project(pv, Q), pv)  # idempotent
    assert np.allclose(sm.principal_angles_deg(Q, Q), 0.0, atol=1e-3)
    R = sm.random_subspace(6, 3, rng)
    assert np.allclose(R.T @ R, np.eye(3))
    assert all(0 <= a <= 90 + 1e-9 for a in sm.principal_angles_deg(Q, R))
    assert sm.recovered_fraction(np.array([1.0, 0.0]), np.zeros(2), np.array([2.0, 0.0])) == 0.5


def test_gate_and_retrieval_include_sonar_blocks() -> None:
    rows, cache = _synthetic_cache(18, good_model=True)
    res = cv.evaluate(cache, rows, stimulus_dir=None)
    assert "sonar" in res and "all" in res["sonar"]["groups"]
    assert res["sonar"]["groups"]["all"]["status"] == "ok"
    assert isinstance(res["interpretation"], str) and "leave-one-scene-out" in res["interpretation"]
    rb_res = rb.analyze(cache, rows, 1.0, 3, 1e-2)
    assert "k4" in rb_res["sonar"]["pooled"] and "model" in rb_res["sonar"]["pooled"]["k4"]
    assert rb_res["sonar"]["per_token"]["k4"]["persistence"]["model_minus_baseline"]["mean"] > 0


# --------------------------------------------------------------------------- #
# planner currency
# --------------------------------------------------------------------------- #

import planner_currency as pc  # noqa: E402


def test_spearman_and_ci() -> None:
    x = np.arange(10.0)
    assert pc.spearman(x, x) == 1.0 and pc.spearman(x, -x) == -1.0
    r = pc.spearman_ci(x, x + 0.1 * np.random.RandomState(0).randn(10))
    assert r["rho"] > 0.9 and r["ci_low"] <= r["rho"] <= r["ci_high"]


def _planner_scene(hazard_aware: bool, seed: int = 0, n_tok: int = 9, d: int = 6):
    rng = np.random.RandomState(seed)
    grid = 4
    cells = {(h, a): i for i, (h, a) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])}
    true = np.zeros((6, grid, grid, d), np.float32)
    pred = np.zeros((6, grid, grid, d), np.float32)
    ctx = rng.randn(grid, grid, d)
    e_a = np.zeros((grid, grid, d)); e_a[..., 0] = 1.0
    e_c = np.zeros((grid, grid, d)); e_c[:2, :2, 1] = 1.0  # contact signature on some tokens
    for (h, a), i in cells.items():
        t = ctx + e_a * a + (e_c * 2.0 if (h == 1 and a == 1) else 0) + 0.05 * rng.randn(grid, grid, d)
        true[i] = t
        if hazard_aware:
            pred[i] = t + 0.05 * rng.randn(grid, grid, d)
        else:
            pred[i] = ctx + e_a * a + 0.05 * rng.randn(grid, grid, d)  # never predicts contact
    tokens = np.arange(n_tok)
    return cells, pred, true, tokens


def test_planner_currency_distinguishes_hazard_aware_from_blind() -> None:
    aware = {f"p{s}": pc.scene_planner_currency(*_planner_scene(True, s)) for s in range(18)}
    blind = {f"p{s}": pc.scene_planner_currency(*_planner_scene(False, s)) for s in range(18)}
    force = {f"p{s}": 10.0 + s for s in range(18)}
    pa = pc.pool_planner_currency(aware, force)
    pb = pc.pool_planner_currency(blind, force)
    # hazard-aware: Rec_h1 ~ 1 and Rec_h0 ~ 0.9ish, G_h1 < 0 (closer to contact future), a1 costs more under H1
    assert pa["rec_h1_minus_h0"]["mean"] > -0.2 and pa["rec_h1"]["median"] > 0.8
    assert pa["energy_gap"]["fraction_h1_prediction_closer_to_contact_future"] == 1.0
    assert pa["energy_gap"]["h1_vs_h0"]["sign_flip_p"] < 0.01
    assert pa["planner_ranking"]["delta_h1_minus_h0"]["mean"] > 0 and pa["planner_ranking"]["delta_h1_minus_h0"]["sign_flip_p"] < 0.01
    assert "rec_h1_minus_h0prime" in pa and "h1_vs_h0prime" in pa["energy_gap"]
    assert pa["planner_ranking"]["perturbed_candidates"]["status"] == "deferred"
    # hazard-blind: Rec_h1 << Rec_h0 (specificity negative), prediction closer to the no-contact future, no extra penalty
    assert pb["rec_h1_minus_h0"]["mean"] < -0.3 and pb["rec_h1_minus_h0"]["sign_flip_p"] < 0.01
    assert pb["energy_gap"]["fraction_h1_prediction_closer_to_contact_future"] == 0.0
    assert abs(pb["planner_ranking"]["delta_h1_minus_h0"]["mean"]) < 0.5 * abs(pa["planner_ranking"]["delta_h1_minus_h0"]["mean"])
    assert pa["force_did_correlation"]["n_with_force"] == 18


def test_perturbed_rank_correlation() -> None:
    rng = np.random.RandomState(0)
    goal = rng.randn(4, 4, 6).astype(np.float32)
    direction = rng.randn(4, 4, 6).astype(np.float32)
    preds = [goal + direction * s for s in np.linspace(0.1, 2.0, 8)]
    scene = {"pred": {"h1": preds, "h0": preds, "h0prime": list(reversed(preds))}, "goal": {"h1": goal, "h0": goal, "h0prime": goal}}
    r = pc.perturbed_rank_correlations({"p0": scene}, "p0", np.arange(16))
    assert r["n_candidates"] == 8 and r["rho_h1_h0"] == 1.0 and r["rho_h0_h0prime"] == -1.0
    assert pc.perturbed_rank_correlations({"p0": scene}, "missing", np.arange(16)) is None


def test_gate_has_planner_currency_and_specificity_verdict() -> None:
    rows, cache = _synthetic_cache(18, good_model=True)
    res = cv.evaluate(cache, rows, stimulus_dir=None)
    assert "planner_currency" in res and res["checks"]["hazard_specificity_ok"] in (True, False)
    assert res["verdict_with_specificity"] in ("PASS", "FAIL")
    assert "Hazard specificity" in res["statement"]
    prim = res["planner_currency"]["token_groups"][res["planner_currency"]["primary_group"]]["pooled"]
    assert "energy_gap" in prim and "planner_ranking" in prim and "force_did_correlation" in prim


# --------------------------------------------------------------------------- #
# driving domain (protocol v0.8): synthetic 8-cell scene end to end on CPU
# --------------------------------------------------------------------------- #

import behavior_gate as bg  # noqa: E402
import token_groups as tg  # noqa: E402
from protocol import OBJECT_HAZARD, PROTOCOL_VERSION_V08, V08_CELL_ORDER, did_by_identity  # noqa: E402

D_LEVELS = (0, 1, 2, OBJECT_HAZARD)


def _drive_pair_id(seed: int) -> str:
    return f"drive_seed{seed:06d}"


def _drive_cell_id(seed: int, h: int, a: int) -> str:
    return f"{_drive_pair_id(seed)}__h{h}a{a}"


def _drive_min_distance(h: int, a: int) -> float:
    """Arm A physics: contact only in (1, throttle); the ghost cone (3) and sidewalk poses keep clearance."""

    if a == 0:
        return 6.0
    return 0.0 if h == 1 else 5.0


def _write_driving_seed(root: Path, seed: int, cells=V08_CELL_ORDER, pixel_error: int = 0, differing: int | None = None,
                        positioning: bool = True, write_masks: bool = True, shift: int = 0) -> None:
    sdir = root / f"seed_{seed}"
    (sdir / "cells").mkdir(parents=True)
    if write_masks:
        (sdir / "masks").mkdir()
    rows = []
    for h, a in cells:
        cid = _drive_cell_id(seed, h, a)
        np.savez(sdir / "cells" / f"{cid}.npz", x=np.zeros(1))
        jit = pixel_error if h == 1 else 0
        row = {
            "pair_id": _drive_pair_id(seed), "cell_id": cid, "seed": seed, "hazard": h, "candidate_action": a,
            "artifact": f"cells/{cid}.npz", "replay_frames_bit_exact": jit == 0, "replay_max_pixel_error": jit,
            "replay_max_state_error": 0.0, "replay_initial_state_error": 0.0, "replay_contact_count_equal": True,
            "min_distance_m": _drive_min_distance(h, a), "contact": _drive_min_distance(h, a) == 0.0,
            "contact_step": 2 if _drive_min_distance(h, a) == 0.0 else None,
            "crash_human": (h, a) == (1, 1), "crash_object": False,
            "hazard_centroid_xy": [128.0 + shift, 176.0], "hazard_bbox_xyxy": [100 + shift, 140, 156 + shift, 212],
        }
        if jit and differing is not None:
            row["replay_max_differing_pixels_per_frame"] = differing
        rows.append(row)
        if write_masks:
            hm = np.zeros((4, 256, 256), bool)
            cm = np.zeros((4, 256, 256), bool)
            col = 100 + shift + (40 if h in (0, 2) else 0)  # sidewalk poses sit to the right of the lane
            hm[:, 140:212, col : col + 56] = True
            cm[:, 128:256, 96:160] = True  # ego lane polygon ahead
            np.savez_compressed(sdir / "masks" / f"{cid}.npz", hazard_mask=hm, corridor_mask=cm,
                                hazard_centroid_px=np.array([[col + 28, 176]] * 4, np.float32), hazard_bbox_xyxy=np.array([col, 140, col + 56, 212], np.float32))
    (sdir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    d = {(h, a): _drive_min_distance(h, a) for h, a in V08_CELL_ORDER}
    mdd = (d[(1, 1)] - d[(1, 0)]) - (d[(0, 1)] - d[(0, 0)])
    summary = {"pairs": [{"pair_id": _drive_pair_id(seed), "positioning_gate": positioning, "state_contrast_gate": True,
                          "contact_pattern_gate": True, "min_distance_did_m": mdd, "contact_did": 1.0,
                          "hazard_bbox_px": [56, 72]}], "arm": "A", "protocol_version": PROTOCOL_VERSION_V08}
    (sdir / "summary.json").write_text(json.dumps(summary))


def test_merge_driving_v08_octet_completeness_and_partial_scenes(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    for s in (1, 2, 3):
        _write_driving_seed(root, s)
    _write_driving_seed(root, 4, cells=V08_CELL_ORDER[:6])  # partial: level-3 cells missing
    _write_driving_seed(root, 5, pixel_error=8, differing=16)  # at the v0.8 frame-gate limit -> admitted
    _write_driving_seed(root, 6, pixel_error=8, differing=17)  # one pixel too many -> excluded
    _write_driving_seed(root, 7, pixel_error=8)  # unverified count -> fails closed
    _write_driving_seed(root, 8, positioning=False)
    _write_driving_seed(root, 9)
    val = tmp_path / "labels.json"
    val.write_text(json.dumps({"protocol": PROTOCOL_VERSION_V08, "seeds": [
        {"seed": s, "passed": True, "failures": []} for s in (1, 2, 3, 4, 5, 6, 7, 8)] + [{"seed": 9, "passed": False, "failures": ["h1a1:no_contact_under_throttle"]}]}))
    with pytest.raises(ValueError):
        mh.merge(root, tmp_path / "bad", None, None, PROTOCOL_VERSION_V03, [], val, 0, None, (), "driving")
    res = mh.merge(root, tmp_path / "out", None, (1, 10), PROTOCOL_VERSION_V08, [], val, 0, None, (), "driving")
    assert res["protocol_version"] == PROTOCOL_VERSION_V08 and res["domain"] == "driving"
    assert res["admitted_seeds"] == [1, 2, 3, 5]
    t = res["tally"]
    assert t["partial_scenes"] == [4] and t["complete_octets"] == 8
    ps = res["per_seed"]
    assert ps["4"]["partial"] and ps["4"]["cells_missing"] == ["h3a0", "h3a1"] and ps["4"]["exclusion_reasons"] == ["partial_scene:missing_2_of_8"]
    assert ps["6"]["exclusion_reasons"] == ["replay:more_than_16_differing_pixels"]
    assert ps["7"]["exclusion_reasons"] == ["replay:differing_pixel_count_unverified"]
    assert ps["8"]["exclusion_reasons"] == ["positioning"]
    assert ps["9"]["exclusion_reasons"] == ["labels:h1a1:no_contact_under_throttle"]
    assert ps["10"]["status"] == "not_run_or_no_log"
    # physical DiD: min-distance DiD from summary, approach sign flipped, identity DiD from the per-cell distances
    assert ps["1"]["min_distance_did_m"] == pytest.approx(-5.0) and ps["1"]["approach_did_m"] == pytest.approx(5.0)
    assert ps["1"]["did_by_identity_approach_m"]["identity_x_action"] == pytest.approx(5.0)
    assert ps["1"]["did_by_identity_min_distance_m"] == did_by_identity({k: _drive_min_distance(*k) for k in V08_CELL_ORDER})
    assert ps["1"]["per_cell"]["h1a1"]["contact"] is True and ps["1"]["per_cell"]["h3a1"]["crash_object"] is False
    pg = res["phase0_stimulus_gate"]
    assert pg["complete_octets"] and pg["nonzero_min_distance_did_pairs"] == 4 and "force_interaction_at_least_5n" not in pg
    assert res["physical_did"]["name"] == "min_distance_did_m" and "approach_did_m = -min_distance_did_m" in res["physical_did"]["definition"]
    rows = [json.loads(l) for l in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * 8 and sorted({int(r["hazard"]) for r in rows}) == [0, 1, 2, 3]
    assert set(res["split"]["discovery"]) | set(res["split"]["confirmation"]) == {1, 2, 3, 5}


def test_token_groups_driving_aliases_and_detection(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    _write_driving_seed(root, 1)
    row = {"cell_id": _drive_cell_id(1, 1, 1), "hazard_bbox_xyxy": [100, 140, 156, 212]}
    g = tg.load_cell_groups(root / "seed_1", row)  # detected from the mask keys even in the default (egg) domain
    assert g.domain == "driving" and g.source == "masks" and g.n_frames == 4
    hz, co, bg_, hc = (g.group(3, k) for k in ("hazard", "corridor", "background", "hazard_corridor"))
    assert len(hz) > 0 and len(co) > 0 and not set(hz) & set(co)
    assert set(hc) == set(hz) | set(co) and len(hz) + len(co) + len(bg_) == 256
    np.testing.assert_array_equal(g.group(3, "egg"), hz)
    np.testing.assert_array_equal(g.group(3, "gripper_corridor"), co)
    assert len(g.group(3, "gripper")) == 0
    with pytest.raises(KeyError):
        g.group(0, "robot")
    s = tg.summarize(g)
    assert s["domain"] == "driving" and s["token_group_source"]["aliases"]["egg"] == "hazard" and s["sizes"][0]["gripper_corridor"] == s["sizes"][0]["corridor"]
    # bbox fallback and domain switch
    fb = tg.load_cell_groups(tmp_path / "nowhere", row, domain="driving")
    assert fb.source == "bbox_fallback" and len(fb.group(0, "hazard")) > 0 and len(fb.group(0, "corridor")) == 0
    assert tg.group_names("driving") == tg.DRIVING_GROUP_NAMES and "egg" in tg.group_names("driving", include_aliases=True)
    assert tg.group_names("egg") == tg.GROUP_NAMES and tg.alias_note("egg") is None
    assert tg.candidate_groups("egg") == tg.CANDIDATE_GROUPS and "hazard_corridor" in tg.candidate_groups("driving")
    assert tg.set_domain("driving") == "driving" and tg.current_domain() == "driving"
    tg.set_domain("egg")
    with pytest.raises(ValueError):
        tg.set_domain("boats")
    # egg masks are still read as egg masks
    egg = {"egg_mask": np.zeros((4, 256, 256), bool), "robot_mask": np.zeros((4, 256, 256), bool), "eef_px": np.zeros((4, 2)), "egg_px": np.full((4, 2), 128.0)}
    egg["egg_mask"][:, 120:128, 120:128] = True  # one patch (7, 7); dilated by one -> 9 tokens
    (root / "eggmask").mkdir()
    (root / "eggmask" / "masks").mkdir()
    np.savez(root / "eggmask" / "masks" / "c.npz", **egg)
    eg = tg.load_cell_groups(root / "eggmask", {"cell_id": "c"}, domain="driving")
    assert eg.domain == "egg" and len(eg.group(0, "egg")) == 9 and len(eg.group(0, "hazard")) == 9


def _synthetic_driving_cache(rows: list[dict], seed: int = 0, grid: int = 16, d: int = 4, arm: str = "A", good_model: bool = True):
    """8-cell scenes: the pedestrian consequence (arm A) lives on a few lane tokens; cone and sidewalk poses are inert.
    ``good_model=False`` = a hazard-blind predictor that never predicts the contact future."""

    rng = np.random.RandomState(seed)
    e0 = np.zeros((grid, grid, d)); e0[..., 0] = 1.0
    e1 = np.zeros((grid, grid, d)); e1[9:13, 7:10, 1] = 1.0
    solid = {1: 1.0, 3: 0.0} if arm == "A" else {1: 0.0, 3: 1.0}
    base_by_scene: dict[str, np.ndarray] = {}
    ctx, tgt, pred, zero, act, mask = [], [], [], [], [], []
    for r in rows:
        h, a = int(r["hazard"]), int(r["candidate_action"])
        base = base_by_scene.setdefault(r["pair_id"], rng.randn(grid, grid, d))
        c = base + 0.02 * h * np.random.RandomState(seed + h).randn(grid, grid, d)
        true = c + e0 * (0.5 + a) + e1 * solid.get(h, 0.0) * a * 2.0 + 0.01 * rng.randn(grid, grid, d)  # contact signature on lane tokens
        p = true + 0.02 * rng.randn(grid, grid, d) if good_model else c + e0 * (0.5 + a) + 0.02 * rng.randn(grid, grid, d)
        ctx.append(c); tgt.append(true[None]); pred.append(p[None]); zero.append((c + 0.001 * rng.randn(grid, grid, d))[None])
        aa = np.zeros((3, 2), np.float32); aa[:, 1] = 1.0 if a else -1.0
        act.append(aa)
        m = np.zeros((2, grid, grid), bool); m[:, 8:16, 6:10] = True
        mask.append(m)
    return {
        "cell_id": np.array([r["cell_id"] for r in rows]),
        "context": np.stack(ctx).astype(np.float16), "target": np.stack(tgt).astype(np.float16),
        "prediction": np.stack(pred).astype(np.float16), "zero_prediction": np.stack(zero).astype(np.float16),
        "actions": np.stack(act), "token_mask": np.stack(mask), "has_mask": np.ones(len(rows), bool), "meta": {"domain": "driving"},
    }


def test_driving_pipeline_end_to_end_cpu(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    seeds = list(range(1, 19))
    for s in seeds:
        _write_driving_seed(root, s, shift=(s % 3) * 4)
    val = tmp_path / "labels.json"
    val.write_text(json.dumps({"seeds": [{"seed": s, "passed": True, "failures": []} for s in seeds]}))
    out = tmp_path / "merged"
    res = mh.merge(root, out, None, None, PROTOCOL_VERSION_V08, [], val, 0, None, (), "driving")
    assert res["admitted_seeds"] == seeds
    (out / "masks").mkdir()
    for s in seeds:  # the runner copies per-seed masks into the merged dir
        for p in (root / f"seed_{s}" / "masks").glob("*.npz"):
            (out / "masks" / p.name).write_bytes(p.read_bytes())
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    cache = _synthetic_driving_cache(rows)

    # 1. counterfactual-validity gate, driving currency
    gate = cv.evaluate(cache, rows, stimulus_dir=out, domain="driving")
    assert gate["domain"] == "driving" and gate["n_scenes"] == 18 and gate["verdict"] == "PASS"
    assert gate["token_mask"].startswith("hazard|corridor") and gate["token_group_source"]["aliases"]["gripper_corridor"] == "corridor"
    pcb = gate["planner_currency"]
    assert pcb["primary_group"] == "hazard_corridor" and pcb["physical_did_label"] == "approach_did"
    assert set(pcb["token_groups"]) == {"hazard_corridor", "hazard", "corridor", "all", "egg_gripper"}
    assert pcb["token_groups"]["egg_gripper"]["alias_of"] == "hazard_corridor"
    prim = pcb["token_groups"]["hazard_corridor"]["pooled"]
    assert "approach_did_correlation" in prim and "force_did_correlation" not in prim
    assert prim["approach_did_correlation"]["n_with_approach_did"] == 18
    ident = prim["identity_contrast"]
    # a hazard-aware model: the throttle chunk ranks worse under the pedestrian than under the inert cone (planner currency),
    # and its H1 prediction sits closer to the contact future than to the cone future; Rec alone is ~1 at every level
    assert ident["delta_h1_minus_h3"]["mean"] > 0 and ident["delta_h1_minus_h3"]["sign_flip_p"] < 0.01
    assert ident["energy_gap_h1_vs_h3"]["mean"] < 0 and ident["energy_gap_h1_vs_h3"]["sign_flip_p"] < 0.01
    assert ident["flip_rate_h1_vs_h3"] >= 0.0 and abs(ident["rec_h1_minus_h3"]["mean"]) < 0.1
    assert ident["identity_did_correlation"]["n_with_identity_did"] == 18
    assert "rec_h1_minus_h0prime" in prim and gate["verdict_with_specificity"] in ("PASS", "FAIL")
    # a hazard-blind model never predicts the contact: Rec_h1 collapses while Rec_h3 (inert cone) stays ~1
    blind = cv.evaluate(_synthetic_driving_cache(rows, good_model=False), rows, stimulus_dir=out, domain="driving")
    bid = blind["planner_currency"]["token_groups"]["hazard_corridor"]["pooled"]["identity_contrast"]
    assert bid["rec_h1_minus_h3"]["mean"] < -0.3 and bid["rec_h1_minus_h3"]["sign_flip_p"] < 0.01 and blind["verdict"] == "FAIL"
    assert "hazard/corridor latent space" in blind["statement"] and "egg" not in blind["statement"]
    assert gate["identity_contrast"]["status"] == "ok" and gate["physical_did"]["label"] == "approach_did"
    assert set(gate["sonar"]["groups"]) == set(tg.DRIVING_GROUP_NAMES) and gate["sonar"]["groups"]["hazard_corridor"]["status"] == "ok"
    assert gate["hazard_levels"]["3"] == "cone_in_lane" and "egg" not in gate["statement"] and "Hazard specificity (primary, hazard_corridor tokens)" in gate["statement"]
    per = pcb["token_groups"]["hazard_corridor"]["per_scene"][rows[0]["pair_id"]]
    assert per["levels"] == ["h0", "h1", "h0prime", "h3"] and "rank_flip_h1_vs_h3" in per and "interaction_cosine_object" in per
    # arm B relabelling: the cone becomes H1 and the pedestrian level 3 -> specificity flips sign
    gate_b = cv.evaluate(cache, rows, stimulus_dir=out, domain="driving", primary_hazard_level=3)
    assert gate_b["primary_hazard_level"] == 3 and "relabelled" in gate_b["statement"]
    ident_b = gate_b["planner_currency"]["token_groups"]["hazard_corridor"]["pooled"]["identity_contrast"]
    # the planner-ranking contrast flips sign under the relabelling; the energy gap (own true future vs the other identity's)
    # stays negative for an accurate model under either labelling, so only its magnitude is compared
    assert ident_b["delta_h1_minus_h3"]["mean"] < 0 and ident_b["delta_h1_minus_h3"]["mean"] == pytest.approx(-ident["delta_h1_minus_h3"]["mean"], rel=0.2)
    assert ident_b["energy_gap_h1_vs_h3"]["mean"] < 0
    # gate JSON round-trips through canonical json (no NaN) and the B-gate reads the driving primary group
    gate_json = json.loads(json.dumps(su.finite(gate)))
    b = bg.evaluate(gate_json)
    assert b["verdict"] in ("PASS", "PASS_PLANNER", "FAIL") and b["domain"] == "driving" and b["token_group"] == "hazard_corridor"
    assert b["T1b_ok"] and b["identity_contrast"]["status"] == "ok" and b["identity_contrast"]["delta_h1_minus_h3"]["mean"] > 0
    assert b["T1c_rec_h1_minus_h0"] == pytest.approx(prim["rec_h1_minus_h0"]["mean"])  # read from the driving primary group
    assert bg.primary_pooled({"planner_currency": {"token_groups": {"egg_gripper": {"pooled": {"x": 1}}}}}) == ("egg_gripper", {"x": 1})

    # 2. retrieval control with the identity block
    rbr = rb.analyze(cache, rows, 1.0, 3, 1e-2, domain="driving")
    assert rbr["domain"] == "driving" and rbr["identity"]["n_scenes"] == 18
    assert rbr["identity"]["identity"]["model"]["mean_cosine"]["mean"] > 0.8
    assert rbr["identity"]["identity"]["baselines"]["persistence"]["sign_flip_p"] < 0.05
    assert rbr["interaction_gate"]["passed"] is True

    # 3. audit helpers on 2-D driving actions and 4 hazard levels
    hazard = np.array([int(r["hazard"]) for r in rows]); level = np.array([int(r["candidate_action"]) for r in rows])
    aud = ac.action_audit(cache["actions"], hazard, level, [r["pair_id"] for r in rows], chance=0.25)
    assert aud["chance_hazard"] == 0.25 and aud["actions_identical_across_hazard"] and aud["hazard_from_actions_at_chance"]
    ref = np.concatenate([cache["actions"][:2]] * 30) + 0.01 * np.random.RandomState(3).randn(60, 3, 2).astype(np.float32)
    nov = ac.action_novelty(cache["actions"], ref, np.repeat(np.arange(6), 10), 60, 0, scale=None)
    assert nov["scaling"].startswith("unscaled") and nov["dtw"]["distinct_stimulus_chunks"] == 2
    assert np.isfinite(nov["euclidean"]["ratio_median"]) and nov["euclidean"]["ratio_median"] < 2.0
    with pytest.raises(ValueError):
        ac.scale_actions(cache["actions"])  # 2-D actions must not be scaled by the 7-D DROID max_norms

    # 4. identity DiD in the merge equals protocol.did_by_identity on the per-cell distances
    ident = res["physical_did"]["per_seed"]["1"]["did_by_identity_approach_m"]
    assert ident["identity_x_action"] == pytest.approx(5.0) and ident["pedestrian"] == pytest.approx(5.0) and ident["object"] == pytest.approx(0.0)


def test_driving_egg_gripper_alias_is_read_by_behavior_gate_without_native_group() -> None:
    gate = {"pooled": {"interaction_nmse": {"median": 0.1}}, "n_scenes": 20,
            "planner_currency": {"primary_group": "hazard_corridor", "token_groups": {"hazard_corridor": {"pooled": {
                "rec_h1_minus_h0": {"mean": 0.4, "sign_flip_p": 0.001}, "rec_h1_minus_h0prime": {"mean": 0.3, "sign_flip_p": 0.01},
                "energy_gap": {"normalized_h1_vs_h0": {"mean": 0.2, "sign_flip_p": 0.01}}, "planner_ranking": {"flip_rate_h1_vs_h0": 0.5}}}}}}
    r = bg.evaluate(gate)
    assert r["verdict"] == "PASS_PLANNER" and "domain" not in r
