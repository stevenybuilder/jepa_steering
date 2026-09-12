#!/usr/bin/env python3
"""Aggregate the driving-cell (protocol v0.7/v0.8, arms A/B) gate results, the cross-truth 2x2x2 table and the
substrate comparison against the egg cells into one traceable summary.

Outputs (re-runnable; every number carries the artifact path + JSON key it was read from):
  artifacts/cgs_pilot/drive_results_summary.json
  paper/results_tables.md                (Tables R1, R2, R3 + seed replication + paired-seed sign test)
  paper/figures/fig2_bgate_substrates.{png,pdf}     (Fig 2: captured interaction fraction and planner flip rate)
  paper/figures/figR2_crosstruth_dissociation.{png,pdf}  (Fig R2: model arm x truth arm per seed)

Reads ONLY the JSON artifacts (never prose):
  artifacts/cgs_pilot/drive_eval/arm{A,B}_seed{s}/
      behavior_gate_discovery.json              primary identity B-gate (A: level 1 pedestrian; B: level 3 cone)
      behavior_gate_discovery_level{3|1}.json   cross (ghost) identity B-gate
      cf_gate_discovery*.json                   pooled validity-gate blocks (medians, CI of the MEAN, planner currency)
      retrieval_baseline.json, contamination.json, summary.json, EVAL_SCOPE.json
  artifacts/cgs_pilot/drive_eval_crosstruth/arm{X}_seed{s}_on_arm{Y}/crosstruth_summary.json
  artifacts/cgs_pilot/heldout_v1_merged_w1|w2|w2_vj2ac/{behavior_gate_*,cf_gate_*,retrieval_baseline*}.json (egg cells)

Missing models / cells are skipped with a note (the box is still producing arm B seed 2 and its cross-truth cells).

Usage:  /usr/local/bin/python3.9 scripts/cgs_pilot/aggregate_drive_results.py [--no-figures] [--root DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "cgs_pilot"

B_GATE_THRESH = {"max_interaction_nmse_median": 0.75, "min_captured_fraction": 0.25, "min_flip_rate": 0.25, "spec_p": 0.05}

ARMS = {
    # primary (solid) identity level, cross (ghost) identity level, names
    "A": {"primary_level": 1, "cross_level": 3, "solid": "pedestrian", "ghost": "cone",
          "description": "pedestrian solid / cone ghost"},
    "B": {"primary_level": 3, "cross_level": 1, "solid": "cone", "ghost": "pedestrian",
          "description": "cone solid / pedestrian ghost"},
}
LEVEL_NAME = {1: "pedestrian", 3: "cone"}

EGG_CELLS = [
    {"label": "JEPA-WM n=21", "substrate": "JEPA-WM DROID (egg, wave 1 all)", "n": 21,
     "bgate": "heldout_v1_merged_w1/behavior_gate_cf_gate_sonar_all.json",
     "cf": "heldout_v1_merged_w1/cf_gate_sonar_all.json",
     "retrieval": "heldout_v1_merged_w1/retrieval_baseline_sonar.json"},
    {"label": "JEPA-WM n=34", "substrate": "JEPA-WM DROID (egg, wave 2 discovery)", "n": 34,
     "bgate": "heldout_v1_merged_w2/behavior_gate_cf_gate_discovery.json",
     "cf": "heldout_v1_merged_w2/cf_gate_discovery.json", "retrieval": None},
    {"label": "JEPA-WM n=66", "substrate": "JEPA-WM DROID (egg, wave 2 all)", "n": 66,
     "bgate": "heldout_v1_merged_w2/behavior_gate_cf_gate_all.json",
     "cf": "heldout_v1_merged_w2/cf_gate_all.json",
     "retrieval": "heldout_v1_merged_w2/retrieval_baseline.json"},
    {"label": "V-JEPA 2-AC n=34", "substrate": "V-JEPA 2-AC DROID (egg, wave 2 discovery)", "n": 34,
     "bgate": "heldout_v1_merged_w2_vj2ac/behavior_gate_vj2ac.json",
     "cf": "heldout_v1_merged_w2_vj2ac/cf_gate_discovery_vj2ac.json",
     "retrieval": "heldout_v1_merged_w2_vj2ac/retrieval_baseline_vj2ac.json"},
]

NOTES: list[str] = []
_NOTE_GROUPS: dict[str, int] = {}


def note(msg: str, group: str | None = None) -> None:
    """Record a note; notes sharing `group` are printed once with a multiplicity count (the JSON keeps the count too)."""
    if group is not None:
        if group in _NOTE_GROUPS:
            _NOTE_GROUPS[group] += 1
            idx = next(i for i, n in enumerate(NOTES) if n.startswith(f"[{group}]"))
            base = NOTES[idx].split(" (x")[0]
            NOTES[idx] = f"{base} (x{_NOTE_GROUPS[group]} occurrences; first instance shown)"
            return
        _NOTE_GROUPS[group] = 1
        msg = f"[{group}] {msg}"
    NOTES.append(msg)
    print("NOTE:", msg, file=sys.stderr)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def mean_sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else None
    return m, sd, len(xs)


# ----------------------------------------------------------------------------------------------------------------- egg
def read_egg_cell(spec: dict) -> dict:
    bg_path, cf_path = ART / spec["bgate"], ART / spec["cf"]
    bg, cf = load(bg_path), load(cf_path)
    if bg is None:
        note(f"egg cell {spec['label']}: missing {rel(bg_path)} (skipped)")
        return {"label": spec["label"], "missing": True}
    row = {
        "label": spec["label"], "substrate": spec["substrate"], "n_scenes": bg["n_scenes"],
        "domain": "manipulation (RoboCasa egg)",
        "interaction_nmse_median": bg["T1b_interaction_nmse_median"],
        "captured_fraction": bg["model_captured_interaction_fraction"],
        "rec_h1_minus_h0": bg["T1c_rec_h1_minus_h0"], "rec_h1_minus_h0_p": bg["T1c_p"],
        "rec_h1_minus_h0prime": bg["T1c_rec_h1_minus_h0prime"],
        "flip_rate_h1_vs_h0": bg["T2a_flip_rate_h1_vs_h0"],
        "energy_gap_norm_h1_vs_h0": bg["T2b_energy_gap_norm_h1_vs_h0"], "energy_gap_p": bg["T2b_p"],
        "b_gate_verdict": bg["verdict"],
        "sources": {"b_gate": rel(bg_path), "cf_gate": rel(cf_path) if cf else None},
    }
    if cf:
        po = cf["pooled"]
        row.update({
            "rec_h1_median": g(po, "rec_h1", "median"), "rec_h0_median": g(po, "rec_h0_control", "median"),
            "interaction_cosine_median": g(po, "interaction_cosine", "median"),
            "validity_verdict": cf.get("verdict"), "validity_with_specificity": cf.get("verdict_with_specificity"),
        })
    if spec.get("retrieval"):
        rp = ART / spec["retrieval"]
        r = load(rp)
        if r:
            row["retrieval"] = {
                "model_mean_cosine": g(r, "model", "mean_cosine", "mean"),
                "baselines": {k: v.get("baseline_mean_cosine") for k, v in r["baselines"].items()},
                "passed": g(r, "interaction_gate", "passed"), "source": rel(rp),
            }
    return row


# ------------------------------------------------------------------------------------------------------------- driving
def level_block(bg: dict, cf: dict | None, level: int) -> dict:
    """One identity level of one model: B-gate fields + pooled validity-gate fields."""
    out = {
        "level": level, "identity": LEVEL_NAME[level], "n_scenes": bg["n_scenes"],
        "interaction_nmse_median": bg["T1b_interaction_nmse_median"],
        "captured_fraction": bg["model_captured_interaction_fraction"],
        "T1b_ok": bg["T1b_ok"],
        "rec_h1_minus_h0": bg["T1c_rec_h1_minus_h0"], "rec_h1_minus_h0_p": bg["T1c_p"],
        "rec_h1_minus_h0prime": bg["T1c_rec_h1_minus_h0prime"], "T1c_ok": bg["T1c_ok"],
        "T1c_prime": bg.get("T1c_prime") or None, "t1c_route": bg.get("t1c_route"),
        "flip_rate_h1_vs_h0": bg["T2a_flip_rate_h1_vs_h0"], "T2a_ok": bg["T2a_ok"],
        "energy_gap_norm_h1_vs_h0": bg["T2b_energy_gap_norm_h1_vs_h0"], "energy_gap_p": bg["T2b_p"], "T2b_ok": bg["T2b_ok"],
        "b_gate_verdict": bg["verdict"],
    }
    if cf:
        po = cf["pooled"]
        pc_name = g(cf, "planner_currency", "primary_group", default="hazard_corridor")
        pcp = g(cf, "planner_currency", "token_groups", pc_name, "pooled", default={})
        out.update({
            "interaction_nmse_mean": g(po, "interaction_nmse", "mean"),
            "interaction_nmse_ci_of_mean": [g(po, "interaction_nmse", "ci_low"), g(po, "interaction_nmse", "ci_high")],
            "interaction_cosine_median": g(po, "interaction_cosine", "median"),
            "interaction_cosine_ci_of_mean": [g(po, "interaction_cosine", "ci_low"), g(po, "interaction_cosine", "ci_high")],
            "rec_h1_median": g(po, "rec_h1", "median"), "rec_h0_median": g(po, "rec_h0_control", "median"),
            "validity_verdict": cf.get("verdict"), "validity_with_specificity": cf.get("verdict_with_specificity"),
            "fraction_h1_prediction_closer_to_contact_future": g(pcp, "energy_gap", "fraction_h1_prediction_closer_to_contact_future"),
            "planner_delta_h1_minus_h0_mean": g(pcp, "planner_ranking", "delta_h1_minus_h0", "mean"),
            "token_group": pc_name,
        })
        med, lo, hi = out["interaction_nmse_median"], *out["interaction_nmse_ci_of_mean"]
        if lo is not None and hi is not None and not (lo <= med <= hi):
            out["ci_note"] = "pooled CI brackets the mean, not the median"
            note(f"{cf.get('model_label')} level {level} n={bg['n_scenes']}: interaction NMSE median {med:.4f} lies outside the pooled CI "
                 f"[{lo:.4f}, {hi:.4f}] (CI is of the scene mean {out['interaction_nmse_mean']:.4f}); tables label CIs as CI-of-mean", group="ci_of_mean")
        frac = out["fraction_h1_prediction_closer_to_contact_future"]
        if (not bg["T2b_ok"]) and frac is not None and frac >= 0.5 and bg["T2b_p"] is not None and bg["T2b_p"] < 0.05 and (bg["T2b_energy_gap_norm_h1_vs_h0"] or 0) < 0:
            note(f"T2b sign convention: level {level} energy gap normalised {bg['T2b_energy_gap_norm_h1_vs_h0']:.3f} (p {bg['T2b_p']:.1e}) is reported T2b_ok=False by "
                 f"behavior_gate.py (requires > 0) although planner_currency.py defines gap < 0 as 'H1 prediction closer to the contact future' "
                 f"(fraction closer = {frac:.2f}); verdict unaffected because T2a passes", group="t2b_sign")
    return out


def read_driving_model(mdir: Path) -> dict | None:
    name = mdir.name  # armA_seed0
    arm, seed = name[3], int(name.split("seed")[1])
    spec = ARMS[arm]
    p_lv, c_lv = spec["primary_level"], spec["cross_level"]
    bg_p = load(mdir / "behavior_gate_discovery.json")
    if bg_p is None:
        note(f"driving model {name}: no behavior_gate_discovery.json yet (eval not finished; skipped)")
        return None
    if bg_p.get("primary_hazard_level") != p_lv:
        note(f"driving model {name}: primary_hazard_level {bg_p.get('primary_hazard_level')} != expected {p_lv}")
    bg_c = load(mdir / f"behavior_gate_discovery_level{c_lv}.json")
    cf_p = load(mdir / "cf_gate_discovery.json")
    cf_c = load(mdir / f"cf_gate_discovery_level{c_lv}.json")
    summ = load(mdir / "summary.json") or {}
    scope = load(mdir / "EVAL_SCOPE.json") or {}
    rec = {
        "model": name, "arm": arm, "seed": seed, "arm_description": spec["description"],
        "solid_identity": spec["solid"], "ghost_identity": spec["ghost"],
        "n_discovery_scenes": bg_p["n_scenes"],
        "split": {"admitted": g(summ, "tally", "admitted"), "discovery": g(summ, "split", "discovery"),
                  "confirmation": g(summ, "split", "confirmation"), "scope": scope.get("scope")},
        "primary": level_block(bg_p, cf_p, p_lv),
        "cross": level_block(bg_c, cf_c, c_lv) if bg_c else None,
        "sources": {
            "primary_b_gate": rel(mdir / "behavior_gate_discovery.json"),
            "cross_b_gate": rel(mdir / f"behavior_gate_discovery_level{c_lv}.json") if bg_c else None,
            "primary_cf_gate": rel(mdir / "cf_gate_discovery.json") if cf_p else None,
            "cross_cf_gate": rel(mdir / f"cf_gate_discovery_level{c_lv}.json") if cf_c else None,
            "summary": rel(mdir / "summary.json"), "eval_scope": rel(mdir / "EVAL_SCOPE.json"),
        },
    }
    if bg_c is None:
        note(f"driving model {name}: cross-identity gate level {c_lv} missing")

    # Identity contrast with FIXED labelling (pedestrian = level 1, cone = level 3), read from whichever gate file has
    # primary level 1 (arm A: the primary file; arm B: the level-1 cross file). identity_contrast.rec_h1_minus_h3 there is
    # rec(pedestrian) - rec(cone); delta_h1_minus_h3 is the planner-cost DiD pedestrian - cone.
    lv1_bg = bg_p if p_lv == 1 else bg_c
    lv1_src = rec["sources"]["primary_b_gate"] if p_lv == 1 else rec["sources"]["cross_b_gate"]
    ic = (lv1_bg or {}).get("identity_contrast") or {}
    nmse_ped = rec["primary"]["interaction_nmse_median"] if p_lv == 1 else (rec["cross"] or {}).get("interaction_nmse_median")
    nmse_cone = rec["primary"]["interaction_nmse_median"] if p_lv == 3 else (rec["cross"] or {}).get("interaction_nmse_median")
    t1cp = rec["primary"]["T1c_prime"] or {}
    rec["identity_contrast_fixed_labels"] = {
        "definition": "pedestrian (level 1) minus cone (level 3); predicted: rec/delta > 0 and NMSE diff < 0 in arm A, reversed in arm B",
        "rec_ped_minus_cone_mean": g(ic, "rec_h1_minus_h3", "mean"),
        "rec_ped_minus_cone_median": g(ic, "rec_h1_minus_h3", "median"),
        "rec_ped_minus_cone_p": g(ic, "rec_h1_minus_h3", "sign_flip_p"),
        "planner_delta_ped_minus_cone_mean": g(ic, "delta_h1_minus_h3", "mean"),
        "planner_delta_ped_minus_cone_p": g(ic, "delta_h1_minus_h3", "sign_flip_p"),
        "identity_interaction_cosine_median": g(ic, "identity_interaction_cosine", "median"),
        "flip_rate_ped_vs_cone": ic.get("flip_rate_h1_vs_h3"),
        "nmse_ped_median": nmse_ped, "nmse_cone_median": nmse_cone,
        "nmse_ped_minus_cone_median_diff": (nmse_ped - nmse_cone) if (nmse_ped is not None and nmse_cone is not None) else None,
        "solid_minus_ghost_nmse_median_diff": (rec["primary"]["interaction_nmse_median"] - rec["cross"]["interaction_nmse_median"]) if rec["cross"] else None,
        "T1c_prime_scene_level": {"diff_mean_solid_minus_ghost": t1cp.get("diff_mean"), "p": t1cp.get("p"), "ok": t1cp.get("ok"), "n": t1cp.get("n")},
        "source": lv1_src,
    }
    # retrieval + contamination controls
    r = load(mdir / "retrieval_baseline.json")
    if r:
        rec["retrieval"] = {
            "model_mean_cosine": g(r, "model", "mean_cosine", "mean"),
            "baselines": {k: v.get("baseline_mean_cosine") for k, v in r["baselines"].items()},
            "passed": g(r, "interaction_gate", "passed"), "per_baseline": g(r, "interaction_gate", "per_baseline"),
            "deferred": g(r, "interaction_gate", "deferred_baselines"), "source": rel(mdir / "retrieval_baseline.json"),
        }
        if rec["retrieval"]["passed"] is False:
            failing = [k for k, v in (rec["retrieval"]["per_baseline"] or {}).items() if v is False]
            note(f"driving model {name}: retrieval control NOT passed (failing baselines {failing}: "
                 f"{ {k: round(rec['retrieval']['baselines'][k], 3) for k in failing} } vs model {rec['retrieval']['model_mean_cosine']:.3f}); "
                 f"the other-scene copy-delta baseline is high because the driving stimulus is stereotyped (hazard always 9.0 m ahead), so the control cannot discriminate copying here", group="retrieval_copy_delta")
    c = load(mdir / "contamination.json")
    if c:
        rec["contamination"] = {
            "n_cells": c.get("n_cells"),
            "near_duplicate_context_cells": g(c, "duplicate_audit", "n_near_duplicate_context_cells"),
            "cells_with_hazard_centroid_within_2px_of_another_scene": g(c, "duplicate_audit", "n_cells_with_hazard_centroid_within_threshold"),
            "fraction_stimuli_beyond_training_p95_frames": g(c, "training_reference_audit", "frame_domain_shift", "fraction_stimuli_beyond_reference_p95"),
            "actions_identical_across_hazard": g(c, "action_audit", "actions_identical_across_hazard"),
            "hazard_from_actions_at_chance": g(c, "action_audit", "hazard_from_actions_at_chance"),
            "source": rel(mdir / "contamination.json"),
        }
    return rec


def read_crosstruth(cdir: Path) -> dict | None:
    cs = load(cdir / "crosstruth_summary.json")
    if cs is None:
        note(f"cross-truth cell {cdir.name}: no crosstruth_summary.json yet (skipped)")
        return None
    out = {"cell": cdir.name, "model_arm": cs["model_arm"], "model_seed": int(cs["model_seed"]), "truth_arm": cs["truth_arm"],
           "levels": {}, "own_truth_reference": {}, "source": rel(cdir / "crosstruth_summary.json")}
    for lv in ("level1", "level3"):
        L = cs["levels"][lv]
        gt = L["gate"]
        out["levels"][lv] = {
            "identity": LEVEL_NAME[int(lv[-1])], "n": gt["n"],
            "interaction_nmse_median": gt["interaction_nmse_median"], "interaction_nmse_ci_of_mean": gt.get("interaction_nmse_ci"),
            "captured_fraction_signed": gt["captured_fraction"], "captured_fraction": max(0.0, gt["captured_fraction"]),
            "rec_h1_minus_h0": gt["rec_h1_minus_h0"], "rec_h1_minus_h0_p": gt["rec_h1_minus_h0_p"],
            "flip_rate_h1_vs_h0": gt["flip_rate_h1_vs_h0"],
            "validity_verdict": gt["gate_verdict"], "validity_with_specificity": gt["verdict_with_specificity"],
            "b_gate_verdict": L["b_gate_verdict"], "b_gate_verdict_crossgate": L.get("b_gate_verdict_crossgate"),
            "b_gate_route_crossgate": L.get("b_gate_route_crossgate"),
        }
        ref = g(cs, "own_truth_reference", lv, "gate")
        if ref:
            out["own_truth_reference"][lv] = {k: ref.get(k) for k in ("interaction_nmse_median", "captured_fraction", "rec_h1_minus_h0", "flip_rate_h1_vs_h0")}
    return out


# ------------------------------------------------------------------------------------------------------------ stats
def sign_test_one_sided(n_success: int, n: int) -> float:
    """Exact one-sided binomial p for n_success of n in the predicted direction (p = 0.5 each)."""
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(n_success, n + 1)) / 2 ** n


def fmt(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 1e-3:
        return f"{p:.0e}".replace("e-0", "e-")
    return f"{p:.3f}"


def fmt_ms(m, sd, n, nd=3, signed=False):
    if m is None:
        return "—"
    if sd is None:
        return f"{fmt(m, nd, signed)} (n={n})"
    return f"{fmt(m, nd, signed)} ± {sd:.{nd}f} (n={n})"


# ------------------------------------------------------------------------------------------------------------ main
def build(root: Path) -> dict:
    global ART
    ART = root
    egg = [read_egg_cell(s) for s in EGG_CELLS]
    models = []
    for mdir in sorted(root.glob("drive_eval/arm[AB]_seed*")):
        m = read_driving_model(mdir)
        if m:
            models.append(m)
    cross = []
    for cdir in sorted(root.glob("drive_eval_crosstruth/arm[AB]_seed*_on_arm[AB]")):
        c = read_crosstruth(cdir)
        if c:
            cross.append(c)

    # consistency: cross-truth own_truth_reference vs the main run; stale aggregate summary
    by_model = {(m["arm"], m["seed"]): m for m in models}
    for c in cross:
        m = by_model.get((c["model_arm"], c["model_seed"]))
        if not m:
            continue
        for lv, blk in c["own_truth_reference"].items():
            level = int(lv[-1])
            own = m["primary"] if m["primary"]["level"] == level else m["cross"]
            if own and abs(own["interaction_nmse_median"] - blk["interaction_nmse_median"]) > 1e-9:
                note(f"{c['cell']}: own_truth_reference {lv} NMSE {blk['interaction_nmse_median']:.4f} != main run {own['interaction_nmse_median']:.4f}")
    agg = load(root / "drive_eval_crosstruth/summary.json")
    if agg:
        for arm in "AB":
            for lv in ("level1", "level3"):
                per = g(agg, "table", f"model_arm{arm}", f"truth_arm{arm}", lv, "per_seed", default={})
                for s, blk in per.items():
                    m = by_model.get((arm, int(s)))
                    if m is None:
                        continue
                    level = int(lv[-1])
                    own = m["primary"] if m["primary"]["level"] == level else m["cross"]
                    if blk is None and own is not None:
                        note(f"drive_eval_crosstruth/summary.json is STALE: model_arm{arm}/truth_arm{arm}/{lv} seed {s} is null there but "
                             f"{m['sources']['primary_b_gate' if own is m['primary'] else 'cross_b_gate']} exists locally (verdict {own['b_gate_verdict']})")
                    elif isinstance(blk, dict) and own is not None and blk.get("b_gate_verdict") is None and own.get("b_gate_verdict"):
                        note(f"drive_eval_crosstruth/summary.json is STALE: b_gate_verdict null for model_arm{arm} seed {s} {lv}; local file says {own['b_gate_verdict']}")

    # ---- Table R2: model arm x truth arm x level
    r2 = {}
    for ma in "AB":
        for ta in "AB":
            for level in (1, 3):
                key = f"model_arm{ma}|truth_arm{ta}|level{level}"
                per_seed = {}
                if ma == ta:
                    for m in models:
                        if m["arm"] != ma:
                            continue
                        blk = m["primary"] if m["primary"]["level"] == level else m["cross"]
                        if blk is None:
                            continue
                        per_seed[m["seed"]] = {
                            "interaction_nmse_median": blk["interaction_nmse_median"], "captured_fraction_signed": 1.0 - blk["interaction_nmse_median"],
                            "captured_fraction": blk["captured_fraction"], "rec_h1_minus_h0": blk["rec_h1_minus_h0"],
                            "rec_h1_minus_h0_p": blk["rec_h1_minus_h0_p"], "flip_rate_h1_vs_h0": blk["flip_rate_h1_vs_h0"],
                            "b_gate_verdict": blk["b_gate_verdict"], "n": blk["n_scenes"],
                            "source": m["sources"]["primary_b_gate" if blk is m["primary"] else "cross_b_gate"],
                        }
                else:
                    for c in cross:
                        if c["model_arm"] != ma or c["truth_arm"] != ta:
                            continue
                        blk = c["levels"][f"level{level}"]
                        per_seed[c["model_seed"]] = {
                            "interaction_nmse_median": blk["interaction_nmse_median"], "captured_fraction_signed": blk["captured_fraction_signed"],
                            "captured_fraction": blk["captured_fraction"], "rec_h1_minus_h0": blk["rec_h1_minus_h0"],
                            "rec_h1_minus_h0_p": blk["rec_h1_minus_h0_p"], "flip_rate_h1_vs_h0": blk["flip_rate_h1_vs_h0"],
                            "b_gate_verdict": blk["b_gate_verdict"], "n": blk["n"], "source": c["source"],
                        }
                pooled = {}
                for q in ("interaction_nmse_median", "captured_fraction_signed", "captured_fraction", "rec_h1_minus_h0", "flip_rate_h1_vs_h0"):
                    mn, sd, n = mean_sd([v[q] for v in per_seed.values()])
                    pooled[q] = {"mean": mn, "sd": sd, "n_seeds": n}
                verdicts = [v["b_gate_verdict"] for v in per_seed.values()]
                pooled["b_gate_pass_count"] = sum(1 for v in verdicts if v in ("PASS", "PASS_PLANNER"))
                pooled["n_seeds"] = len(verdicts)
                r2[key] = {"model_arm": ma, "truth_arm": ta, "level": level, "identity": LEVEL_NAME[level],
                           "truth_kind": "own" if ma == ta else "cross",
                           "solid_in_truth": (ARMS[ta]["primary_level"] == level),
                           "per_seed": {str(k): per_seed[k] for k in sorted(per_seed)}, "pooled": pooled}

    # ---- seed replication + paired-seed sign tests
    rep = {"per_arm": {}, "paired_seed_sign_test": {}, "all_models_solid_better_than_ghost": {}}
    for arm in "AB":
        ms = [m for m in models if m["arm"] == arm]
        rep["per_arm"][arm] = {
            "n_seeds_evaluated": len(ms), "seeds": [m["seed"] for m in ms],
            "primary_b_gate_pass": sum(m["primary"]["b_gate_verdict"] in ("PASS", "PASS_PLANNER") for m in ms),
            "primary_b_gate_pass_planner": sum(m["primary"]["b_gate_verdict"] == "PASS_PLANNER" for m in ms),
            "cross_identity_b_gate_pass": sum((m["cross"] or {}).get("b_gate_verdict") in ("PASS", "PASS_PLANNER") for m in ms),
            "solid_nmse_below_ghost_nmse": sum(1 for m in ms if m["cross"] and m["primary"]["interaction_nmse_median"] < m["cross"]["interaction_nmse_median"]),
            "T1c_prime_scene_level_p_below_0.05": sum(1 for m in ms if (m["primary"]["T1c_prime"] or {}).get("ok")),
            "T1c_prime_p_values": {str(m["seed"]): (m["primary"]["T1c_prime"] or {}).get("p") for m in ms},
            "cross_truth_cells_evaluated": sum(1 for c in cross if c["model_arm"] == arm),
            "cross_truth_b_gate_fail_both_levels": sum(1 for c in cross if c["model_arm"] == arm and all(c["levels"][lv]["b_gate_verdict"] == "FAIL" for lv in ("level1", "level3"))),
        }
    # arm A: D_A(s) = NMSE_ped - NMSE_cone (predicted < 0); arm B: D_B(s) = NMSE_cone - NMSE_ped (predicted < 0)
    # DD(s) = (NMSE_ped - NMSE_cone)_A - (NMSE_ped - NMSE_cone)_B (predicted < 0); unit = seed pair
    pairs = {}
    for s in sorted({m["seed"] for m in models}):
        a, b = by_model.get(("A", s)), by_model.get(("B", s))
        if a and b and a["cross"] and b["cross"]:
            dA = a["identity_contrast_fixed_labels"]["nmse_ped_minus_cone_median_diff"]
            dB = b["identity_contrast_fixed_labels"]["nmse_ped_minus_cone_median_diff"]
            pairs[str(s)] = {"armA_nmse_ped_minus_cone": dA, "armB_nmse_cone_minus_ped": -dB, "armB_nmse_ped_minus_cone": dB,
                             "double_dissociation_DD": dA - dB, "predicted_sign": "negative",
                             "armA_source": a["sources"]["primary_b_gate"], "armB_source": b["sources"]["primary_b_gate"]}
    k = sum(1 for v in pairs.values() if v["double_dissociation_DD"] < 0)
    n = len(pairs)
    rep["paired_seed_sign_test"] = {
        "definition": "per paired seed s: DD_s = (NMSE_ped - NMSE_cone)[arm A, seed s] - (NMSE_ped - NMSE_cone)[arm B, seed s] on discovery-scene medians; "
                      "predicted DD_s < 0 (pedestrian better captured in A, cone better captured in B); exact one-sided sign test over seed pairs",
        "pairs": pairs, "n_pairs": n, "n_pairs_in_predicted_direction": k,
        "p_one_sided": sign_test_one_sided(k, n) if n else None,
        "min_attainable_p_with_3_pairs": 0.125,
        "armA_seeds_ped_better": sum(1 for v in pairs.values() if v["armA_nmse_ped_minus_cone"] < 0),
        "armB_seeds_cone_better": sum(1 for v in pairs.values() if v["armB_nmse_cone_minus_ped"] < 0),
    }
    allm = [m for m in models if m["cross"]]
    ks = sum(1 for m in allm if m["primary"]["interaction_nmse_median"] < m["cross"]["interaction_nmse_median"])
    rep["all_models_solid_better_than_ghost"] = {
        "definition": "count of trained models (both arms, all seeds) whose solid-identity interaction NMSE median is below its ghost-identity NMSE median; "
                      "models of the same seed share initialisation/clips across arms, so this is a descriptive tally, not an independent-unit test",
        "count": ks, "n_models": len(allm), "p_one_sided_if_independent": sign_test_one_sided(ks, len(allm)) if allm else None,
    }

    # ---- Table R3 substrate rows
    r3 = []
    for e in egg:
        if e.get("missing"):
            continue
        r3.append({"substrate": e["label"], "domain": "egg (released checkpoint, zero-shot)", "n_scenes": e["n_scenes"], "n_seeds": 1,
                   "identity": "on-path egg (H1)", "captured_fraction": {"mean": e["captured_fraction"], "sd": None, "n_seeds": 1},
                   "rec_h1_minus_h0": {"mean": e["rec_h1_minus_h0"], "sd": None, "n_seeds": 1, "p": [e["rec_h1_minus_h0_p"]]},
                   "flip_rate_h1_vs_h0": {"mean": e["flip_rate_h1_vs_h0"], "sd": None, "n_seeds": 1},
                   "b_gate": {"verdicts": [e["b_gate_verdict"]], "pass_count": int(e["b_gate_verdict"] != "FAIL"), "n": 1},
                   "sources": e["sources"]})
    for arm in "AB":
        for which in ("primary", "cross"):
            ms = [m for m in models if m["arm"] == arm and m[which]]
            if not ms:
                continue
            blks = [m[which] for m in ms]
            cf_m, cf_sd, nn = mean_sd([b["captured_fraction"] for b in blks])
            sp_m, sp_sd, _ = mean_sd([b["rec_h1_minus_h0"] for b in blks])
            fr_m, fr_sd, _ = mean_sd([b["flip_rate_h1_vs_h0"] for b in blks])
            verdicts = [b["b_gate_verdict"] for b in blks]
            r3.append({"substrate": f"Driving arm {arm} ({ARMS[arm]['description']})",
                       "domain": "driving (trained predictor, MetaDrive)", "n_scenes": blks[0]["n_scenes"], "n_seeds": nn,
                       "identity": f"{blks[0]['identity']} in lane ({'solid' if which == 'primary' else 'ghost'} identity, level {blks[0]['level']})",
                       "captured_fraction": {"mean": cf_m, "sd": cf_sd, "n_seeds": nn},
                       "rec_h1_minus_h0": {"mean": sp_m, "sd": sp_sd, "n_seeds": nn, "p": [b["rec_h1_minus_h0_p"] for b in blks]},
                       "flip_rate_h1_vs_h0": {"mean": fr_m, "sd": fr_sd, "n_seeds": nn},
                       "b_gate": {"verdicts": verdicts, "pass_count": sum(v != "FAIL" for v in verdicts), "n": len(verdicts)},
                       "sources": [m["sources"]["primary_b_gate" if which == "primary" else "cross_b_gate"] for m in ms]})

    return {"generated_by": rel(Path(__file__).resolve()), "artifact_root": rel(root),
            "b_gate_thresholds": B_GATE_THRESH,
            "definitions": {
                "captured_fraction": "max(0, 1 - median interaction NMSE) = model_captured_interaction_fraction (behavior_gate.py); signed version 1 - NMSE kept for cross-truth cells",
                "rec_h1_minus_h0": "hazard specificity of recovered fraction, primary token group (driving: hazard_corridor), scene-level sign-flip p (min attainable 2e-4 at 5000 flips)",
                "flip_rate_h1_vs_h0": "CEM brake/throttle ranking flip rate H1 vs H0 (T2a)",
                "interaction_nmse_ci_of_mean": "the pooled ci_low/ci_high in cf_gate_*.json bracket the scene MEAN (scene-clustered bootstrap); the point estimate used by the gate is the MEDIAN",
                "T1c_prime": "scene-level sign-flip test that the solid identity's interaction NMSE is below the ghost identity's (B-gate amendment T1c'); the T1c route was taken by every driving model because T1c already passed",
            },
            "egg_cells": egg, "driving_models": models, "cross_truth_cells": cross, "table_R2": r2,
            "seed_replication": rep, "table_R3": r3, "notes": NOTES}


# ------------------------------------------------------------------------------------------------------- markdown
def write_markdown(S: dict, out: Path) -> None:
    L = []
    L.append("# Results tables — driving cell (arms A/B) and substrate comparison\n")
    L.append(f"Generated by `{S['generated_by']}` from `{S['artifact_root']}/` JSON artifacts only. Every number is traceable via `drive_results_summary.json` "
             "(`sources` fields). Medians are the gate's point estimates; CIs stored in the artifacts bracket the scene *mean* and are labelled as such. "
             "p = scene-level sign-flip (minimum attainable 2e-4). Driving n = 54 discovery scenes per model (confirmation 66 sealed).\n")

    # R1
    L.append("## Table R1 — per trained driving model (discovery scenes)\n")
    L.append("| Model | Identity | NMSE median (mean CI) | Captured | Rec_h1−Rec_h0 (p) | vs H0′ | Flip rate | T1c′ p (solid<ghost) | B-gate | Retrieval |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in S["driving_models"]:
        for which in ("primary", "cross"):
            b = m[which]
            if not b:
                continue
            tag = "solid" if which == "primary" else "ghost"
            ci = b.get("interaction_nmse_ci_of_mean")
            ci_s = f" ({fmt(ci[0])}, {fmt(ci[1])})" if ci and ci[0] is not None else ""
            t1cp = m["primary"]["T1c_prime"] or {}
            t1cp_s = (f"{fmt_p(t1cp.get('p'))}{'' if t1cp.get('ok') else ' (n.s.)'}") if which == "primary" and t1cp else "—"
            retr = m.get("retrieval") or {}
            retr_s = "—"
            if retr:
                fails = [k for k, v in (retr.get("per_baseline") or {}).items() if v is False]
                retr_s = ("PASS" if retr.get("passed") else f"FAIL ({', '.join(fails)} {', '.join(fmt(retr['baselines'][k]) for k in fails)} > model {fmt(retr['model_mean_cosine'])})") if which == "primary" else "—"
            L.append(f"| arm {m['arm']} seed {m['seed']} | {b['identity']} ({tag}) | {fmt(b['interaction_nmse_median'])}{ci_s} | {fmt(b['captured_fraction'])} | "
                     f"{fmt(b['rec_h1_minus_h0'], signed=True)} ({fmt_p(b['rec_h1_minus_h0_p'])}) | {fmt(b['rec_h1_minus_h0prime'], signed=True)} | "
                     f"{fmt(b['flip_rate_h1_vs_h0'])} | {t1cp_s} | {b['b_gate_verdict']} | {retr_s} |")
    L.append("")
    L.append("Sources: `drive_eval/arm{X}_seed{s}/behavior_gate_discovery.json` (solid identity), `behavior_gate_discovery_level{3|1}.json` (ghost identity), "
             "`cf_gate_discovery*.json` (pooled CI of the mean), `retrieval_baseline.json`. Identity contrast with fixed labels (pedestrian − cone):\n")
    L.append("| Model | Rec_ped − Rec_cone (p) | planner-cost DiD ped − cone (p) | NMSE_ped − NMSE_cone | identity interaction cosine | predicted sign (rec/Δ, NMSE) |")
    L.append("|---|---|---|---|---|---|")
    for m in S["driving_models"]:
        ic = m["identity_contrast_fixed_labels"]
        pred = "+, −" if m["arm"] == "A" else "−, +"
        L.append(f"| arm {m['arm']} seed {m['seed']} | {fmt(ic['rec_ped_minus_cone_mean'], signed=True)} ({fmt_p(ic['rec_ped_minus_cone_p'])}) | "
                 f"{fmt(ic['planner_delta_ped_minus_cone_mean'], 0, signed=True)} ({fmt_p(ic['planner_delta_ped_minus_cone_p'])}) | "
                 f"{fmt(ic['nmse_ped_minus_cone_median_diff'], signed=True)} | {fmt(ic['identity_interaction_cosine_median'])} | {pred} |")
    L.append("\nSource: `identity_contrast` block of the level-1 gate file per model (`rec_h1_minus_h3`, `delta_h1_minus_h3`, `identity_interaction_cosine`); NMSE difference from the two per-level medians.\n")

    # R2
    L.append("## Table R2 — cross-truth 2×2×2 (model arm × truth arm × in-lane identity), discovery scenes\n")
    L.append("Own-truth cells (diagonal) from `drive_eval/`, cross-truth cells from `drive_eval_crosstruth/arm{X}_seed{s}_on_arm{Y}/crosstruth_summary.json`. "
             "Captured = 1 − NMSE median (negative = worse than predicting no interaction; clipped to 0 for the gate). Pooled = mean ± sd over seeds.\n")
    L.append("| Model arm | Truth arm | Identity in lane | Solid in truth? | Seed | NMSE median | Captured (signed) | Rec_h1−Rec_h0 (p) | Flip rate | B-gate |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, cell in S["table_R2"].items():
        for s, v in cell["per_seed"].items():
            L.append(f"| {cell['model_arm']} | {cell['truth_arm']} ({cell['truth_kind']}) | {cell['identity']} | {'yes' if cell['solid_in_truth'] else 'no (ghost)'} | {s} | "
                     f"{fmt(v['interaction_nmse_median'])} | {fmt(v['captured_fraction_signed'], signed=True)} | {fmt(v['rec_h1_minus_h0'], signed=True)} ({fmt_p(v['rec_h1_minus_h0_p'])}) | "
                     f"{fmt(v['flip_rate_h1_vs_h0'])} | {v['b_gate_verdict']} |")
        p = cell["pooled"]
        if p["n_seeds"]:
            L.append(f"| {cell['model_arm']} | {cell['truth_arm']} ({cell['truth_kind']}) | {cell['identity']} | {'yes' if cell['solid_in_truth'] else 'no (ghost)'} | **pooled** | "
                     f"{fmt_ms(p['interaction_nmse_median']['mean'], p['interaction_nmse_median']['sd'], p['n_seeds'])} | "
                     f"{fmt_ms(p['captured_fraction_signed']['mean'], p['captured_fraction_signed']['sd'], p['n_seeds'], signed=True)} | "
                     f"{fmt_ms(p['rec_h1_minus_h0']['mean'], p['rec_h1_minus_h0']['sd'], p['n_seeds'], signed=True)} | "
                     f"{fmt_ms(p['flip_rate_h1_vs_h0']['mean'], p['flip_rate_h1_vs_h0']['sd'], p['n_seeds'])} | {p['b_gate_pass_count']}/{p['n_seeds']} PASS |")
        else:
            L.append(f"| {cell['model_arm']} | {cell['truth_arm']} ({cell['truth_kind']}) | {cell['identity']} | {'yes' if cell['solid_in_truth'] else 'no (ghost)'} | — | not evaluated yet | | | | |")
    L.append("")

    # R3
    L.append("## Table R3 — substrate comparison (B-gate quantities)\n")
    L.append("| Substrate | Identity / cell | n scenes | seeds | Captured fraction | Specificity Rec_h1−Rec_h0 | Flip rate | B-gate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in S["table_R3"]:
        cf, sp, fr, bg = r["captured_fraction"], r["rec_h1_minus_h0"], r["flip_rate_h1_vs_h0"], r["b_gate"]
        ps = ", ".join(fmt_p(p) for p in sp["p"])
        L.append(f"| {r['substrate']} | {r['identity']} | {r['n_scenes']} | {r['n_seeds']} | {fmt_ms(cf['mean'], cf['sd'], cf['n_seeds'])} | "
                 f"{fmt_ms(sp['mean'], sp['sd'], sp['n_seeds'], signed=True)} (p {ps}) | {fmt_ms(fr['mean'], fr['sd'], fr['n_seeds'])} | "
                 f"{bg['pass_count']}/{bg['n']} {'PASS' if bg['pass_count'] else 'FAIL'} ({', '.join(bg['verdicts'])}) |")
    L.append("\nEgg sources: `heldout_v1_merged_w1/behavior_gate_cf_gate_sonar_all.json`, `heldout_v1_merged_w2/behavior_gate_cf_gate_discovery.json`, "
             "`heldout_v1_merged_w2/behavior_gate_cf_gate_all.json`, `heldout_v1_merged_w2_vj2ac/behavior_gate_vj2ac.json`. Driving: as Table R1.\n")

    # replication
    rep = S["seed_replication"]
    L.append("## Seed-level replication and paired-seed sign test\n")
    for arm, r in rep["per_arm"].items():
        L.append(f"- **Arm {arm}** ({ARMS[arm]['description']}): seeds evaluated {r['seeds']}; solid-identity B-gate PASS {r['primary_b_gate_pass']}/{r['n_seeds_evaluated']} "
                 f"(PASS_PLANNER {r['primary_b_gate_pass_planner']}); ghost-identity B-gate also PASS {r['cross_identity_b_gate_pass']}/{r['n_seeds_evaluated']}; "
                 f"solid NMSE < ghost NMSE in {r['solid_nmse_below_ghost_nmse']}/{r['n_seeds_evaluated']} seeds; scene-level T1c′ p < 0.05 in {r['T1c_prime_scene_level_p_below_0.05']}/{r['n_seeds_evaluated']} "
                 f"(p = {', '.join(f'{s}: {fmt_p(p)}' for s, p in r['T1c_prime_p_values'].items())}); cross-truth cells evaluated {r['cross_truth_cells_evaluated']}, "
                 f"B-gate FAIL at both identities in {r['cross_truth_b_gate_fail_both_levels']}/{r['cross_truth_cells_evaluated']}.")
    st = rep["paired_seed_sign_test"]
    L.append(f"- **Paired-seed sign test** ({st['definition']}):")
    for s, v in st["pairs"].items():
        L.append(f"  - seed {s}: arm A NMSE_ped − NMSE_cone = {fmt(v['armA_nmse_ped_minus_cone'], signed=True)}; arm B NMSE_cone − NMSE_ped = {fmt(v['armB_nmse_cone_minus_ped'], signed=True)}; DD = {fmt(v['double_dissociation_DD'], signed=True)}")
    L.append(f"  - {st['n_pairs_in_predicted_direction']}/{st['n_pairs']} seed pairs in the predicted direction; exact one-sided p = {fmt_p(st['p_one_sided'])} "
             f"(resolution 1/2^{st['n_pairs']}; the preregistered 3 pairs give at best 0.125). Arm A pedestrian better in {st['armA_seeds_ped_better']}/{st['n_pairs']}, arm B cone better in {st['armB_seeds_cone_better']}/{st['n_pairs']} paired seeds.")
    am = rep["all_models_solid_better_than_ghost"]
    L.append(f"- Descriptive tally over all trained models: solid-identity NMSE below ghost-identity NMSE in {am['count']}/{am['n_models']} models "
             f"(binomial p {fmt_p(am['p_one_sided_if_independent'])} only if models were independent; they are not, seeds are shared across arms).\n")

    if S["notes"]:
        L.append("## Notes and inconsistencies detected while aggregating\n")
        for n in S["notes"]:
            L.append(f"- {n}")
        L.append("")
    out.write_text("\n".join(L))


# -------------------------------------------------------------------------------------------------------- figures
COL = {"JEPA-WM": "#0072B2", "V-JEPA 2-AC": "#E69F00", "A": "#009E73", "B": "#CC79A7"}
TEXT, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#d9d9d9"


def make_figures(S: dict, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    figdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Fig 2
    items = []  # (label, group, solid value dict, ghost value dict or None)
    for e in S["egg_cells"]:
        if e.get("missing"):
            continue
        grp = "V-JEPA 2-AC" if "V-JEPA" in e["label"] else "JEPA-WM"
        items.append({"label": e["label"].replace("JEPA-WM ", "").replace("V-JEPA 2-AC ", ""), "group": grp,
                      "cf": e["captured_fraction"], "fr": e["flip_rate_h1_vs_h0"], "cf_ghost": None, "fr_ghost": None, "n": e["n_scenes"], "src": e["sources"]["b_gate"]})
    for m in S["driving_models"]:
        c = m["cross"] or {}
        items.append({"label": f"seed {m['seed']}", "group": m["arm"], "cf": m["primary"]["captured_fraction"], "fr": m["primary"]["flip_rate_h1_vs_h0"],
                      "cf_ghost": c.get("captured_fraction"), "fr_ghost": c.get("flip_rate_h1_vs_h0"), "n": m["n_discovery_scenes"], "src": m["sources"]["primary_b_gate"]})
    groups = ["JEPA-WM", "V-JEPA 2-AC", "A", "B"]
    gname = {"JEPA-WM": "JEPA-WM DROID\n(egg, zero-shot)", "V-JEPA 2-AC": "V-JEPA 2-AC\n(egg, zero-shot)", "A": "Driving arm A\n(ped solid / cone ghost)", "B": "Driving arm B\n(cone solid / ped ghost)"}
    xs, gap, x = [], 0.9, 0.0
    last = None
    for it in items:
        if last is not None and it["group"] != last:
            x += gap
        xs.append(x)
        x += 1.0
        last = it["group"]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, gridspec_kw={"hspace": 0.18})
    w = 0.34
    print("Fig 2 values drawn:")
    for ax, key, ylab, thr in ((axes[0], "cf", "captured interaction fraction\n(1 − interaction NMSE median)", 0.25),
                               (axes[1], "fr", "planner ranking flip rate\n(H1 vs H0, CEM)", 0.25)):
        for it, xx in zip(items, xs):
            col = COL[it["group"]]
            ghost = it[key + "_ghost"]
            if ghost is None:
                ax.bar(xx, it[key], width=w * 1.6, color=col, edgecolor=col, linewidth=0.8)
                ax.text(xx, it[key] + 0.02, f"{it[key]:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
            else:
                ax.bar(xx - w / 2 - 0.01, it[key], width=w, color=col, edgecolor=col, linewidth=0.8)
                ax.bar(xx + w / 2 + 0.01, ghost, width=w, facecolor="white", edgecolor=col, linewidth=1.0, hatch="////")
                if abs(ghost - it[key]) < 0.005:  # equal values: one label over both bars
                    ax.text(xx, it[key] + 0.02, f"{it[key]:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
                else:
                    ax.text(xx - w / 2 - 0.01, it[key] + 0.02, f"{it[key]:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
                    ax.text(xx + w / 2 + 0.01, ghost + 0.02, f"{ghost:.2f}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
            print(f"  {key} {it['group']} {it['label']}: solid={it[key]:.4f} ghost={ghost} n={it['n']} src={it['src']}")
        ax.axhline(thr, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.text(xs[0] - 0.55, thr + 0.015, f"B-gate {thr:.2f}", ha="left", va="bottom", fontsize=7, color=MUTED)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([it["label"] for it in items], fontsize=7)
    # group captions under the axis
    for grp in groups:
        gx = [xx for it, xx in zip(items, xs) if it["group"] == grp]
        if gx:
            axes[1].text(sum(gx) / len(gx), -0.30, gname[grp], ha="center", va="top", fontsize=7, color=COL[grp], transform=axes[1].get_xaxis_transform())
    axes[0].set_title("(a) captured interaction fraction per cell (B-gate tier 1)", loc="left", fontsize=8.5, color=TEXT)
    axes[1].set_title("(b) planner flip rate per cell (B-gate tier 2)", loc="left", fontsize=8.5, color=TEXT)
    handles = [Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f", label="egg cell / driving solid identity (own-arm truth)"),
               Patch(facecolor="white", edgecolor="#7f7f7f", hatch="////", label="driving ghost identity (visual pass-through interaction)")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=6.5, frameon=False)
    fig.text(0.01, 0.005, "egg: n = 21 / 34 / 66 / 34 scenes; driving: n = 54 discovery scenes per model. Sources: behavior_gate_*.json per cell (see drive_results_summary.json).",
             fontsize=6, color=MUTED)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig2_bgate_substrates.{ext}", dpi=300)
    plt.close(fig)

    # ---------------- Fig R2: per seed, rows = model arm, columns = truth arm x identity
    seeds = sorted({m["seed"] for m in S["driving_models"]} | {c["model_seed"] for c in S["cross_truth_cells"]})
    R2 = S["table_R2"]
    cmap = matplotlib.colormaps["Blues"]
    fig, axes = plt.subplots(1, len(seeds), figsize=(2.6 * len(seeds) + 0.6, 2.9), squeeze=False)
    print("Fig R2 values drawn (captured fraction, signed):")
    cols = [("A", 1), ("A", 3), ("B", 1), ("B", 3)]  # truth arm, level
    for ax, s in zip(axes[0], seeds):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        for i, ma in enumerate("AB"):
            y = 1 - i
            for j, (ta, lv) in enumerate(cols):
                cell = R2[f"model_arm{ma}|truth_arm{ta}|level{lv}"]
                v = cell["per_seed"].get(str(s))
                own = ma == ta
                if v is None:
                    ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor="#f2f2f2", edgecolor="white", linewidth=2, hatch="..."))
                    ax.text(j + 0.5, y + 0.5, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
                    print(f"  seed {s} model {ma} truth {ta} {LEVEL_NAME[lv]}: not evaluated")
                    continue
                val = v["captured_fraction_signed"]
                shade = cmap(0.15 + 0.75 * max(0.0, min(1.0, val)))
                ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=shade, edgecolor="white", linewidth=2))
                txtcol = "white" if val > 0.55 else TEXT
                ax.text(j + 0.5, y + 0.58, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=txtcol, fontweight="bold")
                ax.text(j + 0.5, y + 0.28, v["b_gate_verdict"].replace("PASS_PLANNER", "PASS+pl."), ha="center", va="center", fontsize=5.5, color=txtcol)
                if own:
                    ax.add_patch(plt.Rectangle((j + 0.04, y + 0.04), 0.92, 0.92, fill=False, edgecolor=COL[ma], linewidth=1.6))
                print(f"  seed {s} model {ma} truth {ta} {LEVEL_NAME[lv]} ({'own' if own else 'cross'}): {val:+.4f} {v['b_gate_verdict']} src={v['source']}")
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(["ped", "cone", "ped", "cone"], fontsize=7)
        ax.set_yticks([1.5, 0.5])
        ax.set_yticklabels(["model\narm A", "model\narm B"], fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(1.0, 2.08, "truth: arm A\n(ped solid)", ha="center", va="bottom", fontsize=6.5, color=COL["A"])
        ax.text(3.0, 2.08, "truth: arm B\n(cone solid)", ha="center", va="bottom", fontsize=6.5, color=COL["B"])
        ax.set_title(f"seed {s}", fontsize=8.5, y=1.22, color=TEXT)
    fig.suptitle("Cross-truth double dissociation: captured interaction fraction (1 − NMSE median) per in-lane identity;\n"
                 "outlined cells = own-truth (diagonal), plain = cross-truth; negative = worse than predicting no interaction",
                 fontsize=7.5, color=TEXT, y=1.02)
    fig.text(0.01, 0.01, "n = 54 discovery scenes. Sources: drive_eval/*/behavior_gate_discovery*.json (own), drive_eval_crosstruth/*/crosstruth_summary.json (cross).",
             fontsize=6, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.72, bottom=0.14, wspace=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"figR2_crosstruth_dissociation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ART)
    ap.add_argument("--out-json", type=Path, default=ART / "drive_results_summary.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "paper" / "results_tables.md")
    ap.add_argument("--figdir", type=Path, default=REPO / "paper" / "figures")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    S = build(a.root.resolve())
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(S, fh, indent=1)
    write_markdown(S, a.out_md)
    print(f"wrote {rel(a.out_json)} and {rel(a.out_md)}; {len(S['driving_models'])} driving models, {len(S['cross_truth_cells'])} cross-truth cells, {len(S['notes'])} notes")
    if not a.no_figures:
        try:
            make_figures(S, a.figdir)
            print(f"wrote figures to {rel(a.figdir)}/")
        except ImportError as e:
            print(f"figures skipped: {e} (use an interpreter with matplotlib, e.g. /usr/local/bin/python3.9)", file=sys.stderr)


if __name__ == "__main__":
    main()
