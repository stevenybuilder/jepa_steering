#!/usr/bin/env python3
"""Behavioural-target gate (B-gate): is there an unambiguous, model-native hazard-conditional behaviour to explain?

Preregistered 2026-09-02 (see experiment_design.md, "Behavioural-target gate"). Runs on a counterfactual-validity gate
JSON (discovery scenes) and decides whether mechanism arms (localization, patching, geometry, Jacobian, modulation)
are licensed. All thresholds are fixed here; nothing is tuned on results.

Tier 1 (required, all three):
  T1a  n_scenes >= 16 (the gate's own rule)
  T1b  interaction NMSE median <= 0.75  -> the model reproduces >= 25% of the true hazard x action interaction energy
  T1c  Rec_h1 - Rec_h0 > 0 with sign-flip p < 0.05 (hazard-specific recovery; the H0' contrast must not reverse the sign)
Tier 2 (reported; needed for a planner-level claim):
  T2a  CEM ranking flip rate (H1 vs H0) >= 0.25   or
  T2b  normalised energy gap H1 vs H0 < 0 with sign-flip p < 0.05 (H1 prediction sits closer to the contact future; planner_currency sign convention)

T1c' (identity contrast; added 2026-09-02 ~19:50 UTC before any full driving model, motivated by the 2-minute dry-run
  pilot: with an immovable hazard the collision future is a larger displacement, so recovered-fraction specificity
  can be negative while the interaction is reproduced): with --cross-gate (the same gate computed with the OTHER
  in-lane identity as primary), per discovery scene the primary (solid) identity's interaction NMSE is lower than
  the cross (ghost) identity's, scene-level sign-flip p < 0.05. Tier 1 = T1a AND T1b AND (T1c OR T1c').
  Without --cross-gate the rule is exactly the original (egg verdicts unchanged).

Verdict: PASS (tier 1) / PASS_PLANNER (tier 1 + tier 2) / FAIL. On FAIL the runner stops before mechanism arms and the
loop's finding is "no behavioural target on this stimulus/model", to be fixed by changing stimulus or model, not by
adding interpretability methods.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THRESH = {"min_scenes": 16, "max_interaction_nmse_median": 0.75, "spec_p": 0.05, "min_flip_rate": 0.25, "energy_p": 0.05}


def primary_pooled(gate: dict) -> tuple[str, dict]:
    """Planner-currency pooled block of the gate's primary token group (egg: ``egg_gripper``; driving:
    ``hazard_corridor``, also exposed under the ``egg_gripper`` alias by the gate)."""

    planner = gate.get("planner_currency", {})
    tg = planner.get("token_groups", {})
    name = planner.get("primary_group", "egg_gripper")
    block = tg.get(name) or tg.get("egg_gripper") or {}
    return name, (block.get("pooled") or {})


def _f(x) -> float:
    """None-tolerant float (run_drive_arms.sh compatibility, 2026-09-03): driving gate JSONs carry ``"mean": null`` for a
    block that could not be computed (e.g. the energy gap of the cross-identity level); a missing value is treated exactly
    like an absent key (NaN -> the corresponding criterion is False). Thresholds unchanged."""
    return float("nan") if x is None else float(x)


def _sign_flip_p(d, n_mc: int = 20000, seed: int = 0) -> float:
    """One-sided (mean < 0) scene-level sign-flip p for paired differences d (exact if n <= 16, else Monte Carlo). Stdlib only."""
    import itertools
    import random
    d = [float(x) for x in d]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = sum(d) / n
    if n <= 16:
        cnt = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            if sum(x * s for x, s in zip(d, signs)) / n <= obs + 1e-12:
                cnt += 1
        return cnt / 2 ** n
    rng = random.Random(seed)
    cnt = 0
    for _ in range(n_mc):
        if sum(x * rng.choice((-1.0, 1.0)) for x in d) / n <= obs + 1e-12:
            cnt += 1
    return (cnt + 1) / (n_mc + 1)


def identity_contrast(gate: dict, cross: dict) -> dict:
    """T1c': per-scene interaction NMSE, primary (solid identity) minus cross (ghost identity)."""
    import math
    from statistics import median
    a = gate.get("per_scene", {})
    b = cross.get("per_scene", {})
    common = sorted(k for k in a if k in b and "interaction_nmse" in a[k] and "interaction_nmse" in b[k])
    pairs = [(float(a[k]["interaction_nmse"]), float(b[k]["interaction_nmse"])) for k in common]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    d = [x - y for x, y in pairs]
    p = _sign_flip_p(d) if d else float("nan")
    mean = sum(d) / len(d) if d else float("nan")
    return {
        "n": len(d),
        "nmse_primary_median": median([x for x, _ in pairs]) if pairs else float("nan"),
        "nmse_cross_median": median([y for _, y in pairs]) if pairs else float("nan"),
        "diff_mean": mean,
        "p": p,
        "ok": bool(len(d) >= THRESH["min_scenes"] and mean < 0 and p < THRESH["spec_p"]),
    }


def evaluate(gate: dict, cross: dict | None = None) -> dict:
    p = gate["pooled"]
    n = int(gate["n_scenes"])
    nmse = _f(p["interaction_nmse"]["median"])
    group_name, pc = primary_pooled(gate)
    spec = pc.get("rec_h1_minus_h0", {})
    spec_null = pc.get("rec_h1_minus_h0prime", {})
    spec_mean = _f(spec.get("mean"))
    spec_p = _f(spec.get("sign_flip_p"))
    null_mean = _f(spec_null.get("mean")) if spec_null else float("nan")
    eg = pc.get("energy_gap", {}).get("normalized_h1_vs_h0", {})
    rk = pc.get("ranking", pc.get("planner_ranking", {})) or {}
    flip = _f(rk.get("flip_rate_h1_vs_h0"))
    e_mean = _f(eg.get("mean"))
    e_p = _f(eg.get("sign_flip_p"))
    t1a = n >= THRESH["min_scenes"]
    t1b = nmse <= THRESH["max_interaction_nmse_median"]
    t1c = spec_mean > 0 and spec_p < THRESH["spec_p"] and not (null_mean == null_mean and null_mean < 0 and spec_mean > 0 and abs(null_mean) > spec_mean)
    t2a = flip == flip and flip >= THRESH["min_flip_rate"]
    # sign convention (planner_currency.py): normalised energy gap H1 vs H0 < 0 == the H1 prediction sits CLOSER to the contact future.
    # Corrected 2026-09-03 02:35 UTC (was > 0; tier-2 verdicts unaffected because T2a carried them; egg T2b stays false).
    t2b = e_mean < 0 and e_p < THRESH["energy_p"]
    t1c_prime = identity_contrast(gate, cross) if cross is not None else None
    t1c_any = t1c or bool(t1c_prime and t1c_prime["ok"])
    tier1 = t1a and t1b and t1c_any
    verdict = "PASS_PLANNER" if tier1 and (t2a or t2b) else ("PASS" if tier1 else "FAIL")
    extra = {}
    if gate.get("domain", "egg") != "egg":  # driving: pass the domain, group and identity contrast through (reported, never thresholded)
        ident = gate.get("identity_contrast", {})
        extra = {
            "domain": gate.get("domain"), "primary_hazard_level": gate.get("primary_hazard_level", 1), "token_group": group_name,
            "identity_contrast": {k: ident.get(k) for k in ("status", "rec_h1_minus_h3", "delta_h1_minus_h3", "flip_rate_h1_vs_h3", "identity_interaction_cosine") if k in ident},
        }
    return {
        "verdict": verdict, "thresholds": THRESH, "n_scenes": n, **extra,
        "T1a_enough_scenes": t1a, "T1b_interaction_nmse_median": nmse, "T1b_ok": t1b,
        "T1c_rec_h1_minus_h0": spec_mean, "T1c_p": spec_p, "T1c_rec_h1_minus_h0prime": null_mean, "T1c_ok": t1c,
        "T1c_prime": t1c_prime, "t1c_route": ("T1c" if t1c else ("T1c_prime" if (t1c_prime and t1c_prime["ok"]) else None)),
        "T2a_flip_rate_h1_vs_h0": flip, "T2a_ok": t2a, "T2b_energy_gap_norm_h1_vs_h0": e_mean, "T2b_p": e_p, "T2b_ok": t2b,
        "model_captured_interaction_fraction": max(0.0, 1.0 - nmse),
        "statement": ("Mechanism arms licensed." if tier1 else
                      "No unambiguous hazard-conditional behaviour: mechanism arms NOT licensed; change stimulus/model, not methods."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cross-gate", type=Path, default=None, help="gate JSON with the other in-lane identity as primary (enables T1c')")
    a = ap.parse_args()
    cross = json.load(open(a.cross_gate)) if a.cross_gate else None
    res = evaluate(json.load(open(a.gate)), cross)
    res["cross_gate_file"] = str(a.cross_gate) if a.cross_gate else None
    res["gate_file"] = str(a.gate)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["verdict"] != "FAIL" else 3)


if __name__ == "__main__":
    main()
