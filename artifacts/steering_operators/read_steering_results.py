#!/usr/bin/env python3
"""Read the steering-operator JSONs on box 2 and print a compact, re-derived summary (numbers from the files only)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/cgs-pilot/artifacts/steering_operators")
r = lambda x: None if x is None else round(x, 3)  # noqa: E731

for arm in ("A", "B"):
    print(f"\n=================== ARM {arm} ===================")
    f = root / f"arm{arm}_seed0" / "min_distortion" / "causal_metric_fit.json"
    if f.exists():
        d = json.load(open(f))
        print(f"[min_distortion fit] n_scenes={d['n_scenes']} sites={len(d['sites'])} runtime={d['runtime_s']:.0f}s file={f}")
        for e in d["entries"]:
            if e.get("status") != "ok":
                print("  ", e.get("site_id"), e.get("group"), e.get("status")); continue
            tw, lo, c = e["twin_comparison"], e.get("loso") or {}, e["variants"]["main"]["constraints"]
            nu = e["nuisance"]
            print(f"   {e['site_id']:>14s} {e['group']:>15s} n={e['n_scenes']} cos(main,eucl)={r(tw['cos_main_vs_euclid_top'])} cos(main,mrel)={r(tw['cos_main_top_vs_mean_rel'])} cos(eucl,mrel)={r(tw['cos_euclid_top_vs_mean_rel'])}"
                  f" | LOSO main {r(lo.get('main',{}).get('mean'))} [{r(lo.get('main',{}).get('ci_low'))},{r(lo.get('main',{}).get('ci_high'))}] p={lo.get('main',{}).get('sign_flip_p')} eucl {r(lo.get('euclid',{}).get('mean'))} p={lo.get('euclid',{}).get('sign_flip_p')}"
                  f" | w_B={r(c['w_B'])} w_S={r(c['w_S'])} Dact={c['worst_case_D_action']:.2e}/tau2={c['tau2']:.2e} mahal_inc={r(c['worst_case_mahal_increment'])} | eps0 main={r(e['variants']['main']['eps0'])} eucl={r(e['variants']['euclid']['eps0'])}"
                  f" | nuis effrank={r(nu['effective_rank'])} lam={nu['lambda']:.3g} rows={ {g: v['n_rows'] for g, v in nu['groups'].items()} } factors_used={e['factors'].get('used')}")
    else:
        print("[min_distortion fit] missing", f)
    fv = root / "v09" / f"arm{arm}_seed0" / "min_distortion" / "causal_metric_fit.json"
    if fv.exists():
        d = json.load(open(fv))
        ok = [e for e in d["entries"] if e.get("status") == "ok"]
        same = sum(1 for e in ok if e["twin_comparison"]["cos_main_vs_euclid_top"] > 0.999)
        print(f"[v0.9 min_distortion fit] n_scenes={d['n_scenes']} cells_ok={len(ok)} twins_coincide={same} factors_used={[e['factors'].get('used') for e in ok][:1]} file={fv}")
        for e in ok:
            tw, lo = e["twin_comparison"], e.get("loso") or {}
            print(f"   {e['site_id']:>14s} {e['group']:>15s} cos(main,eucl)={r(tw['cos_main_vs_euclid_top'])} cos(main,mrel)={r(tw['cos_main_top_vs_mean_rel'])} cos(eucl,mrel)={r(tw['cos_euclid_top_vs_mean_rel'])} LOSO main {r(lo.get('main',{}).get('mean'))} eucl {r(lo.get('euclid',{}).get('mean'))} factors={e['factors']}")
    f = root / f"arm{arm}_seed0" / "modulation" / "modulation_operator_fit.json"
    if f.exists():
        d = json.load(open(f))
        print(f"[modulation fit] n_scenes={d['n_scenes']} blocks={d['blocks']} runtime={d['runtime_s']:.0f}s file={f}")
        for site, s in d["per_block"].items():
            g = s["gate"]
            top = sorted(((sl, e["point"], e.get("p_maxt_fwer"), e.get("ci_low"), e.get("ci_high")) for sl, e in s["selection"].items()), key=lambda t: -abs(t[1] or 0))[:3]
            print(f"   {site}: gate solid={r(g['solid'])} ghost={r(g['ghost'])} h0={r(g['h0'])} h0'={r(g['h0prime'])} d'={r(g['separation_d'])} | top slices " + "; ".join(f"{sl} {r(v)} [{r(lo)},{r(hi)}] pT={r(p)}" for sl, v, p, lo, hi in top))
    else:
        print("[modulation fit] missing", f)
    for kind, f in (("min_distortion", root / f"arm{arm}_seed0" / "steer_min_distortion" / "discovery" / "steered_ranking.json"),
                    ("min_distortion_full", root / f"arm{arm}_seed0" / "steer_min_distortion_full" / "discovery" / "steered_ranking.json"),
                    ("modulation", root / f"arm{arm}_seed0" / "steer_modulation" / "discovery" / "steered_ranking.json"),
                    ("v09_min_distortion", root / "v09" / f"arm{arm}_seed0" / "steer_min_distortion" / "discovery" / "steered_ranking.json")):
        if not f.exists():
            print(f"[steer {kind}] missing", f); continue
        d = json.load(open(f))
        cal = d.get("calibration") or {}
        print(f"[steer {kind}] n_scenes={d['n_scenes']} seeds={d['seeds']} configs={len(d['results']['progress'])} runtime={d['runtime_s']:.0f}s file={f}")
        res = d["results"]["progress"]
        u = res["unsteered"]["safe_choice"]
        print("   unsteered safe-choice (progress): " + " ".join(f"{k}={r(v['mean'])}" for k, v in u.items()))
        for fam, c in cal.get("families", {}).items():
            print(f"   family {fam}: calibrated beta={c['beta']}")
            for cand in c["candidates"]:
                e = res[cand["config"]]
                ctrl = {k: (r(res[n]["did_primary"].get("mean")) if (n := next((nn for nn, ee in res.items() if ee["config"]["family"] == fam and ee["config"]["kind"] == k and (k == "sham" or abs(ee["config"]["beta"] - cand["beta"]) < 1e-9)), None)) else None) for k in e.get("controls_fail", {})}
                hf = e["hazard_free_checks"]
                ex = {k: r(v["mean"]) for k, v in (e.get("extra") or {}).items()}
                print(f"      beta={cand['beta']}: DiD={r(cand['did_mean'])} [{r(cand['ci_low'])},{r(cand['ci_high'])}] p={cand['sign_flip_p']} selective={cand['selective']} controls_fail={e.get('all_controls_fail')} "
                      f"nontarget_eq={e['nontarget_equivalent']['all']} hf_within={hf['all_within_margin']} editRMS={r((e.get('activation_edit_relative_rms') or {}).get('mean'))} ctrlDiD={ctrl} extra={ex} "
                      f"safe: " + " ".join(f"{k}={r(v['mean'])}" for k, v in e["safe_choice"].items()))
