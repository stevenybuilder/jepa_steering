#!/usr/bin/env python3
"""Markdown table of per-cell continuous margin changes (steered - unsteered, scene-bootstrap CI) for main vs every control at
every beta, from a steered_ranking.json (numbers from the file only)."""
import json
import sys

d = json.load(open(sys.argv[1]))
goal = sys.argv[2] if len(sys.argv) > 2 else d["primary_goal"]
res = d["results"][goal]
u = res["unsteered"]
print(f"Goal `{goal}`, n_scenes = {d['n_scenes']}, unsteered safe-choice: " + ", ".join(f"{k} {v['mean']:.2f}" for k, v in u["safe_choice"].items())
      + "; unsteered margin: " + ", ".join(f"{k} {v['mean']:+.3f}" for k, v in u["margin"].items()) + ".")
print()
print("| family | mode | beta | kind | h1 Δmargin | h3 Δmargin | h0 Δmargin | h0′ Δmargin | DiD margin (solid−h0) | DiD CI excl. 0 (exp. sign) | safe-choice DiD | edit RMS |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")


def cell(e, lvl):
    v = e["vs_unsteered"].get(lvl)
    if not v:
        return "-"
    m = v["margin_change"]
    return f"{m['mean']:+.4f} [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"


rows = [(e["config"]["family"], e["config"]["mode"], e["config"]["beta"], e["config"]["kind"], name, e) for name, e in res.items() if e["config"]["kind"] != "unsteered"]
order = {"main": 0, "sham": 1}
rows.sort(key=lambda r: (r[0], r[2], order.get(r[3], 2), r[3]))
for fam, mode, beta, kind, name, e in rows:
    dm = e["did"][e["did_primary"]["contrast"]]["margin"]
    ds = e["did_primary"]
    rms = (e.get("activation_edit_relative_rms") or {}).get("mean")
    print(f"| {fam.split('|')[0]}:{fam.split('|')[-1]} | {mode} | {beta:g} | {kind} | {cell(e, 'h1')} | {cell(e, 'h3')} | {cell(e, 'h0')} | {cell(e, 'h0prime')} | "
          f"{dm['mean']:+.4f} [{dm['ci_low']:+.4f}, {dm['ci_high']:+.4f}] p={dm['sign_flip_p']:.3g} | {int(bool(e['did_margin_ci_excludes_zero_expected']))} | "
          f"{ds.get('mean', 0):+.2f} [{ds.get('ci_low', 0):+.2f}, {ds.get('ci_high', 0):+.2f}] | {'-' if rms is None else f'{rms:.3f}'} |")
