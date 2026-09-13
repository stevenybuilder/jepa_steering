"""Run every mechanism-analysis script on one results/freeze/analysis triple and write OUT/INDEX.md.

  .venv/bin/python analysis/mechanism/run_all.py --results R --freeze F --analysis A --out OUT

Each script runs as a subprocess; a failure is recorded in INDEX.md and does not stop the rest.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["regime_report", "specificity_pathway", "scenario_heterogeneity",
           "forecast_decision_outcome", "representation_geometry", "planner_dynamics"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "freeze", "analysis", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    status = {}
    for s in SCRIPTS:
        out = a.out / s
        cmd = [sys.executable, str(HERE / f"{s}.py"), "--results", str(a.results), "--freeze", str(a.freeze),
               "--analysis", str(a.analysis), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (a.out / f"{s}.log").write_text(r.stdout + r.stderr)
        status[s] = r.returncode
        print(f"{s:28s} {'OK' if r.returncode == 0 else 'FAIL rc=' + str(r.returncode)}", flush=True)
    regime = "unknown"
    rj = a.out / "regime_report" / "regime_report.json"
    if rj.exists():
        try:
            j = json.loads(rj.read_text())
            regime = j.get("overall_regime") or j.get("regime", {}).get("overall") or json.dumps(j.get("regime", "unknown"))[:200]
        except Exception as e:  # noqa: BLE001
            regime = f"unreadable ({e})"
    lines = [f"# Mechanism analyses — overall regime: **{regime}**", "",
             "Read next: `docs/MECHANISM_HYPOTHESES.md` (decision table §D before writing any claim).", ""]
    for s in SCRIPTS:
        d = a.out / s
        lines.append(f"## {s} — {'OK' if status[s] == 0 else 'FAILED (see ' + s + '.log)'}")
        if d.exists():
            for f in sorted(d.iterdir()):
                lines.append(f"- [{f.name}]({s}/{f.name})")
        lines.append("")
    (a.out / "INDEX.md").write_text("\n".join(lines))
    print("INDEX:", a.out / "INDEX.md")
    return 0 if all(v == 0 for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
