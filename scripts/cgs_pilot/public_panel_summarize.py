#!/usr/bin/env python3
"""Panel P screen summary (Label-first principle, amendment 3).

Reads ``screen/<env>_<model>/seed*/episodes.jsonl`` written by ``public_panel_eval.py``, merges
seeds into ``screen/<env>_<model>/episodes.jsonl``, writes ``summary.json`` per cell and the
cross-cell ``c0_table.json`` / ``c0_table.md``.

Frozen eligibility (amendment 3): 20-80 % success, and >= 3 successes and >= 3 failures per
15-episode fit cell; a model advances with >= 2 eligible benchmarks. The fit cell is the first 15
episodes of seed 1 in the evaluator's own episode order (the split stage 2 will use), and the
minimum class count over every consecutive 15-episode block is reported as well.

Every number comes from the jsonl; the Table 2 reference values are copied from the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TABLE2 = {
    ("mz", "jepa-wm"): 83.9, ("mz", "dino-wm"): 81.6,
    ("wall", "jepa-wm"): 78.8, ("wall", "dino-wm"): 64.1,
    ("pt", "jepa-wm"): 70.2, ("pt", "dino-wm"): 66.0,
    ("mw-reach", "jepa-wm"): 58.2, ("mw-reach", "dino-wm"): 44.8,
    ("mw-reach-wall", "jepa-wm"): 41.6, ("mw-reach-wall", "dino-wm"): 35.1,
}
ENV_NAMES = {"mz": "PointMaze", "wall": "Wall", "pt": "Push-T", "mw": "MetaWorld"}
SECONDARY = ("ep_reward", "end_distance", "state_dist", "success_dist", "n_env_steps", "wall_s")
INFO_KEYS = ("max_coverage", "final_coverage", "near_object", "obj_goal_dist")


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def block_counts(succ: list[int], block: int = 15) -> list[dict]:
    out = []
    for i in range(0, len(succ) - block + 1, block):
        b = succ[i : i + block]
        out.append({"start": i, "n": len(b), "successes": int(sum(b)), "failures": int(len(b) - sum(b))})
    return out


def summarize_cell(cell_dir: Path, ref_key_env: str | None = None) -> dict | None:
    seed_dirs = sorted(d for d in cell_dir.glob("seed*") if (d / "episodes.jsonl").exists())
    if not seed_dirs:
        return None
    rows = []
    per_seed = {}
    for d in seed_dirs:
        r = load_jsonl(d / "episodes.jsonl")
        complete = (d / "DONE.json").exists()
        for x in r:
            x["seed_dir"] = d.name
            x["seed_complete"] = complete
        rows.extend(r)
        s = np.array([x["success"] for x in r], dtype=float)
        per_seed[d.name] = {
            "n": int(len(s)),
            "successes": int(s.sum()),
            "rate": float(s.mean()) if len(s) else None,
            "complete": complete,
            "wilson95": wilson(int(s.sum()), len(s)),
        }
    with open(cell_dir / "episodes.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    env, model = rows[0]["env"], rows[0]["model"]
    task = rows[0].get("task_cfg") or rows[0].get("task")
    succ = np.array([x["success"] for x in rows], dtype=float)
    n = int(len(succ))
    k = int(succ.sum())
    rate = k / n if n else float("nan")
    ci = bootstrap_ci(succ)
    ref_key = (task if env == "mw" else env, model)
    ref = TABLE2.get(ref_key)
    ref_rate = ref / 100 if ref is not None else None
    seed1 = [x for x in rows if x.get("meta_seed") == 1]
    seed1 = sorted(seed1, key=lambda x: (x["task"], x["ep"]))
    s1 = [int(x["success"]) for x in seed1]
    first15 = {"n": min(15, len(s1)), "successes": int(sum(s1[:15])), "failures": int(min(15, len(s1)) - sum(s1[:15]))}
    blocks = block_counts(s1)
    min_block = min([min(b["successes"], b["failures"]) for b in blocks], default=None)
    in_band = 0.20 <= rate <= 0.80 if n else False
    fit_ok = len(s1) >= 15 and first15["successes"] >= 3 and first15["failures"] >= 3
    eligible = bool(in_band and fit_ok)
    secondary = {}
    for key in SECONDARY:
        v = np.array([x.get(key) for x in rows if isinstance(x.get(key), (int, float))], dtype=float)
        if len(v):
            secondary[key] = {"mean": float(np.nanmean(v)), "n": int(len(v))}
    for key in INFO_KEYS:
        v = np.array([x["last_info"].get(key) for x in rows if isinstance((x.get("last_info") or {}).get(key), (int, float))], dtype=float)
        if len(v):
            secondary[f"last_info.{key}"] = {"mean": float(np.nanmean(v)), "n": int(len(v))}
    for key in ("end_distance", "state_dist", "last_info.max_coverage"):
        parts = key.split(".")
        vals = {}
        for lab, want in (("success", 1), ("failure", 0)):
            sel = [x for x in rows if x["success"] == want]
            if parts[0] == "last_info":
                v = [x["last_info"].get(parts[1]) for x in sel if isinstance((x.get("last_info") or {}).get(parts[1]), (int, float))]
            else:
                v = [x.get(key) for x in sel if isinstance(x.get(key), (int, float))]
            if v:
                vals[lab] = float(np.mean(v))
        if vals:
            secondary[f"{key}_by_class"] = vals
    summary = {
        "env": env,
        "env_name": ENV_NAMES.get(env, env),
        "model": model,
        "task": task,
        "config": rows[0].get("config"),
        "config_sha256": rows[0].get("config_sha256"),
        "checkpoint_sha256": rows[0].get("checkpoint_sha256"),
        "n_episodes": n,
        "n_success": k,
        "n_failure": n - k,
        "success_rate": rate,
        "bootstrap95": ci,
        "wilson95": wilson(k, n),
        "per_seed": per_seed,
        "seeds_complete": [d for d, v in per_seed.items() if v["complete"]],
        "table2_reference_pct": ref,
        "table2_within_bootstrap95": (ci[0] <= ref_rate <= ci[1]) if ref_rate is not None and n else None,
        "table2_delta_pp": (rate * 100 - ref) if ref is not None and n else None,
        "eligibility": {
            "band_20_80": bool(in_band),
            "first15_seed1": first15,
            "first15_ok_3_3": bool(fit_ok),
            "blocks15_seed1": blocks,
            "min_class_count_over_blocks": min_block,
            "eligible": eligible,
        },
        "secondary": secondary,
        "mean_wall_s_per_episode": secondary.get("wall_s", {}).get("mean"),
    }
    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="artifacts/public_panel/screen")
    args = ap.parse_args()
    screen = Path(args.screen)
    cells = []
    for d in sorted(screen.iterdir()):
        if d.is_dir() and "_" in d.name:
            s = summarize_cell(d)
            if s:
                cells.append(s)
    per_model = {}
    for s in cells:
        per_model.setdefault(s["model"], []).append(s)
    advancement = {}
    for model, ss in per_model.items():
        elig = [f"{s['env']}:{s['task']}" for s in ss if s["eligibility"]["eligible"]]
        advancement[model] = {"eligible_benchmarks": elig, "n_eligible": len(elig), "advances": len(elig) >= 2}
    table = {"cells": cells, "advancement": advancement}
    (screen / "c0_table.json").write_text(json.dumps(table, indent=2))
    lines = [
        "# Panel P C0 screen (Label-first principle, amendment 3)",
        "",
        "Rates from `screen/<env>_<model>/episodes.jsonl`; CI = episode bootstrap (10k, seed 0) / Wilson; Table 2 = arXiv:2512.24497 (96 ep x 3 seeds).",
        "",
        "| cell | task | seeds (n) | succ/n | rate | boot95 | Wilson95 | Table 2 | ref in CI | first-15 (s/f) | min class/15-block | eligible |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in cells:
        seeds = ", ".join(f"{d}({v['n']}{'' if v['complete'] else '*'})" for d, v in s["per_seed"].items())
        f15 = s["eligibility"]["first15_seed1"]
        lines.append(
            f"| {s['env_name']} / {s['model']} | {s['task']} | {seeds} | {s['n_success']}/{s['n_episodes']} | {s['success_rate']*100:.1f} % | "
            f"[{s['bootstrap95'][0]*100:.1f}, {s['bootstrap95'][1]*100:.1f}] | [{s['wilson95'][0]*100:.1f}, {s['wilson95'][1]*100:.1f}] | "
            f"{s['table2_reference_pct']} | {s['table2_within_bootstrap95']} | {f15['successes']}/{f15['failures']} | "
            f"{s['eligibility']['min_class_count_over_blocks']} | {'YES' if s['eligibility']['eligible'] else 'no'} |"
        )
    lines += ["", "`*` = seed run incomplete at summary time.", ""]
    for model, a in advancement.items():
        lines.append(f"- **{model}**: eligible benchmarks = {a['eligible_benchmarks']} (n={a['n_eligible']}); advances (>= 2): **{a['advances']}**")
    (screen / "c0_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
