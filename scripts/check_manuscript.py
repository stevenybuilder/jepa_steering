"""Check manuscript tables, graphics and BibTeX keys using only public evidence.

Standard library only; no TeX installation, model assets or GPU access required.
This is a narrow checker for the two complete result tables, not a TeX parser.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA = "8123d71497835fc164f09f5094c430308647a3ee2462c66746baea32633a0b15"
TASKS = ("reach", "reach-wall", "pointmaze", "wall")
HISTORICAL_TASKS = ("reach", "reach-wall", "pusht", "pointmaze", "wall", "droid")
ARMS = ("native", "fixed_rank4", "matched_random_fixed_rank4", "coupling_only",
        "matched_random_coupling", "joint", "visual_only", "action_condition_only")
TASK_LABELS = {
    "Reach": "reach", "Reach-Wall": "reach-wall", "R.-Wall": "reach-wall",
    "Push-T": "pusht", "PointMaze": "pointmaze", "P.Maze": "pointmaze",
    "Wall": "wall", "DROID": "droid",
}
ARM_LABELS = {
    "Native": "native", "Unsteered": "native",
    "Refined": "fixed_rank4", "Refined rank-four": "fixed_rank4",
    "Refined four-direction": "fixed_rank4",
    "Random subspace": "matched_random_fixed_rank4",
    "Calibrated random subspace": "matched_random_fixed_rank4",
    "Coupling": "coupling_only", "Equal-budget coupling": "coupling_only",
    "Random directions": "matched_random_coupling",
    "Dose-matched random directions": "matched_random_coupling",
    "Joint": "joint", "Unscaled joint": "joint",
    "Visual": "visual_only", "Visual only": "visual_only",
    "Action": "action_condition_only", "Action conditioning only": "action_condition_only",
    "Action-conditioning only": "action_condition_only",
}


class CheckError(ValueError):
    """Actionable evidence or manuscript mismatch."""


def rounded(value):
    try:
        result = Decimal(str(value))
        if not result.is_finite():
            raise InvalidOperation
        return result.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise CheckError(f"Nonfinite or invalid numeric evidence: {value!r}") from exc


def without_comments(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def load_evidence(root):
    report_path = root / "reports/fresh-confirmation/report.json"
    raw = report_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != REPORT_SHA:
        raise CheckError(f"{report_path}: immutable report SHA mismatch: {actual_sha}")
    report = json.loads(raw)
    fresh = {(arm, task): rounded(report["results"][task]["success_percent"][arm])
             for arm in ARMS for task in TASKS}
    csv_path = root / "paper/data/benchmark_comparison.csv"
    historical = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["population"] != "development":
                continue
            key = row["arm"], row["task"]
            if key in historical:
                raise CheckError(f"{csv_path}: duplicate development evidence for {key}")
            historical[key] = rounded(row["value_percent"])
    expected = {(a, t) for a in ARMS for t in HISTORICAL_TASKS}
    if set(historical) != expected:
        raise CheckError(f"{csv_path}: development coverage differs from 48 cells; "
                         f"missing={sorted(expected-set(historical))}, "
                         f"extra={sorted(set(historical)-expected)}")
    return fresh, historical


def check_tables(tex, fresh, historical):
    """Identify the 4- and 6-task tables by column headers, not table position."""
    found = {"fresh": 0, "historical": 0}
    checked = 0
    for block in re.findall(r"\\begin\{tabular\}[^\n]*\n(.*?)\\end\{tabular\}", tex, re.S):
        cleaned = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", block)
        rows = [[re.sub(r"\s+", " ", c).strip() for c in row.split("&")]
                for row in re.split(r"\\\\(?:\[[^\]]*\])?", cleaned) if "&" in row]
        if not rows or rows[0][0] != "Arm":
            continue
        tasks = tuple(TASK_LABELS.get(label) for label in rows[0][1:])
        if len(tasks) not in (4, 6):
            continue
        kind = "fresh" if len(tasks) == 4 else "historical"
        required_tasks = TASKS if kind == "fresh" else HISTORICAL_TASKS
        if len(set(tasks)) != len(tasks) or set(tasks) != set(required_tasks):
            raise CheckError(f"{kind} table: unknown, duplicate or missing task headers {rows[0]}")
        found[kind] += 1
        if found[kind] != 1:
            raise CheckError(f"Duplicate complete {kind} result table")
        evidence = fresh if kind == "fresh" else historical
        seen = set()
        for cells in rows[1:]:
            label = cells[0]
            arm = ARM_LABELS.get(label)
            if arm is None or arm in seen:
                raise CheckError(f"{kind} table: unknown or duplicate arm label {label!r}")
            seen.add(arm)
            if len(cells) != len(tasks)+1:
                raise CheckError(f"{kind} table / {label}: expected {len(tasks)} numeric cells")
            for task, cell in zip(tasks, cells[1:]):
                # Historical component-native provenance markers are not rate digits.
                value = re.sub(r"\\\(\s*\^\{\*\}\s*\\\)", "", cell).strip()
                if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                    raise CheckError(f"{kind} table / {label} / {task}: invalid cell {cell!r}")
                actual, expected = Decimal(value), evidence[arm, task]
                if actual != expected:
                    raise CheckError(f"{kind} table / {label} / {task}: found {actual}, "
                                     f"expected {expected} from public evidence")
                checked += 1
        if seen != set(ARMS):
            raise CheckError(f"{kind} table: missing arms {sorted(set(ARMS)-seen)}")
    for kind, count in found.items():
        if count != 1:
            raise CheckError(f"Missing complete {kind} result table with Arm/task headers")
    return checked


def check_figures(tex, base):
    macros = dict(re.findall(r"\\(?:newcommand|renewcommand)\s*\{(\\\w+)\}\s*\{([^{}]*)\}", tex))
    paths = re.findall(r"\\includegraphics\*?(?:\[[^\]]*\])?\s*\{([^{}]+)\}", tex)
    for source in paths:
        expanded = source
        for _ in range(len(macros)+1):
            replacement = re.sub(r"\\[A-Za-z]+", lambda m: macros.get(m[0], m[0]), expanded)
            if replacement == expanded:
                break
            expanded = replacement
        if "\\" in expanded:
            raise CheckError(f"Figure {source!r}: unresolved path macro; expanded to {expanded!r}")
        target = base / expanded
        choices = [target] if target.suffix else [target.with_suffix(x) for x in (".pdf", ".png", ".jpg", ".jpeg", ".eps")]
        if not any(path.is_file() for path in choices):
            raise CheckError(f"Missing figure: {source} resolves to {target}")
    if not paths:
        raise CheckError("No includegraphics paths found; check manuscript input")
    return len(paths)


def check_citations(tex, bib):
    defined_list = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib)
    defined = set(defined_list)
    if len(defined) != len(defined_list):
        raise CheckError("Duplicate BibTeX entry key")
    citations = re.findall(r"\\(?:cite[a-zA-Z]*|nocite)\*?(?:\s*\[[^\]]*\])*\s*\{([^}]+)\}", tex)
    used = {key.strip() for group in citations for key in group.split(",")} - {"*"}
    missing = used-defined
    if missing:
        raise CheckError(f"Undefined BibTeX citation keys: {', '.join(sorted(missing))}")
    return len(used)


def check(root=ROOT, manuscript=None):
    root = Path(root)
    manuscript = Path(manuscript) if manuscript else root / "paper/workshop/main.tex"
    tex = without_comments(manuscript.read_text())
    fresh, historical = load_evidence(root)
    cells = check_tables(tex, fresh, historical)
    figures = check_figures(tex, manuscript.parent)
    names = re.findall(r"\\bibliography\{([^}]+)\}", tex)
    if not names:
        raise CheckError("No bibliography declaration found")
    bib = "\n".join((manuscript.parent / (name.strip() + ".bib")).read_text()
                    for group in names for name in group.split(","))
    citations = check_citations(tex, bib)
    return {"fresh_cells": 32, "historical_cells": cells-32,
            "figure_includes": figures, "citation_keys": citations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manuscript", type=Path)
    args = parser.parse_args()
    try:
        counts = check(args.root, args.manuscript)
    except (CheckError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: immutable report SHA; "
          f"{counts['fresh_cells']} fresh + {counts['historical_cells']} historical cells; "
          f"{counts['figure_includes']} figure includes; {counts['citation_keys']} citation keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
