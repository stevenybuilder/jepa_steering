#!/usr/bin/env python3
"""Merge per-seed held-out stimulus directories into one gated stimulus set.

Input layout (produced by ``run_heldout_seeds.sh``):

    <root>/seed_<s>/manifest.jsonl
    <root>/seed_<s>/summary.json
    <root>/seed_<s>/cells/*.npz

Seeds that aborted inside the generator (positioning gate) have no
``summary.json``; their failure reason is recovered from the per-seed log.

Gates applied per seed (all must pass):

- complete quartet (h0a0, h0a1, h1a0, h1a1); the optional null-factor cells
  ``h2a0/h2a1`` (H0': a second off-path placement, added by the augment script)
  are reported as a *sextet* when present but are never required;
- generator gates from ``summary.json``: positioning, state-contrast isolation,
  intended contact pattern, force DiD >= 5 N;
- replay gate via ``protocol.replay_gate``: v0.2 = bit-identical frames;
  v0.3 = exact physics replay (state, force, contact count) and at most one
  differing pixel per frame with |diff| <= 1 LSB. The differing-pixel count is
  NOT recorded by the generator, so for v0.3 it must come from
  ``verify_render_jitter.py`` output (``--jitter-json``); cells that are not
  bit-exact and have no verified count fail closed
  (``differing_pixel_count_unverified``);
- hazard-label validation from ``validate_hazard_labels.py``
  (``--label-validation``): any seed whose label checks failed is excluded with
  its failure list. Without the file, seeds are admitted but the summary says
  ``label_validation: not_provided``.

Also writes a deterministic discovery/confirmation split of admitted seeds.

``--domain driving`` (protocol v0.8, MetaDrive): a scene is the 8-cell factorial
{levels 0 sidewalk, 1 pedestrian in lane, 2 H0' matched-displacement sidewalk
pose, 3 cone in lane} x {A0 brake, A1 throttle}; it is admitted only when all
8 cells exist and pass. Scenes with some but not all cells are reported as
``partial`` (``cells_missing`` per seed, ``partial_scenes`` in the tally).
Generator gates read from ``summary.json`` pairs: ``positioning_gate`` (hazard
bbox >= 48 px), ``state_contrast_gate`` (identical context across cells),
``contact_pattern_gate`` (contact only where the arm permits it) and
``min_distance_interaction_gate`` (falls back to a nonzero, finite
``min_distance_did_m`` when the generator does not write the flag). The
per-scene physical DiD is ``min_distance_did_m`` = (d11 - d10) - (d01 - d00) on
the minimum ego-hazard distance; ``approach_did_m = -min_distance_did_m`` is the
consequence-currency version (larger = closer approach under throttle when the
hazard is on the path). Per-cell ``min_distance_m`` / ``contact`` /
``crash_human`` / ``crash_object`` from the manifest rows are carried through
and, when all 8 cells carry ``min_distance_m``, ``protocol.did_by_identity`` is
reported on them. Label validation JSON keeps the egg validator's shape
(``{"seeds": [{"seed", "passed", "failures"}]}``). Replay gate = v0.8 frame gate
(``protocol.replay_gate``): physics bit-exact, <= 16 differing pixels per frame
with |diff| <= 8/255, count from ``--jitter-json`` or the row field
``replay_max_differing_pixels_per_frame``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import (  # noqa: E402
    CELL_ORDER,
    HAZARD_LEVELS,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_V03,
    PROTOCOL_VERSION_V08,
    PROTOCOL_VERSION_V09,
    V08_CELL_ORDER,
    canonical_json,
    did_by_identity,
    replay_gate,
    sha256_file,
)

CALIBRATION_SEEDS = (101, 102)
NULL_FACTOR_CELLS = ((2, 0), (2, 1))
DOMAINS = ("egg", "driving")
PROTOCOLS = (PROTOCOL_VERSION, PROTOCOL_VERSION_V03, PROTOCOL_VERSION_V08, PROTOCOL_VERSION_V09)  # v0.9 = v0.8 gates, varied stimulus
# driving: per-cell metrics carried from the manifest rows (generator convention, v0.8)
DRIVING_CELL_FIELDS = ("min_distance_m", "contact", "contact_step", "crash_human", "crash_object")

FAILURE_PATTERNS = (
    ("hazard_not_visible", re.compile(r"hazard target is not visible")),
    ("approach_error", re.compile(r"gripper-to-approach error")),
    ("no_control_pose", re.compile(r"no visible, supported off-path control pose")),
    ("camera_install", re.compile(r"failed to install JEPA-WM evaluation camera")),
    # driving generator (metadrive_hazard_pilot.py) abort messages
    ("hazard_bbox_too_small", re.compile(r"hazard bbox", re.IGNORECASE)),
    ("positioning_gate", re.compile(r"positioning gate", re.IGNORECASE)),
    ("replay_mismatch", re.compile(r"replay (mismatch|not bit-exact)", re.IGNORECASE)),
)


def classify_failure(log_text: str) -> tuple[str, str]:
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    error_line = ""
    for ln in reversed(lines):
        if "Error" in ln or "error" in ln:
            error_line = ln.strip()
            break
    for label, pattern in FAILURE_PATTERNS:
        if pattern.search(log_text):
            return label, error_line
    if error_line:
        return "other_error", error_line
    return "no_error_in_log", ""


def seed_from_dir(path: Path) -> int | None:
    match = re.fullmatch(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_jitter(patterns: list[str]) -> dict[str, int]:
    """cell_id -> max over frames of differing pixels (replay vs replay), from verify_render_jitter.py."""

    counts: dict[str, int] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for rec in json.loads(Path(path).read_text()):
                frames = rec.get("replay_vs_replay_frames", []) + rec.get("saved_vs_replay_frames", [])
                if not frames:
                    continue
                n = max(int(f.get("differing_pixels", 10**9)) for f in frames)
                if rec.get("verdict") != "render_only":
                    n = 10**9  # physics_or_unknown: never admit
                counts[rec["cell_id"]] = max(counts.get(rec["cell_id"], 0), n)
    return counts


def load_label_validation(path: Path | None) -> dict[int, dict[str, Any]] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    return {int(r["seed"]): r for r in data.get("seeds", [])}


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def driving_scene_physics(cells: dict[tuple[int, int], dict[str, Any]], pair: dict[str, Any]) -> dict[str, Any]:
    """Per-cell driving metrics and the identity x action DiD on the measured minimum distance."""

    per_cell = {}
    for (h, a), r in sorted(cells.items()):
        per_cell[f"h{h}a{a}"] = {k: r.get(k) for k in DRIVING_CELL_FIELDS if k in r}
    dist = {k: _finite(r.get("min_distance_m")) for k, r in cells.items()}
    out: dict[str, Any] = {"per_cell": per_cell}
    mdd = _finite(pair.get("min_distance_did_m"))
    if mdd is None and all(dist.get(k) is not None for k in CELL_ORDER):
        mdd = (dist[(1, 1)] - dist[(1, 0)]) - (dist[(0, 1)] - dist[(0, 0)])
        out["min_distance_did_source"] = "derived_from_cells"
    else:
        out["min_distance_did_source"] = "summary.json" if mdd is not None else None
    out["min_distance_did_m"] = mdd
    out["approach_did_m"] = (-mdd) if mdd is not None else None
    if all(dist.get(k) is not None for k in V08_CELL_ORDER):
        raw = did_by_identity({k: dist[k] for k in V08_CELL_ORDER})
        out["did_by_identity_min_distance_m"] = raw
        out["did_by_identity_approach_m"] = {k: -v for k, v in raw.items()}
    return out


def evaluate_seed(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    protocol: str,
    jitter: dict[str, int],
    label_record: dict[str, Any] | None,
    label_validation_provided: bool,
    domain: str = "egg",
) -> dict[str, Any]:
    cells = {(int(r["hazard"]), int(r["candidate_action"])): r for r in rows}
    required = V08_CELL_ORDER if domain == "driving" else CELL_ORDER
    complete = set(required) <= set(cells)
    sextet = complete and set(NULL_FACTOR_CELLS) <= set(cells)
    pairs = summary.get("pairs", [])
    pair = pairs[0] if pairs else {}

    replay = {}
    for key in sorted(cells):
        r = cells[key]
        n = jitter.get(r["cell_id"])
        replay[r["cell_id"]] = replay_gate(r, protocol, n if n is None or n < 10**9 else 10**9)
    replay_ok = complete and all(v["passed"] for v in replay.values())
    replay_reasons = sorted({v["reason"] for v in replay.values() if not v["passed"]})

    physics = driving_scene_physics(cells, pair) if domain == "driving" else {}
    if domain == "driving":
        dist_gate = pair.get("min_distance_interaction_gate")
        if dist_gate is None:  # derived: the generator did not write the flag
            dist_gate = physics["min_distance_did_m"] is not None and abs(physics["min_distance_did_m"]) > 1e-9
        gates = {
            "complete_octet": complete,
            "positioning": bool(pair.get("positioning_gate", False)),
            "state_contrast_isolated": bool(pair.get("state_contrast_gate", False)),
            "intended_contact_pattern": bool(pair.get("contact_pattern_gate", False)),
            "min_distance_interaction": bool(dist_gate),
            "replay": replay_ok,
            "hazard_labels": True,
        }
    else:
        gates = {
            "complete_quartet": complete,
            "positioning": bool(pair.get("positioning_gate", False)),
            "state_contrast_isolated": bool(pair.get("state_contrast_gate", False)),
            "intended_contact_pattern": bool(pair.get("contact_pattern_gate", False)),
            "force_interaction_at_least_5n": bool(pair.get("force_interaction_gate", False)),
            "replay": replay_ok,
            "hazard_labels": True,
        }
    label_failures: list[str] = []
    if label_validation_provided:
        if label_record is None:
            gates["hazard_labels"] = False
            label_failures = ["not_validated"]
        elif not label_record.get("passed", False):
            gates["hazard_labels"] = False
            label_failures = list(label_record.get("failures", [])) or ["failed"]

    reasons = [k for k, v in gates.items() if not v]
    if not replay_ok:
        reasons = [r for r in reasons if r != "replay"] + [f"replay:{x}" for x in replay_reasons or ["incomplete"]]
    if label_failures:
        reasons = [r for r in reasons if r != "hazard_labels"] + [f"labels:{x}" for x in label_failures]
    out = {
        "gates": gates,
        "admitted": all(gates.values()),
        "exclusion_reasons": reasons,
        "has_null_factor_sextet": sextet,
        "replay_per_cell": replay,
        "max_replay_pixel_error": max((int(r.get("replay_max_pixel_error", 255)) for r in rows), default=255),
        "physics_exact_frames_not_bit_exact": bool(
            complete
            and all(v["physics_ok"] for v in replay.values())
            and not all(v["frames_bit_exact"] for v in replay.values())
        ),
        "contact_did": pair.get("contact_did"),
        "force_did_n": pair.get("force_did_n"),
        "hazard_to_control_centroid_px": pair.get("control_pose_selection", {}).get("hazard_to_control_centroid_px"),
        "label_validation": (
            "not_provided" if not label_validation_provided else ("passed" if not label_failures else "failed")
        ),
    }
    if domain == "driving":
        present = sorted(cells)
        missing = sorted(set(required) - set(cells))
        out.update(
            {
                "cells_present": [f"h{h}a{a}" for h, a in present],
                "cells_missing": [f"h{h}a{a}" for h, a in missing],
                "partial": bool(cells) and not complete,
                "force_did_n": None,
                **physics,
            }
        )
        if out["partial"]:  # incompleteness is reported once, as the partial-scene reason (replay of the present cells is still checked)
            cells_replay_ok = all(v["passed"] for v in replay.values())
            keep = [r for r in reasons if r != "complete_octet" and not (r == "replay:incomplete" and cells_replay_ok)]
            out["exclusion_reasons"] = keep + [f"partial_scene:missing_{len(missing)}_of_{len(required)}"]
    return out


def split_seeds(passing: list[int], split_seed: int, n_discovery: int | None) -> tuple[list[int], list[int]]:
    """Deterministic discovery/confirmation split that is STABLE under batch growth.

    Membership is a per-seed hash (sha256 of "<split_seed>:<seed>"), so adding
    more admitted seeds later never moves an existing seed between arms. This
    matters because discovery seeds may be used for site selection before the
    full batch exists; a permutation of the admitted list would reshuffle them.
    ``n_discovery`` is ignored except to warn (kept for CLI compatibility).
    """

    import hashlib

    discovery, confirmation = [], []
    for s in sorted(passing):
        h = hashlib.sha256(f"{split_seed}:{s}".encode()).digest()[0]
        (discovery if h < 128 else confirmation).append(int(s))
    return discovery, confirmation


def merge(
    root: Path,
    output: Path,
    log_dir: Path | None,
    seed_range: tuple[int, int] | None,
    protocol: str,
    jitter_globs: list[str],
    label_validation: Path | None,
    split_seed: int,
    n_discovery: int | None,
    exclude: tuple[int, ...],
    domain: str = "egg",
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol}")
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain}")
    if domain == "driving" and protocol not in (PROTOCOL_VERSION_V08, PROTOCOL_VERSION_V09):
        raise ValueError(f"domain driving requires protocol {PROTOCOL_VERSION_V08} or {PROTOCOL_VERSION_V09}")
    output.mkdir(parents=True, exist_ok=True)
    jitter = load_jitter(jitter_globs)
    if label_validation is None:
        default = root / "_validation" / "hazard_label_validation.json"
        label_validation = default if default.exists() else None
    labels = load_label_validation(label_validation)

    seed_dirs = {s: p for p in sorted(root.iterdir()) if p.is_dir() and (s := seed_from_dir(p)) is not None}
    candidate_seeds = set(seed_dirs)
    if seed_range is not None:
        candidate_seeds |= set(range(seed_range[0], seed_range[1] + 1))

    per_seed: dict[int, dict[str, Any]] = {}
    merged_rows: list[dict[str, Any]] = []
    merged_pairs: list[dict[str, Any]] = []
    for seed in sorted(candidate_seeds):
        if seed in exclude:
            per_seed[seed] = {"status": "excluded_calibration", "exclusion_reasons": ["calibration_seed"]}
            continue
        sdir = seed_dirs.get(seed)
        summary_path = sdir / "summary.json" if sdir else None
        if sdir is None or summary_path is None or not summary_path.exists():
            log_path = (log_dir / f"heldout_seed_{seed}.log") if log_dir else None
            if log_path is not None and log_path.exists():
                label, line = classify_failure(log_path.read_text(errors="replace"))
                status = "generator_aborted" if label != "no_error_in_log" else "in_progress_or_incomplete"
            else:
                label, line, status = "not_run_or_no_log", "", "not_run_or_no_log"
            per_seed[seed] = {"status": status, "failure": label, "error_line": line, "exclusion_reasons": [f"generator:{label}"]}
            continue
        summary = json.loads(summary_path.read_text())
        rows = [json.loads(ln) for ln in (sdir / "manifest.jsonl").read_text().splitlines() if ln.strip()]
        ev = evaluate_seed(rows, summary, protocol, jitter, labels.get(seed) if labels else None, labels is not None, domain)
        ev["status"] = "admitted" if ev["admitted"] else "gate_failed"
        ev["source_dir"] = str(sdir)
        per_seed[seed] = ev
        if ev["admitted"]:
            for r in rows:
                row = dict(r)
                row["artifact"] = str(Path(os.path.relpath(sdir.resolve(), output.resolve())) / r["artifact"])
                row["source_seed_dir"] = sdir.name
                row["replay_gate"] = ev["replay_per_cell"][r["cell_id"]]["reason"]
                merged_rows.append(row)
            for p in summary.get("pairs", []):
                merged_pairs.append(dict(p, seed=seed))

    passing = [s for s, v in per_seed.items() if v.get("status") == "admitted"]
    discovery, confirmation = split_seeds(passing, split_seed, n_discovery)
    attempted = [
        s
        for s, v in per_seed.items()
        if v.get("status") not in ("excluded_calibration", "not_run_or_no_log", "in_progress_or_incomplete")
    ]
    reason_counts: dict[str, int] = {}
    for s in attempted:
        for r in per_seed[s].get("exclusion_reasons", []):
            reason_counts[r] = reason_counts.get(r, 0) + 1
    tally = {
        "attempted": len(attempted),
        "admitted": len(passing),
        "generator_aborted": sum(v.get("status") == "generator_aborted" for v in per_seed.values()),
        "gate_failed": sum(v.get("status") == "gate_failed" for v in per_seed.values()),
        "not_run_or_no_log": sum(v.get("status") == "not_run_or_no_log" for v in per_seed.values()),
        "in_progress_or_incomplete": sum(v.get("status") == "in_progress_or_incomplete" for v in per_seed.values()),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "physics_exact_frames_not_bit_exact": sum(v.get("physics_exact_frames_not_bit_exact", False) for v in per_seed.values()),
        "admitted_with_null_factor_sextet": sorted(s for s in passing if per_seed[s].get("has_null_factor_sextet")),
        "exclusion_rate": (1.0 - len(passing) / len(attempted)) if attempted else None,
        "jitter_verified_cells": len(jitter),
        "label_validation": "provided" if labels is not None else "not_provided",
    }
    if domain == "driving":
        tally["partial_scenes"] = sorted(s for s in attempted if per_seed[s].get("partial"))
        tally["complete_octets"] = sum(1 for s in attempted if per_seed[s].get("gates", {}).get("complete_octet"))

    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text("".join(canonical_json(r) + "\n" for r in merged_rows), encoding="utf-8")
    (output / "discovery_seeds.txt").write_text("".join(f"{s}\n" for s in discovery))
    (output / "confirmation_seeds.txt").write_text("".join(f"{s}\n" for s in confirmation))

    n_core = sum(1 for r in merged_rows if int(r["hazard"]) in (0, 1))
    frame_gate_note = {
        PROTOCOL_VERSION: "v0.2: replay frames must be bit-identical",
        PROTOCOL_VERSION_V03: "v0.3: physics replay exact; renders may differ by <= 1 pixel per frame with |diff| <= 1 LSB, "
        "verified per cell by verify_render_jitter.py",
        PROTOCOL_VERSION_V08: "v0.8: physics/ego state bit-exact on replay; renders may differ on <= 16 pixels per frame with "
        "|diff| <= 8/255 (count from --jitter-json or the row field replay_max_differing_pixels_per_frame); worse fails closed",
        PROTOCOL_VERSION_V09: "v0.9: identical gates to v0.8 (physics bit-exact, <= 16 differing pixels per frame); stimulus factors "
        "(hazard distance, ego prefix throttle, in-lane lateral offset, FOV, settle steps) vary per the generator flags / per-seed draws",
    }[protocol]
    result = {
        "protocol_version": protocol,
        "frame_gate_note": frame_gate_note,
        "root": str(root),
        "excluded_calibration_seeds": list(exclude),
        "label_validation_file": str(label_validation) if label_validation else None,
        "tally": tally,
        "per_seed": {str(k): {kk: vv for kk, vv in v.items() if kk != "replay_per_cell"} for k, v in sorted(per_seed.items())},
        "admitted_seeds": sorted(passing),
        "split": {"split_seed": split_seed, "discovery": discovery, "confirmation": confirmation},
        "pairs": merged_pairs,
        "phase0_stimulus_gate": {
            "complete_quartets": n_core == 4 * len(passing),
            "replay_gate_all_passed": True if passing else False,
            "frames_bit_exact_all": all(bool(r.get("replay_frames_bit_exact", False)) for r in merged_rows) if merged_rows else False,
            "positioning": True if passing else False,
            "state_contrast_isolated": True if passing else False,
            "intended_contact_pattern": True if passing else False,
            "force_interaction_at_least_5n": True if passing else False,
            "nonzero_contact_did_pairs": sum(abs(float(p.get("contact_did", 0.0))) > 0 for p in merged_pairs),
            "nonzero_force_did_pairs": sum(abs(float(p.get("force_did_n", 0.0))) > 1e-9 for p in merged_pairs),
        },
        "manifest_sha256": sha256_file(manifest_path),
    }
    if domain == "driving":
        n_all = sum(1 for r in merged_rows if int(r["hazard"]) in HAZARD_LEVELS)
        result["domain"] = domain
        result["hazard_levels"] = {"0": "sidewalk (off-path)", "1": "pedestrian in lane", "2": "H0' matched-displacement sidewalk pose", "3": "cone in lane"}
        result["action_levels"] = {"0": "brake", "1": "throttle"}
        result["physical_did"] = {
            "name": "min_distance_did_m",
            "definition": "(d11 - d10) - (d01 - d00) on the minimum ego-hazard distance; approach_did_m = -min_distance_did_m "
            "(larger = closer approach under throttle when the hazard is on the path)",
            "per_seed": {str(s): {"min_distance_did_m": per_seed[s].get("min_distance_did_m"), "approach_did_m": per_seed[s].get("approach_did_m"),
                                  "did_by_identity_approach_m": per_seed[s].get("did_by_identity_approach_m")} for s in passing},
        }
        pg = result["phase0_stimulus_gate"]
        pg.pop("force_interaction_at_least_5n", None)
        pg.pop("nonzero_force_did_pairs", None)
        pg["complete_octets"] = n_all == 8 * len(passing)
        pg["min_distance_interaction"] = True if passing else False
        pg["nonzero_min_distance_did_pairs"] = sum(1 for s in passing if (per_seed[s].get("min_distance_did_m") or 0.0) != 0.0)
    (output / "summary.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="directory containing seed_<s>/ subdirs")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=None, help="dir with heldout_seed_<s>.log (for aborted seeds)")
    ap.add_argument("--seed-range", type=int, nargs=2, default=None, help="inclusive range of attempted seeds")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default=PROTOCOL_VERSION)
    ap.add_argument("--domain", choices=list(DOMAINS), default="egg", help="driving = v0.8 8-cell MetaDrive scenes")
    ap.add_argument("--jitter-json", nargs="*", default=[], help="glob(s) of verify_render_jitter.py outputs")
    ap.add_argument("--label-validation", type=Path, default=None, help="hazard_label_validation.json (default: <root>/_validation/)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-discovery", type=int, default=None, help="default: half of admitted seeds")
    ap.add_argument("--exclude-seeds", type=int, nargs="*", default=list(CALIBRATION_SEEDS))
    args = ap.parse_args()
    result = merge(
        args.root, args.output, args.log_dir, tuple(args.seed_range) if args.seed_range else None, args.protocol,
        args.jitter_json, args.label_validation, args.split_seed, args.n_discovery, tuple(args.exclude_seeds), args.domain,
    )
    print(json.dumps({k: result[k] for k in ("protocol_version", "tally", "admitted_seeds", "split")}, indent=2))


if __name__ == "__main__":
    main()
