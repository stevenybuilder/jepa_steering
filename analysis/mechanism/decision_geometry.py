"""Hash-bound cost geometry on the archived common-action diagnostic.

This is exploratory development analysis, not a fresh efficacy evaluation.
Candidate scores are repeated measurements within a scenario, never replicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from src.offline_study.decision_diagnostic import ARMS, rank_agreement, validate_records

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/decision_geometry_reanalysis_20260913.json"
BASE = "decision/artifacts/offline_study/decision-diagnostic-20260910-v2-retry1/"
LABELS = dict(zip(ARMS, ("native", "refined", "random_refined", "coupling", "random_coupling")))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_entropy(scores, temperature):
    """A fixed-temperature descriptive score diagnostic, not CEM entropy."""
    if not np.isfinite(temperature) or temperature <= 0:
        return None
    logits = -(scores - scores.min()) / temperature
    weights = np.exp(logits)
    weights /= weights.sum()
    positive = weights > 0
    return float(-np.sum(weights[positive] * np.log(weights[positive])))


def compare_scores(native, edited):
    native, edited = np.asarray(native, dtype=float), np.asarray(edited, dtype=float)
    if native.ndim != 1 or native.size < 2 or native.shape != edited.shape:
        raise ValueError("Paired candidate vectors required")
    if not np.isfinite(native).all() or not np.isfinite(edited).all():
        raise ValueError("Finite candidate vectors required")
    order = np.argsort(native, kind="stable")
    winner = int(order[0])
    chosen = int(np.argmin(edited))
    margin = float(native[order[1]] - native[winner])
    delta = edited - native
    centered = delta - delta.mean()
    energy = float(np.mean(delta ** 2))
    perturbation_range = float(np.ptp(delta))
    # Algebraically exact relative margins; native and edited ties select lowest ID.
    gaps = native - native[winner]
    perturbations = delta - delta[winner]
    competitors = np.arange(native.size) != winner
    actual_gaps = edited - edited[winner]
    if not np.allclose(gaps + perturbations, actual_gaps, rtol=1e-12, atol=1e-12):
        raise ValueError("Relative-margin identity failed")
    certificate = perturbation_range < margin
    if certificate and chosen != winner:
        raise ValueError("No-flip certificate contradicted by selected candidate")
    temperature = float(native.std())
    h0, h1 = score_entropy(native, temperature), score_entropy(edited, temperature)
    k = min(10, native.size)
    return {
        "native_selected": winner, "edited_selected": chosen,
        "changed": int(chosen != winner), "native_margin": margin,
        "delta_mean": float(delta.mean()),
        "centered_delta_rms": float(np.sqrt(np.mean(centered ** 2))),
        "centered_delta_energy_fraction": float(np.mean(centered ** 2) / energy) if energy else None,
        "perturbation_range": perturbation_range,
        "range_to_margin": perturbation_range / margin if margin else None,
        "no_flip_certified": int(certificate),
        "strict_margin_crossings": int(np.sum((actual_gaps < 0) & competitors)),
        "native_score_std": temperature,
        "centered_rms_to_native_std": float(np.sqrt(np.mean(centered ** 2)) / temperature) if temperature else None,
        "spearman": rank_agreement(native, edited),
        "top10_overlap": len(set(order[:k]) & set(np.argsort(edited, kind="stable")[:k])) / k,
        "native_score_entropy_nats": h0,
        "score_entropy_delta_nats": h1 - h0 if h0 is not None and h1 is not None else None,
    }


def load_records(archive, plan):
    if digest(archive) != plan["archive_sha256"]:
        raise ValueError("Archive SHA256 mismatch")
    records, receipts = [], {}
    # One sequential decompression pass: random access inside a gzip archive
    # otherwise repeatedly inflates the entire preceding payload.
    members = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name
            if not name.startswith(BASE) or not member.isfile():
                continue
            relative = name[len(BASE):]
            selected = relative == "protocol.json" or (
                "/episode-" in relative and relative.endswith(("/record.json", "/DONE.json")))
            if selected:
                if relative in members:
                    raise ValueError("Duplicate archive member")
                members[relative] = tar.extractfile(member).read()
    if len([k for k in members if k.endswith("/record.json")]) != 192:
        raise ValueError("Unexpected scientific record count")
    if members:
        def read(name):
            if name not in members:
                raise ValueError("Missing archive member: " + name)
            return members[name]
        protocol_bytes = read("protocol.json")
        protocol = json.loads(protocol_bytes)
        if hashlib.sha256(protocol_bytes).hexdigest() != plan["original_protocol_sha256"]:
            raise ValueError("Protocol hash mismatch")
        for task in plan["tasks"]:
            for episode in range(96):
                stem = f"{task}/episode-{episode:03d}/"
                raw = read(stem + "record.json")
                checksum = hashlib.sha256(raw).hexdigest()
                done = json.loads(read(stem + "DONE.json"))
                if done["files"]["record.json"] != checksum:
                    raise ValueError("Record hash mismatch")
                record = json.loads(raw)
                if record["engineering_only"] is not False:
                    raise ValueError("Engineering record in scientific subset")
                receipts[stem + "record.json"] = checksum
                records.append(record)
    validate_records(records, protocol["episodes"], plan["original_protocol_sha256"])
    return records, receipts


def analyze(archive):
    plan = json.loads(PLAN.read_text())
    records, receipts = load_records(archive, plan)
    rows = []
    for record in records:
        for arm in ARMS[1:]:
            values = compare_scores(record["arms"]["native"]["scores"], record["arms"][arm]["scores"])
            rows.append({"task": record["task"], "episode": record["episode"],
                         "arm": LABELS[arm], **values})
    rng = np.random.default_rng(2026091321)
    summaries = []
    for task in plan["tasks"]:
        indices = rng.integers(0, 96, size=(20000, 96))
        for arm in plan["arms"][1:]:
            subset = [row for row in rows if row["task"] == task and row["arm"] == arm]
            if len(subset) != 96:
                raise ValueError("Incomplete arm")
            summary = {"task": task, "arm": arm, "n_scenarios": 96,
                       "changed": sum(x["changed"] for x in subset),
                       "no_flip_certified": sum(x["no_flip_certified"] for x in subset),
                       "metrics": {}}
            for metric in ("native_margin", "delta_mean", "centered_delta_rms", "centered_delta_energy_fraction",
                           "range_to_margin", "centered_rms_to_native_std", "spearman", "top10_overlap",
                           "native_score_entropy_nats", "score_entropy_delta_nats"):
                values = [x[metric] for x in subset]
                defined = np.array([x for x in values if x is not None])
                entry = {"undefined": 96 - len(defined), "mean": float(defined.mean()) if len(defined) else None,
                         "median": float(np.median(defined)) if len(defined) else None}
                # Do not silently bootstrap a selected finite subset with the full-scenario draws.
                entry["marginal_95_mean_interval"] = np.quantile(defined[indices].mean(1), [.025, .975]).tolist() if len(defined) == 96 else None
                summary["metrics"][metric] = entry
            summaries.append(summary)
    output = ROOT / "paper/data"
    with (output / "decision_geometry_scenarios.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"status": "complete_exploratory_development_reanalysis", "scenarios": len(records),
               "paired_arm_comparisons": len(rows), "candidate_scores_per_arm": 300,
               "plan": plan, "plan_sha256": digest(PLAN), "analysis_source_sha256": digest(__file__),
               "record_sha256": receipts, "summaries": summaries}
    (output / "decision_geometry.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


def plots():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    data = json.loads((ROOT / "paper/data/decision_geometry.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))
    labels = ["Refined", "Random\nsubspace", "V–A\ncoupling", "Random\ncoupling"]
    colors = ["#286E78", "#9DACB2"]
    for t, task in enumerate(("reach", "reach-wall")):
        group = [x for x in data["summaries"] if x["task"] == task]
        x = np.arange(4) + (t - .5) * .36
        for a, key in enumerate(("no_flip_certified", "changed")):
            values = [s[key] / 96 for s in group]
            axes[a].bar(x, values, .34, color=colors[t], label=("Reach", "Reach-Wall")[t])
            for xp, yp, s in zip(x, values, group):
                axes[a].text(xp, yp + (.015 if a == 0 else .0008), str(s[key]), ha="center", fontsize=8)
        values = [100 * s["metrics"]["centered_rms_to_native_std"]["mean"] for s in group]
        axes[2].bar(x, values, .34, color=colors[t])
    axes[0].set_title("Decision margins absorb the edit", loc="left", fontsize=11)
    axes[0].set_ylabel("Scenarios with sufficient no-flip certificate")
    axes[0].set_ylim(0, 1.10)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Few first choices change", loc="left", fontsize=11)
    axes[1].set_ylabel("Scenarios changing selected candidate")
    axes[1].set_ylim(0, .034)
    axes[1].set_yticks([0, .01, .02, .03])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[2].set_title("Small changes to candidate costs", loc="left", fontsize=11)
    axes[2].set_ylabel("Centered Δcost RMS / native cost std. (%)")
    for ax in axes:
        ax.set_xticks(np.arange(4), labels)
        ax.tick_params(axis="x", length=0, labelsize=8)
        ax.grid(axis="y", alpha=.13)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(.5, .012, "96 development scenarios/task · same 300 actions across arms · labels are scenario counts · initial population, not full CEM", ha="center", fontsize=8, color="#53636A")
    fig.tight_layout(rect=(0, .06, 1, 1), w_pad=2)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"docs/figures/decision_geometry.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.archive:
        analyze(args.archive)
    plots()


if __name__ == "__main__":
    main()
