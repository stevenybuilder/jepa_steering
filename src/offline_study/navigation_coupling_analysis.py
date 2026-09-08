"""Complete two-task coupling planning panel, fixed 32-contrast interval family.

Never selects from an incomplete task/arm scope. Original offline losses are not
loaded. This reports planning development, not untouched confirmation or training
seed robustness. Hardware-dependent runtime is reported, not silently normalized.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .behavioral_development import schedule, verified_report
from .navigation_coupling_behavior import ARMS, TASKS, load_native, paired_inputs
from .navigation_replication import validate_records
from .protocol import sha256, write_json


def load_panel(root, reference):
    panels, protocols, bindings = {}, {}, {}
    for task in TASKS:
        freeze = root / task / "freeze"
        fp = freeze / "protocol.json"
        protocol = json.loads(fp.read_text())
        if (sha256(fp) != json.loads((freeze / "FROZEN.json").read_text())["protocol_sha256"] or
                protocol["role"] != "full_static_coupling_navigation_development" or
                protocol["task"] != task or protocol["arms"] != list(ARMS) or
                protocol["episodes"] != schedule() or protocol["fresh_confirmation"] is not False or
                protocol["analysis"]["interval_family_size"] != 32 or
                protocol["analysis"]["bootstrap_draws"] != 20000 or
                protocol["analysis"]["bootstrap_seed"] != 2026090801):
            raise ValueError("Changed task/condition/analysis freeze")
        _, native, native_hash = load_native(reference / task / "native", reference / task / "freeze")
        if native_hash != protocol["bindings"]["native_report_sha256"]:
            raise ValueError("Wrong native reference")
        panels[task] = {"native": native}
        protocols[task] = protocol
        bindings[str(fp)] = sha256(fp)
        bindings[str(reference / task / "native/report.json")] = native_hash
        for arm in ARMS[1:]:
            shards = sorted((root / task / "conditions" / arm).glob("shard-*"))
            if not shards:
                raise ValueError("Incomplete panel: " + task + "/" + arm)
            rows = []
            for shard in shards:
                report, report_hash = verified_report(shard)
                launch = json.loads((shard / "protocol.json").read_text())
                if (report["status"] != "complete_coupling_navigation_shard" or
                        report["task"] != task or report["arm"] != arm or
                        report["parameters_unchanged"] is not True or
                        report["scientific_efficacy_measurement"] is not True or
                        report["fresh_confirmation"] is not False or
                        report["source_sha256"] != protocol["source_sha256"] or
                        report["freeze_sha256"] != sha256(fp) or
                        report["protocol_sha256"] != sha256(shard / "protocol.json") or
                        launch["freeze_sha256"] != sha256(fp) or launch["arm"] != arm):
                    raise ValueError("Unbound candidate completion")
                records = []
                for name, digest in report["episode_files_sha256"].items():
                    if Path(name).name != name or not name.startswith("episode-") or sha256(shard / name) != digest:
                        raise ValueError("Candidate episode checksum mismatch")
                    row = json.loads((shard / name).read_text())
                    if row["arm"] != arm:
                        raise ValueError("Wrong treatment label")
                    records.append(row)
                records.sort(key=lambda row: row["episode"])
                if len(records) != report["episodes"]:
                    raise ValueError("Wrong completed episode count")
                validate_records(records, launch["expected_episodes"])
                rows.extend(records)
                bindings[str(shard / "report.json")] = report_hash
            rows.sort(key=lambda row: row["episode"])
            validate_records(rows, schedule())
            for baseline, candidate in zip(native, rows, strict=True):
                paired_inputs(baseline, candidate)
            panels[task][arm] = rows
    return panels, protocols, bindings


def analyze(panels, protocols):
    if tuple(panels) != TASKS or tuple(protocols) != TASKS:
        raise ValueError("Both complete tasks required")
    rng = np.random.default_rng(2026090801)
    results, summaries = [], {}
    for task in TASKS:
        arms = panels[task]
        if tuple(arms) != ARMS or any(len(arms[a]) != 96 for a in ARMS):
            raise ValueError("Every condition must contain exactly 96 episodes")
        success = {arm: np.array([r["result"]["native_success"] for r in arms[arm]], dtype=float)
                   for arm in ARMS}
        groups = {}
        for i, row in enumerate(arms["native"]):
            key = tuple(row["result"][k] for k in ("initial_sha256", "goal_sha256"))
            groups.setdefault(key, []).append(i)
        clusters = list(groups.values())
        if len(clusters) < 2:
            raise ValueError("Too few distinct initial/goal clusters")
        sizes = np.array([len(c) for c in clusters])
        draws = rng.integers(len(clusters), size=(20000, len(clusters)))
        denominator = sizes[draws].sum(1)
        summaries[task] = {}
        for arm in ARMS:
            if not np.isin(success[arm], [0., 1.]).all():
                raise ValueError("Invalid binary task-success endpoint")
            summaries[task][arm] = {"episodes": 96, "scenario_clusters": len(clusters),
                "successes": int(success[arm].sum()), "success_percent": float(success[arm].mean() * 100),
                "mean_episode_seconds": float(np.mean([r["seconds"] for r in arms[arm]])),
                "runtime_hardware_parity_not_assumed": True}
        contrasts = protocols[task]["pairwise_contrasts"]
        if len(contrasts) != 15:
            raise ValueError("Original pairwise contrast family changed")
        pairs = [(c["name"], success[c["candidate"]] - success[c["control"]]) for c in contrasts]
        interaction = protocols[task]["factorial_success_interaction"]
        if interaction != {"joint": 1, "visual_only": -1, "action_condition_only": -1, "native": 1}:
            raise ValueError("Factorial interaction changed")
        pairs.append(("factorial_success_interaction", sum(success[a] * weight for a, weight in interaction.items())))
        for name, delta in pairs:
            totals = np.array([delta[cluster].sum() for cluster in clusters])
            samples = totals[draws].sum(1) / denominator * 100
            bounds = np.quantile(samples, [.05 / 64, 1 - .05 / 64])
            results.append({"task": task, "contrast": name,
                "difference_percentage_points": float(delta.mean() * 100),
                "simultaneous_95_interval_percentage_points": bounds.tolist(),
                "scenario_clusters": len(clusters)})
    if len(results) != 32:
        raise ValueError("Incomplete 32-contrast family")
    return {"status": "complete_navigation_coupling_planning_development_analysis",
        "task_summaries": summaries, "contrasts": results, "fresh_confirmation": False,
        "three_training_seeds_complete": False, "automatic_confirmation_authorized": False,
        "offline_scores_used_for_admission": False, "selection_from_partial_results": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    panels, protocols, bindings = load_panel(args.root, args.reference)
    report = analyze(panels, protocols)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "bindings.json", bindings)
    report["bindings_sha256"] = sha256(args.output / "bindings.json")
    write_json(args.output / "report.json", report)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
