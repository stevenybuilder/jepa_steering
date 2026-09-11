"""Complete paired eight-contrast analysis of the separate five-arm successor panel."""
import argparse
import json
from pathlib import Path

import numpy as np

from .behavioral_analysis import exact_discordance, scenario_key, shard_paths
from .behavioral_development import schedule, validate_coverage, verified_report
from .fixed_response_behavior import ANALYSIS, ARMS, CONTRASTS, TASKS, validate_protocol
from .fixed_response_smoke import verify_episode
from .protocol import sha256, write_json


def holm(values):
    count = ANALYSIS["family"]
    if len(values) != count or any(not 0 <= p <= 1 for p in values):
        raise ValueError("Require the complete eight-test family")
    adjusted, last = [0.] * count, 0.
    for rank, index in enumerate(sorted(range(count), key=lambda i: (values[i], i))):
        last = max(last, min(1., (count - rank) * values[index]))
        adjusted[index] = last
    return adjusted


def validate_panel(panel, expected=None):
    expected = schedule() if expected is None else expected
    if len(expected) != 96:
        raise ValueError("This analysis requires exactly 96 paired scenarios per task")
    if tuple(panel) != TASKS:
        raise ValueError("Both predeclared tasks required")
    for task in TASKS:
        if tuple(panel[task]) != ARMS:
            raise ValueError("All five arms required; partial selection forbidden")
        for arm in ARMS:
            rows = panel[task][arm]
            validate_coverage(rows, expected)
            for reference, row, expected_row in zip(panel[task]["native"], rows, expected):
                if row["arm"] != arm or row["local_seed"] != expected_row["local_seed"]:
                    raise ValueError("Wrong treatment or planner seed stream")
                if (row["initial_state_vector"] != reference["initial_state_vector"] or
                        any(row["result"][k] != reference["result"][k] for k in ("initial_sha256", "goal_sha256"))):
                    raise ValueError("Unpaired initial/goal stimuli")
                diagnostics = [row["seconds"], row["result"]["native_state_distance"], row["result"]["native_reward"]]
                if not np.isfinite(diagnostics).all() or row["seconds"] <= 0:
                    raise ValueError("Invalid complete-episode diagnostics")


def load_panel(root, freeze):
    protocol = validate_protocol(freeze)
    freeze_hash = sha256(freeze / "protocol.json")
    panel, bindings = {}, {}
    for task in TASKS:
        panel[task] = {}
        device_classes = set()
        for arm in ARMS:
            records = []
            for shard in shard_paths(root / task / arm):
                report, digest = verified_report(shard)
                launch = json.loads((shard / "protocol.json").read_text())
                if ((shard / "FAILED.json").exists() or report.get("status") != "fixed_response_behavioral_shard_complete" or
                        report["task"] != task or report["arm"] != arm or
                        report["protocol_sha256"] != sha256(shard / "protocol.json") or
                        launch["freeze_sha256"] != freeze_hash or launch["source_sha256"] != protocol["source_sha256"] or
                        report["scientific_efficacy_measurement"] is not True or report["fresh_confirmation"] is not False or
                        report["parameters_unchanged"] is not True):
                    raise ValueError("Unbound/failed behavioral shard")
                device_classes.add(report["device_name"])
                bindings[str(shard / "report.json")] = digest
                shard_rows = []
                for name, expected in report["episode_files_sha256"].items():
                    if Path(name).name != name or not name.startswith("episode-") or sha256(shard / name) != expected:
                        raise ValueError("Changed or unsafe episode reference")
                    row = json.loads((shard / name).read_text())
                    calls_root = shard / f"calls-{row['episode']:03d}"
                    for file, key in (("unroll_calls.json", "unroll_calls_sha256"), ("action_trace.json", "action_trace_sha256")):
                        if sha256(calls_root / file) != row[key]:
                            raise ValueError("Behavioral action/call trace changed")
                    calls = json.loads((calls_root / "unroll_calls.json").read_text())
                    verify_episode(row["result"], calls)
                    if any(c["backend_calls"] != 1 for c in calls):
                        raise ValueError("Extra online model calls")
                    shard_rows.append(row)
                shard_rows.sort(key=lambda r: r["episode"])
                validate_coverage(shard_rows, launch["expected_episodes"])
                if len(shard_rows) != report["episodes"]:
                    raise ValueError("Shard episode count mismatch")
                records.extend(shard_rows)
            records.sort(key=lambda r: r["episode"])
            panel[task][arm] = records
        if len(device_classes) != 1:
            raise ValueError("Mixed worker classes need a separate paired contract")
    validate_panel(panel)
    return panel, bindings


def analyze(panel, expected=None):
    validate_panel(panel, expected)
    rng = np.random.default_rng(ANALYSIS["seed"])
    rows, tasks = [], {}
    for task in TASKS:
        arms, groups = panel[task], {}
        for i, row in enumerate(arms["native"]):
            groups.setdefault(scenario_key(row), []).append(i)
        clusters = list(groups.values())
        if len(clusters) < 2:
            raise ValueError("Insufficient independent scenario clusters")
        sizes = np.array([len(c) for c in clusters])
        draws = rng.integers(len(clusters), size=(ANALYSIS["replicates"], len(clusters)))
        denominator = sizes[draws].sum(1)
        success = {arm: np.array([r["result"]["native_success"] for r in arms[arm]], float) for arm in ARMS}
        summary = {arm: {"successes": int(success[arm].sum()), "episodes": 96,
            "success_percent": float(success[arm].mean() * 100),
            "mean_episode_seconds": float(np.mean([r["seconds"] for r in arms[arm]])),
            "mean_task_distance": float(np.mean([r["result"]["native_state_distance"] for r in arms[arm]])),
            "mean_reward": float(np.mean([r["result"]["native_reward"] for r in arms[arm]]))} for arm in ARMS}
        for arm, control in CONTRASTS:
            delta = success[arm] - success[control]
            sums = np.array([delta[c].sum() for c in clusters])
            samples = sums[draws].sum(1) / denominator * 100
            tail = .05 / (2 * ANALYSIS["family"])
            interval = np.quantile(samples, [tail, 1-tail])
            homogeneous = all(len(set(success[a][c])) == 1 for a in (arm, control) for c in clusters)
            wins = int(sum(delta[c[0]] > 0 for c in clusters)) if homogeneous else None
            losses = int(sum(delta[c[0]] < 0 for c in clusters)) if homogeneous else None
            rows.append({"task": task, "arm": arm, "control": control,
                "gain_percentage_points": float(delta.mean() * 100),
                "simultaneous_95_interval_percentage_points": interval.tolist(),
                "scenario_clusters": len(clusters), "episodes": 96,
                "discordant_win_clusters": wins, "discordant_loss_clusters": losses,
                "exact_discordance_p": exact_discordance(wins, losses) if homogeneous else None})
        tasks[task] = {"arms": summary, "scenario_clusters": len(clusters),
                       "duplicate_scenario_clusters": [c for c in clusters if len(c) > 1]}
    for row, p in zip(rows, holm([r["exact_discordance_p"] if r["exact_discordance_p"] is not None else 1. for r in rows])):
        row["holm_p"] = p if row["exact_discordance_p"] is not None else None
    for task in TASKS:
        eligible = []
        for arm, control in (("fixed_rank4", "matched_random_fixed_rank4"), ("coupling_only", "matched_random_coupling")):
            native = next(r for r in rows if (r["task"], r["arm"], r["control"]) == (task, arm, "native"))
            random = next(r for r in rows if (r["task"], r["arm"], r["control"]) == (task, arm, control))
            if (native["gain_percentage_points"] >= ANALYSIS["minimum_useful_success_gain_percentage_points"] and
                    native["simultaneous_95_interval_percentage_points"][0] > 0 and
                    random["simultaneous_95_interval_percentage_points"][0] > 0):
                eligible.append((-native["simultaneous_95_interval_percentage_points"][0],
                    tasks[task]["arms"][arm]["mean_episode_seconds"], arm))
        tasks[task]["selected_for_later_confirmation_freeze"] = min(eligible)[-1] if eligible else "native"
    return {"status": "complete_fixed_response_five_arm_development_analyzed", "tasks": tasks,
        "contrasts": rows, "analysis": ANALYSIS, "requested_tasks": 6, "measured_tasks": 2,
        "fresh_confirmation": False, "confirmation_outcomes_authorized": False,
        "training_seed_histories_complete": False, "full_study_complete": False,
        "estimand": "episode-weighted success; paired whole-scenario cluster resampling",
        "selection_on_development_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "freeze", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    panel, inputs = load_panel(args.root, args.freeze)
    report = analyze(panel)
    report.update(freeze_sha256=sha256(args.freeze / "protocol.json"), input_reports_sha256=inputs,
                  analysis_source_sha256=sha256(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
