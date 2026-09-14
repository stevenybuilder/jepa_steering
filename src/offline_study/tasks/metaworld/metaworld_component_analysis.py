"""Frozen, complete-panel analysis of the MetaWorld table component extension."""
import argparse
from pathlib import Path

import numpy as np

from offline_study.evaluation.behavioral_analysis import scenario_key, shard_paths
from offline_study.evaluation.behavioral_development import assigned_rows, schedule, validate_coverage, verified_report
from offline_study.validation.fixed_response_smoke import verify_episode
from offline_study.tasks.metaworld.metaworld_component_behavior import ANALYSIS, ARMS, CONTRASTS, METHOD, TASKS, read, validate_protocol, verify_engineering
from offline_study.core.protocol import sha256, write_json


def validate_panel(panel):
    if set(panel) != set(TASKS):
        raise ValueError("Both component tasks are required")
    for task in TASKS:
        if set(panel[task]) != set(ARMS):
            raise ValueError("All four conditions required; no partial selection")
        native = panel[task]["native"]
        for arm in ARMS:
            rows = panel[task][arm]
            validate_coverage(rows, schedule())
            for row, base, expected in zip(rows, native, schedule()):
                if any(row.get(k) != value for k, value in expected.items()):
                    raise ValueError("Changed canonical episode or persistent RNG seed")
                if row["arm"] != arm or row["initial_state_vector"] != base["initial_state_vector"]:
                    raise ValueError("Unpaired treatment/scenario")
                if any(row["result"][k] != base["result"][k] for k in ("initial_sha256", "goal_sha256")):
                    raise ValueError("Unpaired initial/goal observations")
                if row["result"]["elementary_steps"] != 100 or type(row["result"]["native_success"]) is not bool:
                    raise ValueError("Require complete binary full-episode outcomes")
                if (row["seconds"] <= 0 or not np.isfinite([row["seconds"], row["result"]["native_reward"],
                                                          row["result"]["native_state_distance"]]).all()):
                    raise ValueError("Invalid diagnostic values")


def load_panel(root, freeze, engineering_root):
    protocol = validate_protocol(freeze)
    panel, bindings, engineering = {}, {}, {}
    for task in TASKS:
        panel[task] = {}
        devices_by_episode = {}
        for arm in ARMS:
            rows = []
            for shard in shard_paths(root / task / arm):
                report, digest = verified_report(shard)
                launch = read(shard / "protocol.json")
                if ((shard / "FAILED.json").exists() or report["status"] != METHOD + "_shard_complete" or
                        report["task"] != task or report["arm"] != arm or launch["method"] != METHOD or
                        launch["task"] != task or launch["arm"] != arm or
                        report["protocol_sha256"] != sha256(shard / "protocol.json") or
                        launch["freeze_sha256"] != sha256(freeze / "protocol.json") or
                        launch["source_sha256"] != protocol["source_sha256"] or
                        report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False or
                        report["scientific_efficacy_measurement"] is not True):
                    raise ValueError("Unbound or failed component shard")
                uuid = launch["device_uuid"]
                if Path(uuid).name != uuid:
                    raise ValueError("Unsafe receiving-device reference")
                key = (uuid, task)
                if key not in engineering:
                    engineering[key] = verify_engineering(engineering_root / uuid / task, freeze, task, uuid)
                if engineering[key] != launch["engineering_report_sha256"]:
                    raise ValueError("Receiving proof changed")
                expected = assigned_rows(protocol["episodes"], launch["logical_ranks"])
                if launch["expected_episodes"] != expected:
                    raise ValueError("Altered logical stream membership")
                current = []
                for filename, file_hash in report["episode_files_sha256"].items():
                    if Path(filename).name != filename or not filename.startswith("episode-") or sha256(shard / filename) != file_hash:
                        raise ValueError("Changed or unsafe episode record")
                    row = read(shard / filename)
                    out = shard / f"calls-{row['episode']:03d}"
                    for file, hash_key in (("unroll_calls.json", "unroll_calls_sha256"), ("action_trace.json", "action_trace_sha256")):
                        if sha256(out / file) != row[hash_key]:
                            raise ValueError("Changed raw call/action trace")
                    calls = read(out / "unroll_calls.json")
                    verify_episode(row["result"], calls)
                    if any(c["backend_calls"] != 1 for c in calls):
                        raise ValueError("Additional online forecasts")
                    # Each logical stream stays on the same physical device
                    # for native and every treatment in this extension.
                    ep = row["episode"]
                    if ep in devices_by_episode and devices_by_episode[ep] != uuid:
                        raise ValueError("Paired arms executed on different devices")
                    devices_by_episode[ep] = uuid
                    current.append(row)
                current.sort(key=lambda r: r["episode"])
                validate_coverage(current, expected)
                if report["episodes"] != len(current):
                    raise ValueError("Reported episode count differs")
                rows.extend(current)
                bindings[str(shard / "report.json")] = digest
            panel[task][arm] = sorted(rows, key=lambda r: r["episode"])
    validate_panel(panel)
    return panel, bindings


def analyze(panel):
    validate_panel(panel)
    rng = np.random.default_rng(ANALYSIS["seed"])
    tasks, contrasts = {}, []
    for task in TASKS:
        arms, groups = panel[task], {}
        for i, row in enumerate(arms["native"]):
            groups.setdefault(scenario_key(row), []).append(i)
        clusters = list(groups.values())
        if len(clusters) < 2:
            raise ValueError("Insufficient independent scenario clusters")
        draws = rng.integers(len(clusters), size=(ANALYSIS["replicates"], len(clusters)))
        sizes = np.array([len(c) for c in clusters])
        denominator = sizes[draws].sum(1)
        values = {arm: np.array([r["result"]["native_success"] for r in arms[arm]], dtype=float) for arm in ARMS}
        tasks[task] = {"scenario_clusters": len(clusters), "arms": {arm: {
            "episodes": len(arms[arm]), "successes": int(values[arm].sum()), "success_percent": float(values[arm].mean() * 100),
            "mean_episode_seconds": float(np.mean([r["seconds"] for r in arms[arm]])),
            "mean_reward": float(np.mean([r["result"]["native_reward"] for r in arms[arm]])),
            "mean_task_distance": float(np.mean([r["result"]["native_state_distance"] for r in arms[arm]]))}
            for arm in ARMS}}
        for name, weights in CONTRASTS.items():
            delta = sum(weight * values[arm] for arm, weight in weights.items())
            sums = np.array([delta[c].sum() for c in clusters])
            samples = sums[draws].sum(1) / denominator * 100
            tail = .05 / (2 * ANALYSIS["family"])
            interval = np.quantile(samples, [tail, 1-tail])
            contrasts.append({"task": task, "contrast": name, "weights": weights,
                "gain_percentage_points": float(delta.mean() * 100),
                "simultaneous_95_interval_percentage_points": interval.tolist(), "scenario_clusters": len(clusters)})
    return {"status": METHOD + "_complete_panel_analyzed", "tasks": tasks, "contrasts": contrasts,
        "analysis": ANALYSIS, "fresh_confirmation": False, "full_six_task_table_complete": False,
        "automatic_candidate_selection": False, "historical_native_equivalence_verified": False,
        "historical_core_and_new_panel_may_not_be_pooled_without_separate_verification": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "freeze", "engineering-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    panel, bindings = load_panel(args.root, args.freeze, args.engineering_root)
    report = analyze(panel)
    report.update(freeze_sha256=sha256(args.freeze / "protocol.json"), input_reports_sha256=bindings,
                  analysis_source_sha256=sha256(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
