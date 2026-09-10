"""Frozen seven-arm/two-task behavioral analysis; refuses incomplete panels.

Paired scenario-cluster bootstrap, twelve Bonferroni intervals and Holm-adjusted
exact discordance tests. This is planning-development selection, not confirmation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .behavioral_development import ARMS, CONTRASTS, schedule, validate_coverage, verified_report
from .protocol import sha256, write_json

TASKS = ("reach", "reach-wall")
NAMES = tuple(arm[0] for arm in ARMS)
FAMILY = 12
REPLICATES = 20000
SEED = 2026090722


def shard_paths(root):
    """Physical shard groups and intact single logical streams share one schema."""
    return sorted(p for p in root.iterdir() if p.is_dir() and
                  (p.name.startswith("shard-") or p.name.startswith("rank-"))) if root.is_dir() else []


def exact_discordance(wins, losses):
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.


def holm(values):
    if len(values) != FAMILY or any(not 0 <= p <= 1 for p in values):
        raise ValueError("The complete twelve-test family is required")
    order = sorted(range(FAMILY), key=lambda i: (values[i], i))
    adjusted, previous = [0.] * FAMILY, 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (FAMILY - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def scenario_key(record):
    # Native initial vectors include task goal coordinates. Retain goal pixels as
    # an additional collision guard. Any duplicate is resampled as one cluster.
    payload = [record["initial_state_vector"], record["result"]["goal_sha256"]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_panel(root, freeze):
    frozen = json.loads((freeze / "FROZEN.json").read_text())
    freeze_hash = sha256(freeze / "protocol.json")
    protocol = json.loads((freeze / "protocol.json").read_text())
    if (freeze_hash != frozen["protocol_sha256"] or protocol["arms"] != [list(a) for a in ARMS] or
            protocol["episodes"] != schedule() or protocol["primary_contrasts_per_task"] != [list(c) for c in CONTRASTS]):
        raise ValueError("Changed scientific panel, scenarios or contrast registry")
    # Validate the numerical analysis contract, rather than silently accepting a
    # revised freeze with different prose settings.
    analysis = protocol["analysis"]
    if (analysis["minimum_useful_success_gain_percentage_points"] != 5 or
            "20000 draws, seed 2026090722" not in analysis["intervals"] or
            "Holm across 12" not in analysis["p_values"]):
        raise ValueError("Unrecognized analysis settings")
    panel, bindings = {}, {}
    for task in TASKS:
        panel[task] = {}
        for arm in NAMES:
            records = []
            shards = shard_paths(root / task / arm)
            if not shards:
                raise ValueError(f"Incomplete panel: missing {task}/{arm}")
            for shard in shards:
                report, report_hash = verified_report(shard)
                launch = json.loads((shard / "protocol.json").read_text())
                status = ("native_planning_development_shard_complete" if arm == "native" else
                          "fixed_planning_development_candidate_shard_complete")
                if (report["protocol_sha256"] != sha256(shard / "protocol.json") or
                        report["status"] != status or report["task"] != task or report["arm"] != arm or
                        launch["freeze_sha256"] != freeze_hash or launch["task"] != task or
                        launch["arm"] != arm or report["fresh_confirmation"] is not False or
                        report["scientific_efficacy_measurement"] is not True or
                        report["parameters_unchanged"] is not True or
                        report["episodes"] != len(report["episode_files_sha256"])):
                    raise ValueError("Unbound or malformed completed behavioral shard")
                bindings[str(shard / "report.json")] = report_hash
                for name, digest in report["episode_files_sha256"].items():
                    if Path(name).name != name or not name.startswith("episode-"):
                        raise ValueError("Unsafe episode reference")
                    if sha256(shard / name) != digest:
                        raise ValueError("Episode checksum mismatch")
                    row = json.loads((shard / name).read_text())
                    if row["arm"] != arm or row["local_seed"] != schedule()[row["episode"]]["local_seed"]:
                        raise ValueError("Wrong treatment label or logical RNG stream")
                    records.append(row)
                if "unroll_calls_sha256" in records[-1]:
                    for row in records[-report["episodes"]:]:
                        path = shard / f"calls-{row['episode']:03d}" / "unroll_calls.json"
                        if sha256(path) != row["unroll_calls_sha256"]:
                            raise ValueError("Missing candidate call trace")
            records.sort(key=lambda r: r["episode"])
            validate_coverage(records, schedule())
            panel[task][arm] = records
        native = panel[task]["native"]
        for arm in NAMES:
            for baseline, candidate in zip(native, panel[task][arm]):
                if (candidate["initial_state_vector"] != baseline["initial_state_vector"] or
                        any(candidate["result"][key] != baseline["result"][key]
                            for key in ("initial_sha256", "goal_sha256"))):
                    raise ValueError("Candidate and baseline stimuli are unpaired")
    return protocol, panel, bindings


def analyze(panel):
    rng = np.random.default_rng(SEED)
    rows, task_reports = [], {}
    for task in TASKS:
        arms = panel[task]
        if tuple(arms) != NAMES or any(len(arms[name]) != 96 for name in NAMES):
            raise ValueError("Require all 96 episodes for all seven conditions")
        groups = {}
        for index, row in enumerate(arms["native"]):
            groups.setdefault(scenario_key(row), []).append(index)
        clusters = list(groups.values())
        if len(clusters) < 2:
            raise ValueError("Insufficient distinct paired scenario clusters")
        sizes = np.array([len(c) for c in clusters])
        draws = rng.integers(len(clusters), size=(REPLICATES, len(clusters)))
        denominator = sizes[draws].sum(1)
        success = {arm: np.array([r["result"]["native_success"] for r in arms[arm]], dtype=float) for arm in NAMES}
        if any(not np.isin(value, [0., 1.]).all() for value in success.values()):
            raise ValueError("Invalid native binary success endpoint")
        summary = {}
        for arm in NAMES:
            seconds = [r["seconds"] for r in arms[arm]]
            distances = [r["result"]["native_state_distance"] for r in arms[arm]]
            rewards = [r["result"]["native_reward"] for r in arms[arm]]
            if not np.isfinite(seconds + distances + rewards).all() or min(seconds) <= 0:
                raise ValueError("Nonfinite or invalid complete-episode diagnostics")
            summary[arm] = {"successes": int(success[arm].sum()), "episodes": 96,
                "success_percent": float(success[arm].mean() * 100),
                "mean_episode_seconds": float(np.mean(seconds)),
                "mean_native_distance": float(np.mean(distances)), "mean_native_reward": float(np.mean(rewards))}
        for arm, control in CONTRASTS:
            delta = success[arm] - success[control]
            sums = np.array([delta[c].sum() for c in clusters])
            samples = sums[draws].sum(1) / denominator * 100
            interval = np.quantile(samples, [.05 / (2 * FAMILY), 1 - .05 / (2 * FAMILY)])
            homogeneous = all(len(set(success[a][c])) == 1 for a in (arm, control) for c in clusters)
            # Repeated scenarios with differing binary results do not become
            # independent Bernoulli trials. Their bootstrap still uses clusters;
            # do not fabricate an exact McNemar p-value from fractional outcomes.
            wins = sum(delta[c[0]] > 0 for c in clusters) if homogeneous else None
            losses = sum(delta[c[0]] < 0 for c in clusters) if homogeneous else None
            p = exact_discordance(int(wins), int(losses)) if homogeneous else 1.
            rows.append({"task": task, "arm": arm, "control": control,
                "gain_percentage_points": float(delta.mean() * 100),
                "simultaneous_95_interval_percentage_points": interval.tolist(),
                "scenario_clusters": len(clusters), "episodes": 96,
                "discordant_win_clusters": int(wins) if homogeneous else None,
                "discordant_loss_clusters": int(losses) if homogeneous else None,
                "exact_discordance_p": p if homogeneous else None,
                "p_for_holm": p, "exact_test_valid": homogeneous})
        task_reports[task] = {"arms": summary, "scenario_clusters": len(clusters),
                              "duplicate_scenario_clusters": [c for c in clusters if len(c) > 1]}
    for row, adjusted in zip(rows, holm([row["p_for_holm"] for row in rows])):
        row["holm_p"] = adjusted if row["exact_test_valid"] else None
        del row["p_for_holm"]
    for task in TASKS:
        eligible = []
        for arm, control in (("coupling_only", "matched_random_coupling"),
                             ("rank4_only", "matched_random_rank4"), ("combined", "matched_random_combined")):
            native = next(r for r in rows if (r["task"], r["arm"], r["control"]) == (task, arm, "native"))
            random = next(r for r in rows if (r["task"], r["arm"], r["control"]) == (task, arm, control))
            if (native["gain_percentage_points"] >= 5 and native["simultaneous_95_interval_percentage_points"][0] > 0
                    and random["simultaneous_95_interval_percentage_points"][0] > 0):
                eligible.append((-native["simultaneous_95_interval_percentage_points"][0],
                    task_reports[task]["arms"][arm]["mean_episode_seconds"], 2 if arm == "combined" else 1, arm))
        task_reports[task]["selected_for_later_confirmation_freeze"] = min(eligible)[-1] if eligible else "native"
        task_reports[task]["eligible_candidates"] = sorted(item[-1] for item in eligible)
    return {"status": "complete_metaworld_planning_development_panel_analyzed",
        "tasks": task_reports, "contrasts": rows, "multiplicity_family": FAMILY,
        "bootstrap_replicates": REPLICATES, "bootstrap_seed": SEED,
        "estimand": "native episode-weighted success; paired whole-scenario cluster resampling",
        "fresh_confirmation": False, "confirmation_outcomes_authorized": False,
        "requested_tasks": 6, "measured_tasks": 2, "full_study_complete": False,
        "training_seed_histories_complete": False, "selection_on_development_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "freeze", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _, panel, bindings = load_panel(args.root, args.freeze)
    report = analyze(panel)
    report.update(freeze_sha256=sha256(args.freeze / "protocol.json"), input_reports_sha256=bindings,
                  analysis_source_sha256=sha256(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
