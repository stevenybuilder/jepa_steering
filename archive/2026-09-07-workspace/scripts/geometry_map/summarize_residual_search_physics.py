"""Paired, episode-weighted summaries of completed residual-search physics only."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


METRICS = ("hand_goal_progress_m", "final_hand_goal_distance_m",
           "minimum_hand_goal_distance_m", "native_reward_sum",
           "native_reward_max", "native_ever_success", "native_final_success")


def summarize(rows, episodes, *, seen_development_repeat=False):
    episodes = sorted(episodes)
    allowed = (8, 9, 10, 11) if seen_development_repeat else (0, 4, 7)
    if len(set(episodes)) != len(episodes) or not episodes or any(e not in allowed for e in episodes):
        raise ValueError("Expected distinct starts from explicitly declared development group")
    table = {}
    for row in rows:
        key = (row["episode"], row["arm"])
        if key in table or key[0] not in episodes:
            raise ValueError("Duplicate or unexpected episode/arm")
        table[key] = row
    arms = sorted({arm for _, arm in table})
    if "unsteered" not in arms or set(table) != {(e, a) for e in episodes for a in arms}:
        raise ValueError("Incomplete paired design")
    result = []
    for arm in arms:
        values = [table[e, arm] for e in episodes]
        comparators = ["unsteered"]
        if "projected-unsteered" in arms:
            comparators.append("projected-unsteered")
        sham = arm.removesuffix("-edit")+"-sham"
        if arm.endswith("-edit") and sham in arms:
            comparators.append(sham)
        metrics = {}
        for metric in METRICS:
            present = [metric in row for row in values]
            if not all(present):
                metrics[metric] = {"status": "not_captured", "available_count": sum(present)}
                continue
            vals = [row[metric] for row in values]
            metrics[metric] = {"by_episode": vals, "mean": mean(vals), "minimum": min(vals), "maximum": max(vals)}
            if metric.endswith("success"):
                metrics[metric]["count"] = sum(bool(v) for v in vals)
        comparisons = {}
        for other in comparators:
            paired = {}
            for metric in METRICS:
                if all(metric in table[e, a] for e in episodes for a in (arm, other)):
                    differences = [float(table[e, arm][metric])-float(table[e, other][metric]) for e in episodes]
                    paired[metric] = {"difference_by_episode": differences, "mean_difference": mean(differences),
                                      "positive_count": sum(d > 0 for d in differences),
                                      "negative_count": sum(d < 0 for d in differences)}
            comparisons[other] = paired
        forecasts = []
        if all("replans" in row for row in values):
            for horizon in (1, 2, 3):
                by_episode = []
                for row in values:
                    errors = [error for replan in row["replans"] for error in replan["prediction_errors"]
                              if error["imagined_horizon"] == horizon]
                    if not errors:
                        raise ValueError("Missing executed-prefix forecast horizon")
                    by_episode.append({"episode": row["episode"], "replan_count": len(errors),
                                       **{key: mean(error[key] for error in errors) for key in ("visual_mse", "proprio_mse")}})
                forecasts.append({"imagined_horizon": horizon, "by_episode": by_episode,
                                  **{key: mean(row[key] for row in by_episode) for key in ("visual_mse", "proprio_mse")}})
        result.append({"arm": arm, "candidate": values[0].get("candidate"), "metrics": metrics,
                       "comparisons": comparisons, "executed_prefix_forecast_errors": forecasts,
                       "forecast_interpretation": "Each arm's own selected actions and visited states; error against actual encoded frames. Episode-weighted, not a same-action causal comparison. Missing for short reports."})
    return {"complete": True, "episodes": episodes, "episode_count": len(episodes), "arm_count": len(arms),
            "seen_development_repeat": seen_development_repeat,
            "interpretation": "Adaptive DEVELOPMENT, descriptive paired effects; not untouched confirmation. Positive distance difference is harm, positive progress/reward difference is benefit. No endpoint substitution.",
            "rows": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 4, 7])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seen-development-repeat", action="store_true", help="Explicit previously seen8..11 group, never confirmation")
    args = parser.parse_args()
    rows, sources = [], []
    for path in args.inputs:
        receipt = json.loads((path.parent/"DONE.json").read_text())
        record = next(r for r in receipt["outputs"] if r["path"] == path.name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not receipt["complete"] or digest != record["sha256"]:
            raise ValueError("Incomplete or invalid report receipt")
        report = json.loads(path.read_text())
        if not report["complete"]:
            raise ValueError("Partial report not accepted")
        rows.extend(report["rows"])
        sources.append({"path": str(path), "sha256": digest, "seconds": report.get("seconds")})
    result = summarize(rows, args.episodes, seen_development_repeat=args.seen_development_repeat)
    result["sources"] = sources
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "arms": result["arm_count"]}))


if __name__ == "__main__":
    main()
