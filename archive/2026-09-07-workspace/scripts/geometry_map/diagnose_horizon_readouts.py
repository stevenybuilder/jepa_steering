#!/usr/bin/env python3
"""NPZ-only decomposition of frozen visual readout errors on12 H6 references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CANDIDATE_SHA = "07325ea079345e65d961cb8c598955ade195de51f1a8e9b9c85a1236b4cf9543"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_npz(path):
    receipt = json.loads((path.parent/"DONE.json").read_text())
    row = next(r for r in receipt["outputs"] if r["path"] == path.name)
    if not receipt["complete"] or sha(path) != row["sha256"]:
        raise RuntimeError("Incomplete or checksum-mismatched NPZ before access")
    return dict(np.load(path, allow_pickle=False))


def decomposition(predicted, encoded_actual, truth):
    predicted, encoded_actual, truth = (np.asarray(x, dtype=np.float64) for x in (predicted, encoded_actual, truth))
    total, observer, projected = predicted-truth, encoded_actual-truth, predicted-encoded_actual
    square = lambda x: np.square(x).sum(axis=-1)
    cross = 2*np.sum(observer*projected, axis=-1)
    vector_error = float(np.max(np.abs(total-observer-projected)))
    square_error = float(np.max(np.abs(square(total)-square(observer)-square(projected)-cross)))
    if vector_error > 1e-12 or square_error > 1e-12:
        raise RuntimeError("Float64 vector/squared-norm decomposition failed")
    return {"total_error": total, "observer_error": observer, "projected_forecast_discrepancy": projected,
            "total_squared_norm": square(total), "observer_squared_norm": square(observer),
            "projected_squared_norm": square(projected), "cross_term": cross,
            "vector_identity_max_abs": vector_error, "squared_identity_max_abs": square_error}


def persistence_error(start, truth):
    start, truth = np.asarray(start), np.asarray(truth)
    if start.shape != truth.shape or not np.array_equal(start, np.repeat(start[:1], len(start), axis=0)):
        raise RuntimeError("Persistence must repeat horizon0 hand, not oracle preceding future state")
    return start.astype(np.float64)-truth


def grouped_summary(values, indices):
    values = np.asarray(values, dtype=np.float64)
    means = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "episode_range": [float(values.min()), float(values.max())],
            "whole_episode_bootstrap95": np.quantile(means, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("horizon-root", "readout", "candidate", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if sha(args.candidate) != CANDIDATE_SHA:
        raise RuntimeError("Frozen discovery candidate SHA mismatch")
    candidate = dict(np.load(args.candidate, allow_pickle=False))
    if str(candidate["target_name"]) != "world":
        raise RuntimeError("This fixed diagnostic expects the frozen world-coordinate P3 readout")
    readout = verified_npz(args.readout)
    def decode(x):
        return ((x.astype(np.float64)-readout["mean"])/readout["scale"])@readout["coef"].T+readout["intercept"]
    sources, rows, parts, p3_errors, persistence = [], [], [], [], []
    for episode in range(12):
        relative = ("canary-episode0-v1" if episode == 0 else "worker-49766237/results-v1" if episode < 4
                    else "worker-49155754/results-v2" if episode < 8 else "worker-49902461/results-v2")
        path = args.horizon_root/relative/f"episode-{episode:03d}.npz"
        value = verified_npz(path)
        if not np.array_equal(value["episode"], np.full(6, episode)) or not np.array_equal(value["imagined_step"], np.arange(1, 7)):
            raise RuntimeError("Wrong episode/horizon axes")
        if not np.array_equal(value["context_real_raw_step"], np.ones(6)):
            raise RuntimeError("Only initial contexts belong in this frozen diagnostic")
        truth = value["next_hand_xyz"].astype(np.float64)
        predicted, actual = decode(value["predicted_visual_pooled"]), decode(value["actual_visual_pooled"])
        p3 = value["p3_pooled"].astype(np.float64)@candidate["coordinate_coef_raw"].T+candidate["coordinate_intercept_raw"]
        part = decomposition(predicted, actual, truth)
        persist = persistence_error(value["start_hand_xyz"], truth)
        parts.append(part); p3_errors.append(p3-truth); persistence.append(persist)
        sources.append({"episode": episode, "path": str(path), "sha256": sha(path)})
        for h in range(6):
            rows.append({"episode": episode, "imagined_step": h+1, "real_raw_step": int(value["real_raw_step"][h]),
                         "truth_xyz": truth[h].tolist(), "visual_forecast_decoded_xyz": predicted[h].tolist(),
                         "actual_image_decoded_xyz": actual[h].tolist(), "p3_decoded_xyz": p3[h].tolist(),
                         "known_start_hand_xyz": value["start_hand_xyz"][h].tolist(),
                         **{k: np.asarray(part[k][h]).tolist() for k in part if not k.endswith("max_abs")},
                         "p3_error": (p3[h]-truth[h]).tolist(), "persistence_error": persist[h].tolist()})
    indices = np.random.default_rng(2026090507).integers(0, 12, size=(1000, 12))
    p3_squared = np.square(p3_errors).sum(axis=-1)
    persistence_squared = np.square(persistence).sum(axis=-1)
    summaries = []
    for h in range(6):
        metrics = {k: np.array([p[k][h] for p in parts]) for k in ("total_squared_norm", "observer_squared_norm", "projected_squared_norm", "cross_term")}
        metrics.update(p3_squared_norm=p3_squared[:, h], persistence_squared_norm=persistence_squared[:, h])
        summary = {"imagined_step": h+1, "real_raw_step": 1+5*(h+1), "episode_count": 12,
                   "squared_error_decomposition_m2": {k: grouped_summary(v, indices) for k, v in metrics.items()},
                   "errors": {}}
        for name, squared in metrics.items():
            if name == "cross_term":
                continue
            boot = np.sqrt(squared[indices].mean(axis=1))
            summary["errors"][name.removesuffix("_squared_norm")] = {"rms_vector_error_m": float(np.sqrt(squared.mean())),
                     "rmse_coordinate_m": float(np.sqrt(squared.mean()/3)),
                     "rms_vector_whole_episode_bootstrap95_m": np.quantile(boot, [.025, .975]).tolist()}
        summary["p3_minus_visual_squared_error_m2"] = grouped_summary(p3_squared[:, h]-metrics["total_squared_norm"], indices)
        summary["p3_better_than_visual_episode_count"] = int(np.sum(p3_squared[:, h] < metrics["total_squared_norm"]))
        summaries.append(summary)
    result = {"complete": True, "sources": sources, "readout_sha256": sha(args.readout), "candidate_sha256": CANDIDATE_SHA,
              "script_sha256": sha(Path(__file__)), "episode_count": 12, "rows": rows, "horizon_summaries": summaries,
              "identities": {"vector_max_abs": max(p["vector_identity_max_abs"] for p in parts), "squared_max_abs": max(p["squared_identity_max_abs"] for p in parts), "float64_tolerance": 1e-12},
              "bootstrap": {"unit": "whole development episode; same1000 resamples across six horizons", "seed": 2026090507, "replicates": 1000, "interpretation": "descriptive95% percentile intervals, no significance or independent-horizon claim"},
              "definitions": {"total": "g(predicted_visual)-true_hand", "observer": "g(actual_encoded_visual)-true_hand", "projected": "g(predicted_visual)-g(actual_encoded_visual)", "cross": "2*dot(observer,projected); squared norms do not add without this term", "persistence": "known horizon0 hand for all futures, never true previous future hand", "p3": "frozen discovery world-coordinate linear readout on imagined P3"},
              "paper_context": {"url": "https://arxiv.org/html/2608.12959", "section": "7.4", "version": "v1,2026-08-13", "scope": "A separate TwoRoom/LeWorldModel study motivates checking encoded-to-imagined readout transfer; its result is not assumed for JEPA-WM Reach"},
              "limitations": "Projected forecast discrepancy is observer-coordinate disagreement, not a direct estimate of physical dynamics error; observer extrapolation can matter. Actual-image observer error measures this frozen observer on our real rendered frames. No refit, GPU/model/simulator pass, held study, or candidate selection."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir/"horizon_readout_diagnosis.json"
    with output.open("x") as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
    with (args.output_dir/"DONE.json").open("x") as f:
        json.dump({"complete": True, "outputs": [{"path": output.name, "sha256": sha(output), "bytes": output.stat().st_size}]}, f)
    print(json.dumps({"event": "horizon_readout_diagnosis_complete", "sha256": sha(output), "identities": result["identities"], "horizon_summaries": summaries}), flush=True)


if __name__ == "__main__":
    main()
