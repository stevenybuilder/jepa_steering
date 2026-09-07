"""Conditional fixed-bank contrast pilot: distinguish action information from bias.

Uses verified loading, score replacement and promotion helpers from run_push.
Only prediction features and TRAIN targets are centered; no DEVELOPMENT outcome
enters fitting. Every run first checks all three failed original family summaries.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from correction_math import fit_predict, cap_edit, norm_match, corrected_cost
from run_push import SEED, CAPS, sha, array_sha, write, load_verified, values, promotion


def center_features(features, group):
    output = np.array(features, copy=True)
    for identity in np.unique(group):
        index = np.flatnonzero(group == identity)
        block = output[index, 384:].astype(np.float64)
        output[index, 384:] = (block - block.mean(0, keepdims=True)).astype(output.dtype)
    return output


def main(a):
    started = time.monotonic()
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    protocol_path = Path(__file__).with_name("protocol-contrast-v1.json")
    protocol = json.loads(protocol_path.read_text())
    prerequisites = []
    prior_families = set()
    for summary_path in a.prerequisite_summaries:
        previous = json.loads(summary_path.read_text())
        previous_family = previous.get("family")
        gates = previous.get("correction_gates", [])
        if (not previous.get("complete") or previous.get("held_access", True)
                or previous_family not in ("linear", "rbf", "knn")
                or previous_family in prior_families or len(gates) != 3
                or sorted(gate.get("cap_fraction") for gate in gates) != list(CAPS)
                or any(gate.get("passes", True) for gate in gates)
                or previous.get("candidate_promoted_to_fresh_development", True)):
            raise ValueError("Contrast pilot requires all three completed original correction families to fail")
        prior_families.add(previous_family)
        prerequisites.append(dict(path=str(summary_path), family=previous_family,
                                  sha256=sha(summary_path), all_correction_gates_failed=True))
    if prior_families != {"linear", "rbf", "knn"}:
        raise ValueError("Missing original correction-family prerequisite")
    if tuple(protocol["output_edit_cap_fractions"]) != CAPS or a.family not in protocol["families"]:
        raise ValueError("Frozen family/cap protocol differs")
    if len(a.fit) != 2 or len(a.development) != 4:
        raise ValueError("Exactly two FIT shards and four DEVELOPMENT banks required")
    fit_parts, sources = [], []
    for path in a.fit:
        load_started = time.monotonic()
        print(json.dumps(dict(event="load_start", kind="fit", path=str(path))), flush=True)
        part, source = load_verified(path, "fit")
        print(json.dumps(dict(event="load_complete", kind="fit", path=str(path),
                              seconds=time.monotonic()-load_started)), flush=True)
        fit_parts.append(part)
        sources.append(source)
    train = {key: np.concatenate([part[key] for part in fit_parts]) for key in fit_parts[0]}
    order = np.lexsort((train["candidate"], train["group"]))
    train = {key: value[order] for key, value in train.items()}
    groups, counts = np.unique(train["group"], return_counts=True)
    if len(groups) != 16 or len(train["x"]) != 112 or not np.all(counts == 7):
        raise ValueError("Must retain exactly16 FIT groups with7 candidates each")
    for group in groups:
        if not np.array_equal(train["candidate"][train["group"] == group], np.arange(7)):
            raise ValueError("FIT action rows duplicated or missing")
    banks = {}
    for path in a.development:
        load_started = time.monotonic()
        print(json.dumps(dict(event="load_start", kind="development", path=str(path))), flush=True)
        bank, source = load_verified(path, "development")
        print(json.dumps(dict(event="load_complete", kind="development", path=str(path),
                              seconds=time.monotonic()-load_started)), flush=True)
        episode = int(bank["group"][0])
        if episode in banks:
            raise ValueError("Duplicate DEVELOPMENT episode")
        banks[episode] = bank
        sources.append(source)
    if sorted(banks) != list(range(4)):
        raise ValueError("Missing DEVELOPMENT episode")
    original_context = train["x"][:, :384].copy()
    train["x"] = center_features(train["x"], train["group"])
    if not np.array_equal(train["x"][:, :384], original_context):
        raise ValueError("Current context must remain unchanged before TRAIN standardization")
    for episode in range(4):
        banks[episode]["x"] = center_features(banks[episode]["x"], banks[episode]["group"])
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / "protocol.json", protocol)
    write(a.output / "sources.json", dict(sources=sources, prerequisites=prerequisites, family=a.family, seed=SEED,
         protocol_sha256=sha(protocol_path), runner_sha256=sha(__file__),
         shared_runner_helpers_sha256=sha(Path(__file__).with_name("run_push.py")),
         math_sha256=sha(Path(__file__).with_name("correction_math.py")), held_access=False))
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_started = time.monotonic()
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, device=device, dtype=dtype)
    train_x, train_y = tensor(train["x"]), tensor(train["residual"])
    train_y = train_y.clone()
    for group in groups:
        index = tensor(np.flatnonzero(train["group"] == group), torch.long)
        target = train_y[index].double()
        train_y[index] = (target - target.mean(0, keepdim=True)).float()
    print(json.dumps(dict(event="fit_start", family=a.family, rows=112)), flush=True)
    test_x = tensor(np.concatenate([banks[ep]["x"] for ep in range(4)]))
    rng = np.random.default_rng(SEED)
    label_permutation = np.arange(112)
    for group in groups:
        index = np.flatnonzero(train["group"] == group)
        label_permutation[index] = rng.permutation(index)
    if not np.array_equal(train["group"][label_permutation], train["group"]):
        raise ValueError("Target permutation crossed a TRAIN state")
    coordinate_permutation = rng.permutation(98304)
    coordinate_signs = rng.choice(np.asarray([-1., 1.], dtype=np.float32), size=98304)
    semantic, semantic_meta = fit_predict(train_x, train_y, test_x, train["group"], a.family)
    shuffled, shuffled_meta = fit_predict(train_x, train_y[tensor(label_permutation, torch.long)],
                                          test_x, train["group"], a.family)
    for episode in range(4):
        block = slice(episode * 64, (episode + 1) * 64)
        for predicted in (semantic, shuffled):
            value = predicted[block].double()
            predicted[block] = (value - value.mean(0, keepdim=True)).float()
    print(json.dumps(dict(event="fit_complete", family=a.family)), flush=True)
    fit_receipt = dict(complete=True, family=a.family, pilot="within_state_contrast_v1", fit_groups=groups.tolist(),
        fit_rows=112, development_targets_used=False, fit_before_physical_scoring=True,
        semantic=semantic_meta, shuffled=shuffled_meta, train_label_permutation=label_permutation.tolist(),
        coordinate_permutation_seed=SEED, random_draw_order=["within_state_permutations_in_sorted_TRAIN_group_order", "coordinate_permutation98304", "coordinate_signs98304"],
        coordinate_permutation_sha256=array_sha(coordinate_permutation),
        coordinate_signs_sha256=array_sha(coordinate_signs),
        semantic_predictions_sha256=array_sha(semantic), shuffled_predictions_sha256=array_sha(shuffled),
        targets="TRAIN residuals centered within each seven-action initial state",
        predictions="Semantic and shuffled predicted edits centered within each64candidateDEVbank before capping",
        feature_transform="Current context first384 kept; predictedmean/actions remaining444 centered within each initialstate",
        test_transform="Unlabeled fixed-bank transduction; no test outcome enters centering",
        centered_train_targets_sha256=array_sha(train_y),
        centered_train_features_sha256=array_sha(train["x"]))
    write(a.output / "FROZEN_FIT.json", fit_receipt)
    # Evaluation labels are read only after the estimator/predictions receipt.
    coverage_document = json.loads(a.coverage.read_text())
    if not coverage_document.get("complete") or coverage_document.get("held_access", False):
        raise ValueError("Coverage must be complete original DEVELOPMENT only")
    coverage_rows = {int(row["episode"]): row for row in coverage_document["rows"]}
    if sorted(coverage_rows) != list(range(4)) or len(coverage_document["rows"]) != 4:
        raise ValueError("Coverage groups differ")
    rows, bank_reports, native_mse = [], {}, {}
    perm = tensor(coordinate_permutation, torch.long)
    signs = tensor(coordinate_signs)

    def score(ep, arm, cost, edit=None, parameter=None, penalty=None):
        b = banks[ep]
        costs = values(cost)
        if costs.shape != (64,) or not np.isfinite(costs).all():
            raise ValueError("Expected64 finite scores")
        truth = tensor(b["residual"], torch.float64)
        mse_vector = truth.square().mean(1) if edit is None else (edit.double() - truth).square().mean(1)
        centered_truth = truth - truth.mean(0, keepdim=True)
        centered_edit = torch.zeros_like(truth) if edit is None else edit.double() - edit.double().mean(0, keepdim=True)
        contrast_mse_vector = values((centered_edit - centered_truth).square().mean(1))
        mse_vector = values(mse_vector)
        picked = int(np.argmin(costs))  # candidate rows sorted: stable first tie
        baseline = int(np.argmin(b["native_cost"]))
        coverage = np.asarray(coverage_rows[ep]["goal_aligned_coverage_by_candidate"], dtype=float)
        default = np.asarray(coverage_rows[ep]["default_raw_coverage_by_candidate"], dtype=float)
        if coverage.shape != (64,) or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1 + 1e-8)):
            raise ValueError("Invalid coverage outcome vector")
        xy = np.linalg.norm(b["physical_state"][:, :4] - b["goal_state"][:, :4], axis=1)
        row = dict(episode=ep, arm=arm, selected_index=picked, native_selected_index=baseline,
            choice_changed=picked != baseline, selected_goal_coverage=float(coverage[picked]),
            coverage_gain_vs_native=float(coverage[picked] - coverage[baseline]),
            selected_xy_distance_px=float(xy[picked]), xy_distance_delta_vs_native_px=float(xy[picked] - xy[baseline]),
            selected_default_painted_goal_raw_coverage=float(default[picked]),
            forecast_mse_all64=float(mse_vector.mean()), selected_forecast_mse=float(mse_vector[picked]),
            residual_contrast_mse_all64=float(contrast_mse_vector.mean()),
            residual_contrast_mse_by_candidate=contrast_mse_vector.tolist(),
            selected_native_forecast_mse=float(native_mse[ep][picked]) if ep in native_mse else float(mse_vector[picked]),
            cost_by_candidate=costs.tolist(), forecast_mse_by_candidate=mse_vector.tolist(),
            goal_coverage_by_candidate=coverage.tolist(), xy_distance_by_candidate_px=xy.tolist(),
            default_painted_goal_raw_coverage_by_candidate=default.tolist())
        if parameter:
            row.update(parameter)
        if edit is not None:
            row["edit_l2_by_candidate"] = values(edit.double().norm(dim=1)).tolist()
        if penalty is not None:
            row["penalty_by_candidate"] = values(penalty).tolist()
        rows.append(row)
        return row, mse_vector

    for ep in range(4):
        b = banks[ep]
        pred, goal = tensor(b["predicted"]), tensor(b["goal"])
        native = tensor(b["native_cost"], torch.float64)
        zero = torch.zeros_like(pred)
        identity = corrected_cost(native, pred, goal, zero)
        if not torch.equal(identity, native) or int(identity.argmin()) != int(native.argmin()):
            raise ValueError("Exact no-edit scalar/argmin invariant failed")
        _, native_mse[ep] = score(ep, "native", native)
        oracle_delta = tensor(b["residual"])
        oracle_cost = corrected_cost(native, pred, goal, oracle_delta)
        oracle_row, _ = score(ep, "oracle_actual_visual_correction", oracle_cost, oracle_delta)
        oracle_row.update(privileged_diagnostic_only=True, runtime_candidate=False,
                          promotion_eligible=False,
                          scope="Actual terminal visual substituted; native predicted proprio unchanged")
        semantic_ep, shuffled_ep = semantic[ep * 64:(ep + 1) * 64], shuffled[ep * 64:(ep + 1) * 64]
        for fraction in CAPS:
            edit = cap_edit(semantic_ep, pred, fraction)
            sham1 = norm_match(shuffled_ep, edit)
            sham2 = norm_match(edit[:, perm] * signs, edit)
            for name, delta in [("semantic", edit), ("within_state_permutation_sham", sham1),
                                ("signed_coordinate_sham", sham2)]:
                cost = corrected_cost(native, pred, goal, delta)
                score(ep, name, cost, delta, dict(cap_fraction=fraction))
            for sham in (sham1, sham2):
                if not torch.allclose(sham.double().norm(dim=1), edit.double().norm(dim=1), atol=1e-7, rtol=1e-6):
                    raise ValueError("Matched sham norm guard failed")
        bank_reports[ep] = dict(episode=ep, original_cem_action_index=0,
            original_cem_goal_coverage=float(coverage_rows[ep]["goal_aligned_coverage_by_candidate"][0]),
            native_bank_argmin=int(native.argmin()),
            physical_states=b["physical_state"].tolist(), goals=b["goal_state"].tolist(),
            native_cost_by_candidate=b["native_cost"].tolist())
        print(json.dumps(dict(event="contrast_state_scored", family=a.family, episode=ep)), flush=True)
    gates = [promotion(rows, "semantic", ["within_state_permutation_sham", "signed_coordinate_sham"],
                       "cap_fraction", fraction) for fraction in CAPS]
    passing = [gate for gate in gates if gate["passes"]]
    correction_rows = list(rows)
    write(a.output / "correction.json", dict(complete=True, rows=correction_rows, gates=gates,
        smallest_passing_cap=passing[0]["cap_fraction"] if passing else None))
    risk_report = dict(status="not_part_of_contrast_protocol", rows=[], gates=[])
    write(a.output / "risk.json", risk_report)
    if device.type == "cuda":
        torch.cuda.synchronize()
    process_gpu_seconds = time.monotonic() - gpu_started if device.type == "cuda" else 0.
    result = dict(complete=True, family=a.family, pilot="within_state_contrast_v1", fit_groups=16, fit_rows=112,
        development_states=4, development_candidates=256, held_access=False,
        exploratory_only=True, confirmation_claim=False, correction_gates=gates,
        smallest_passing_cap=passing[0]["cap_fraction"] if passing else None,
        candidate_promoted_to_fresh_development=bool(passing),
        risk=risk_report, correction_rows=correction_rows, banks=bank_reports,
        coverage_sha256=sha(a.coverage), sources_sha256=sha(a.output / "sources.json"),
        elapsed_seconds=time.monotonic() - started, elapsed_process_gpu_seconds=process_gpu_seconds,
        gpu_time_definition="Wall time owning CUDA process during fit/scoring; not kernel busy-time", device=str(device),
        limitations=["Four adaptively reused states; candidate actions are dependent",
                     "Output correction does not identify an internal causal mechanism",
                     "Fixed-bank native argmin and original CEM action are separately reported",
                     "Predictedfuture/actions features and predicted edits centered over64candidateDEVbank; composition-dependent unlabeled transduction",
                     "Within-state TRAIN target permutation preserves state residual distribution",
                     "Currentcontext conditioning retained; linear additive context effects cancel on output centering",
                     "Mean-zero raw correction can acquire a small mean after per-row norm capping",
                     "Risk branch is not part of this conditional contrast protocol"])
    write(a.output / "summary.json", result)
    outputs = [path for path in sorted(a.output.iterdir()) if path.is_file()]
    write(a.output / "DONE.json", dict(complete=True, family=a.family, held_access=False,
        elapsed_seconds=result["elapsed_seconds"], elapsed_process_gpu_seconds=process_gpu_seconds,
        outputs=[dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size) for path in outputs]))
    print(json.dumps(dict(event="complete", family=a.family, correction_passes=len(passing),
                         output=str(a.output), elapsed_seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--development", type=Path, nargs="+", required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--prerequisite-summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--family", choices=("linear", "rbf", "knn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
