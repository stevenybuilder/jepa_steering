"""Frozen-protocol Push-T residual correction and conditional risk screening.

Only TRAIN residuals enter estimators. DEVELOPMENT physical outcomes and true
residuals are accessed for scoring after predicted edits have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from correction_math import fit_predict, cap_edit, norm_match, corrected_cost


SEED = 20260906
CAPS = (0.005, 0.02, 0.1)
STRENGTHS = (0.1, 0.3, 1.0)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def array_sha(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_verified(path, kind):
    path = Path(path)
    root = path if path.is_dir() else path.parent
    receipt_path = root / "DONE.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("complete") or receipt.get("kind") != kind:
        raise ValueError("Incomplete or wrong split before array loading: " + str(root))
    if receipt.get("held_access", False) or not receipt.get("split_checked_before_tensor_load"):
        raise ValueError("Missing split access gate")
    entries = [entry for entry in receipt["outputs"] if entry["path"] == "data.npz"]
    if len(entries) != 1:
        raise ValueError("Exactly one data.npz checksum expected")
    npz = root / "data.npz"
    if sha(npz) != entries[0]["sha256"]:
        raise ValueError("Extracted array checksum mismatch")
    with np.load(npz, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    required = {"x", "residual", "predicted", "goal", "group", "candidate",
                "native_cost", "physical_state", "goal_state", "raw_action_norm"}
    if set(arrays) != required:
        raise ValueError("Extracted fields changed")
    rows = len(arrays["x"])
    if rows != receipt["rows"] or arrays["x"].shape != (rows, 828):
        raise ValueError("Feature/receipt row contract changed")
    if any(arrays[key].shape != (rows, 98304) for key in ("residual", "predicted", "goal")):
        raise ValueError("Native visual support changed")
    for key, value in arrays.items():
        if value.shape[0] != rows or not np.isfinite(value).all():
            raise ValueError("Nonfinite or inconsistent array: " + key)
    for key in ("group", "candidate", "native_cost", "raw_action_norm"):
        if arrays[key].ndim != 1:
            raise ValueError("Expected scalar row field: " + key)
    if kind == "development":
        episode = receipt.get("episode")
        if episode not in range(4) or rows != 64 or not np.all(arrays["group"] == episode):
            raise ValueError("Only four original DEVELOPMENT groups allowed")
        if not np.array_equal(np.sort(arrays["candidate"]), np.arange(64)):
            raise ValueError("Candidate completeness failure")
    elif set(np.unique(arrays["group"])) != set(receipt["groups"]):
        raise ValueError("FIT receipt groups differ")
    order = np.lexsort((arrays["candidate"], arrays["group"]))
    arrays = {key: value[order] for key, value in arrays.items()}
    source = dict(root=str(root), done_sha256=sha(receipt_path),
                  data_sha256=entries[0]["sha256"], receipt=receipt)
    return arrays, source


def values(tensor):
    return tensor.detach().cpu().double().numpy()


def promotion(rows, semantic_arm, controls, parameter_name, parameter):
    relevant = {arm: sorted([row for row in rows if row["arm"] == arm and
                             row.get(parameter_name) == parameter], key=lambda r: r["episode"])
                for arm in [semantic_arm, "native"] + controls}
    relevant["native"] = sorted([row for row in rows if row["arm"] == "native"],
                                key=lambda r: r["episode"])
    if any([row["episode"] for row in arm_rows] != list(range(4)) for arm_rows in relevant.values()):
        raise ValueError("Gate requires all four states under every control")
    coverage = {arm: np.asarray([row["selected_goal_coverage"] for row in arm_rows])
                for arm, arm_rows in relevant.items()}
    gains = {arm: coverage[semantic_arm] - coverage[arm] for arm in ["native"] + controls}
    mse = float(np.mean([row["forecast_mse_all64"] for row in relevant[semantic_arm]]))
    baseline_mse = float(np.mean([row["forecast_mse_all64"] for row in relevant["native"]]))
    checks = {"mean_coverage_gain_ge_002_vs_" + arm: bool(delta.mean() >= .02)
              for arm, delta in gains.items()}
    checks.update(positive_in_at_least_three_states=bool(np.sum(gains["native"] > 0) >= 3),
                  no_state_native_coverage_harm_over_005=bool(np.min(gains["native"]) >= -.05),
                  mean_forecast_mse_le_105_native=bool(mse <= 1.05 * baseline_mse))
    return dict(arm=semantic_arm, **{parameter_name: parameter}, passes=all(checks.values()),
                checks=checks, gains_by_state={arm: delta.tolist() for arm, delta in gains.items()},
                mean_gains={arm: float(delta.mean()) for arm, delta in gains.items()},
                mean_forecast_mse=mse, native_mean_forecast_mse=baseline_mse,
                forecast_mse_ratio=mse / baseline_mse if baseline_mse else None)


def main(a):
    started = time.monotonic()
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    protocol_path = Path(__file__).with_name("protocol-v1.json")
    protocol = json.loads(protocol_path.read_text())
    if tuple(protocol["output_edit_cap_fractions"]) != CAPS or a.family not in protocol["families"]:
        raise ValueError("Frozen family/cap protocol differs")
    if len(a.fit) != 2 or len(a.development) != 4:
        raise ValueError("Exactly two FIT shards and four DEVELOPMENT banks required")
    fit_parts, sources = [], []
    for path in a.fit:
        part, source = load_verified(path, "fit")
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
        bank, source = load_verified(path, "development")
        episode = int(bank["group"][0])
        if episode in banks:
            raise ValueError("Duplicate DEVELOPMENT episode")
        banks[episode] = bank
        sources.append(source)
    if sorted(banks) != list(range(4)):
        raise ValueError("Missing DEVELOPMENT episode")
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / "protocol.json", protocol)
    write(a.output / "sources.json", dict(sources=sources, family=a.family, seed=SEED,
         protocol_sha256=sha(protocol_path), runner_sha256=sha(__file__),
         math_sha256=sha(Path(__file__).with_name("correction_math.py")), held_access=False))
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_started = time.monotonic()
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, device=device, dtype=dtype)
    train_x, train_y = tensor(train["x"]), tensor(train["residual"])
    test_x = tensor(np.concatenate([banks[ep]["x"] for ep in range(4)]))
    rng = np.random.default_rng(SEED)
    label_permutation = rng.permutation(112)
    coordinate_permutation = rng.permutation(98304)
    coordinate_signs = rng.choice(np.asarray([-1., 1.], dtype=np.float32), size=98304)
    semantic, semantic_meta = fit_predict(train_x, train_y, test_x, train["group"], a.family)
    shuffled, shuffled_meta = fit_predict(train_x, train_y[tensor(label_permutation, torch.long)],
                                          test_x, train["group"], a.family)
    mean_delta = train_y.double().mean(0).float()
    fit_receipt = dict(complete=True, family=a.family, fit_groups=groups.tolist(),
        fit_rows=112, development_targets_used=False, fit_before_physical_scoring=True,
        semantic=semantic_meta, shuffled=shuffled_meta, train_label_permutation=label_permutation.tolist(),
        coordinate_permutation_seed=SEED, random_draw_order=["train_label_permutation112", "coordinate_permutation98304", "coordinate_signs98304"],
        coordinate_permutation_sha256=array_sha(coordinate_permutation),
        coordinate_signs_sha256=array_sha(coordinate_signs),
        semantic_predictions_sha256=array_sha(semantic), shuffled_predictions_sha256=array_sha(shuffled),
        mean_delta_sha256=array_sha(mean_delta))
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
            mean_edit = cap_edit(mean_delta.expand_as(pred), pred, fraction)
            for name, delta in [("semantic", edit), ("train_permutation_sham", sham1),
                                ("signed_coordinate_sham", sham2), ("train_mean", mean_edit)]:
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
        print(json.dumps(dict(event="correction_state_scored", family=a.family, episode=ep)), flush=True)
    gates = [promotion(rows, "semantic", ["train_permutation_sham", "signed_coordinate_sham"],
                       "cap_fraction", fraction) for fraction in CAPS]
    passing = [gate for gate in gates if gate["passes"]]
    correction_rows = list(rows)
    write(a.output / "correction.json", dict(complete=True, rows=correction_rows, gates=gates,
        smallest_passing_cap=passing[0]["cap_fraction"] if passing else None))
    risk_report = dict(status="skipped_because_this_family_correction_passed" if passing else "executed_conditional_branch",
        global_eligibility="Only eligible if no correction passes across any of the three family summaries",
        strengths=list(STRENGTHS), rows=[], gates=[], model_weights_changed=False,
        visual_forecasts_unchanged=True)
    if not passing:
        train_error = train_y.double().square().mean(1)
        risk, risk_meta = fit_predict(train_x, train_error[:, None], test_x, train["group"], a.family)
        risk = risk[:, 0].clamp_min(0).double()
        native_median = float(np.median(train["native_cost"]))
        error_median = float(np.median(values(train_error)))
        action_median = float(np.median(train["raw_action_norm"]))
        if native_median <= 0 or error_median <= 0 or action_median <= 0:
            raise ValueError("Positive TRAIN median scales required for risk branch")
        risk_report.update(metadata=risk_meta, train_native_cost_median=native_median,
            train_forecast_error_median=error_median, train_action_norm_median=action_median,
            risk_scale=native_median / error_median, action_scale=native_median / action_median,
            risk_prediction_sha256=array_sha(risk), candidate_permutation_seed_rule="20260906 + episode")
        write(a.output / "FROZEN_RISK_FIT.json", {key: value for key, value in risk_report.items() if key not in ("rows", "gates")})
        begin = len(rows)
        for ep in range(4):
            b = banks[ep]
            native = tensor(b["native_cost"], torch.float64)
            risk_ep = risk[ep * 64:(ep + 1) * 64] * (native_median / error_median)
            action_ep = tensor(b["raw_action_norm"], torch.float64) * (native_median / action_median)
            candidate_perm = np.random.default_rng(SEED + ep).permutation(64)
            permuted = risk_ep[tensor(candidate_perm, torch.long)]
            for strength in STRENGTHS:
                for arm, base_penalty in [("risk", risk_ep), ("risk_candidate_permutation", permuted),
                                          ("risk_action_norm", action_ep)]:
                    penalty = strength * base_penalty
                    score(ep, arm, native + penalty, parameter=dict(strength=strength), penalty=penalty)
        risk_report["rows"] = rows[begin:]
        risk_report["gates"] = [promotion(rows, "risk", ["risk_candidate_permutation", "risk_action_norm"],
                                          "strength", strength) for strength in STRENGTHS]
    write(a.output / "risk.json", risk_report)
    if device.type == "cuda":
        torch.cuda.synchronize()
    process_gpu_seconds = time.monotonic() - gpu_started if device.type == "cuda" else 0.
    result = dict(complete=True, family=a.family, fit_groups=16, fit_rows=112,
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
                     "Risk eligibility requires aggregation across all three family runs"])
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
    parser.add_argument("--family", choices=("linear", "rbf", "knn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
