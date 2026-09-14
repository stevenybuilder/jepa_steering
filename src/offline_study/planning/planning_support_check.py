"""Bounded, fit-only validation and throughput of primary-bank planning transfer."""
import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.models.backends import JepaBackend
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.interventions.interventions import PredictorIntervention, compile_edits, window_key
from offline_study.planning.planning_native_smoke import CHECKPOINTS
from offline_study.planning.planning_support import PlanningSupportIntervention, SupportRolloutBackend, TRANSFER_POLICY, select_context, selected_support_fields
from offline_study.core.protocol import sha256, write_json
from offline_study.interventions.support_operator import prepare_support


def fit_candidate_actions(action_clips, count):
    """Cycle distinct recorded fit-only H6 sequences; never generate outcomes."""
    if action_clips.ndim != 3 or action_clips.shape[1] < 6 or count < 1:
        raise ValueError("Invalid fit action clips")
    sequences = torch.stack([clip[start:start + 6] for clip in action_clips
                             for start in range(clip.shape[0] - 5)])
    if not torch.isfinite(sequences).all() or torch.unique(sequences.flatten(1), dim=0).shape[0] < 2:
        raise ValueError("Diverse candidate check needs at least two distinct fit sequences")
    return sequences[torch.arange(count) % len(sequences)].transpose(0, 1).contiguous()


def source_coupling_fields(coupling, batch, device):
    """Independent original compiler, not the planning static_edits helper."""
    if coupling is None:
        return []
    protocol, bank, arm = coupling
    metadata = [{"trajectory_id": "fit_only_layout_check", "start": i} for i in range(batch)]
    selected = [row for row in protocol["arms"] if row["name"] == arm]
    if len(selected) != 1:
        raise ValueError("Unknown source coupling arm")
    reference_bank = {"global_tensors": bank["global_tensors"],
                      "rows": {window_key(row): {"tensors": {}} for row in metadata}}
    return compile_edits({**protocol, "arms": selected}, reference_bank, metadata, device)


@torch.no_grad()
def probe_batch_diagnostic(args, backend, protocol, bank, context, action):
    """Measure an alternative batching axis; this never grants a launch gate."""
    rows = []
    names = [a["name"] for a in protocol["arms"]]
    for count in (8, 19, 300):
        actions = action.repeat(1, count, 1)
        candidates = {}
        for probe_chunk in ((8, 1) if count < 300 else (1,)):
            configuration = copy.deepcopy(protocol)
            configuration["support_operator"]["probe_chunk_size"] = probe_chunk
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            before = time.monotonic()
            fields, diagnostics = prepare_support(SupportRolloutBackend(backend), context, actions, configuration, bank)
            selected = selected_support_fields(fields, names, "rank4", count)
            with PredictorIntervention(backend.predictor, selected):
                result = backend.predict(context, actions)
            torch.cuda.synchronize()
            candidates[probe_chunk] = (selected, result)
            row = {"candidates": count, "probe_chunk_size": probe_chunk, "candidate_axis_unsliced": True,
                "seconds": time.monotonic() - before, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "response_probes_per_candidate": diagnostics[0]["response_probe_rollouts"],
                "all_common_eligible": all(r["common_energy_edit_eligible"] for r in diagnostics)}
            if probe_chunk == 1 and 8 in candidates:
                old_fields, old_result = candidates[8]
                row["source_probe_chunking_parity"] = {key: {
                    "maximum_absolute_difference": float((old_result[key] - result[key]).abs().max()),
                    "within_existing_tolerance": bool(torch.allclose(old_result[key], result[key], atol=1e-6, rtol=1e-5))}
                    for key in ("visual", "proprio")}
                row["maximum_edit_difference"] = max(float((a.delta - b.delta).abs().max()) for a, b in zip(old_fields, selected))
            rows.append(row)
            write_json(args.output / "diagnostic_progress.json", {"completed_checks": rows, "confirmation_authorized": False})
            print(json.dumps(row), flush=True)
        del candidates, fields, selected, result
        torch.cuda.empty_cache()
    write_json(args.output / "diagnostic_report.json", {"status": "fit_only_probe_batch_diagnostic_complete",
        "checks": rows, "preserved_candidate_count": 300, "fit_only": True,
        "scientific_efficacy_measurement": False, "confirmation_authorized": False,
        "not_a_production_transfer_gate": True})
    write_json(args.output / "DIAGNOSTIC_DONE.json", {"report_sha256": sha256(args.output / "diagnostic_report.json")})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "fixture", "fit-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--diagnostic-probe-batching", action="store_true")
    parser.add_argument("--diverse-fit-actions", action="store_true")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--arm", choices=("rank4", "matched_random_rank4"), default="rank4")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        fixture_done = json.loads((args.fixture / "DONE.json").read_text())
        fixture = json.loads((args.fixture / "report.json").read_text())
        if sha256(args.fixture / "report.json") != fixture_done["report_sha256"]:
            raise ValueError("Fixture receipt changed")
        for name, suffix in (("cohort", ".json"), ("inputs", ".pt")):
            if sha256(args.fixture / (name + suffix)) != fixture[name + "_sha256"]:
                raise ValueError("Fit-only fixture checksum changed")
        cohort = json.loads((args.fixture / "cohort.json").read_text())
        validate_cohort(cohort)
        if (cohort["dataset"] != "metaworld" or fixture["selected"] != cohort["fit"][:1] or
                fixture["development_or_protected_outcomes_accessed"]):
            raise ValueError("This check is restricted to one previously eligible MW fitting trajectory")
        fit = args.fit_root / "bfloat16" / cohort["task"].removeprefix("mw-") / "operator_rank"
        _, protocol = verify_fit(fit, fixture["source_cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
        bank = torch.load(fit / "operator_bank.pt", weights_only=True, map_location="cpu")
        coupling_components, coupling_binding = None, None
        if args.combined:
            if cohort["task"] != "mw-reach" or args.diagnostic_probe_batching:
                raise ValueError("Combined transfer checks are for the fixed Reach recipe only")
            coupling_fit = fit.parent / "vision_action_coupling"
            _, coupling_protocol = verify_fit(coupling_fit, fixture["source_cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
            coupling_bank = torch.load(coupling_fit / "operator_bank.pt", weights_only=True, map_location="cpu")
            coupling_components = coupling_protocol, coupling_bank
            coupling_binding = {"protocol_sha256": sha256(coupling_fit / "protocol.json"),
                                "bank_sha256": sha256(coupling_fit / "operator_bank.pt")}
        def coupling_for(arm):
            if coupling_components is None or arm in ("native", "zero_dose"):
                return None
            name = "matched_random_equal_standardized_energy" if arm.startswith("matched_random") else "joint_equal_standardized_energy"
            return (*coupling_components, name)
        # Freeze engineering behavior before any benchmark/fit-check outputs.
        write_json(args.output / "transfer_contract.json", {"policy": TRANSFER_POLICY,
            "source_protocol_sha256": sha256(fit / "protocol.json"),
            "source_bank_sha256": sha256(fit / "operator_bank.pt"),
            "fixture_report_sha256": fixture_done["report_sha256"],
            "fit_only": True, "candidate_selection": False,
            "profiled_rank_arm": args.arm, "combined_coupling_binding": coupling_binding,
            "candidate_actions": "all contiguous H6 sequences from existing fit fixture, cycled in fixed order" if args.diverse_fit_actions else "repeated first fit sequence",
            "parity_atol": 1e-6, "parity_rtol": 1e-5,
            "diagnostic_probe_batching_only": args.diagnostic_probe_batching,
            "diagnostic_policy": {"candidate_counts": [8, 19, 300], "probe_chunks": [8, 1],
                "all_32_probes_retained": True, "native_and_final_candidate_axis_unsliced": True,
                "no_300_by_8_allocation": True} if args.diagnostic_probe_batching else None})
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        inputs = torch.load(args.fixture / "inputs.pt", weights_only=True, map_location="cpu")
        from tensordict import TensorDict
        with backend.autocast():
            visual, proprio, _ = backend.model.model.encode(
                {k: inputs[k].to(backend.device) for k in ("visual", "proprio")}, inputs["action"].to(backend.device))
        context = TensorDict({"visual": visual[:1, :1], "proprio": proprio[:1, :1]}, batch_size=[])
        action = inputs["action"][:1, :6].transpose(0, 1).contiguous().to(backend.device)
        if args.diagnostic_probe_batching:
            probe_batch_diagnostic(args, backend, protocol, bank, context, action)
            return
        checks, timings = [], []
        for horizon in (2, 5, 6):
            native = backend.predict(context, action[:horizon])
            for arm in ("native", "zero_dose", "rank4", "matched_random_rank4"):
                adapter = PlanningSupportIntervention(backend, protocol, bank, arm, coupling_for(arm))
                actual = adapter(context, action[:horizon])
                if horizon < 6 or arm in ("native", "zero_dose"):
                    expected = native
                else:
                    fields, _ = prepare_support(SupportRolloutBackend(backend), context, action, protocol, bank)
                    selected = selected_support_fields(fields, [a["name"] for a in protocol["arms"]], arm, 1)
                    selected += source_coupling_fields(coupling_for(arm), 1, backend.device)
                    with PredictorIntervention(backend.predictor, selected):
                        expected = backend.predict(context, action)
                for key in ("visual", "proprio"):
                    if not torch.equal(actual[key], expected[key]):
                        raise ValueError(f"Unchanged source/no-op planning parity failed: {arm}/H{horizon}/{key}")
                checks.append({"horizon": horizon, "arm": arm, "bitwise_equal_to_source_or_true_native": True})
        # Fit actions test engineering layout, not statistical sample size or
        # a counterfactual physical prediction endpoint.
        for count in (8, 19, 300):
            actions = (fit_candidate_actions(inputs["action"], count).to(backend.device)
                       if args.diverse_fit_actions else action.repeat(1, count, 1))
            torch.cuda.synchronize()
            before = time.monotonic()
            native = backend.predict(context, actions)
            torch.cuda.synchronize()
            native_seconds = time.monotonic() - before
            parts = {k: [] for k in ("visual", "proprio")}
            for start in range(0, count, TRANSFER_POLICY["candidate_chunk_size"]):
                stop = min(start + TRANSFER_POLICY["candidate_chunk_size"], count)
                chunk = backend.predict(select_context(context, count, start, stop), actions[:, start:stop].contiguous())
                for key in parts:
                    parts[key].append(chunk[key])
            # Retain the failed strategy as a diagnostic, not an accepted
            # execution path. Production now keeps ACTUAL forecasts unsliced.
            rejected_strategy = {key: bool(torch.allclose(torch.cat(parts[key], 1), native[key], atol=1e-6, rtol=1e-5))
                                 for key in parts}
            for arm in ("native", "zero_dose"):
                unchanged = PlanningSupportIntervention(backend, protocol, bank, arm)(context, actions)
                if any(not torch.equal(unchanged[key], native[key]) for key in parts):
                    raise ValueError("Full-population native/zero-dose identity failed")
            checks.append({"candidate_count": count, "actual_full_population_noop_bitwise_equal": True,
                "rejected_forecast_slicing_diagnostic": rejected_strategy,
                "forecast_slicing_used_in_production": False, "no_candidate_reordering_or_padding": True})
            adapter = PlanningSupportIntervention(backend, protocol, bank, args.arm, coupling_for(args.arm))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            before = time.monotonic()
            batch_calls, original_predict = [], backend.predict
            def observed_predict(z, a, **kwargs):
                batch_calls.append(a.shape[1])
                return original_predict(z, a, **kwargs)
            backend.predict = observed_predict
            try:
                actual = adapter(context, actions)
            finally:
                backend.predict = original_predict
            if batch_calls[-1] != count or adapter.energy[-1]["final_forecasts_chunked"]:
                raise ValueError("Actual edited forecast did not retain the full candidate batch")
            torch.cuda.synchronize()
            timings.append({"candidates": count, "native_seconds": native_seconds,
                "rank4_seconds": time.monotonic() - before, "profiled_rank_arm": args.arm,
                "combined": args.combined, "distinct_action_sequences": int(torch.unique(actions.transpose(0, 1).flatten(1), dim=0).shape[0]),
                "energy": adapter.energy,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "fit_inputs_only_not_actual_cem_throughput": True})
            print(json.dumps(timings[-1]), flush=True)
            if count in (19, 300):
                # Independently compile the unmodified SOURCE fields in bounded
                # probe batches, then apply them to one full-population unroll.
                # This tests every returned candidate, including a nonpadded tail.
                assembled = []
                for start in range(0, count, 8):
                    stop = min(start + 8, count)
                    z = select_context(context, count, start, stop)
                    a = actions[:, start:stop].contiguous()
                    fields, _ = prepare_support(SupportRolloutBackend(backend), z, a, protocol, bank)
                    names = [r["name"] for r in protocol["arms"]]
                    index = names.index(args.arm)
                    assembled.append([(e, e.delta[index::len(names)].contiguous()) for e in fields])
                reference = []
                for index in range(len(assembled[0])):
                    original = assembled[0][index][0]
                    delta = torch.cat([chunk[index][1] for chunk in assembled], dim=0)
                    reference.append(replace(original, delta=delta, applications=0, realized_l2=None,
                        delivered_l2=delta.double().flatten(1).norm(dim=1).cpu().tolist()))
                reference += source_coupling_fields(coupling_for(args.arm), count, backend.device)
                with PredictorIntervention(backend.predictor, reference):
                    expected = backend.predict(context, actions)
                for key in parts:
                    if not torch.equal(actual[key], expected[key]):
                        raise ValueError("Full-population forecast differs from the unchanged source correction")
                checks.append({"candidate_count": count, "actual_final_batch_verified": batch_calls[-1],
                    "full_population_forecast_bitwise_matches_source_hook_compilation": True})
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen model parameters changed")
        write_json(args.output / "report.json", {"status": "primary_rank_planning_transfer_fit_only_passed",
            "checks": checks, "timings": timings, "seconds": time.monotonic() - started,
            "transfer_contract_sha256": sha256(args.output / "transfer_contract.json"),
            "fit_trajectory": fixture["selected"][0]["trajectory_id"], "bank_precision": "bfloat16",
            "planning_precision": "float32_strict_no_tf32", "planning_context": 2,
            "parameters_unchanged": True, "development_or_protected_outcomes_accessed": False,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False,
            "combined_planning_transfer_validated": args.combined,
            "diverse_fit_actions_checked": args.diverse_fit_actions, "profiled_rank_arm": args.arm,
            "actual_cem_integration_validated": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
