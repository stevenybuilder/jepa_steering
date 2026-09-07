"""Bounded, fit-only validation and throughput of primary-bank planning transfer."""
import argparse
import json
import time
from pathlib import Path

import torch

from .author_evaluate import verify_fit
from .author_runtime import validate_cohort
from .backends import JepaBackend
from .intervention_runner import _model_versions
from .interventions import PredictorIntervention
from .planning_native_smoke import CHECKPOINTS
from .planning_support import PlanningSupportIntervention, TRANSFER_POLICY, select_context, selected_support_fields
from .protocol import sha256, write_json
from .support_operator import prepare_support


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "fixture", "fit-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
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
        # Freeze engineering behavior before any benchmark/fit-check outputs.
        write_json(args.output / "transfer_contract.json", {"policy": TRANSFER_POLICY,
            "source_protocol_sha256": sha256(fit / "protocol.json"),
            "source_bank_sha256": sha256(fit / "operator_bank.pt"),
            "fixture_report_sha256": fixture_done["report_sha256"],
            "fit_only": True, "candidate_selection": False,
            "parity_atol": 1e-6, "parity_rtol": 1e-5})
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        inputs = torch.load(args.fixture / "inputs.pt", weights_only=True, map_location="cpu")
        from tensordict import TensorDict
        with backend.autocast():
            visual, proprio, _ = backend.model.model.encode(
                {k: inputs[k].to(backend.device) for k in ("visual", "proprio")}, inputs["action"].to(backend.device))
        context = TensorDict({"visual": visual[:1, :1], "proprio": proprio[:1, :1]}, batch_size=[])
        action = inputs["action"][:1, :6].transpose(0, 1).contiguous().to(backend.device)
        checks, timings = [], []
        for horizon in (2, 5, 6):
            native = backend.predict(context, action[:horizon])
            for arm in ("native", "zero_dose", "rank4", "matched_random_rank4"):
                adapter = PlanningSupportIntervention(backend, protocol, bank, arm)
                actual = adapter(context, action[:horizon])
                if horizon < 6 or arm in ("native", "zero_dose"):
                    expected = native
                else:
                    fields, _ = prepare_support(backend, context, action, protocol, bank)
                    selected = selected_support_fields(fields, [a["name"] for a in protocol["arms"]], arm, 1)
                    with PredictorIntervention(backend.predictor, selected):
                        expected = backend.predict(context, action)
                for key in ("visual", "proprio"):
                    if not torch.equal(actual[key], expected[key]):
                        raise ValueError(f"Unchanged source/no-op planning parity failed: {arm}/H{horizon}/{key}")
                checks.append({"horizon": horizon, "arm": arm, "bitwise_equal_to_source_or_true_native": True})
        # Validate candidate slicing on repeated FIT actions, without CEM or new
        # simulator outcomes. Repetition here tests layout, not statistical n.
        for count in (8, 19, 300):
            actions = action.repeat(1, count, 1)
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
            for key in parts:
                if not torch.allclose(torch.cat(parts[key], 1), native[key], atol=1e-6, rtol=1e-5):
                    raise ValueError("Candidate chunking violates existing native fidelity tolerance: " + str(count) + "/" + key)
            checks.append({"candidate_count": count, "native_chunking_within_existing_tolerance": True,
                           "no_candidate_reordering_or_padding": True})
            adapter = PlanningSupportIntervention(backend, protocol, bank, "rank4")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            before = time.monotonic()
            actual = adapter(context, actions)
            torch.cuda.synchronize()
            timings.append({"candidates": count, "native_seconds": native_seconds,
                "rank4_seconds": time.monotonic() - before, "energy": adapter.energy,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "fit_inputs_only_not_actual_cem_throughput": True})
            print(json.dumps(timings[-1]), flush=True)
            if count == 19:
                # Same-size independent source compilation includes a nonpadded
                # tail; checks mapping of every returned candidate, not just shape.
                for start in (0, 8, 16):
                    stop = min(start + 8, count)
                    z = select_context(context, count, start, stop)
                    a = actions[:, start:stop].contiguous()
                    fields, _ = prepare_support(backend, z, a, protocol, bank)
                    selected = selected_support_fields(fields, [r["name"] for r in protocol["arms"]], "rank4", stop - start)
                    with PredictorIntervention(backend.predictor, selected):
                        expected = backend.predict(z, a)
                    for key in parts:
                        if not torch.equal(actual[key][:, start:stop], expected[key]):
                            raise ValueError("Candidate slicing changes the source rank correction")
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen model parameters changed")
        write_json(args.output / "report.json", {"status": "primary_rank_planning_transfer_fit_only_passed",
            "checks": checks, "timings": timings, "seconds": time.monotonic() - started,
            "transfer_contract_sha256": sha256(args.output / "transfer_contract.json"),
            "fit_trajectory": fixture["selected"][0]["trajectory_id"], "bank_precision": "bfloat16",
            "planning_precision": "float32_strict_no_tf32", "planning_context": 2,
            "parameters_unchanged": True, "development_or_protected_outcomes_accessed": False,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False,
            "combined_planning_transfer_validated": False, "actual_cem_integration_validated": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
