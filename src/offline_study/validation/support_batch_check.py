"""Fit-only test of larger response-probe candidate chunks on the96GB worker.

Keep all32 probes, all ranks, strict FP32 and the final300-candidate forecast.
Numerical differences are measured and rejected for adoption, not hidden by a
changed tolerance. This is computational scheduling, not an intervention sweep.
"""
from offline_study._paths import source_path
import argparse
from contextlib import contextmanager
import json
import time
from pathlib import Path

import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.models.backends import JepaBackend
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.planning.planning_native_smoke import CHECKPOINTS
from offline_study.planning.planning_support import TRANSFER_POLICY
from offline_study.planning.planning_support_check import fit_candidate_actions
from offline_study.core.protocol import sha256, write_json
from offline_study.evaluation.support_prefix_cache import CachedPlanningSupportIntervention


@contextmanager
def response_chunk_size(size):
    if size not in (8, 16, 32) or TRANSFER_POLICY["candidate_chunk_size"] != 8:
        raise ValueError("Require original candidate8 scheduling and one fixed engineering alternative")
    TRANSFER_POLICY["candidate_chunk_size"] = size
    try:
        yield
    finally:
        TRANSFER_POLICY["candidate_chunk_size"] = 8


class CaptureFields(CachedPlanningSupportIntervention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields = {}

    def _prepare_support(self, backend, context, actions):
        edits, diagnostics = super()._prepare_support(backend, context, actions)
        for edit in edits:
            key = (edit.site, edit.block, edit.horizon, edit.token_start, edit.token_end)
            self.fields.setdefault(key, []).append(edit.delta.reshape(
                actions.shape[1], len(self.protocol["arms"]), *edit.delta.shape[1:]).clone())
        return edits, diagnostics


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(b).all():
        raise ValueError("Changed shape/dtype or nonfinite engineering output")
    delta = a - b
    return {"bitwise_equal": torch.equal(a, b), "maximum_absolute_difference": float(delta.abs().max()),
            "relative_l2_difference": float(delta.norm() / a.norm().clamp_min(1e-30))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        fixture_root = args.inputs / "reach-wall-worker-fit-fixture-v1"
        fixture = json.loads((fixture_root / "report.json").read_text())
        if sha256(fixture_root / "report.json") != json.loads((fixture_root / "DONE.json").read_text())["report_sha256"]:
            raise ValueError("Unbound fit-only fixture")
        for name, suffix in (("inputs", ".pt"), ("cohort", ".json")):
            if sha256(fixture_root / (name + suffix)) != fixture[name + "_sha256"]:
                raise ValueError("Changed fixture")
        cohort = json.loads((fixture_root / "cohort.json").read_text())
        validate_cohort(cohort)
        if fixture["selected"] != cohort["fit"][:1] or fixture["development_or_protected_outcomes_accessed"]:
            raise ValueError("Only existing fitting fixture permitted")
        fit = args.inputs / "author-correction-20260907/fits-v1/bfloat16/reach-wall/operator_rank"
        _, protocol = verify_fit(fit, fixture["source_cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
        bank = torch.load(fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        write_json(args.output / "protocol.json", {"role": "fixed_fit_only_response_batch_scheduling_check",
            "source_sha256": {name: sha256(source_path(name)) for name in
                ("support_batch_check.py", "support_prefix_cache.py", "planning_support.py", "support_operator.py")},
            "fixture_report_sha256": sha256(fixture_root / "report.json"),
            "bank_sha256": sha256(fit / "operator_bank.pt"), "fit_protocol_sha256": sha256(fit / "protocol.json"),
            "candidate_counts": [19, 300], "response_candidate_chunks": [8, 16, 32],
            "full_population_timing_orders": [[8, 16, 32], [32, 16, 8]],
            "all_source_probes_and_ranks_retained": True, "final_forecast_batch_unchanged": True,
            "precision": "strict_float32_no_tf32", "acceptance": "bitwise_all_arm_fields_forecasts_and_energy",
            "no_relaxation_after_results": True, "full_cem_gate_still_required": True,
            "scientific_efficacy_measurement": False, "confirmation_authorized": False})
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.85)
        backend = JepaBackend(args.vendor, args.inputs / "checkpoints/jepa_wm_metaworld.pth.tar",
            CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        inputs = torch.load(fixture_root / "inputs.pt", map_location="cpu", weights_only=True)
        visual, proprio, _ = backend.model.model.encode(
            {k: inputs[k].to(backend.device) for k in ("visual", "proprio")}, inputs["action"].to(backend.device))
        from tensordict import TensorDict
        context = TensorDict({"visual": visual[:1, :1], "proprio": proprio[:1, :1]}, batch_size=[])
        checks = []
        for count, repetition, order in ((19, 0, [8, 16, 32]), (300, 0, [8, 16, 32]), (300, 1, [32, 16, 8])):
            actions = fit_candidate_actions(inputs["action"], count).to(backend.device)
            values, timings = {}, {}
            for size in order:
                adapter = CaptureFields(backend, protocol, bank, "rank4")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                before = time.monotonic()
                with response_chunk_size(size):
                    result = adapter(context, actions)
                torch.cuda.synchronize()
                timings[size] = {"seconds": time.monotonic() - before,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                values[size] = {"outputs": result, "energy": adapter.energy,
                    "fields": {key: torch.cat(parts) for key, parts in adapter.fields.items()}}
                del adapter
            parity = {}
            for size in (16, 32):
                expected, actual = values[8], values[size]
                if expected["fields"].keys() != actual["fields"].keys():
                    raise ValueError("Changed intervention sites")
                fields = {str(key): difference(expected["fields"][key], actual["fields"][key]) for key in expected["fields"]}
                outputs = {key: difference(expected["outputs"][key], actual["outputs"][key]) for key in ("visual", "proprio")}
                energy = expected["energy"] == actual["energy"]
                parity[size] = {"all_arm_fields": fields, "forecasts": outputs, "energy_exact": energy,
                    "accepted_numerically": energy and all(v["bitwise_equal"] for v in [*fields.values(), *outputs.values()]),
                    "speedup": timings[8]["seconds"] / timings[size]["seconds"]}
            row = {"candidates": count, "repetition": repetition, "order": order, "timings": timings, "parity": parity}
            checks.append(row)
            write_json(args.output / "progress.json", {"checks": checks})
            print(json.dumps(row), flush=True)
            del values
        if versions != _model_versions(backend.model) or TRANSFER_POLICY["candidate_chunk_size"] != 8:
            raise ValueError("Frozen parameters or global scheduling not restored")
        write_json(args.output / "report.json", {"status": "response_batch_schedule_benchmark_complete_not_adopted",
            "protocol_sha256": sha256(args.output / "protocol.json"), "checks": checks,
            "seconds": time.monotonic() - started, "parameters_unchanged": True,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "production_schedule_changed": False, "scientific_efficacy_measurement": False,
            "confirmation_authorized": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "production_schedule_changed": False})
        raise


if __name__ == "__main__":
    main()
