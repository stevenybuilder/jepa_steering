"""Fixed fit-only parity and timing benchmark for exact prefix memoization."""
import argparse
import json
import time
from pathlib import Path

import torch

from .author_evaluate import verify_fit
from .author_runtime import validate_cohort
from .backends import JepaBackend
from .intervention_runner import _model_versions
from .planning_native_smoke import CHECKPOINTS
from .planning_support import PlanningSupportIntervention, SupportRolloutBackend
from .planning_support_check import fit_candidate_actions
from .protocol import sha256, write_json
from .support_operator import prepare_support
from .support_prefix_cache import CachedPlanningSupportIntervention, cached_prepare_support


def compare_fields(expected, actual):
    if len(expected) != len(actual):
        raise ValueError("Changed support edit count")
    for a, b in zip(expected, actual):
        for name in ("site", "horizon", "block", "token_start", "token_end", "delivered_l2"):
            if getattr(a, name) != getattr(b, name):
                raise ValueError("Changed support edit metadata or dose")
        if not torch.equal(a.delta, b.delta):
            raise ValueError("Cached support fields are not bitwise identical")


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
            raise ValueError("Fit fixture report binding changed")
        for name, suffix in (("inputs", ".pt"), ("cohort", ".json")):
            if sha256(fixture_root / (name + suffix)) != fixture[name + "_sha256"]:
                raise ValueError("Fit fixture inputs changed")
        cohort = json.loads((fixture_root / "cohort.json").read_text())
        validate_cohort(cohort)
        if fixture["selected"] != cohort["fit"][:1] or fixture["development_or_protected_outcomes_accessed"]:
            raise ValueError("Only existing fit-only stimuli permitted")
        fit = args.inputs / "author-correction-20260907/fits-v1/bfloat16/reach-wall/operator_rank"
        _, protocol = verify_fit(fit, fixture["source_cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
        bank = torch.load(fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        write_json(args.output / "protocol.json", {"role": "fit_only_exact_prefix_reuse_engineering",
            "source_sha256": {name: sha256(Path(__file__).parent / name) for name in
                ("support_prefix_check.py", "support_prefix_cache.py", "planning_support.py", "support_operator.py")},
            "fixture_report_sha256": sha256(fixture_root / "report.json"),
            "fit_protocol_sha256": sha256(fit / "protocol.json"), "bank_sha256": sha256(fit / "operator_bank.pt"),
            "candidate_counts": [8, 19, 300], "timing_repetitions_full_population": 2,
            "precision": "float32_no_tf32", "parity": "bitwise_all_outputs_and_energy",
            "all_source_probes_ranks_candidates_retained": True,
            "cache_scope": "identical H1/H2 calls within one prepare_support, same tensor strides and values",
            "confirmation_authorized": False, "scientific_efficacy_measurement": False})
        backend = JepaBackend(args.vendor, args.inputs / "checkpoints/jepa_wm_metaworld.pth.tar",
            CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        inputs = torch.load(fixture_root / "inputs.pt", map_location="cpu", weights_only=True)
        visual, proprio, _ = backend.model.model.encode(
            {k: inputs[k].to(backend.device) for k in ("visual", "proprio")}, inputs["action"].to(backend.device))
        from tensordict import TensorDict
        context = TensorDict({"visual": visual[:1, :1], "proprio": proprio[:1, :1]}, batch_size=[])
        actions = fit_candidate_actions(inputs["action"], 8).to(backend.device)
        reference_fields, reference_diag = prepare_support(SupportRolloutBackend(backend), context, actions, protocol, bank)
        counters = []
        fields, diagnostics = cached_prepare_support(SupportRolloutBackend(backend), context, actions, protocol, bank, counters)
        compare_fields(reference_fields, fields)
        if diagnostics != reference_diag or not counters[0]["cached_prefix_calls"]:
            raise ValueError("Changed diagnostics or no actual prefix reuse")
        del reference_fields, fields
        checks = []
        for count in (8, 19, 300):
            actions = fit_candidate_actions(inputs["action"], count).to(backend.device)
            for repetition in range(2 if count == 300 else 1):
                result, timing, energies, counts = {}, {}, {}, {}
                order = ("source", "cached") if repetition == 0 else ("cached", "source")
                for kind in order:
                    cls = PlanningSupportIntervention if kind == "source" else CachedPlanningSupportIntervention
                    adapter = cls(backend, protocol, bank, "rank4")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    result[kind] = adapter(context, actions)
                    torch.cuda.synchronize()
                    timing[kind] = {"seconds": time.monotonic() - before,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                    energies[kind] = adapter.energy
                    if kind == "cached":
                        counts = adapter.prefix_counters
                if energies["source"] != energies["cached"] or any(
                        not torch.equal(result["source"][key], result["cached"][key]) for key in ("visual", "proprio")):
                    raise ValueError("Cached full-population forecasts or realized energy differ")
                row = {"candidates": count, "repetition": repetition, "order": order,
                    "bitwise_outputs_and_energy": True, "timing": timing, "prefix_counters": counts,
                    "speedup": timing["source"]["seconds"] / timing["cached"]["seconds"]}
                checks.append(row)
                write_json(args.output / "progress.json", {"completed_checks": checks})
                print(json.dumps({k: v for k, v in row.items() if k != "prefix_counters"}), flush=True)
                del result
        if _model_versions(backend.model) != versions:
            raise ValueError("Model weights changed")
        write_json(args.output / "report.json", {"status": "exact_prefix_cache_fit_parity_and_timing_complete",
            "protocol_sha256": sha256(args.output / "protocol.json"), "checks": checks,
            "all_arm_support_fields_bitwise_equal": True, "parameters_unchanged": True,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "seconds": time.monotonic() - started, "scientific_efficacy_measurement": False,
            "full_cem_engineering_still_required_before_production": True, "confirmation_authorized": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "confirmation_authorized": False})
        raise


if __name__ == "__main__":
    main()
