"""Fit-only, full-population engineering for the fixed-map planning adapter.

Checks the new one-pass operator against independently compiled static fields,
not against the old candidate-response method. No simulator/behavioral efficacy.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .author_fit import source_hash
from .author_runtime import open_normalized_dataset, validate_cohort
from .backends import JepaBackend
from .fixed_response import ARMS, FixedResponseIntervention, load_fitted_bank
from .interventions import CompiledEdit, PredictorIntervention
from .operator_fit import _model_versions
from .planning_support_check import fit_candidate_actions
from .protocol import sha256, write_json
from .support_operator import NativeFieldCapture


class CountedBackend:
    def __init__(self, backend):
        self.backend, self.calls = backend, 0

    def __getattr__(self, key):
        return getattr(self.backend, key)

    def predict(self, *args, **kwargs):
        self.calls += 1
        return self.backend.predict(*args, **kwargs)


def reference_fields(field, bank, arm):
    """Compute outside the forward pass, then use the established static hook."""
    score = ((field.float().flatten(1) - bank["mean"]) @ bank["projection"].T) / bank["scale"]
    features = torch.cat((torch.ones_like(score[:, :1]), score), dim=1)
    operator = bank["operators"][arm]
    raw = features @ operator["map"].T
    delta = (raw @ operator["basis"].flatten(1)).reshape_as(field)
    norm = delta.flatten(1).norm(dim=1)
    factor = torch.where(norm > bank["zero_threshold"], bank["dose"] / norm.clamp_min(bank["zero_threshold"]), 0.)
    delta *= factor[:, None, None]
    return [CompiledEdit("block_output", 3, 3, -256, None, delta,
                         delta.flatten(1).norm(dim=1).cpu().tolist())]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "vendor", "checkpoint", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task", choices=("mw-reach", "mw-reach-wall"), required=True)
    args = parser.parse_args()
    bank = load_fitted_bank(args.fit, task=args.task, checkpoint_sha256=args.checkpoint_sha256)
    cohort = json.loads((args.fit / "cohort.json").read_text())
    validate_cohort(cohort)
    selected = cohort["fit"][0]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        contract = {"method": bank["method"], "task": args.task, "source_sha256": source_hash(),
            "fit_done_sha256": sha256(args.fit / "DONE.json"), "fit_bank_sha256": sha256(args.fit / "operator_bank.pt"),
            "fit_trajectory": selected["trajectory_id"], "candidate_counts": [8, 19, 300],
            "horizon_checks": [2, 5, 6], "arms": list(ARMS), "timing_repeats": 3,
            "planning_context": 2, "precision": "float32_strict_no_tf32", "reference_parity": "bitwise",
            "candidate_actions": "cycle contiguous normalized recorded H6 suffixes of first fit trajectory",
            "reference": "same fixed map computed from native capture then independently compiled static hook",
            "response_probe_rollouts": 0, "native_shadows_in_deployed_operator": 0,
            "extra_native_reference_calls_only_for_engineering": True,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False}
        write_json(args.output / "contract.json", contract)
        backend = JepaBackend(args.vendor, args.checkpoint, args.checkpoint_sha256, "metaworld", "cuda:0", "float32")
        if backend.model.ctxt_window != 2:
            raise ValueError("Wrong planning context")
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset("metaworld", args.data_root, cohort["reference_config"], True)
        observations, actions, _, _, _ = dataset[selected["index"]]
        stride = cohort["reference_config"]["data"]["custom"]["frameskip"]
        frames = cohort["reference_config"]["data"]["validation"]["num_frames_val"]
        observations = {k: v[:frames * stride:stride].unsqueeze(0).to(backend.device) for k, v in observations.items()}
        actions = actions[:frames * stride].reshape(1, frames, -1).to(backend.device)
        visual, proprio, _ = backend.model.model.encode(observations, actions)
        from tensordict import TensorDict
        context = TensorDict({"visual": visual[:, :1], "proprio": proprio[:, :1]}, batch_size=[])
        counted = CountedBackend(backend)
        adapters = {arm: FixedResponseIntervention(counted, bank, arm) for arm in ARMS}
        checks, timings = [], []
        for horizon in contract["horizon_checks"]:
            suffix = actions[:, :horizon].transpose(0, 1).contiguous()
            expected = backend.predict(context, suffix)
            for arm in ARMS:
                if horizon == 6 and arm in ARMS[2:]:
                    continue
                before = counted.calls
                actual = adapters[arm](context, suffix)
                if counted.calls - before != 1 or any(not torch.equal(actual[k], expected[k]) for k in ("visual", "proprio")):
                    raise ValueError("Native/short/zero identity or single-call contract failed")
                checks.append({"arm": arm, "horizon": horizon, "exact_native": True, "backend_calls": 1})
        for count in contract["candidate_counts"]:
            suffix = fit_candidate_actions(actions, count)
            with NativeFieldCapture(backend.predictor) as capture:
                backend.predict(context, suffix)
            for arm in ARMS[2:]:
                adapter = adapters[arm]
                fields = reference_fields(capture.values[3], adapter.bank, arm)
                with PredictorIntervention(backend.predictor, fields):
                    expected = backend.predict(context, suffix)
                before = counted.calls
                actual = adapter(context, suffix)
                if counted.calls - before != 1 or any(not torch.equal(actual[k], expected[k]) for k in ("visual", "proprio")):
                    raise ValueError("Full-population fixed-map/static-hook parity failed")
                checks.append({"arm": arm, "candidate_count": count, "static_reference_bitwise_equal": True,
                               "backend_calls": 1, "response_probes": 0, "native_shadows": 0})
            for arm in ("native", *ARMS[2:]):
                adapter = adapters[arm]
                adapter(context, suffix)  # Unreported warmup, same fitting inputs.
                elapsed = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(contract["timing_repeats"]):
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    adapter(context, suffix)
                    torch.cuda.synchronize()
                    elapsed.append(time.monotonic() - before)
                timings.append({"arm": arm, "candidate_count": count, "horizon": 6,
                    "seconds": elapsed, "mean_seconds": sum(elapsed) / len(elapsed),
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
            write_json(args.output / "progress.json", {"completed_candidate_count": count,
                "checks": checks, "timings": timings, "full_cem_episode_run": False})
        if _model_versions(backend.model) != versions or source_hash() != contract["source_sha256"]:
            raise ValueError("Model/source changed during engineering")
        write_json(args.output / "report.json", {"status": "fixed_map_fit_action_planning_engineering_complete",
            "contract_sha256": sha256(args.output / "contract.json"), "checks": checks, "timings": timings,
            "backend": backend.provenance, "device_name": torch.cuda.get_device_name(),
            "parameters_unchanged": True, "wall_seconds": time.monotonic() - started,
            "full_cem_episode_run": False, "behavioral_launch_ready": False,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
