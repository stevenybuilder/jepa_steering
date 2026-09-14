"""Four full author-planner engineering episodes for the frozen Reach combination.

Native/repeat first, then both fixed combined arms. These are excluded engineering
episodes, not development selection, confirmation, or a 96-episode comparison.
"""
from __future__ import annotations
from offline_study._paths import source_path, snapshot_source_files

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.fitting.author_fit import source_hash
from offline_study.models.backends import JepaBackend
from offline_study.interventions.fixed_combined import ARM_COMPONENTS, METHOD, CombinedFixedResponseIntervention
from offline_study.validation.fixed_combined_check import COUNTS, HORIZONS, STATUS, deadline, exclusive_json, hooks_clean, json_tensors, model_signature, validate_coverage, verify_bindings, verify_files
from offline_study.validation.fixed_response_check import CountedBackend
from offline_study.validation.fixed_response_smoke import trace_actor, verify_episode, verify_pair
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.planning.planning_scenarios import close_expert_environments
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


EPISODES = ("native", "native_repeat", *ARM_COMPONENTS)
MAX_SECONDS = 3600


def validate_record(record, arm, horizon, candidates):
    edited = arm in ARM_COMPONENTS and horizon == 6
    expected = {"backend_calls": 1, "response_probe_rollouts": 0,
        "full_native_shadow_rollouts": 0, "main_predictor_blocks": horizon * 6,
        "native_prefix_replays": int(edited), "extra_native_predictor_blocks": 4 * int(edited),
        "rank_applications": int(edited), "horizon": horizon, "candidates": candidates}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Combined execution counts changed")
    coupling = record.get("coupling", [])
    if len(coupling) != 2 * int(edited):
        raise ValueError("Combined coupling scope changed")
    if edited:
        if {(r["site"], r["block"], r["horizon"], r["applications"]) for r in coupling} != {
                ("predictor_visual", None, 3, 1), ("block_condition", 3, 3, 1)}:
            raise ValueError("Combined coupling was not delivered exactly once")
        for key, shape in (("coefficients", (candidates, 4)), ("active", (candidates,)),
                           ("requested_l2", (candidates,)), ("realized_l2", (candidates,))):
            value = torch.as_tensor(record[key])
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("Incomplete/nonfinite combined candidate record")


def verify_numerical(root, expected_hash, checked_source, fit_files, checkpoint, device_uuid):
    root, checked_source = Path(root), Path(checked_source)
    done = json.loads((root / "DONE.json").read_text())
    report = json.loads((root / "report.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    if ((root / "FAILED.json").exists() or done.get("report_sha256") != expected_hash or
            sha256(root / "report.json") != expected_hash or
            sha256(root / "contract.json") != report["contract_sha256"]):
        raise ValueError("Incomplete/changed numerical receipt")
    expected_contract = {"method": METHOD, "task": "mw-reach", "arms": list(ARM_COMPONENTS),
        "candidate_counts": list(COUNTS), "horizons": list(HORIZONS), "planning_context": 2,
        "precision": "strict_float32_no_autocast_no_tf32", "timing_repeats": 3,
        "scientific_efficacy_measurement": False, "fresh_confirmation": False}
    if any(contract.get(k) != v for k, v in expected_contract.items()):
        raise ValueError("Numerical contract differs from the frozen combination")
    expected_report = {"status": STATUS, "parameter_buffer_bytes_modes_unchanged": True,
        "hooks_clean": True, "source_and_bound_inputs_unchanged": True,
        "scientific_efficacy_measurement": False, "fresh_confirmation": False,
        "development_or_protected_outcomes_accessed": False, "device_uuid": device_uuid}
    if (any(report.get(k) != v for k, v in expected_report.items()) or
            report["backend"].get("checkpoint_sha256") != checkpoint):
        raise ValueError("Numerical task/model/device/integrity evidence differs")
    validate_coverage(report["checks"], report["timings"])
    for row in report["checks"]:
        if any(row.get(k) is not True for k in ("all_output_horizons_byte_equal",
                "input_shape_stride_bytes_unchanged", "cpu_cuda_rng_unchanged")):
            raise ValueError("Missing exact numerical equivalence")
        validate_record(row["adapter_record"], row["arm"], row["horizon"], row["candidate_count"])
    bound = contract["bound_files"]
    if not fit_files or any(bound.get(p) != value for p, value in fit_files.items()):
        raise ValueError("Different completed fitted components")
    verify_files(bound)
    before, after = root / "model-before.json", root / "model-after.json"
    if (sha256(before) != report["model_before_sha256"] or
            sha256(after) != report["model_after_sha256"] or before.read_bytes() != after.read_bytes()):
        raise ValueError("Numerical model byte evidence changed")
    digest, files = hashlib.sha256(), {}
    for path in snapshot_source_files(checked_source):
        if path.is_symlink() or path.read_bytes() != (source_path(path.resolve().relative_to(checked_source.resolve()))).read_bytes():
            raise ValueError("Numerically checked source file changed: " + path.name)
        digest.update(path.resolve().relative_to(checked_source.resolve()).as_posix().encode() + b"\0" + path.read_bytes())
        files[path.resolve().relative_to(checked_source.resolve()).as_posix()] = sha256(path)
    if not files or digest.hexdigest() != contract["source_sha256"]:
        raise ValueError("Numerical source snapshot changed")
    return {"report_sha256": expected_hash, "checked_source_sha256": digest.hexdigest(),
            "unchanged_runtime_files": files, "bound_files": bound, "device_uuid": device_uuid}


class ObservedCombined:
    def __init__(self, counted, arm, output, adapter=None):
        self.counted, self.arm, self.output, self.adapter = counted, arm, output, adapter
        self.calls = []

    def __call__(self, context, act_suffix=None, **kwargs):
        horizon, candidates, _ = act_suffix.shape
        before, start = self.counted.calls, time.monotonic()
        write_json(self.output / "progress.json", {"completed_calls": len(self.calls),
            "running_horizon": horizon, "running_candidates": candidates, "engineering_only": True})
        if self.adapter is None:
            result = self.counted.predict(context, act_suffix, **kwargs)
            record = {"backend_calls": 1, "response_probe_rollouts": 0,
                "full_native_shadow_rollouts": 0, "native_prefix_replays": 0,
                "extra_native_predictor_blocks": 0, "main_predictor_blocks": horizon * 6,
                "rank_applications": 0, "horizon": horizon, "candidates": candidates, "coupling": []}
        else:
            result = self.adapter(context, act_suffix, **kwargs)
            record = self.adapter.last_record
        if self.counted.calls - before != 1 or any(
                not torch.isfinite(result[k]).all() for k in ("visual", "proprio")):
            raise ValueError("Extra backend forecast or nonfinite prediction")
        validate_record(record, self.arm, horizon, candidates)
        if not hooks_clean(self.counted.predictor):
            raise ValueError("Leaked predictor hooks")
        torch.cuda.synchronize()
        self.calls.append({"horizon": horizon, "candidates": candidates,
            "seconds": time.monotonic() - start, "record": json_tensors(record)})
        return result


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "fit", "coupling-fit", "numerical-check", "checked-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--numerical-report-sha256", required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    start, frozen_source = time.monotonic(), source_hash()
    try:
        with deadline(MAX_SECONDS):
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly the numerically checked receiving GPU")
            checkpoint = CHECKPOINTS["metaworld"]
            if sha256(args.checkpoint) != checkpoint:
                raise ValueError("Wrong released MetaWorld checkpoint")
            bank, _, coupling_protocol, coupling_bank, _, files = verify_bindings(
                args.fit, args.coupling_fit, "mw-reach", checkpoint)
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            numerical = verify_numerical(args.numerical_check, args.numerical_report_sha256,
                args.checked_source, files, checkpoint, uuid)
            use_vendor(args.vendor)
            from omegaconf import OmegaConf
            from evals.simu_env_planning.envs.init import make_env
            from evals.simu_env_planning.planning.gc_agent import GC_Agent
            from evals.simu_env_planning.planning import plan_evaluator

            prepared = prepare(args.vendor, "reach")
            protocol = {"method": METHOD, "task": "reach", "source_sha256": frozen_source,
                "numerical_check": numerical, "planning_contract": prepared,
                "engineering_seed": SMOKE_SEED, "episode_order": list(EPISODES),
                "full_simulator_episodes": 4, "scenario_count": 1,
                "precision": "float32_strict_no_tf32", "max_seconds": MAX_SECONDS,
                "required_outer_process_cap_seconds": MAX_SECONDS + 10,
                "candidate_selection_from_smoke_forbidden": True, "scientific_efficacy_measurement": False,
                "fresh_confirmation": False, "protected_outcomes_access_authorized": False,
                "historical_worker_goal_pixels_assumed_equal": False}
            exclusive_json(args.output / "protocol.json", protocol)
            backend = JepaBackend(args.vendor, args.checkpoint, checkpoint, "metaworld", "cuda:0", "float32")
            if backend.model.ctxt_window != 2:
                raise ValueError("Wrong official planning context")
            initial_model, counted = model_signature(backend.model), CountedBackend(backend)
            exclusive_json(args.output / "model-before.json", initial_model)
            completed = {}
            for name in EPISODES:
                output = args.output / name
                output.mkdir(exist_ok=False)
                arm = "native" if name == "native_repeat" else name
                random.seed(SMOKE_SEED); np.random.seed(SMOKE_SEED); torch.manual_seed(SMOKE_SEED)
                cfg = OmegaConf.create(prepared["config"])
                cfg.meta.seed = cfg.local_seed = SMOKE_SEED
                agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
                adapter = None if arm == "native" else CombinedFixedResponseIntervention(
                    counted, bank, coupling_protocol, coupling_bank, arm)
                observed = ObservedCombined(counted, arm, output, adapter)
                agent.planner.unroll = observed
                trace, env = trace_actor(agent), make_env(cfg)
                episode_start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                try:
                    with close_expert_environments(plan_evaluator):
                        result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
                finally:
                    env.close()
                    exclusive_json(output / "unroll_calls.json", observed.calls)
                    exclusive_json(output / "action_trace.json", trace)
                verify_episode(result, observed.calls)
                current = {"result": result, "action_trace": trace}
                if completed:
                    verify_pair(completed["native"], current, native_repeat=name == "native_repeat")
                if model_signature(backend.model) != initial_model or not hooks_clean(backend.predictor):
                    raise ValueError("Frozen model bytes/modes or predictor hooks changed")
                report = {"arm": arm, "episode_name": name, "engineering_only": True,
                    "scientific_efficacy_measurement": False, "result": result,
                    "unroll_calls_sha256": sha256(output / "unroll_calls.json"),
                    "action_trace_sha256": sha256(output / "action_trace.json"),
                    "seconds": time.monotonic() - episode_start,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
                exclusive_json(output / "report.json", report)
                exclusive_json(output / "DONE.json", {"report_sha256": sha256(output / "report.json")})
                completed[name] = current
                print(json.dumps({"completed_engineering_episode": name,
                    "seconds": report["seconds"], "scientific_efficacy_measurement": False}), flush=True)
            final_model = model_signature(backend.model)
            exclusive_json(args.output / "model-after.json", final_model)
            verify_files(numerical["bound_files"])
            if final_model != initial_model or source_hash() != frozen_source:
                raise ValueError("Frozen model/source changed")
            exclusive_json(args.output / "report.json", {"status": "fixed_combined_full_cem_engineering_complete",
                "task": "reach", "method": METHOD, "protocol_sha256": sha256(args.output / "protocol.json"),
                "episodes": {name: sha256(args.output / name / "DONE.json") for name in EPISODES},
                "native_repeat_exact": True, "paired_stimuli_exact": True, "full_cem_schedule_verified": True,
                "parameter_buffer_bytes_modes_unchanged": True, "full_simulator_episodes": 4,
                "scenario_count": 1, "backend": backend.provenance, "device_uuid": uuid,
                "model_before_sha256": sha256(args.output / "model-before.json"),
                "model_after_sha256": sha256(args.output / "model-after.json"),
                "seconds": time.monotonic() - start, "scientific_efficacy_measurement": False,
                "fresh_confirmation": False, "behavioral_launch_ready": False})
            exclusive_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except BaseException as error:
        exclusive_json(args.output / "FAILED.json", {"error": repr(error),
            "automatic_retry": False, "scientific_efficacy_measurement": False, "behavioral_launch_ready": False})
        raise


if __name__ == "__main__":
    main()
