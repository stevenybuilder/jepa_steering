"""Complete native CEM episodes for the fixed-response successor.

An excluded engineering scenario, never candidate selection or confirmation.
The repeated native episodes must match before either edited episode is run.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .author_fit import source_hash
from .backends import JepaBackend
from .fixed_response import ARMS, METHOD, FixedResponseIntervention, load_fitted_bank
from .fixed_response_check import CountedBackend
from .operator_fit import _model_versions
from .planning_contract import prepare
from .planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from .planning_support_smoke import verify_call_schedule
from .protocol import sha256, write_json
from .vendor import use_vendor


EPISODES = ("native", "native_repeat", "fixed_rank4", "matched_random_fixed_rank4")


def verify_numerical(root, fit, checked_source, task, checkpoint):
    """Bind the completed real-GPU check to this exact fit and unchanged runtime."""
    root, fit, checked_source = Path(root), Path(fit), Path(checked_source)
    done = json.loads((root / "DONE.json").read_text())
    report = json.loads((root / "report.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    if ((root / "FAILED.json").exists() or sha256(root / "report.json") != done["report_sha256"] or
            sha256(root / "contract.json") != report["contract_sha256"]):
        raise ValueError("Incomplete or changed numerical check")
    if (contract.get("method") != METHOD or contract.get("task") != "mw-" + task or
            contract.get("fit_done_sha256") != sha256(fit / "DONE.json") or
            contract.get("fit_bank_sha256") != sha256(fit / "operator_bank.pt") or
            contract.get("arms") != list(ARMS) or contract.get("planning_context") != 2 or
            contract.get("precision") != "float32_strict_no_tf32" or
            report.get("status") != "fixed_map_fit_action_planning_engineering_complete" or
            report.get("parameters_unchanged") is not True or
            report.get("scientific_efficacy_measurement") is not False or
            report.get("backend", {}).get("checkpoint_sha256") != checkpoint):
        raise ValueError("Numerical receipt does not bind this task, bank and planning contract")
    for arm in ARMS[2:]:
        for count in (8, 19, 300):
            if not any(row.get("arm") == arm and row.get("candidate_count") == count and
                       row.get("static_reference_bitwise_equal") is True and
                       row.get("backend_calls") == 1 and row.get("response_probes") == 0 and
                       row.get("native_shadows") == 0 for row in report["checks"]):
                raise ValueError("Missing exact fixed-map/control population check")
    digest, checked_files = hashlib.sha256(), {}
    for path in sorted(checked_source.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
        if sha256(path) != sha256(Path(__file__).parent / path.name):
            raise ValueError("Previously checked scientific runtime changed: " + path.name)
        checked_files[path.name] = sha256(path)
    if not checked_files or digest.hexdigest() != contract["source_sha256"]:
        raise ValueError("Numerical-check source snapshot changed")
    return {"report_sha256": done["report_sha256"], "checked_source_sha256": digest.hexdigest(),
            "unchanged_runtime_files": checked_files}


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite action/observation in engineering trace")
    return hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode() +
                          value.numpy().tobytes()).hexdigest()


def trace_actor(agent):
    """Observe the unchanged author's action callback without consuming RNG."""
    original, trace = agent.act, []

    def act(observation, steps_left):
        inputs = {key: tensor_digest(observation[key]) for key in sorted(observation.keys())}
        action = original(observation, steps_left=steps_left)
        trace.append({"steps_left": int(steps_left), "observation_sha256": inputs,
                      "actions_sha256": tensor_digest(action)})
        return action

    agent.act = act
    return trace


def verify_episode(result, calls):
    verify_call_schedule(result["planning_calls"], calls)
    if result["elementary_steps"] != 100 or result["published_candidate_count"] != 300:
        raise ValueError("Incomplete/shortened official MetaWorld episode")


def verify_pair(reference, current, native_repeat=False):
    for key in ("initial_sha256", "goal_sha256", "expert_success"):
        if reference["result"][key] != current["result"][key]:
            raise ValueError("Paired engineering stimuli changed: " + key)
    if native_repeat:
        for key in ("native_success", "native_state_distance", "native_reward", "elementary_steps",
                    "observed_frames", "published_candidate_count"):
            if reference["result"][key] != current["result"][key]:
                raise ValueError("Receiving-worker native repeat changed: " + key)
        if reference["action_trace"] != current["action_trace"]:
            raise ValueError("Receiving-worker native actions/observations did not repeat exactly")


class ObservedFixed:
    def __init__(self, adapter, counted, output):
        self.adapter, self.counted, self.output, self.calls = adapter, counted, output, []

    def __call__(self, context, act_suffix=None, **kwargs):
        horizon, candidates, _ = act_suffix.shape
        before_calls, started = self.counted.calls, time.monotonic()
        write_json(self.output / "progress.json", {"completed_calls": len(self.calls),
            "running_horizon": horizon, "running_candidates": candidates, "scientific_efficacy": False})
        result = self.adapter(context, act_suffix, **kwargs)
        if self.counted.calls - before_calls != 1:
            raise ValueError("Successor performed extra model calls")
        if any(not torch.isfinite(result[key]).all() for key in ("visual", "proprio")):
            raise ValueError("Nonfinite planner forecast")
        torch.cuda.synchronize()
        record = {key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                  for key, value in self.adapter.last_record.items()}
        if record["response_probe_rollouts"] or record["native_shadow_rollouts"]:
            raise ValueError("Successor probe/shadow contract changed")
        self.calls.append({"horizon": horizon, "candidates": candidates,
                           "seconds": time.monotonic() - started, "record": record})
        # Complete call records are written once per episode. The small progress
        # file is sufficient for liveness; no growing-file rewrite per forecast.
        return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "fit", "numerical-check", "checked-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS["metaworld"]
    bank = load_fitted_bank(args.fit, task="mw-" + args.task, checkpoint_sha256=checkpoint)
    numerical = verify_numerical(args.numerical_check, args.fit, args.checked_source, args.task, checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        use_vendor(args.vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent

        prepared = prepare(args.vendor, args.task)
        protocol = {"method": METHOD, "task": args.task, "source_sha256": source_hash(),
            "numerical_check": numerical, "fit_done_sha256": sha256(args.fit / "DONE.json"),
            "fit_bank_sha256": sha256(args.fit / "operator_bank.pt"), "planning_contract": prepared,
            "precision": "float32_strict_no_tf32", "engineering_seed": SMOKE_SEED,
            "episode_order": list(EPISODES), "episodes_in_this_check": 4,
            "scenario_count": 1, "new_receiving_worker_native_reference": True,
            "historical_worker_goal_pixels_assumed_equal": False,
            "candidate_selection_from_smoke_forbidden": True, "scientific_efficacy_measurement": False,
            "fresh_confirmation": False, "protected_outcomes_access_authorized": False}
        write_json(args.output / "protocol.json", protocol)
        backend = JepaBackend(args.vendor, args.checkpoint, checkpoint, "metaworld", "cuda:0", "float32")
        if backend.model.ctxt_window != 2:
            raise ValueError("Wrong author planning context")
        versions, counted = _model_versions(backend.model), CountedBackend(backend)
        completed = {}
        for name in EPISODES:
            output = args.output / name
            output.mkdir(exist_ok=False)
            arm = "native" if name == "native_repeat" else name
            random.seed(SMOKE_SEED)
            np.random.seed(SMOKE_SEED)
            torch.manual_seed(SMOKE_SEED)
            cfg = OmegaConf.create(prepared["config"])
            cfg.meta.seed = cfg.local_seed = SMOKE_SEED
            agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
            observed = ObservedFixed(FixedResponseIntervention(counted, bank, arm), counted, output)
            agent.planner.unroll = observed
            trace = trace_actor(agent)
            env = make_env(cfg)
            torch.cuda.reset_peak_memory_stats()
            episode_start = time.monotonic()
            try:
                result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
            finally:
                env.close()
                write_json(output / "unroll_calls.json", observed.calls)
                write_json(output / "action_trace.json", trace)
            verify_episode(result, observed.calls)
            current = {"result": result, "action_trace": trace}
            if completed:
                verify_pair(completed["native"], current, native_repeat=name == "native_repeat")
            if _model_versions(backend.model) != versions or source_hash() != protocol["source_sha256"]:
                raise ValueError("Frozen model/source changed")
            report = {"arm": arm, "episode_name": name, "engineering_only": True,
                "scientific_efficacy_measurement": False, "result": result,
                "unroll_calls_sha256": sha256(output / "unroll_calls.json"),
                "action_trace_sha256": sha256(output / "action_trace.json"),
                "seconds": time.monotonic() - episode_start,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
            write_json(output / "report.json", report)
            write_json(output / "DONE.json", {"report_sha256": sha256(output / "report.json")})
            completed[name] = current
            print(json.dumps({"completed_engineering_episode": name, "seconds": report["seconds"],
                              "task": args.task, "scientific_efficacy_measurement": False}), flush=True)
        write_json(args.output / "report.json", {"status": "fixed_response_full_cem_engineering_complete",
            "task": args.task, "method": METHOD, "protocol_sha256": sha256(args.output / "protocol.json"),
            "episodes": {name: sha256(args.output / name / "DONE.json") for name in EPISODES},
            "native_repeat_exact": True, "paired_stimuli_exact": True, "full_cem_schedule_verified": True,
            "parameters_unchanged": True, "full_simulator_episodes": 4, "scenario_count": 1,
            "backend": backend.provenance, "device_name": torch.cuda.get_device_name(),
            "seconds": time.monotonic() - started, "scientific_efficacy_measurement": False,
            "fresh_confirmation": False, "engineering_ready_not_behavioral_access_authorization": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
