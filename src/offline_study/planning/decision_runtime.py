"""GPU executor for the frozen, one-decision diagnostic. No old queues are run."""
from offline_study._paths import PACKAGE_ROOT, source_files
import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.experiments.decision_diagnostic import ANALYSIS, ARMS, ROLE, TASKS, candidate_seed, score_summary
from offline_study.core.protocol import sha256, write_json


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_hash(value):
    a = value.detach().cpu().contiguous()
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.numpy().tobytes()).hexdigest()


def physics(env):
    data = env.proprio_env.unwrapped.data
    values = {k: np.asarray(getattr(data, k)).copy() for k in
              ("qpos", "qvel", "act", "time", "mocap_pos", "mocap_quat", "ctrl", "qacc_warmstart")}
    h = hashlib.sha256()
    for key, a in values.items():
        h.update(key.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return values, h.hexdigest()


def reset_initial(env, seed):
    """The exact initial part of native set_episode; no expert rollout is needed."""
    from evals.simu_env_planning.planning.utils import make_td
    env.reset(seed=seed, task_idx=0)
    unwrapped = env.proprio_env.unwrapped
    unwrapped._freeze_rand_vec = False
    unwrapped.seeded_rand_vec = True
    env.seed(seed)
    obs, info = env.reset_warmup(seed=seed)
    return make_td(obs, info), info


def capture_population(planner, context, seed):
    """Interrupt upstream at its FIRST cost call, after its unchanged sampler/clipping."""
    class Captured(Exception):
        pass
    original = planner.cost_function
    captured = []
    def intercept(actions, z_init):
        captured.append(actions.detach().clone())
        raise Captured()
    planner.local_generator.manual_seed(seed)
    planner.cost_function = intercept
    try:
        try:
            planner.plan(context, steps_left=6)
        except Captured:
            pass
    finally:
        planner.cost_function = original
    if len(captured) != 1 or captured[0].shape != (6, 300, 20) or not torch.isfinite(captured[0]).all():
        raise ValueError("Wrong native initial CEM candidate population")
    if torch.count_nonzero(captured[0][:, 0]):
        raise ValueError("Native zero-mean candidate was not retained")
    return captured[0]


def execute_prefix(env, seed, actions, initial_hash, physics_hash, goal_state):
    from offline_study.planning.planning_env_smoke import observation_digest
    initial, info = reset_initial(env, seed)
    state, digest = physics(env)
    if observation_digest(initial) != initial_hash or digest != physics_hash:
        raise ValueError("Simulator reset mismatch: do not evaluate an unpaired arm")
    initial_state = np.asarray(info["state"]).copy()
    _, rewards, dones, infos = env.step_multiple(actions)
    if len(infos) != 15 or not all(np.isfinite(np.asarray(x["state"])).all() for x in infos):
        raise ValueError("Incomplete/nonfinite fifteen-step prefix")
    final = np.asarray(infos[-1]["state"])
    goal = np.asarray(goal_state)
    return {"initial_sha256": initial_hash, "physics_sha256": digest,
            "initial_physics": jsonable(state), "initial_state": initial_state.tolist(),
            "states": [np.asarray(x["state"]).tolist() for x in infos],
            "executed_actions": actions.tolist(), "elementary_steps": len(infos),
            "terminal_ee_distance": float(np.linalg.norm(final[:3] - goal[:3])),
            "terminal_full_state_distance": float(np.linalg.norm(final - goal)),
            "initial_ee_distance": float(np.linalg.norm(initial_state[:3] - goal[:3])),
            "rewards": jsonable(rewards), "prefix_success": bool(infos[-1]["success"]),
            "dones": jsonable(dones)}


def source_binding(vendor):
    sources = {p.relative_to(PACKAGE_ROOT).as_posix(): sha256(p) for p in source_files()}
    upstream = {str(p.relative_to(vendor)): sha256(p) for base in ("evals", "app", "src", "configs")
                for p in sorted((vendor / base).rglob("*")) if p.is_file() and p.suffix in (".py", ".yaml")}
    return {"local": sources, "vendor": upstream}


def freeze(args):
    from offline_study.evaluation.behavioral_development import schedule
    from offline_study.interventions.fixed_response import load_fitted_bank
    from offline_study.evaluation.fixed_response_behavior import coupling_binding, stimulus_contract
    from offline_study.planning.planning_contract import prepare
    from offline_study.planning.planning_native_smoke import CHECKPOINTS
    from offline_study.models.vendor import use_vendor
    use_vendor(args.vendor)
    task_bindings = {}
    for task in TASKS:
        fit = args.fits / task
        load_fitted_bank(fit, task="mw-" + task, checkpoint_sha256=CHECKPOINTS["metaworld"])
        legacy, _, stimuli = stimulus_contract(args.stimuli, task)
        _, _, coupling = coupling_binding(args.original, legacy, task)
        task_bindings[task] = {"stimuli": stimuli, "coupling": coupling,
            "fit_done_sha256": sha256(fit / "DONE.json"), "fit_bank_sha256": sha256(fit / "operator_bank.pt"),
            "planning": prepare(args.vendor, task)}
    protocol = {"role": ROLE, "analysis": ANALYSIS, "arms": list(ARMS), "episodes": schedule(),
        "tasks": task_bindings, "source": source_binding(args.vendor),
        "checkpoint_sha256": CHECKPOINTS["metaworld"], "precision": "strict_float32_no_tf32",
        "population": "native_CEM_first_300_H6_var1_before_update", "selected_prefix_steps": 15,
        "candidate_rng": "SHA256 jepa-decision-v1/task/environment_seed first8hex",
        "fresh_confirmation": False, "full_CEM": False, "max_new_spend_usd": 5,
        "design_sha256": sha256(args.design)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
        "new_diagnostic_outcomes_observed_before_freeze": False})
    print("Frozen diagnostic inputs; no model/simulator evaluations performed", flush=True)


@torch.no_grad()
def run(args):
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from offline_study.models.backends import JepaBackend
    from offline_study.interventions.fixed_response import FixedResponseIntervention, load_fitted_bank
    from offline_study.evaluation.fixed_response_behavior import coupling_binding, stimulus_contract
    from offline_study.runtime.intervention_runner import _model_versions
    from offline_study.planning.planning_env_smoke import observation_digest
    from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED
    from offline_study.planning.planning_panel_engineering import COUPLING_ARMS, H6StaticPlanningIntervention
    from offline_study.planning.planning_scenarios import prepare_episode
    from offline_study.models.vendor import use_vendor
    use_vendor(args.vendor)
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.utils import prepare_obs

    protocol = json.loads((args.output / "protocol.json").read_text())
    digest = sha256(args.output / "protocol.json")
    if (digest != json.loads((args.output / "FROZEN.json").read_text())["protocol_sha256"] or
            protocol["role"] != ROLE or protocol["analysis"] != ANALYSIS or
            protocol["source"] != source_binding(args.vendor)):
        raise ValueError("Changed scientific freeze or source")
    # Bound this invocation even if the client disconnects. Provider billing is
    # separately bounded by the operator's lease and preservation deadline.
    deadline = time.monotonic() + args.max_seconds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
    versions = _model_versions(backend.model)
    runtime = {"packages": {p: importlib.metadata.version(p) for p in
               ("torch", "numpy", "mujoco", "metaworld", "gym", "gymnasium", "tensordict")},
               "device": str(torch.cuda.get_device_properties(0)), "protocol_sha256": digest,
               "backend": backend.provenance}
    write_json(args.output / "runtime.json", runtime)
    for task in TASKS:
        legacy, goals, stimulus = stimulus_contract(args.stimuli, task)
        fit = args.fits / task
        bank = load_fitted_bank(fit, task="mw-" + task, checkpoint_sha256=CHECKPOINTS["metaworld"])
        coupling_root, cp, cb = coupling_binding(args.original, legacy, task)
        expected = protocol["tasks"][task]
        if (stimulus != expected["stimuli"] or cb != expected["coupling"] or
                sha256(fit / "DONE.json") != expected["fit_done_sha256"] or
                sha256(fit / "operator_bank.pt") != expected["fit_bank_sha256"]):
            raise ValueError("Changed task fit/stimulus input")
        coupling_bank = torch.load(coupling_root / "operator_bank.pt", map_location="cpu", weights_only=True)
        adapters = {name: (H6StaticPlanningIntervention(backend, cp, coupling_bank, COUPLING_ARMS[name])
                          if name in COUPLING_ARMS else FixedResponseIntervention(backend, bank, name)) for name in ARMS}
        zero = FixedResponseIntervention(backend, bank, "zero_dose")
        cfg = OmegaConf.create(expected["planning"]["config"])
        cfg.local_seed = SMOKE_SEED
        agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
        env = make_env(cfg)
        task_root = args.output / task
        task_root.mkdir(exist_ok=True)
        try:
            engineering = {"episode": -1, "environment_seed": SMOKE_SEED, "logical_rank": -1, "local_seed": SMOKE_SEED}
            rows = [engineering] + protocol["episodes"]
            for row in rows:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded worker time elapsed; preserve partial results, do not expand budget")
                is_check = row["episode"] == -1
                out = task_root / ("engineering" if is_check else f"episode-{row['episode']:03d}")
                if (out / "DONE.json").exists():
                    # Each invocation revalidates engineering; never silently
                    # skip a new receiving-worker gate.
                    if is_check:
                        raise ValueError("Use a fresh run directory for a new worker attempt")
                    raise ValueError("Explicit verified resume required; no implicit outcome retry")
                out.mkdir(exist_ok=False)
                started = time.monotonic()
                cfg.local_seed = row["local_seed"]
                agent.local_generator.manual_seed(row["local_seed"])
                if is_check:
                    metadata, tensors = prepare_episode(cfg, agent, env, row)
                else:
                    metadata, tensors = goals.load(row)
                initial, _ = reset_initial(env, row["environment_seed"])
                state, ph = physics(env)
                if (observation_digest(initial) != metadata["initial_sha256"] or
                        observation_digest(tensors["goal"]) != metadata["goal_sha256"] or
                        not np.array_equal(env.proprio_env.unwrapped._last_rand_vec, metadata["rand_vec"])):
                    raise ValueError("Canonical initial/goal mismatch")
                goal = TensorDict(tensors["goal"], batch_size=[])
                agent.set_goal(prepare_obs(cfg.task_specification.obs, goal))
                context = backend.model.encode(prepare_obs(cfg.task_specification.obs, initial).to(backend.device).unsqueeze(0), act=True)
                context_hashes = {k: tensor_hash(context[k]) for k in ("visual", "proprio")}
                seed = candidate_seed(task, row["environment_seed"])
                actions = capture_population(agent.planner, context, seed)
                ah = tensor_hash(actions)
                if is_check and not torch.equal(actions, capture_population(agent.planner, context, seed)):
                    raise ValueError("Native population capture not deterministic")
                torch.save({"actions": actions.cpu(), "initial": tensors["initial"], "goal": tensors["goal"]}, out / "inputs.pt")
                record = {**row, "task": task, "protocol_sha256": digest, "candidate_seed": seed,
                    "candidate_sha256": ah, "initial_sha256": metadata["initial_sha256"],
                    "goal_sha256": metadata["goal_sha256"], "physics_sha256": ph,
                    "initial_physics": jsonable(state), "goal_state": metadata["goal_state"], "arms": {}}
                outcomes = {}
                native_prediction = None
                for name, adapter in adapters.items():
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    calls = [0]
                    original = backend.model.unroll
                    def counted(*a, **kw):
                        calls[0] += 1
                        return original(*a, **kw)
                    backend.model.unroll = counted
                    try:
                        prediction = adapter(context.clone(), actions.clone())
                    finally:
                        backend.model.unroll = original
                    scores = agent.objective(prediction, actions).detach().cpu().numpy()
                    torch.cuda.synchronize()
                    inference_seconds = time.monotonic() - before
                    summary = score_summary(scores)
                    if calls[0] != 1 or any(not torch.isfinite(prediction[k]).all() for k in ("visual", "proprio")):
                        raise ValueError("Extra online forecast or nonfinite prediction")
                    if is_check and name == "native":
                        for repeat_adapter in (adapter, zero):
                            repeat = repeat_adapter(context.clone(), actions.clone())
                            if any(not torch.equal(prediction[k], repeat[k]) for k in ("visual", "proprio")):
                                raise ValueError("Native-repeat/zero-dose identity failed")
                        native_prediction = True
                    chosen = summary["selected"]
                    model_actions = actions[:3, chosen].cpu()
                    elementary = backend.preprocessor.denormalize_actions(model_actions.reshape(15, 4))
                    if is_check:
                        from einops import rearrange
                        independent = backend.preprocessor.denormalize_actions(rearrange(model_actions, "t (f d) -> (t f) d", d=4))
                        if not torch.equal(elementary, independent):
                            raise ValueError("Native action conversion changed")
                    reused = chosen in outcomes
                    if not reused:
                        outcomes[chosen] = execute_prefix(env, row["environment_seed"], elementary,
                            metadata["initial_sha256"], ph, metadata["goal_state"])
                    result = outcomes[chosen]
                    if is_check and name == "native":
                        repeat = execute_prefix(env, row["environment_seed"], elementary,
                            metadata["initial_sha256"], ph, metadata["goal_state"])
                        if repeat != result:
                            raise ValueError("Physical native prefix not exactly repeatable")
                    energy = adapter.last_record if hasattr(adapter, "last_record") else adapter.energy[-1]
                    record["arms"][name] = {**result, **summary, "scores": scores.tolist(),
                        "candidate_sha256": ah, "goal_sha256": metadata["goal_sha256"],
                        "outcome_reused_same_candidate": reused, "inference_seconds": inference_seconds,
                        "backend_calls": calls[0], "intervention_audit": jsonable(energy)}
                    write_json(out / "partial.json", record)
                if (tensor_hash(actions) != ah or _model_versions(backend.model) != versions or
                        any(tensor_hash(context[k]) != context_hashes[k] for k in context_hashes)):
                    raise ValueError("Mutated candidate bank, context, or parameters")
                record["seconds"] = time.monotonic() - started
                record["engineering_only"] = is_check
                record["native_zero_repeat_passed"] = native_prediction if is_check else None
                write_json(out / "record.json", record)
                write_json(out / "DONE.json", {"files": {n: sha256(out / n) for n in ("inputs.pt", "record.json")}})
                print(json.dumps({"task": task, "episode": row["episode"], "seconds": record["seconds"], "arms": 5}), flush=True)
        finally:
            env.close()
    write_json(args.output / "RUN_DONE.json", {"protocol_sha256": digest, "scenarios": 192, "condition_prefixes": 960})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    for name in ("vendor", "fits", "stimuli", "original", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("docs/PLANNER_DECISION_DIAGNOSTIC.md"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=int, default=10800)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    else:
        try:
            run(args)
        except Exception as exc:
            write_json(args.output / "FAILED.json", {"error": str(exc), "new_outcomes_must_be_preserved": True})
            raise


if __name__ == "__main__":
    main()
