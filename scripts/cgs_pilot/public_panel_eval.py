#!/usr/bin/env python3
"""Panel P driver: run the released jepa-wms planning evaluator, unmodified, on a released
public checkpoint and log the evaluator's own per-episode fields.

Governed by "Label-first principle, amendment 3" (``cross model design jepa.md``): the label is the
official evaluator's per-episode success flag (``succ_def: simu``); nothing here changes the
evaluator, the goal sampler, the horizon, the planner configuration or the success definition.
The only edits to the released config are recorded as ``config_deviations`` in
``config_provenance.json`` next to the output and are limited to:

1. ``model_kwargs.checkpoint`` -> absolute path of the released public checkpoint (the config
   expects ``jepa-latest.pth.tar`` inside the authors' cluster training folder);
2. ``folder`` / ``checkpoint_folder`` -> a local log directory (paths only);
3. ``heads_cfg.pretrain_dec_path.state_head`` -> null when the referenced state-decoder head is
   not part of the public release (the state head is constructed by the wrapper but never read by
   the planning evaluator; ``grep -rn state_head evals/`` finds no consumer);
4. ``meta.seed`` when ``--seed`` differs from the released value (extra evaluation seeds);
5. ``meta.eval_episodes`` only for the smoke test (``--episodes``), never for the screen;
6. ``task_specification.task`` only when ``--task`` is given (a released MetaWorld config exists
   for one task per model; the swapped task is a labelled deviation).

Per-episode logging is a read-only wrapper around ``PlanEvaluator.eval`` /
``PlanEvaluator.unroll_agent``: it records what the evaluator returns and nothing else.

Reproduction note: the released configs run 8 ranks per node (``tasks_per_node: 8``), each rank
with ``local_seed = seed + rank * horizon * 1000``. This driver runs the evaluator in the
single-process ``--debug`` mode of ``evals/main.py`` (rank 0, world size 1), so every episode
draws from ``local_seed = meta.seed``; the goal sampler, planner and label are unchanged but the
per-episode RNG stream differs from an 8-GPU run. Recorded in the provenance.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import difflib
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repository_provenance(repo: Path) -> dict:
    """Return commit plus a content-addressed record of every dirty path."""
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "-z")
    diff = git("diff", "--binary", "HEAD", "--")
    dirty_files = []
    for entry in (item for item in status.decode(errors="replace").split("\0") if item):
        code, name = entry[:2], entry[3:]
        path = repo / name
        dirty_files.append({
            "status": code,
            "path": name,
            "sha256": sha256_file(path) if path.is_file() else None,
        })
    return {
        "commit": commit,
        "clean": not bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "raw_status": status.decode(errors="replace").replace("\0", "\\0"),
        "dirty_files": dirty_files,
    }


def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray) and v.size <= 16:
        return v.astype(float).ravel().tolist()
    if isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(x, (int, float, np.number)) for x in v):
        return [float(x) for x in v]
    return None


def scalarize(info: dict) -> dict:
    out = {}
    for k, v in (info or {}).items():
        s = _scalar(v)
        if s is not None:
            out[str(k)] = s
    return out


def stable_value_sha256(value) -> str:
    """Hash an observation/state by semantic value, independent of device/storage layout."""
    h = hashlib.sha256()

    def update(v) -> None:
        if hasattr(v, "detach") and hasattr(v, "cpu"):
            tensor = v.detach().cpu().contiguous()
            # NumPy does not expose every torch dtype (notably bfloat16).
            if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
                tensor = tensor.float()
            update(tensor.numpy())
        elif isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(b"array\0")
            h.update(str(a.dtype).encode())
            h.update(b"\0")
            h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
            h.update(b"\0")
            h.update(a.tobytes(order="C"))
        elif hasattr(v, "keys") and hasattr(v, "__getitem__"):
            h.update(b"mapping\0")
            for key in sorted((str(k) for k in v.keys())):
                h.update(key.encode())
                h.update(b"\0")
                update(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(b"sequence\0")
            h.update(str(len(v)).encode())
            for item in v:
                update(item)
        elif v is None or isinstance(v, (str, bool, int, float, np.number)):
            h.update(json.dumps(_scalar(v), sort_keys=True, allow_nan=True).encode())
        else:
            raise TypeError(f"unsupported realization value {type(v)!r}")

    update(value)
    return h.hexdigest()


def install_realization_hashing(pe) -> None:
    """Record immutable hashes of the exact initialized observation and sampled goal.

    This is deliberately read-only.  Paired arms can later prove that the seed adapter
    produced the same physical realization instead of trusting integer seed equality.
    """
    original_set_episode = pe.PlanEvaluator.set_episode

    def set_episode_with_hashes(self, cfg, agent, env, ep_seed, task_idx=-1):
        # The released MetaWorld evaluator does two resets.  PlanEvaluator.eval
        # retains ``info`` from the first, discarded reset, while set_episode uses
        # reset_warmup for the initial observation that the policy actually sees.
        # Capture the latter without changing either reset or the evaluator flow.
        captured_warmup_info = None
        reset_warmup = getattr(env, "reset_warmup", None)
        env_dict = getattr(env, "__dict__", {})
        had_instance_reset_warmup = "reset_warmup" in env_dict
        instance_reset_warmup = env_dict.get("reset_warmup")

        if callable(reset_warmup):
            def reset_warmup_with_capture(*args, **kwargs):
                nonlocal captured_warmup_info
                output = reset_warmup(*args, **kwargs)
                if isinstance(output, tuple) and len(output) >= 2:
                    captured_warmup_info = output[1]
                return output

            setattr(env, "reset_warmup", reset_warmup_with_capture)
        try:
            result = original_set_episode(self, cfg, agent, env, ep_seed, task_idx=task_idx)
        finally:
            if callable(reset_warmup):
                if had_instance_reset_warmup:
                    setattr(env, "reset_warmup", instance_reset_warmup)
                else:
                    delattr(env, "reset_warmup")

        init_obs, goal_obs, _expert_obses, _expert_success = result
        record = {
            "initial_observation_sha256": stable_value_sha256(init_obs),
            "goal_observation_sha256": stable_value_sha256(goal_obs),
            "initial_visual_sha256": stable_value_sha256(init_obs["visual"]),
            "goal_visual_sha256": stable_value_sha256(goal_obs["visual"]),
            "goal_state_sha256": stable_value_sha256(self.state_g),
        }
        if "proprio" in init_obs.keys():
            record["initial_proprio_sha256"] = stable_value_sha256(init_obs["proprio"])
        if "proprio" in goal_obs.keys():
            record["goal_proprio_sha256"] = stable_value_sha256(goal_obs["proprio"])
        initial_state = (
            captured_warmup_info.get("state")
            if isinstance(captured_warmup_info, dict)
            else None
        )
        if initial_state is not None:
            self._panelp_initial_simulator_state = np.array(initial_state, copy=True)
            record["initial_simulator_state_sha256"] = stable_value_sha256(initial_state)
        elif hasattr(self, "_panelp_initial_simulator_state"):
            delattr(self, "_panelp_initial_simulator_state")
        self._panelp_realization_hashes = record
        return result

    pe.PlanEvaluator.set_episode = set_episode_with_hashes


def simulator_state_trace(evaluator, initial_info: dict, step_infos: list[dict]) -> np.ndarray | None:
    """Return the policy-realized initial state followed by post-action states.

    MetaWorld's released PlanEvaluator passes stale metadata from a discarded
    preliminary reset to ``unroll_agent``.  The state captured from the later
    reset_warmup is authoritative because its observation is the one supplied to
    the planner.  Other environments retain the vendor-provided initial info as a
    fallback.
    """
    realized_initial = getattr(evaluator, "_panelp_initial_simulator_state", None)
    if realized_initial is None:
        realized_initial = (initial_info or {}).get("state")
    values = []
    if realized_initial is not None:
        values.append(realized_initial)
    values.extend(info["state"] for info in step_infos if info.get("state") is not None)
    if not values:
        return None
    try:
        return np.stack([np.asarray(value) for value in values]).astype(np.float32)
    except (TypeError, ValueError):
        return None


def install_logging_hooks(pe, jsonl_path: Path, meta: dict, episode_rows: list[dict] | None = None) -> dict:
    """Wrap PlanEvaluator.eval / unroll_agent so that every episode appends one JSON line with
    the evaluator's returned fields. The wrapped functions call the originals and return their
    results untouched."""
    orig_unroll = pe.PlanEvaluator.unroll_agent
    orig_eval = pe.PlanEvaluator.eval
    state: dict = {}
    counter = {"n": 0, "n_success": 0}
    trace_dir = jsonl_path.parent / "behavior_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    def unroll_agent(self, env, obs, info, actor, preprocessor=None):
        res = orig_unroll(self, env, obs, info, actor, preprocessor=preprocessor)
        _ep_obs, _ep_reward, actions, infos_list, _success, _state_dist = res
        state["n_env_steps"] = int(len(infos_list))
        state["n_plan_calls"] = int(len(actions))
        if actions:
            action_arrays = [
                action.detach().float().cpu().numpy() if hasattr(action, "detach") else np.asarray(action, dtype=float)
                for action in actions
            ]
            flat_actions = np.concatenate([np.asarray(action).reshape(-1, np.asarray(action).shape[-1]) for action in action_arrays])
            state["executed_action_sha256"] = stable_value_sha256(action_arrays)
            state["executed_action_l2_mean"] = float(np.linalg.norm(flat_actions, axis=1).mean())
            state["executed_action_linf"] = float(np.abs(flat_actions).max())
            state["executed_action_total_variation"] = float(
                np.linalg.norm(np.diff(flat_actions, axis=0), axis=1).sum() if len(flat_actions) > 1 else 0.0
            )
            trace = {
                "executed_actions": flat_actions.astype(np.float32),
                "plan_boundaries": np.cumsum([0] + [len(np.asarray(action).reshape(-1, np.asarray(action).shape[-1])) for action in action_arrays]).astype(np.int32),
            }
            simulator_states = simulator_state_trace(self, info, list(infos_list))
            if simulator_states is not None:
                trace["simulator_states"] = simulator_states
            if hasattr(self, "state_g"):
                try:
                    trace["goal_state"] = np.asarray(self.state_g, dtype=np.float32)
                except (TypeError, ValueError):
                    pass
            for key in ("success", "near_object", "grasp_success", "grasp_reward", "in_place_reward", "obj_to_target"):
                values = [_scalar(step.get(key)) for step in infos_list]
                if values and all(value is not None and not isinstance(value, list) for value in values):
                    trace[f"info_{key}"] = np.asarray(values)
            trace_path = trace_dir / f"ep{int(state['ep']):03d}.npz"
            np.savez_compressed(trace_path, **trace)
            state["behavior_trace"] = str(trace_path.relative_to(jsonl_path.parent))
            state["behavior_trace_sha256"] = sha256_file(trace_path)
        state["last_info"] = scalarize(infos_list[-1]) if infos_list else {}
        return res

    def eval_ep(self, cfg, agent, env, task_idx=-1, ep=0):
        state.clear()
        state["ep"] = int(ep)
        t0 = time.time()
        res = orig_eval(self, cfg, agent, env, task_idx=task_idx, ep=ep)
        (
            expert_success,
            success,
            ep_reward,
            success_dist,
            end_distance,
            end_distance_xyz,
            end_distance_orientation,
            end_distance_closure,
            state_dist,
            total_lpips,
            total_emb_l2,
        ) = res
        local_seed = int(cfg.local_seed)
        vendor_ep_seed = (local_seed * local_seed + int(ep) * local_seed) % (2**32 - 2)
        manifest_row = episode_rows[int(ep)] if episode_rows is not None else None
        ep_seed = int(manifest_row["env_seed"]) if manifest_row else vendor_ep_seed
        rec = dict(meta)
        rec.update(
            {
                "task": str(cfg.tasks[task_idx]),
                "ep": int(ep),
                "ep_seed": int(ep_seed),
                "env_seed": int(ep_seed),
                "planner_seed": int(manifest_row["planner_seed"]) if manifest_row else local_seed,
                "pair_id": str(manifest_row["pair_id"]) if manifest_row else None,
                "split": str(manifest_row["split"]) if manifest_row else None,
                "vendor_ep_seed_ignored": int(vendor_ep_seed) if manifest_row else None,
                "local_seed": local_seed,
                "meta_seed": int(cfg.meta.seed),
                "rank": int(cfg.rank),
                "world_size": int(cfg.world_size),
                "success": int(bool(success)),
                "expert_success": _scalar(expert_success),
                "ep_reward": _scalar(ep_reward),
                "success_dist": _scalar(success_dist),
                "end_distance": _scalar(end_distance),
                "end_distance_xyz": _scalar(end_distance_xyz),
                "end_distance_orientation": _scalar(end_distance_orientation),
                "end_distance_closure": _scalar(end_distance_closure),
                "state_dist": _scalar(state_dist),
                "total_lpips": _scalar(total_lpips),
                "total_emb_l2": _scalar(total_emb_l2),
                "n_env_steps": state.get("n_env_steps"),
                "n_plan_calls": state.get("n_plan_calls"),
                "executed_action_sha256": state.get("executed_action_sha256"),
                "executed_action_l2_mean": state.get("executed_action_l2_mean"),
                "executed_action_linf": state.get("executed_action_linf"),
                "executed_action_total_variation": state.get("executed_action_total_variation"),
                "behavior_trace": state.get("behavior_trace"),
                "behavior_trace_sha256": state.get("behavior_trace_sha256"),
                "last_info": state.get("last_info", {}),
                "wall_s": round(time.time() - t0, 3),
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
        )
        realization = dict(getattr(self, "_panelp_realization_hashes", {}))
        rec["realization_hashes"] = realization
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        realization_rec = {
            key: rec.get(key)
            for key in (
                "env", "model", "task", "split", "pair_id", "ep", "env_seed", "planner_seed",
                "episode_manifest_sha256", "config_sha256", "checkpoint_sha256",
            )
        }
        realization_rec.update(realization)
        with open(jsonl_path.parent / "realizations.jsonl", "a") as f:
            f.write(json.dumps(realization_rec, sort_keys=True) + "\n")
            f.flush()
        counter["n"] += 1
        counter["n_success"] += int(bool(success))
        logging.getLogger("panelP").info(
            f"[panelP] {meta['env']}/{meta['model']} seed={meta['seed']} pair={rec['pair_id']} "
            f"task={rec['task']} ep={ep} "
            f"success={int(bool(success))} running={counter['n_success']}/{counter['n']} wall={rec['wall_s']}s"
        )
        return res

    pe.PlanEvaluator.unroll_agent = unroll_agent
    pe.PlanEvaluator.eval = eval_ep
    return counter


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="path to vendor/jepa-wms (unmodified)")
    ap.add_argument("--env", required=True, choices=["mz", "wall", "pt", "mw"])
    ap.add_argument("--model", required=True, choices=["jepa-wm", "dino-wm"])
    ap.add_argument("--checkpoint", required=True, help="released public checkpoint (.pth.tar)")
    ap.add_argument("--out", required=True, help="output directory for this (env, model, seed) run")
    ap.add_argument("--seed", type=int, default=None, help="meta.seed; default = released config value")
    ap.add_argument("--episodes", type=int, default=None, help="SMOKE ONLY: override meta.eval_episodes")
    ap.add_argument("--task", default=None, help="DEVIATION: override task_specification.task")
    ap.add_argument("--label", default="screen")
    ap.add_argument("--episode-manifest", type=Path, default=None,
                    help="immutable JSONL with independent env_seed/planner_seed pairs")
    ap.add_argument(
        "--disable-optional-visualization",
        action="store_true",
        help=(
            "disable CEM best-frame decoding and optional plots; canary-only until "
            "exact action/outcome equivalence is established"
        ),
    )
    return ap


def apply_optional_visualization_policy(
    params: dict,
    deviations: list[dict],
    *,
    disabled: bool,
) -> None:
    """Disable visualization-only work while recording every resolved-config edit.

    The released CEM implementation selects elites and updates the optimizer before
    optionally decoding the best candidate for plotting.  This helper does not assert
    behavioral equivalence: the separate exact-action canary must establish that on
    the real evaluator before the option may be used for an experimental arm.
    """
    if not disabled:
        return
    edits = (
        ("planner", "decode_each_iteration"),
        ("logging", "optional_plots"),
    )
    for section, key in edits:
        current = params.setdefault(section, {}).get(key)
        if current is not False:
            deviations.append(
                {
                    "field": f"{section}.{key}",
                    "from": current,
                    "to": False,
                    "reason": (
                        "visualization-only runtime canary; admissible for experimental "
                        "runs only after exact paired action/outcome equivalence"
                    ),
                }
            )
        params[section][key] = False


@contextlib.contextmanager
def explicit_environment_seed(env, seed: int):
    """Substitute one explicit seed into the vendor evaluator's environment calls.

    The evaluator computes its own ``ep_seed`` inside ``PlanEvaluator.eval``.  Replacing
    only the seed-bearing methods keeps all evaluator logic intact while ensuring the
    manifest—not the legacy formula—selects the initial state and goal.  Instance
    attributes are restored exactly after the episode.
    """
    method_names = ("reset", "reset_warmup", "seed", "prepare", "sample_random_init_goal_states")
    patched: list[tuple[str, bool, object]] = []
    instance_dict = getattr(env, "__dict__", {})

    for name in method_names:
        original = getattr(env, name, None)
        if not callable(original):
            continue
        had_instance_value = name in instance_dict
        instance_value = instance_dict.get(name)

        def seeded_call(*args, __name=name, __original=original, **kwargs):
            if "seed" in kwargs:
                kwargs["seed"] = int(seed)
            elif args and __name in {
                "reset",
                "reset_warmup",
                "seed",
                "prepare",
                "sample_random_init_goal_states",
            }:
                args = (int(seed), *args[1:])
            elif __name == "seed":
                args = (int(seed),)
            return __original(*args, **kwargs)

        try:
            setattr(env, name, seeded_call)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(f"cannot install explicit seed adapter on env.{name}") from exc
        patched.append((name, had_instance_value, instance_value))

    try:
        yield
    finally:
        for name, had_instance_value, instance_value in reversed(patched):
            if had_instance_value:
                setattr(env, name, instance_value)
            else:
                delattr(env, name)


def install_manifest_seeding(pe, episode_rows: list[dict]) -> None:
    """Wrap ``PlanEvaluator.eval`` with independent per-episode RNG initialization."""
    if not episode_rows:
        raise ValueError("episode manifest is empty")
    original_eval = pe.PlanEvaluator.eval

    def eval_with_manifest(self, cfg, agent, env, task_idx=-1, ep=0):
        ep = int(ep)
        if not 0 <= ep < len(episode_rows):
            raise IndexError(f"episode {ep} outside manifest with {len(episode_rows)} rows")
        row = episode_rows[ep]
        task = str(cfg.tasks[task_idx])
        if str(row["task"]) != task:
            raise RuntimeError(f"manifest task {row['task']!r} does not match evaluator task {task!r}")
        planner_seed = int(row["planner_seed"])
        env_seed = int(row["env_seed"])

        # The CEM/MPPI planners use these private generators.  Resetting all standard
        # generators also makes any auxiliary stochastic code replayable by pair ID.
        random.seed(planner_seed)
        np.random.seed(planner_seed)
        import torch

        torch.manual_seed(planner_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(planner_seed)
        for name in ("local_generator", "local_gpu_generator"):
            generator = getattr(agent, name, None)
            if generator is not None:
                generator.manual_seed(planner_seed)

        with explicit_environment_seed(env, env_seed):
            return original_eval(self, cfg, agent, env, task_idx=task_idx, ep=ep)

    pe.PlanEvaluator.eval = eval_with_manifest


def resolve_config(args, extra_deviations: list[dict] | None = None):
    """Load the released config, apply the recorded path/seed edits, write resolved_config.yaml +
    config_provenance.json. Returns (repo, params, meta, out, jsonl_path). Also chdir()s into the
    repo and puts it on sys.path (what evals/main.py assumes)."""
    driver_path = Path(sys.argv[0]).resolve()
    repo = Path(args.repo).resolve()
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.out = str(Path(args.out).resolve())
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    cfg_dir = repo / "configs" / "evals" / "simu_env_planning" / args.env / args.model
    cfgs = sorted(cfg_dir.glob("*.yaml"))
    if len(cfgs) != 1:
        raise SystemExit(f"expected exactly one released config in {cfg_dir}, found {cfgs}")
    cfg_path = cfgs[0]
    original_text = cfg_path.read_text()
    params = yaml.load(original_text, Loader=yaml.FullLoader)
    original = copy.deepcopy(params)
    repo_provenance = repository_provenance(repo)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "episodes.jsonl"
    if jsonl_path.exists():
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        jsonl_path.rename(out / f"episodes.partial_{stamp}.jsonl")
        realizations_path = out / "realizations.jsonl"
        if realizations_path.exists():
            realizations_path.rename(out / f"realizations.partial_{stamp}.jsonl")

    deviations = []

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    deviations.append(
        {
            "field": "model_kwargs.checkpoint",
            "from": params["model_kwargs"].get("checkpoint"),
            "to": str(ckpt),
            "reason": "released public checkpoint; the config expects jepa-latest.pth.tar in the authors' training folder",
        }
    )
    params["model_kwargs"]["checkpoint"] = str(ckpt)

    eval_logs = out / "eval_logs"
    for key in ("folder", "checkpoint_folder"):
        if key in params:
            deviations.append({"field": key, "from": params[key], "to": str(eval_logs), "reason": "local log directory (paths only)"})
            params[key] = str(eval_logs)

    heads = params["model_kwargs"]["pretrain_kwargs"].get("heads_cfg") or {}
    pdp = heads.get("pretrain_dec_path") or {}
    sh = pdp.get("state_head")
    if sh:
        expanded = os.path.expandvars(str(sh))
        new_path = (heads.get("new_path_heads") or {}).get("state_head", True)
        candidate = expanded.removesuffix(".pth.tar") + "_state_head.pth.tar" if new_path else expanded
        if not (Path(candidate).exists() or Path(expanded).exists()):
            deviations.append(
                {
                    "field": "model_kwargs.pretrain_kwargs.heads_cfg.pretrain_dec_path.state_head",
                    "from": sh,
                    "to": None,
                    "reason": "state-decoder head not in the public release (README lists image decoders only); the head is never read by the planning evaluator (no consumer under evals/)",
                }
            )
            pdp["state_head"] = None

    if args.seed is not None and int(args.seed) != int(params["meta"]["seed"]):
        deviations.append({"field": "meta.seed", "from": params["meta"]["seed"], "to": int(args.seed), "reason": "additional evaluation seed"})
        params["meta"]["seed"] = int(args.seed)

    episode_rows = None
    if args.episode_manifest is not None:
        if args.episodes is not None:
            raise SystemExit("--episodes and --episode-manifest are mutually exclusive")
        manifest_path = Path(args.episode_manifest).resolve()
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from public_panel_manifest import read_jsonl, rows_sha256, validate_rows

        episode_rows = read_jsonl(manifest_path)
        verdict = validate_rows(((manifest_path.stem, episode_rows),))
        if not verdict["passed"]:
            raise SystemExit(f"invalid episode manifest: {verdict['errors']}")
        tasks = {str(row["task"]) for row in episode_rows}
        if tasks != {str(params["task_specification"]["task"])}:
            raise SystemExit(
                f"manifest task(s) {sorted(tasks)} do not match config task "
                f"{params['task_specification']['task']!r}"
            )
        expected_identity = {
            "checkpoint_sha256": sha256_file(ckpt),
            "config_sha256": sha256_file(cfg_path),
            "repo_commit": repo_provenance["commit"],
        }
        for field, actual in expected_identity.items():
            declared = {str(row[field]) for row in episode_rows}
            if declared != {actual}:
                raise SystemExit(f"manifest {field}={sorted(declared)} does not match runtime {actual}")
        deviations.append(
            {
                "field": "meta.eval_episodes / episode RNG",
                "from": {
                    "eval_episodes": params["meta"]["eval_episodes"],
                    "seed_rule": "local_seed**2 + ep*local_seed",
                },
                "to": {
                    "eval_episodes": len(episode_rows),
                    "manifest": str(manifest_path),
                    "manifest_sha256": rows_sha256(episode_rows),
                },
                "reason": "paired preregistration with independent environment and planner seeds",
            }
        )
        params["meta"]["eval_episodes"] = len(episode_rows)
        args._episode_rows = episode_rows
        args._episode_manifest_path = manifest_path

    if args.episodes is not None:
        deviations.append({"field": "meta.eval_episodes", "from": params["meta"]["eval_episodes"], "to": int(args.episodes), "reason": "SMOKE TEST ONLY"})
        params["meta"]["eval_episodes"] = int(args.episodes)

    if args.task is not None and args.task != params["task_specification"]["task"]:
        deviations.append({"field": "task_specification.task", "from": params["task_specification"]["task"], "to": args.task, "reason": "DEVIATION: task not shipped for this model in configs/evals; same planner settings"})
        params["task_specification"]["task"] = args.task

    apply_optional_visualization_policy(
        params,
        deviations,
        disabled=bool(args.disable_optional_visualization),
    )

    params["use_fsdp"] = False

    resolved_path = out / "resolved_config.yaml"
    resolved_text = yaml.dump(params, default_flow_style=False, sort_keys=False)
    resolved_path.write_text(resolved_text)
    norm_original = yaml.dump(original, default_flow_style=False, sort_keys=False)
    diff = "\n".join(
        difflib.unified_diff(norm_original.splitlines(), resolved_text.splitlines(), fromfile=str(cfg_path), tofile=str(resolved_path), lineterm="")
    )

    meta = {
        "env": args.env,
        "model": args.model,
        "seed": int(params["meta"]["seed"]),
        "label": args.label,
        "config": str(cfg_path.relative_to(repo)),
        "config_sha256": sha256_file(cfg_path),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "repo_commit": repo_provenance["commit"],
        "repo_status_sha256": repo_provenance["status_sha256"],
        "repo_diff_sha256": repo_provenance["diff_sha256"],
        "driver": str(driver_path),
        "driver_sha256": sha256_file(driver_path),
        "task_cfg": params["task_specification"]["task"],
        "episode_manifest": str(getattr(args, "_episode_manifest_path", "")) or None,
        "episode_manifest_sha256": (
            __import__("public_panel_manifest").rows_sha256(episode_rows) if episode_rows is not None else None
        ),
        "optional_visualization_disabled": bool(args.disable_optional_visualization),
    }
    prov = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "principle": "Label-first principle, amendment 3 (cross model design jepa.md)",
        "evaluator": "evals.simu_env_planning (unmodified), single-process rank 0 / world size 1 (evals/main.py --debug semantics)",
        "released_config": str(cfg_path),
        "released_config_sha256": meta["config_sha256"],
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "vendor_repository": repo_provenance,
        "driver": str(driver_path),
        "driver_sha256": meta["driver_sha256"],
        "config_deviations": deviations,
        "unified_diff": diff,
        "eval_episodes": int(params["meta"]["eval_episodes"]),
        "meta_seed": int(params["meta"]["seed"]),
        "argv": sys.argv,
        "optional_visualization_disabled": bool(args.disable_optional_visualization),
        "env_vars": {k: os.environ.get(k) for k in ("JEPAWM_DSET", "JEPAWM_LOGS", "JEPAWM_CKPT", "JEPAWM_OSSCKPT", "JEPAWM_HOME", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
    }
    if extra_deviations:
        prov["config_deviations"].extend(extra_deviations)
    (out / "config_provenance.json").write_text(json.dumps(prov, indent=2))
    return repo, params, meta, out, jsonl_path


def main() -> int:
    args = build_parser().parse_args()
    repo, params, meta, out, jsonl_path = resolve_config(args)

    # --- run exactly what evals/main.py --debug does (rank 0, world size 1) -------------------
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    from src.utils.distributed import init_distributed  # noqa: E402
    from evals.scaffold import main as eval_main  # noqa: E402
    import evals.simu_env_planning.planning.plan_evaluator as pe  # noqa: E402

    episode_rows = getattr(args, "_episode_rows", None)
    install_realization_hashing(pe)
    counter = install_logging_hooks(pe, jsonl_path, meta, episode_rows=episode_rows)
    if episode_rows is not None:
        install_manifest_seeding(pe, episode_rows)
    world_size, rank = init_distributed(rank_and_world_size=(0, 1))
    logging.getLogger("panelP").info(f"init_distributed -> rank {rank}/{world_size}")
    t0 = time.time()
    eval_main(params["eval_name"], args_eval=params)
    elapsed = time.time() - t0

    expected = int(params["meta"]["eval_episodes"])
    done = {
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_logged": counter["n"],
        "n_success": counter["n_success"],
        "expected_episodes": expected,
        "wall_s": round(elapsed, 1),
        "complete": counter["n"] == expected,
    }
    (out / ("DONE.json" if done["complete"] else "INCOMPLETE.json")).write_text(json.dumps(done, indent=2))
    print(json.dumps(done))
    return 0 if done["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
