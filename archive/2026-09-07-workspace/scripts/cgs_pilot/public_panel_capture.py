#!/usr/bin/env python3
"""Panel P stage 2 — activation capture during the RELEASED planning evaluator.

Governed by "Label-first principle, amendment 3" (`cross model design jepa.md`): the substrate is the
released ``facebook/jepa-wms`` checkpoints run under the repository's own planning evaluator
(``configs/evals/simu_env_planning/<env>/<model>/*.yaml``, ``succ_def: simu``), executed unmodified.
This script adds a READ-ONLY activation capture on top of the same run; it never changes the planner,
the goal sampler, the horizon, or the success definition, and the per-episode success flags it writes
must be bit-identical to the screen run with capture disabled (``--verify-identity``).

What is captured
----------------
``GC_Agent.plan(z, steps_left)`` is wrapped. By default, during the FIRST plan call of each episode,
read-only hooks retain a bounded sample of the CEM candidate states. The optional ``all_plans``
scope retains a separately tagged candidate population at every physical replan; downstream gates
must still truncate it to the frozen pre-outcome window. This second mode exists to compare the
fitted selected-action population with the exact population an all-replan online hook would edit.
After every unmodified plan call returns its action sequence,
we also run ONE extra ``model.unroll(z, actions)`` forward under ``torch.no_grad()`` and store, per site:

  * ``mean``      mean over all predicted tokens of the last predicted frame        [D]
  * ``mean_all``  mean over every token of every predicted step                     [D]

Sites follow the project's convention: ``L<bb>.resid_post`` (block output), ``L<bb>.attn_out``
(``block.attn``), ``L<bb>.mlp_out`` (``block.mlp``).  The extra forward is identical in shape to the
planner's own evaluation of the chosen chunk, so it adds ~1/num_samples to the episode cost.

Output (one directory per (env, model, seed) run)
-------------------------------------------------
``<out>/activations/ep<NNN>.npz``   arrays ``<site>__planner`` from the first plan call; in
                                    ``all_plans`` mode, tagged arrays
                                    ``<site>__planner_call<NNN>`` for every physical replan; arrays
                                    ``<site>__mean{,_all}`` of shape [n_replans, D]; selected action
                                    plans; and ``replan_index`` [n_replans]
``<out>/activations/index.json``    per episode: ep, ep_seed, success flag, n_replans, sites, dims,
                                    pre-outcome window (see below), checkpoint/config sha256
``<out>/episodes.jsonl``            written by the shared logging hooks (identical schema to the screen)

Pre-outcome truncation (design doc Step 1, "outcome-conditioned geometry may be downstream") applies
to the diagnostic ``mean``/``mean_all`` replan pools. The index records ``n_replans`` per episode and the frozen common window
``[0, min_over_episodes(n_replans) - PRE_OUTCOME_MARGIN)``; fitting code must use only replans inside
that window so that episode length cannot enter an outcome feature.

Usage
-----
    python public_panel_capture.py --repo ... --env pt --model jepa-wm --checkpoint ... \
        --out artifacts/public_panel/fit/pt_jepa-wm/seed1 --seed 1 --episodes 30 [--verify-identity]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PRE_OUTCOME_MARGIN = 3  # replans dropped from the end of the common window


def site_specs(predictor):
    """Yield ``(site_name, module, hook_kind)`` using the same sites as capture.

    ``public_panel_steer`` imports this function so capture and intervention cannot
    silently disagree about layer naming.
    """
    blocks = getattr(predictor, "predictor_blocks", None)
    ff_residual = False
    if blocks is None:
        blocks = getattr(predictor, "blocks", None)
        ff_residual = True
    if blocks is None:
        # Released DINO-WM uses app.plan_common.models.vit.ViTPredictor.  Its
        # transformer stores each residual layer as ModuleList([Attention,
        # FeedForward]) rather than exposing .blocks.  The FeedForward module
        # receives the post-attention residual and returns only the MLP branch,
        # so ff(x) + x is the exact residual-post tensor.
        transformer = getattr(predictor, "transformer", None)
        layers = getattr(transformer, "layers", None)
        if layers is not None:
            for b, layer in enumerate(layers):
                if len(layer) != 2:
                    raise RuntimeError(f"DINO-WM transformer layer {b} is not [attention, feed-forward]")
                attention, feed_forward = layer
                yield f"L{b:02d}.resid_post", feed_forward, "ff_resid"
                yield f"L{b:02d}.attn_out", attention, "tensor"
                yield f"L{b:02d}.mlp_out", feed_forward, "tensor"
            return
    if blocks is None:
        raise RuntimeError(
            "predictor has neither .predictor_blocks, .blocks, nor DINO-WM .transformer.layers"
        )
    for b, block in enumerate(blocks):
        yield f"L{b:02d}.resid_post", block, "ff_resid" if ff_residual else "tensor"
        if hasattr(block, "attn"):
            yield f"L{b:02d}.attn_out", block.attn, "tensor"
        if hasattr(block, "mlp"):
            yield f"L{b:02d}.mlp_out", block.mlp, "tensor"


def dino_attention_specs(predictor):
    """Yield released DINO-WM QKV projections with their head metadata."""
    transformer = getattr(predictor, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        return
    for block, layer in enumerate(layers):
        if len(layer) != 2:
            continue
        attention = layer[0]
        projection = getattr(attention, "to_qkv", None)
        if projection is not None and hasattr(attention, "heads") and hasattr(attention, "dim_head"):
            yield f"L{block:02d}", projection, int(attention.heads), int(attention.dim_head), attention


def infer_n_spatial(predictor, fallback: int = 256) -> int:
    """Infer DINO-WM patches/frame from its registered block-causal mask."""
    try:
        first = next(iter(dino_attention_specs(predictor)))[1]
        del first  # projection itself does not own the mask
        attention = predictor.transformer.layers[0][0]
        bias = attention.bias
        # The first query attends to every patch in frame zero and no later frame.
        value = int(bias[0, 0, 0].sum().item())
        return value if value > 0 else int(fallback)
    except (AttributeError, IndexError, StopIteration, TypeError, ValueError):
        return int(fallback)


# --------------------------------------------------------------------------------------- hooks
class BlockCapture:
    """Forward hooks on the predictor's blocks; one row per unroll() call."""

    def __init__(self, predictor, sites=("resid_post", "attn_out", "mlp_out")):
        import torch

        self.torch = torch
        self.sites = sites
        self.buf: dict[str, list] = {}
        self.plan_buf: dict[tuple[str, int], list] = {}
        self.probe = None            # list -> record tensor shapes
        self.phase = "planner"       # "planner" while the CEM runs, "extra" during our own unroll
        self.collect_planner = False
        self.plan_call_index = 0
        self.per_call_rows = 8        # rows kept per planner forward (CEM candidates, strided)
        self.max_rows_per_site = 256  # cap per episode per site
        gh, gw = getattr(predictor, "grid_height", None), getattr(predictor, "grid_width", None)
        self.n_spatial = int(gh * gw) if gh and gw else infer_n_spatial(predictor)
        self._cur: dict[str, object] = {}
        self._attention_cur: dict[str, dict[str, object]] = {}
        self._handles = []
        specs = list(site_specs(predictor))
        self.n_blocks = len({name.split(".")[0] for name, _module, _kind in specs})
        for name, module, kind in specs:
            if name.split(".")[1] in sites:
                self._handles.append(module.register_forward_hook(self._mk(name, kind)))
        for layer, projection, heads, dim_head, attention in dino_attention_specs(predictor):
            self._handles.append(
                projection.register_forward_hook(self._mk_qkv(layer, heads, dim_head, attention))
            )

    def _mk(self, site, kind="tensor"):
        def hook(_m, args, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not hasattr(t, "detach"):
                return
            if kind == "ff_resid":
                source = args[0]
                if not hasattr(source, "detach") or source.shape != t.shape:
                    raise RuntimeError(f"{site} cannot reconstruct ff(x)+x residual")
                t = t + source
            if self.probe is not None and len(self.probe) < 40:
                self.probe.append({"site": site, "shape": list(t.shape), "during": self.phase})
            if self.phase == "planner" and self.collect_planner:
                # Mirror SonarSteerer._edit exactly: preserve the top-level
                # candidate batch and pool only its final spatial-token block.
                # Flattening leading axes here would silently construct a
                # different population from the one the runtime hook edits.
                x = t.detach().float()
                if x.dim() >= 3:
                    flat = x.reshape(x.shape[0], -1, x.shape[-1])
                    n_spatial = min(self.n_spatial, int(flat.shape[1]))
                    r = flat[:, -n_spatial:].mean(dim=1)
                elif x.dim() == 2:
                    flat = x.unsqueeze(0)
                    n_spatial = min(self.n_spatial, int(flat.shape[1]))
                    r = flat[:, -n_spatial:].mean(dim=1)
                else:
                    r = None
                if r is not None:
                    if len(r) > self.per_call_rows:    # strided, no reordering
                        r = r[:: max(1, len(r) // self.per_call_rows)][: self.per_call_rows]
                    key = (site, int(self.plan_call_index))
                    have = sum(len(c) for c in self.plan_buf.get(key, []))
                    if have < self.max_rows_per_site:
                        remaining = self.max_rows_per_site - have
                        self.plan_buf.setdefault(key, []).append(r[:remaining].cpu().numpy())
            self._cur[site] = t.detach()
        return hook

    def _mk_qkv(self, layer, heads, dim_head, attention):
        """Capture bounded Q/K and attention-probability summaries, not full maps."""
        def hook(_module, _args, out):
            if self.phase != "extra" or not hasattr(out, "detach") or out.dim() != 3:
                return
            q, k, _v = out.detach().float().chunk(3, dim=-1)
            batch, tokens, width = q.shape
            if width != heads * dim_head:
                raise RuntimeError(f"{layer} QKV width {width} != {heads}*{dim_head}")
            q = q.reshape(batch, tokens, heads, dim_head).permute(0, 2, 1, 3)
            k = k.reshape(batch, tokens, heads, dim_head).permute(0, 2, 1, 3)
            n_spatial = min(self.n_spatial, tokens)
            q_tail, k_tail = q[:, :, -n_spatial:], k[:, :, -n_spatial:]
            scores = self.torch.matmul(q_tail, k.transpose(-1, -2)) / (dim_head ** 0.5)
            if hasattr(attention, "_get_sdpa_mask"):
                mask = attention._get_sdpa_mask(tokens, out.device)[-n_spatial:, :tokens]
                scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
            elif hasattr(attention, "bias"):
                mask = attention.bias[:, :, -n_spatial:, :tokens] != 0
                scores = scores.masked_fill(~mask, float("-inf"))
            probability = self.torch.softmax(scores, dim=-1)
            entropy = -(probability * self.torch.log(probability + 1e-30)).sum(-1).mean(-1)
            n_frames = tokens // n_spatial if tokens % n_spatial == 0 else 1
            if n_frames > 1:
                key_mass = probability.reshape(batch, heads, n_spatial, n_frames, n_spatial).sum(-1).mean(2)
            else:
                key_mass = probability.mean(2)
            self._attention_cur[layer] = {
                "query_mean": q_tail.mean(2),
                "query_var": q_tail.var(2, unbiased=False),
                "key_mean": k_tail.mean(2),
                "key_var": k_tail.var(2, unbiased=False),
                "attention_entropy": entropy,
                "attention_key_frame_mass": key_mass,
            }
        return hook

    def start_call(self):
        self._cur = {}
        self._attention_cur = {}

    def end_call(self, n_frames: int):
        """Reduce the tensors captured during one unroll() into two pooled vectors per site."""
        for site, t in self._cur.items():
            x = t.float()
            if x.dim() > 3:
                # Same top-level-batch convention used by the online hook.
                x = x.reshape(x.shape[0], -1, x.shape[-1])
            if x.dim() == 2:                        # [N, D]
                x = x.unsqueeze(0)
            mean_all = x.mean(dim=(0, 1)).cpu().numpy()
            # The active predictor concatenates predicted frames on the token axis:
            # [batch, frames*n_spatial, dim].  The old ``x[-1]`` indexed the batch
            # axis and made mean/mean_all identical for batch size one.  Select the
            # final spatial-token block instead; this is architecture-checked by the
            # divisibility metadata recorded below.
            n_spatial = min(self.n_spatial, int(x.shape[1]))
            last = x[:, -n_spatial:].mean(dim=(0, 1)).cpu().numpy()
            self.buf.setdefault(f"{site}__mean", []).append(last)
            self.buf.setdefault(f"{site}__mean_all", []).append(mean_all)
        for layer, values in self._attention_cur.items():
            for name, tensor in values.items():
                # Extra chosen-action unroll normally has batch one.  Mean over
                # batch keeps exactly one bounded diagnostic row per replan.
                self.buf.setdefault(f"{layer}.{name}__attention", []).append(
                    tensor.mean(0).cpu().numpy()
                )
        self._cur = {}
        self._attention_cur = {}

    def pop(self):
        out = {k: np.stack(v).astype(np.float32) for k, v in self.buf.items() if v}
        self.buf = {}
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------------------- patch
def install_capture(
    out_dir: Path,
    meta: dict,
    verify_identity: bool,
    planner_capture_scope: str = "first_plan",
):
    """Wrap GC_Agent.plan; returns a state dict the caller finalises."""
    import torch
    from evals.simu_env_planning.planning.gc_agent import GC_Agent

    if planner_capture_scope not in {"first_plan", "all_plans"}:
        raise ValueError(f"unknown planner capture scope: {planner_capture_scope}")
    state = {
        "cap": None,
        "probe": [],
        "collect_planner": True,     # rows the CEM itself computed (candidates x imagined steps)
        "planner_subsample": 256,
        "per_call_rows": 8,
        "max_rows_per_site": 256,
        "plan_call": 0,
        "planner_capture_scope": planner_capture_scope,
        "episodes": [],      # index rows
        "cur": [],           # per-replan rows for the current episode
        "chosen_actions": [],
        "steps_left": [],
        "ep": 0,
        "errors": [],
        "extra_forward_s": 0.0,
    }
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    orig_plan = GC_Agent.plan

    def plan(self, z, steps_left=None):
        # Install BEFORE the planner runs. Some released configs replan during the rollout, so the
        # primary planner pool is deliberately restricted to call 0 (strictly before any action).
        if state["cap"] is None and not verify_identity:
            try:
                pred = getattr(self.model, "predictor", None)
                if pred is None:
                    pred = getattr(getattr(self.model, "model", None), "predictor", None)
                cap = BlockCapture(pred)
                cap.probe = state["probe"]
                cap.per_call_rows = int(state.get("per_call_rows", 8))
                cap.max_rows_per_site = int(state.get("max_rows_per_site", 256))
                state["cap"] = cap
            except Exception as exc:
                state["errors"].append(f"install: {type(exc).__name__}: {exc}")
        if state["cap"] is not None:
            state["cap"].phase = "planner"
            state["cap"].plan_call_index = int(state["plan_call"])
            state["cap"].collect_planner = bool(
                state["collect_planner"]
                and (
                    state["planner_capture_scope"] == "all_plans"
                    or state["plan_call"] == 0
                )
            )
        actions = orig_plan(self, z, steps_left=steps_left)
        try:
            chosen = actions.detach().float().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
            if chosen.ndim == 1:
                chosen = chosen[None, :]
            state["chosen_actions"].append(np.asarray(chosen, dtype=np.float32))
            state["steps_left"].append(-1 if steps_left is None else int(steps_left))
        except Exception as exc:
            if len(state["errors"]) < 5:
                state["errors"].append(f"chosen-actions: {type(exc).__name__}: {exc}")
        state["plan_call"] += 1
        if verify_identity:
            return actions
        try:
            if state["cap"] is None:
                raise RuntimeError("capture not installed")
            a = actions
            if hasattr(a, "dim"):
                if a.dim() == 2:      # [T, A] -> [T, 1, A]
                    a = a.unsqueeze(1)
                elif a.dim() == 1:    # [A]    -> [1, 1, A]
                    a = a.view(1, 1, -1)
            t0 = time.time()
            with torch.no_grad():
                state["cap"].phase = "extra"
                state["cap"].collect_planner = False
                state["cap"].start_call()
                self.model.unroll(z, a)
                state["cap"].end_call(int(a.shape[0]))
                state["cap"].phase = "planner"
            state["extra_forward_s"] += time.time() - t0
        except Exception as exc:  # capture must never change the run's outcome
            if len(state["errors"]) < 5:
                state["errors"].append(f"{type(exc).__name__}: {exc}")
        return actions

    GC_Agent.plan = plan
    state["_orig_plan"] = orig_plan
    state["_act_dir"] = act_dir
    state["_meta"] = meta
    return state


def flush_episode(state, ep_row: dict):
    """Called after each episode's logging row is emitted."""
    cap = state["cap"]
    if cap is None:
        return
    arrays = cap.pop()
    chosen_actions = state.pop("chosen_actions", [])
    steps_left = state.pop("steps_left", [])
    state["chosen_actions"] = []
    state["steps_left"] = []
    if chosen_actions:
        max_t = max(action.shape[0] for action in chosen_actions)
        action_dim = max(action.shape[-1] for action in chosen_actions)
        padded = np.full((len(chosen_actions), max_t, action_dim), np.nan, dtype=np.float32)
        lengths = np.zeros(len(chosen_actions), dtype=np.int32)
        for i, action in enumerate(chosen_actions):
            padded[i, : action.shape[0], : action.shape[-1]] = action
            lengths[i] = action.shape[0]
        arrays["chosen_actions"] = padded
        arrays["chosen_action_lengths"] = lengths
        arrays["steps_left"] = np.asarray(steps_left, dtype=np.int32)
    plan_rows = {}
    planner_counts = {}
    for (site, plan_call), chunks in cap.plan_buf.items():
        if not chunks:
            continue
        M = np.concatenate(chunks, axis=0)
        k = int(state.get("planner_subsample", 64))
        if len(M) > k:                                     # frozen stride subsample, no reordering
            M = M[:: max(1, len(M) // k)][:k]
        tagged_key = f"{site}__planner_call{plan_call:03d}"
        plan_rows[tagged_key] = M.astype(np.float32)
        planner_counts[str(plan_call)] = max(planner_counts.get(str(plan_call), 0), int(len(M)))
        if plan_call == 0:
            # Compatibility alias for existing first-plan-only readers.
            plan_rows[f"{site}__planner"] = M.astype(np.float32)
    cap.plan_buf = {}
    arrays.update(plan_rows)
    if not arrays:
        return
    pooled = {k: v for k, v in arrays.items() if "__planner" not in k}
    n = min([v.shape[0] for v in pooled.values()] or [0])
    arrays = {k: (v[:n] if "__planner" not in k else v) for k, v in arrays.items()}
    arrays["replan_index"] = np.arange(n, dtype=np.int32)
    ep = int(ep_row.get("ep", state["ep"]))
    path = Path(state["_act_dir"]) / f"ep{ep:03d}.npz"
    np.savez_compressed(path, **arrays)
    state["episodes"].append(
        {
            "ep": ep,
            "ep_seed": ep_row.get("ep_seed"),
            "success": ep_row.get("success"),
            "n_replans": int(n),
            "n_plan_calls": ep_row.get("n_plan_calls"),
            "n_planner_rows": int(next((v.shape[0] for k, v in arrays.items() if k.endswith("__planner")), 0)),
            "n_planner_rows_by_call": planner_counts,
            "n_chosen_action_plans": int(len(chosen_actions)),
            "planner_capture_scope": (
                "all_plan_calls_tagged_by_physical_replan"
                if state["planner_capture_scope"] == "all_plans"
                else "first_plan_call"
            ),
            "file": path.name,
            "sites": sorted({k.split("__")[0] for k in arrays if "__" in k}),
            "dim": int(next(v for k, v in arrays.items() if "__" in k).shape[-1])
            if any("__" in k for k in arrays) else None,
        }
    )
    state["ep"] += 1


def finalize(state, out_dir: Path, meta: dict):
    eps = state["episodes"]
    window = None
    if eps:
        m = min(e["n_replans"] for e in eps)
        window = [0, max(0, m - PRE_OUTCOME_MARGIN)]
    idx = {
        "meta": meta,
        "probe_shapes": state.get("probe", [])[:40],
        "pre_outcome_window": window,
        "pre_outcome_margin": PRE_OUTCOME_MARGIN,
        "n_episodes": len(eps),
        "n_success": int(sum(1 for e in eps if e.get("success") == 1)),
        "extra_forward_s": round(state["extra_forward_s"], 2),
        "capture_errors": state["errors"],
        "planner_capture_scope": (
            "all_plan_calls_tagged_by_physical_replan"
            if state["planner_capture_scope"] == "all_plans"
            else "first_plan_call_strictly_pre_action"
        ),
        "planner_pooling": (
            "top-level candidate batch; final n_spatial token block, "
            "identical to SonarSteerer._edit"
        ),
        "physical_pooling": {
            "mean": "last n_spatial tokens of the final chosen-action unroll call",
            "mean_all": "all tokens of the final chosen-action unroll call",
            "n_spatial": int(state["cap"].n_spatial) if state.get("cap") is not None else None,
        },
        "episodes": eps,
    }
    (out_dir / "activations" / "index.json").write_text(json.dumps(idx, indent=2))
    return idx


# --------------------------------------------------------------------------------------- main
def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import public_panel_eval as ppe

    ap = ppe.build_parser(description=__doc__)
    ap.add_argument("--verify-identity", action="store_true",
                    help="run with the wrapper installed but capture disabled (success flags must match the screen)")
    ap.add_argument(
        "--planner-capture-scope",
        choices=["first_plan", "all_plans"],
        default="first_plan",
        help="capture first physical plan call or separately tag every physical replan",
    )
    args = ap.parse_args()

    repo, params, meta, out, jsonl = ppe.resolve_config(
        args,
        extra_deviations=[{
            "field": "runtime",
            "from": "evaluator only",
            "to": "evaluator + read-only activation capture (one extra no_grad unroll per replan)",
            "reason": "Panel P stage 2; planner/goal/horizon/success untouched",
        }],
    )
    meta["repo"] = str(repo)
    meta["task"] = str(params["task_specification"]["task"])

    os.environ.setdefault("PYTHONHASHSEED", "0")
    import logging
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logging.basicConfig(); logging.getLogger().setLevel(logging.INFO)
    from src.utils.distributed import init_distributed
    from evals.scaffold import main as eval_main
    import evals.simu_env_planning.planning.plan_evaluator as pe

    episode_rows = getattr(args, "_episode_rows", None)
    ppe.install_realization_hashing(pe)
    ppe.install_logging_hooks(pe, jsonl, meta, episode_rows=episode_rows)  # one row per episode
    if episode_rows is not None:
        ppe.install_manifest_seeding(pe, episode_rows)
    cap_state = install_capture(out, meta, args.verify_identity, args.planner_capture_scope)

    # wrap the (already wrapped) evaluator once more so the capture buffer is flushed at the exact
    # episode boundary, using the row the logging hook just appended.
    _inner_eval = pe.PlanEvaluator.eval

    def eval_with_flush(self, cfg, agent, env, task_idx=-1, ep=0):
        cap_state["plan_call"] = 0
        res = _inner_eval(self, cfg, agent, env, task_idx=task_idx, ep=ep)
        row = {}
        try:
            with open(jsonl) as f:
                last = f.readlines()[-1]
            row = json.loads(last)
        except Exception:
            row = {"ep": ep}
        flush_episode(cap_state, row)
        return res

    pe.PlanEvaluator.eval = eval_with_flush

    init_distributed(rank_and_world_size=(0, 1))   # what evals/main.py --debug does (rank 0, world 1)
    t0 = time.time()
    eval_main(params["eval_name"], args_eval=params)
    wall = time.time() - t0

    rows = [json.loads(l) for l in open(jsonl)] if jsonl.exists() else []
    idx = finalize(cap_state, out, meta)
    done = {
        "finished_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "n_logged": len(rows),
        "n_success": int(sum(1 for r in rows if r.get("success") == 1)),
        "n_captured": idx["n_episodes"],
        "pre_outcome_window": idx["pre_outcome_window"],
        "capture_errors": idx["capture_errors"],
        "wall_s": round(wall, 1),
        "verify_identity": bool(args.verify_identity),
    }
    (out / "DONE.json").write_text(json.dumps(done))
    print(json.dumps(done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
