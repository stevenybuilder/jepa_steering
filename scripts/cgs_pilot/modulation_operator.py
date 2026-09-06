#!/usr/bin/env python3
"""Operator 2 -- AdaLN modulation operator: a donor-free DYNAMICAL intervention on the action-entry path.

    m' = m [ I + g(a, h_hazard) U diag(gamma) V^T ]          (applied to the modulation output of one block / component)

Design-doc references: ``cross model design jepa.md`` Step 4 ("modulation operator: h' = h[I + g(a, h_hazard) U diag(gamma) V^T]
where g is read from the AdaLN (shift, scale, gate) vectors -- the only place the action enters -- so that 'the action gates
which hazard evidence is written' is a testable operator rather than a metaphor"), Step 7 (AdaLN shift/scale/gate vectors as
loci; input-selection modes) and Step 8 row 7; ``representational_geometry_paper_concepts.md`` section 11 (Mante et al.: the
operative representation is the dynamical law h_dot = F(h, x, c); a context-dependent SELECTION VECTOR decides which input
component pushes the state along the slow mode; left/right eigenvectors separate input selection from state evolution),
section 12.5 (dynamical intervention changes F(h, x, c), not a single state) and 12.6 ("a small direction can be
mechanistically important if it strongly changes the subsequent trajectory"), section 6 (matched-spectrum random control),
section 14 ("decodability is not use").

WHERE THE OPERATOR ACTS AND WHY IT IS NOT A RESIDUAL-STREAM CONCEPTOR
----------------------------------------------------------------------
In the JEPA-WM AdaLN predictor the action exists only as the per-block modulation vector
m_b = adaLN_modulation(action_encoder(a)) in R^{6D} = (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp),
broadcast to every token: x <- x + gate_msa * attn(norm1(x)(1 + scale_msa) + shift_msa), then the MLP branch likewise.
A residual-stream conceptor (``steered_planner_ranking`` modes, ``conceptor_patch.py``) rewrites the STATE h of some tokens
at one site: h' = mean + (h - mean) M.  The modulation operator instead rewrites the UPDATE RULE: it changes the
coefficients (shift/scale/gate) with which the block writes hazard evidence into every token, conditionally on (i) the
action, read from m itself, and (ii) the hazard-token content h_hazard read at the block input.  It touches no token state
directly, it is identical for all tokens of the frame (the modulation is broadcast), and it is zero for the brake chunk by
construction (the read (m - m_ref) . u is ~0 when m is the brake modulation).  It is therefore the operator form of Mante's
selection vector: which input (modulation) component is allowed to push the state along the relational mode, gated by
context.

CONSTRUCTION (fitted on discovery scenes only)
----------------------------------------------
- reference m_ref = mean brake modulation (last frame) of the block;  action contrast Delta_c = mean_i (m_b(a1) - m_b(a0))
  restricted to each D-slice c in {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp} of the chosen component
  set (attn -> 3 slices, mlp -> 3, all -> 6);  candidate read/write directions u_c = Delta_c / ||Delta_c||  (U = V: the
  operator amplifies or attenuates the throttle-vs-brake contrast of slice c, the read (m - m_ref) . u_c being the amount
  of throttle in that slice = "g read from the AdaLN vectors themselves").
- selection score (the modulation-path DiD): with the existing forward-AD JVP (``modulation_patch.modulation_jvp``) the
  route-token response to a unit tangent along u_c, evaluated on the solid in-lane context minus the sidewalk (H0) context
  (minus the H0' mirror-pose DiD when present), averaged over the a0 and a1 linearisation points, projected on the TRUE
  consequence I_true = z_fut(solid,a1) - z_fut(solid,a0) - z_fut(0,a1) + z_fut(0,a0) on route tokens (recovery units as
  ``modulation_patch``):  s_{c,i}.  Unit = scene: mean, scene bootstrap CI, sign-flip p, max-T over candidates.
- rank r in {1, 2, 4} (frozen grid): the r slices with the largest |mean s_c| are kept, gamma_c = mean s_c / max_c |mean s_c|
  in [-1, 1] (sign = whether MORE throttle in that slice drives the consequence), so that beta = 1 at full gate writes an
  edit as large as the throttle-brake contrast of the top slice.
- context gate g_h(h_hazard) in [0, 1]: w = unit mean over fitting cells of (x_bar_hazard(solid) - x_bar_hazard(H0)) at the
  block input (last-frame hazard tokens averaged); g_h = clip((w . x_bar - c0) / (c1 - c0), 0, 1) with c0 / c1 the mean
  projections of hazard-free / solid cells.  g(a, h_hazard) = g_h(h_hazard) x [(m - m_ref) . u_c]: the product of the
  hazard content read at the block input and the throttle content read from the modulation.
- runtime:  m'_last = m_last + s_mode * beta * g_h * sum_c gamma_c ((m_last - m_ref) . u_c) u_c,  s = +1 strengthen,
  -1 suppress; only the LAST frame's modulation vector (the frame whose tokens become the prediction) is edited.

CONTROLS (same hook interface, same beta grid): ``sham`` (beta = 0: the hook returns its input, bit-identical);
``ungated`` (g_h frozen at its mean hazard-free value -- the literal twin; near-sham when that value is ~0, which is
reported) and ``ungated_on`` (g_h = 1: the informative twin, must break H0 / H0' preservation if the gate matters);
``random_matched`` (random orthonormal directions inside the same slices, same gamma); ``wrong_block`` (same operator
written at block b + offset); ``wrong_group`` (gate reads the complementary token group).

Files: ``L<bb>.adaln__<comp>__r<r>.npz`` (+ ``.json``) from the ``fit`` sub-command (needs the model; box only);
``steered_planner_ranking.py --modulation-operator-dir`` runs them as mode ``modulation`` at sites ``L<bb>.adaln``.
``--self-test`` builds a fake AdaLN predictor with a planted gate_mlp x hazard mechanism and checks recovery, gating,
controls and bit-identical sham.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgs_stats import cluster_bootstrap, json_safe, sign_flip_maxt  # noqa: E402
from protocol import CELL_ORDER, NULL_CONTROL_HAZARD, OBJECT_HAZARD, canonical_json  # noqa: E402
from stats_utils import finite, sign_flip_p  # noqa: E402

PROTOCOL = "cgs-modulation-operator-v0.1"
SLICES = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
COMPONENT_SLICES = {"attn": SLICES[:3], "mlp": SLICES[3:], "all": SLICES}
RANK_GRID = (1, 2, 4)
VARIANTS = ("main", "random_matched", "ungated", "ungated_on")
INTERPRETATION_SCOPE = (
    "Dynamical intervention on the AdaLN action-entry path of one block: the update rule (shift/scale/gate) is rescaled along the "
    "throttle-brake contrast of the slices whose forward-AD route response is most hazard-dependent, gated by hazard-token content "
    "read at the block input. Fitted on discovery scenes only; descriptive selection statistics; the causal claim comes from the "
    "steered-planner table and the closed-loop label (Label-first principle), never from the selection score."
)


def slice_of(name: str, D: int) -> slice:
    i = SLICES.index(name)
    return slice(i * D, (i + 1) * D)


# --------------------------------------------------------------------------- #
# Operator object
# --------------------------------------------------------------------------- #


class ModulationOperator:
    """Arrays: ``U`` [r, 6D] unit rows (slice-supported), ``U_random`` [r, 6D], ``gamma`` [r], ``m_ref`` [6D], ``gate_w`` [D],
    ``gate_c0``, ``gate_c1``, ``gate_hf`` (mean hazard-free gate), ints ``block``, ``D``; meta JSON."""

    def __init__(self, arrays: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        self.a = arrays
        self.meta = meta or {}
        self.block = int(arrays["block"])
        self.D = int(arrays["D"])
        self.rank = int(arrays["U"].shape[0])

    @staticmethod
    def load(path: Path) -> "ModulationOperator":
        with np.load(Path(path)) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return ModulationOperator(arrays, meta)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(finite(self.meta)), **{k: np.asarray(v) for k, v in self.a.items()})
        return path

    def gate_numpy(self, x_hazard_mean: np.ndarray) -> np.ndarray:
        proj = np.atleast_2d(x_hazard_mean) @ self.a["gate_w"]
        c0, c1 = float(self.a["gate_c0"]), float(self.a["gate_c1"])
        return np.clip((proj - c0) / max(c1 - c0, 1e-12), 0.0, 1.0)

    def apply_numpy(self, m: np.ndarray, g: np.ndarray | float, beta: float, mode: str, variant: str = "main") -> np.ndarray:
        """m [n, 6D], g [n] -> edited modulation (reference implementation of the runtime rule)."""
        m = np.atleast_2d(np.asarray(m, np.float64))
        if beta <= 0:
            return m.copy()
        U = self.a["U_random"] if variant == "random_matched" else self.a["U"]
        g = np.full(m.shape[0], float(self.a["gate_hf"])) if variant == "ungated" else (np.ones(m.shape[0]) if variant == "ungated_on" else np.broadcast_to(np.asarray(g, np.float64), (m.shape[0],)))
        s = 1.0 if mode == "strengthen" else -1.0
        read = (m - self.a["m_ref"]) @ U.T  # [n, r]
        return m + (s * beta) * g[:, None] * ((read * self.a["gamma"]) @ U)


# --------------------------------------------------------------------------- #
# Runtime steerer (same context-manager interface as steered_planner_ranking.PredictorSteerer)
# --------------------------------------------------------------------------- #


class ModulationSteerer:
    """``edits[block] = list of (step | None, gate_index, op, mode, beta, variant, log)``: at the listed imagined steps the
    block's LAST-frame modulation vector is rewritten by ``op`` with the gate read from the block-input hazard tokens
    ``gate_index`` (last frame).  ``applied`` / ``expected`` as in ``PredictorSteerer``."""

    def __init__(self, predictor, edits: dict[int, list[tuple]], n_spatial: int) -> None:
        self.predictor, self.edits, self.n_spatial = predictor, edits, n_spatial
        self.step = -1
        self.applied: list[tuple[str, int]] = []
        self.expected = sum(len(v) for v in edits.values())
        self._gate: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _todo(self, b: int):
        return [e for e in self.edits.get(b, []) if e[0] is None or e[0] == self.step]

    def _pre(self, b: int, x):
        import torch

        todo = self._todo(b)
        if not todo:
            return
        flat = x.reshape(x.shape[0], -1, x.shape[-1])
        last = flat[:, -self.n_spatial:]
        for i, (st, idx, op, mode, beta, variant, log) in enumerate(todo):
            if variant in ("ungated", "ungated_on") or len(idx) == 0:
                g = torch.full((x.shape[0],), float(op.a["gate_hf"]) if variant == "ungated" else 1.0, device=x.device)
            else:
                xm = last[:, torch.as_tensor(np.asarray(idx), dtype=torch.long, device=x.device)].float().mean(dim=1)  # [B, D]
                w = torch.as_tensor(op.a["gate_w"], dtype=torch.float32, device=x.device)
                c0, c1 = float(op.a["gate_c0"]), float(op.a["gate_c1"])
                g = ((xm @ w - c0) / max(c1 - c0, 1e-12)).clamp(0.0, 1.0)
            self._gate[(b, i)] = g

    def _post(self, b: int, out):
        import torch

        todo = self._todo(b)
        if not todo:
            return out
        res = out
        for i, (st, idx, op, mode, beta, variant, log) in enumerate(todo):
            if beta <= 0:  # sham: bit-identical
                self.applied.append((f"L{b:02d}.adaln", self.step))
                continue
            g = self._gate.get((b, i))
            if g is None:
                raise RuntimeError(f"modulation gate for block {b} not captured (block pre-hook did not fire)")
            U = torch.as_tensor(op.a["U_random"] if variant == "random_matched" else op.a["U"], dtype=torch.float32, device=out.device)
            gamma = torch.as_tensor(op.a["gamma"], dtype=torch.float32, device=out.device)
            m_ref = torch.as_tensor(op.a["m_ref"], dtype=torch.float32, device=out.device)
            s = 1.0 if mode == "strengthen" else -1.0
            m = res[:, -1, :].float()  # last frame [B, 6D]
            read = (m - m_ref) @ U.T
            delta = (s * beta) * g[:, None] * ((read * gamma) @ U)
            new = res.clone()
            new[:, -1, :] = (m + delta).to(out.dtype)
            log.append(float(delta.norm() / m.norm().clamp_min(1e-12)))
            res = new
            self.applied.append((f"L{b:02d}.adaln", self.step))
        return res

    def __enter__(self):
        def count(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count, with_kwargs=True))
        for b in self.edits:
            blk = self.predictor.predictor_blocks[b]
            self._handles.append(blk.register_forward_pre_hook(lambda m, args, kwargs, b=b: self._pre(b, args[0]), with_kwargs=True))
            self._handles.append(blk.adaLN_modulation.register_forward_hook(lambda m, a, out, b=b: self._post(b, out)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Fitting (model needed): modulation capture, block-input hazard means, JVP selection scores
# --------------------------------------------------------------------------- #


class BlockInputCapture:
    """Mean of the last-frame tokens ``index`` at block ``b``'s input, at imagined step ``step``."""

    def __init__(self, predictor, blocks: list[int], index: np.ndarray, n_spatial: int, step: int = 0) -> None:
        self.predictor, self.blocks, self.index, self.n_spatial, self.target = predictor, blocks, np.asarray(index, dtype=int), n_spatial, step
        self.step = -1
        self.captured: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _pre(self, b, x):
        import torch

        if self.step != self.target:
            return
        flat = x.reshape(x.shape[0], -1, x.shape[-1])[:, -self.n_spatial:]
        if len(self.index):
            self.captured[b] = flat[0, torch.as_tensor(self.index, dtype=torch.long, device=x.device)].detach().float().mean(dim=0).cpu().numpy()
        else:
            self.captured[b] = flat[0].detach().float().mean(dim=0).cpu().numpy()

    def __enter__(self):
        def count(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count, with_kwargs=True))
        for b in self.blocks:
            self._handles.append(self.predictor.predictor_blocks[b].register_forward_pre_hook(lambda m, args, kwargs, b=b: self._pre(b, args[0]), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def fit_scene(wm, predictor, latents: dict[tuple[int, int], dict[str, Any]], route_idx: np.ndarray, hazard_idx: np.ndarray, k: int, solid: int, blocks: list[int],
              comps: tuple[str, ...], n_spatial: int) -> dict[str, Any]:
    """Per scene: modulation vectors (last frame) per block for a0/a1, block-input hazard means per context level and action,
    and the per-slice JVP selection scores against I_true on route tokens.  Returns numpy-only records."""
    import torch

    from modulation_patch import ModulationHooks, _predict, modulation_jvp

    keys = list(latents)
    D_lat = latents[keys[0]]["z_context"].shape[-1]  # encoder latent width (futures / I_true)
    levels = [lv for lv in (0, solid, NULL_CONTROL_HAZARD, (OBJECT_HAZARD if solid == 1 else 1)) if (lv, 0) in latents]
    a = {0: latents[(0, 0)]["actions"], 1: latents[(0, 1)]["actions"]}
    z = {lv: latents[(lv, 0)]["z_context"] for lv in levels}
    fut = {kk: latents[kk]["z_future"][0, k].reshape(-1, D_lat).float() for kk in ((0, 0), (0, 1), (solid, 0), (solid, 1))}
    I_true = fut[(solid, 1)] - fut[(solid, 0)] - fut[(0, 1)] + fut[(0, 0)]
    Gi = torch.as_tensor(route_idx if len(route_idx) else np.arange(I_true.shape[0]), dtype=torch.long, device=I_true.device)
    It = I_true[Gi].flatten().double()
    It2 = float((It @ It).item())
    T_win = min(k + 1, int(getattr(wm, "ctxt_window", 2)))  # frames in the predictor window at imagined step k
    mods: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    xin: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for lv in levels:
        for ai in (0, 1):
            hooks = ModulationHooks(predictor, capture=True, target_step=k)
            cap = BlockInputCapture(predictor, blocks, hazard_idx, n_spatial, k)
            with torch.no_grad(), hooks, cap:
                _predict(wm, z[lv], a[ai], k)
            mods[(lv, ai)] = {b: hooks.captured[b][0, -1].detach().float().cpu().numpy() for b in blocks}
            xin[(lv, ai)] = dict(cap.captured)
    D = int(mods[(0, 0)][blocks[0]].shape[-1]) // 6  # PREDICTOR width: the modulation vector is 6D (not the encoder latent width)
    scores: dict[int, dict[str, dict[str, float]]] = {}
    for b in blocks:
        dm = 0.5 * ((mods[(0, 1)][b] - mods[(0, 0)][b]) + (mods[(solid, 1)][b] - mods[(solid, 0)][b]))  # action contrast (identical across levels by construction)
        scores[b] = {}
        for sl in SLICES:
            v = np.zeros_like(dm)
            v[slice_of(sl, D)] = dm[slice_of(sl, D)]
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                scores[b][sl] = {"solid_minus_h0": 0.0, "null_minus_h0": 0.0, "norm_delta": 0.0}
                continue
            v /= nv
            resp = {}
            for lv in levels:
                vals = []
                for ai in (0, 1):
                    tangent = torch.zeros(1, T_win, 6 * D, device=z[lv].device, dtype=torch.float32)  # [B, T, 6D]: only the last frame is perturbed
                    tangent[0, -1] = torch.as_tensor(v, dtype=torch.float32, device=z[lv].device)
                    jv = modulation_jvp(wm, predictor, z[lv], a[ai], k, b, "all", tangent)
                    vals.append(float((jv[Gi].flatten().double() @ It).item() / max(It2, 1e-12)))
                resp[lv] = float(np.mean(vals))
            entry = {"solid_minus_h0": resp[solid] - resp[0], "norm_delta": nv, "resp_by_level": {str(lv): r for lv, r in resp.items()}}
            entry["null_minus_h0"] = (resp[NULL_CONTROL_HAZARD] - resp[0]) if NULL_CONTROL_HAZARD in resp else 0.0
            entry["score"] = entry["solid_minus_h0"] - entry["null_minus_h0"]
            scores[b][sl] = entry
    return {"mods": mods, "xin": xin, "scores": scores, "levels": levels, "D": D, "D_latent": int(D_lat), "I_true_route_norm": float(np.sqrt(It2))}


def assemble_operators(records: dict[str, dict[str, Any]], blocks: list[int], comps: tuple[str, ...], ranks: tuple[int, ...], solid: int, n_boot: int, n_perm: int, seed: int,
                       gate_group: str = "hazard") -> tuple[dict[str, ModulationOperator], dict[str, Any]]:
    """Scene records -> per (block, comp, rank) operators + the selection statistics (unit = scene)."""
    rng = np.random.default_rng(seed)
    pids = sorted(records)
    D = int(records[pids[0]]["D"])
    NULL = NULL_CONTROL_HAZARD
    ghost = OBJECT_HAZARD if solid == 1 else 1
    ops: dict[str, ModulationOperator] = {}
    stats: dict[str, Any] = {}
    for b in blocks:
        m_ref = np.mean([records[p]["mods"][(0, 0)][b] for p in pids], axis=0)
        dm_mean = np.mean([0.5 * ((records[p]["mods"][(0, 1)][b] - records[p]["mods"][(0, 0)][b]) + (records[p]["mods"][(solid, 1)][b] - records[p]["mods"][(solid, 0)][b])) for p in pids], axis=0)
        # gate: block-input hazard-token means
        def xbar(lv):
            return np.asarray([records[p]["xin"][(lv, ai)][b] for p in pids for ai in (0, 1) if (lv, ai) in records[p]["xin"]])
        x_solid, x_h0 = xbar(solid), xbar(0)
        x_null, x_ghost = xbar(NULL), xbar(ghost)
        w = (x_solid.mean(0) - x_h0.mean(0))
        w = w / max(np.linalg.norm(w), 1e-12)
        hf = np.concatenate([x_h0] + ([x_null] if len(x_null) else []))
        c0, c1 = float(hf @ w if hf.ndim == 1 else (hf @ w).mean()), float((x_solid @ w).mean())
        gate = lambda X: np.clip(((X @ w) - c0) / max(c1 - c0, 1e-12), 0.0, 1.0) if len(X) else np.zeros(0)  # noqa: E731
        gate_stats = {"solid": float(gate(x_solid).mean()), "h0": float(gate(x_h0).mean()), "h0prime": (float(gate(x_null).mean()) if len(x_null) else None), "ghost": (float(gate(x_ghost).mean()) if len(x_ghost) else None),
                      "separation_d": float((c1 - c0) / max(np.sqrt(0.5 * ((x_solid @ w).var() + (hf @ w).var())), 1e-12)), "gate_hf_mean": float(gate(hf).mean())}
        # selection statistics per slice (unit = scene)
        per_slice = {sl: np.asarray([records[p]["scores"][b][sl].get("score", 0.0) for p in pids]) for sl in SLICES}
        sel: dict[str, Any] = {}
        for sl, vals in per_slice.items():
            sel[sl] = {**cluster_bootstrap(vals, n_boot, rng), "sign_flip_p": sign_flip_p(vals, seed=seed),
                       "solid_minus_h0_mean": float(np.mean([records[p]["scores"][b][sl].get("solid_minus_h0", 0.0) for p in pids])),
                       "null_minus_h0_mean": float(np.mean([records[p]["scores"][b][sl].get("null_minus_h0", 0.0) for p in pids])),
                       "norm_delta_mean": float(np.mean([records[p]["scores"][b][sl].get("norm_delta", 0.0) for p in pids]))}
        maxt = sign_flip_maxt({sl: v.reshape(-1, 1) for sl, v in per_slice.items()}, n_perm, rng, statistic=lambda v, s: v[:, 0] if s is None else v[:, 0] * s)
        for sl in SLICES:
            sel[sl]["p_maxt_fwer"] = maxt["sites"][sl]["p_maxt_fwer"]
        stats[f"L{b:02d}.adaln"] = {"block": b, "gate": gate_stats, "selection": sel, "m_ref_norm": float(np.linalg.norm(m_ref)), "delta_norm_by_slice": {sl: float(np.linalg.norm(dm_mean[slice_of(sl, D)])) for sl in SLICES}}
        for comp in comps:
            cand = list(COMPONENT_SLICES[comp])
            means = np.asarray([sel[sl]["point"] for sl in cand])
            order = np.argsort(-np.abs(means))
            for r in ranks:
                rr = min(r, len(cand))
                keep = [cand[i] for i in order[:rr]]
                U = np.zeros((rr, 6 * D))
                for j, sl in enumerate(keep):
                    v = np.zeros(6 * D)
                    v[slice_of(sl, D)] = dm_mean[slice_of(sl, D)]
                    U[j] = v / max(np.linalg.norm(v), 1e-12)
                gam = np.asarray([sel[sl]["point"] for sl in keep])
                gam = gam / max(np.max(np.abs(gam)), 1e-12)
                Ur = np.zeros_like(U)
                for j, sl in enumerate(keep):  # random direction inside the same slice (matched support and gamma)
                    v = np.zeros(6 * D)
                    v[slice_of(sl, D)] = rng.normal(size=D)
                    Ur[j] = v / np.linalg.norm(v)
                arrays = {"U": U, "U_random": Ur, "gamma": gam, "m_ref": m_ref, "gate_w": w, "gate_c0": np.float64(c0), "gate_c1": np.float64(c1), "gate_hf": np.float64(gate_stats["gate_hf_mean"]),
                          "block": np.int64(b), "D": np.int64(D), "rank": np.int64(rr)}
                meta = {"protocol": PROTOCOL, "block": b, "component": comp, "rank": rr, "rank_requested": r, "slices": keep, "gamma": [float(x) for x in gam], "gate_group": gate_group, "solid_level": solid,
                        "gate": gate_stats, "selection": {sl: sel[sl] for sl in cand}, "n_scenes": len(pids), "scenes": pids, "fitted_on": "discovery scenes only (full-data export)",
                        "rule": "m' = m + s*beta*g_h(h_hazard) * sum_c gamma_c ((m - m_ref).u_c) u_c on the last-frame modulation; u_c = unit throttle-brake contrast of slice c"}
                ops[f"L{b:02d}.adaln__{comp}__r{r}"] = ModulationOperator(arrays, meta)
    return ops, stats


def fit_from_model(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from localize_interaction import IDENTITY_CELLS, NULL_CELLS, encode_cell, merge_manifests, select_pairs
    from model_action_sensitivity import load_model
    from predictor_hooks import n_layers_of, predictor_of, spatial_tokens_of
    from protocol import sha256_file
    from token_groups import REGION_GROUPS, frame_for_step, load_cell_groups, set_domain, union

    t0 = time.time()
    set_domain(args.domain)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    predictor, n_spatial = predictor_of(wm), spatial_tokens_of(wm)
    for p in wm.parameters():
        p.requires_grad_(False)
    n_layers = n_layers_of(predictor)
    blocks = list(range(n_layers)) if not args.blocks else [b for b in args.blocks if b < n_layers]
    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, args.allow_calibration_seeds or args.domain == "driving")
    solid = args.solid_level
    ghost_cells = IDENTITY_CELLS if solid == 1 else ((1, 0), (1, 1))
    need = {(0, 0), (0, 1), (solid, 0), (solid, 1)}
    pids = sorted(p for p, cells in pairs.items() if need <= set(cells))[: args.max_scenes or None]
    if len(pids) < 4:
        raise SystemExit(f"only {len(pids)} scenes with the {{0, {solid}}} x action cells")
    records = {}
    comps = tuple(c for c in args.components if c in COMPONENT_SLICES)
    for i, pid in enumerate(pids):
        cells = pairs[pid]
        keys = [k for k in [(0, 0), (0, 1), (solid, 0), (solid, 1)] + list(NULL_CELLS) + list(ghost_cells) if k in cells]
        latents = {k: encode_cell(wm, *cells[k], device) for k in keys}
        groups = {k: load_cell_groups(*cells[k]) for k in keys}
        frame = frame_for_step(args.step, args.group_frame_offset)
        route = union(*(groups[k].group(frame, g) for k in keys for g in REGION_GROUPS[args.domain]))
        hazard = union(*(groups[k].group(frame, args.gate_group) for k in keys))
        records[pid] = fit_scene(wm, predictor, latents, np.asarray(route, dtype=int), np.asarray(hazard, dtype=int), args.step, solid, blocks, comps, n_spatial)
        print(f"[modop] scene {pid} ({i + 1}/{len(pids)}) done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    ops, stats = assemble_operators(records, blocks, comps, tuple(args.ranks), solid, args.n_boot, args.n_perm, args.seed, args.gate_group)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, op in ops.items():
        op.meta.update({"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "step": args.step, "artifacts": [str(p) for p in args.artifacts], "seeds_file": str(args.seeds_file)})
        op.save(args.out / f"{name}.npz")
        (args.out / f"{name}.json").write_text(json.dumps(finite(op.meta), indent=1) + "\n")
    report = {"protocol": PROTOCOL, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "artifacts": [str(p) for p in args.artifacts], "seeds_file": str(args.seeds_file),
              "n_scenes": len(pids), "scenes": pids, "blocks": blocks, "components": list(comps), "ranks": list(args.ranks), "step": args.step, "solid_level": solid, "gate_group": args.gate_group,
              "per_block": stats, "operators": sorted(ops), "runtime_s": time.time() - t0, "torch_version": torch.__version__, "interpretation_scope": INTERPRETATION_SCOPE,
              "per_scene": {p: {"levels": r["levels"], "I_true_route_norm": r["I_true_route_norm"], "scores": {f"b{b:02d}": {sl: {kk: vv for kk, vv in e.items() if kk != "resp_by_level"} for sl, e in s.items()} for b, s in r["scores"].items()}} for p, r in records.items()}}
    (args.out / "modulation_operator_fit.json").write_text(canonical_json(json_safe(finite(report))) + "\n")
    brief = {site: {"gate": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["gate"].items()}, "top_slices": sorted(((sl, round(e["point"], 4), e["p_maxt_fwer"]) for sl, e in s["selection"].items()), key=lambda t: -abs(t[1]))[:3]} for site, s in stats.items()}
    print(json.dumps(finite(brief), indent=1))
    return report


# --------------------------------------------------------------------------- #
# Self-test: fake AdaLN predictor with a planted gate_mlp x hazard mechanism
# --------------------------------------------------------------------------- #


class _Planted:
    """Fake WM (predictor_embed / action_encoder / predictor_blocks[i].{attn, mlp, adaLN_modulation} / norm / proj, unroll) whose
    block ``PL`` adds  kappa * relu(hazard content) * (gate_mlp . u_gate) * v_rel  to every token: the consequence is
    written only when hazard content is present AND the action's gate_mlp modulation is throttle-like.  shift/scale slices
    carry a hazard-independent action effect (the generic motor geometry)."""

    def __init__(self, d: int = 16, grid: int = 4, depth: int = 3, planted_block: int = 1, seed: int = 0) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        N = grid * grid
        g = torch.Generator().manual_seed(seed)
        u_h = torch.randn(d, generator=g)
        u_h /= u_h.norm()
        v_rel = torch.randn(d, generator=g)
        v_rel -= (v_rel @ u_h) * u_h
        v_rel /= v_rel.norm()
        self.u_h, self.v_rel = u_h, v_rel
        outer = self

        class Block(nn.Module):
            def __init__(self, planted: bool):
                super().__init__()
                self.norm1, self.attn, self.norm2 = nn.LayerNorm(d), nn.Linear(d, d), nn.LayerNorm(d)
                self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
                self.planted = planted
                self.u_gate = torch.randn(d, generator=g)
                self.u_gate /= self.u_gate.norm()

            def forward(self, x, z, T=None):
                mod = self.adaLN_modulation(z)  # [B, T, 6d]
                sh, sc, g1, sh2, sc2, g2 = mod.repeat_interleave(N, dim=1).chunk(6, dim=2)
                x = x + 0.1 * self.attn(self.norm1(x) * (1 + 0.1 * sc) + 0.5 * sh) * (0.1 * g1)
                x = x + 0.1 * g2 * self.mlp(self.norm2(x) * (1 + 0.1 * sc2) + 0.5 * sh2)
                if self.planted:
                    B = x.shape[0]
                    last = x.reshape(B, -1, N, d)[:, -1]  # last frame tokens
                    hz = torch.relu(last @ outer.u_h).mean(dim=1, keepdim=True)  # [B, 1] hazard content
                    gate_last = mod[:, -1, 5 * d: 6 * d] @ self.u_gate  # [B] throttle-ness of gate_mlp
                    add = 2.0 * hz * gate_last[:, None] * outer.v_rel[None, None, :]  # [B, 1, d]
                    xs = x.reshape(B, -1, N, d).clone()
                    xs[:, -1] = xs[:, -1] + add
                    x = xs.reshape(x.shape)
                return x

        class Pred(nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor_embed, self.action_encoder = nn.Linear(d, d), nn.Linear(2, d)
                self.predictor_blocks = nn.ModuleList([Block(i == planted_block) for i in range(depth)])
                self.predictor_norm, self.predictor_proj = nn.Identity(), nn.Identity()
                with torch.no_grad():
                    self.predictor_embed.weight.copy_(torch.eye(d))
                    self.predictor_embed.bias.zero_()

            def forward(self, x, actions):
                x = self.predictor_embed(x).flatten(2, 4)
                B, T = x.shape[:2]
                z = self.action_encoder(actions)
                x = x.flatten(1, 2)
                for blk in self.predictor_blocks:
                    x = blk(x, z, T=T)
                return self.predictor_proj(self.predictor_norm(x)).view(B, T, N, d)

        class VWM(nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = Pred()

        self.model = VWM().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.grid_size, self.ctxt_window, self.d, self.N, self.planted_block = grid, 2, d, N, planted_block

    def unroll(self, z_ctxt, act_suffix=None):
        import torch

        T, B, _ = act_suffix.shape
        vid = z_ctxt.expand(B, *z_ctxt.shape[1:])
        acts = act_suffix.permute(1, 0, 2)
        for h in range(T):
            a = acts[:, : h + 1][:, -self.ctxt_window:]
            pred = self.model.predictor(vid[:, -self.ctxt_window:], a)
            vid = torch.cat([vid, pred[:, -1:].view(B, 1, 1, self.grid_size, self.grid_size, self.d)], dim=1)
        return vid.permute(1, 0, 2, 3, 4, 5)


def _planted_scene(fake: _Planted, rng: np.random.Generator, solid: int = 1) -> tuple[dict, np.ndarray, np.ndarray]:
    """Cells (lv, a) for lv in {0, solid, 2}: hazard content on 3 hazard tokens only in the solid level; futures = the fake
    model's own clean predictions (so I_true is the planted consequence)."""
    import torch

    d, N = fake.d, fake.N
    hazard_idx = np.array([5, 6, 9])
    route_idx = np.array([5, 6, 9, 10, 13, 14])
    base = torch.as_tensor(rng.normal(size=(1, 1, 1, fake.grid_size, fake.grid_size, d)) * 0.5, dtype=torch.float32)
    act = {0: torch.tensor([[[-1.0, 0.0]]] * 3, dtype=torch.float32), 1: torch.tensor([[[1.0, 0.0]]] * 3, dtype=torch.float32)}
    latents = {}
    for lv in (0, solid, NULL_CONTROL_HAZARD):
        z = base.clone()
        if lv == solid:
            zf = z.view(1, 1, 1, N, d)
            zf[0, 0, 0, hazard_idx] += 3.0 * fake.u_h
        elif lv == NULL_CONTROL_HAZARD:
            zf = z.view(1, 1, 1, N, d)
            zf[0, 0, 0, [0, 1, 2]] += 0.5 * torch.as_tensor(rng.normal(size=d), dtype=torch.float32)
        for a in (0, 1):
            with torch.no_grad():
                roll = fake.unroll(z, act_suffix=act[a])
            latents[(lv, a)] = {"z_context": z, "z_future": roll[1:].permute(1, 0, 2, 3, 4, 5).contiguous(), "actions": act[a]}
    return latents, route_idx, hazard_idx


def self_test(out: Path | None = None, seed: int = 0) -> dict[str, Any]:
    import torch

    from modulation_patch import _predict

    t0 = time.time()
    rng = np.random.default_rng(seed)
    fake = _Planted(seed=seed)
    predictor = fake.model.predictor
    blocks = list(range(len(predictor.predictor_blocks)))
    records, scenes = {}, {}
    for i in range(8):
        lat, route, hazard = _planted_scene(fake, rng)
        records[f"s{i}"] = fit_scene(fake, predictor, lat, route, hazard, 0, 1, blocks, ("mlp", "all"), fake.N)
        scenes[f"s{i}"] = (lat, route, hazard)
    ops, stats = assemble_operators(records, blocks, ("mlp", "all"), (1, 2, 4), 1, n_boot=200, n_perm=256, seed=seed)
    pb = fake.planted_block
    sel = stats[f"L{pb:02d}.adaln"]["selection"]
    top = max(sel, key=lambda sl: abs(sel[sl]["point"]))
    checks: dict[str, Any] = {"planted_block": pb, "top_slice": top, "top_score": sel[top]["point"], "top_p_maxt": sel[top]["p_maxt_fwer"],
                              "other_block_max_abs_score": max(abs(e["point"]) for b in blocks if b != pb for e in stats[f"L{b:02d}.adaln"]["selection"].values()),
                              "gate": stats[f"L{pb:02d}.adaln"]["gate"]}
    op = ops[f"L{pb:02d}.adaln__mlp__r1"]
    # apply on the solid throttle cell and the H0 throttle cell: relational recovery vs unsteered
    def recovery(steerer_edits, lat, route):
        D = fake.d
        fut = {kk: lat[kk]["z_future"][0, 0].reshape(-1, D) for kk in CELL_ORDER}
        I_true = (fut[(1, 1)] - fut[(1, 0)] - fut[(0, 1)] + fut[(0, 0)])[route].flatten()
        out = {}
        for lv in (0, 1):
            with torch.no_grad():
                base = _predict(fake, lat[(lv, 0)]["z_context"], lat[(0, 1)]["actions"], 0)
                st = ModulationSteerer(predictor, steerer_edits, fake.N)
                with st:
                    pred = _predict(fake, lat[(lv, 0)]["z_context"], lat[(0, 1)]["actions"], 0)
            out[lv] = {"delta_proj": float(((pred - base)[route].flatten() @ I_true) / (I_true @ I_true)), "changed": bool(not torch.equal(pred, base)), "applied": len(st.applied)}
        return out

    lat, route, hazard = scenes["s0"]
    log: list[float] = []
    main = recovery({pb: [(0, hazard, op, "strengthen", 1.0, "main", log)]}, lat, route)
    supp = recovery({pb: [(0, hazard, op, "suppress", 1.0, "main", [])]}, lat, route)
    sham = recovery({pb: [(0, hazard, op, "strengthen", 0.0, "main", [])]}, lat, route)
    ung_on = recovery({pb: [(0, hazard, op, "strengthen", 1.0, "ungated_on", [])]}, lat, route)
    rand = recovery({pb: [(0, hazard, op, "strengthen", 1.0, "random_matched", [])]}, lat, route)
    wrong_b = (pb + 1) % len(blocks)
    wrong = recovery({wrong_b: [(0, hazard, op, "strengthen", 1.0, "main", [])]}, lat, route)
    checks.update({"main_solid_gain": main[1]["delta_proj"], "main_h0_gain": main[0]["delta_proj"], "suppress_solid_gain": supp[1]["delta_proj"],
                   "sham_bit_identical": (not sham[0]["changed"]) and (not sham[1]["changed"]) and sham[1]["applied"] == 1,
                   "ungated_on_h0_gain": ung_on[0]["delta_proj"], "random_solid_gain": rand[1]["delta_proj"], "wrong_block_solid_gain": wrong[1]["delta_proj"],
                   "edit_relative_rms": float(np.mean(log)) if log else None})
    # numpy rule agrees with the hook
    m = records["s0"]["mods"][(1, 1)][pb][None, :]
    g = op.gate_numpy(records["s0"]["xin"][(1, 1)][pb][None, :])
    m_np = op.apply_numpy(m, g, 1.0, "strengthen")
    checks["numpy_rule_gate_solid"] = float(g[0])
    passed = (top == "gate_mlp" and sel[top]["point"] > 0 and checks["other_block_max_abs_score"] < 0.25 * abs(sel[top]["point"])
              and checks["gate"]["solid"] > 0.8 and checks["gate"]["h0"] < 0.2
              and main[1]["delta_proj"] > 0.3 and abs(main[0]["delta_proj"]) < 0.05 and supp[1]["delta_proj"] < -0.3
              and checks["sham_bit_identical"] and abs(rand[1]["delta_proj"]) < 0.3 * main[1]["delta_proj"] and abs(wrong[1]["delta_proj"]) < 0.3 * main[1]["delta_proj"]
              and abs(ung_on[0]["delta_proj"]) > 0.05 and m_np.shape == m.shape)
    checks["passed"] = bool(passed)
    checks["runtime_s"] = time.time() - t0
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "self_test.json").write_text(json.dumps(finite(checks), indent=1) + "\n")
    print(json.dumps(finite(checks), indent=1))
    print("SELF_TEST_PASSED" if passed else "SELF_TEST_FAILED")
    return checks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit", help="fit modulation operators (needs the model)")
    f.add_argument("--repo", type=Path, required=True)
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--checkpoint", type=Path, required=True)
    f.add_argument("--model-name", default="jepa_wm_driving")
    f.add_argument("--artifacts", type=Path, nargs="+", required=True)
    f.add_argument("--seeds-file", type=Path, default=None, help="DISCOVERY seeds")
    f.add_argument("--allow-calibration-seeds", action="store_true")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--domain", default="driving")
    f.add_argument("--device", default="cuda:0")
    f.add_argument("--solid-level", type=int, choices=[1, OBJECT_HAZARD], default=1)
    f.add_argument("--blocks", type=int, nargs="*", default=None)
    f.add_argument("--components", nargs="*", default=["attn", "mlp", "all"])
    f.add_argument("--ranks", type=int, nargs="*", default=list(RANK_GRID))
    f.add_argument("--step", type=int, default=0)
    f.add_argument("--group-frame-offset", type=int, default=0)
    f.add_argument("--gate-group", default="hazard")
    f.add_argument("--max-scenes", type=int, default=0)
    f.add_argument("--n-boot", type=int, default=2000)
    f.add_argument("--n-perm", type=int, default=2000)
    f.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> Any:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(Path("artifacts/modulation_operator_self_test") if argv is None else None, args.seed)
    if args.cmd == "fit":
        return fit_from_model(args)
    ap.print_help()
    return None


if __name__ == "__main__":
    main()
