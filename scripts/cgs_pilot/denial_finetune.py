#!/usr/bin/env python3
"""Denial retraining (CAFT-inspired, experiment 2): fine-tune a trained arm with a subspace projected out at the
discovered site in the forward AND backward passes, then evaluate with the projection removed (test-time removal) and
with it kept.

*** Evidential status (design doc "Registered additions" (e) and the claim ladder): this is an ENGINEERING / NECESSITY
follow-up.  It asks whether the predictor can re-route the hazard x action consequence around a subspace that the
donor-patch stage found SUFFICIENT.  It cannot count as C3 ("causally used") evidence on its own: C3 requires the donor
patch / operator rule with its full control set (wrong group, time shift, bidirectional, dose) reproduced on the sealed
confirmation scenes with zero refitting.  Every number here is discovery / development evidence; the readouts are the
existing open-loop gates (development readouts) and, when the closed-loop pilot data exist, MetaDrive's native label
(Label-first principle, amendments 1-2). ***

Method (Casademunt et al. 2025, arXiv:2507.16795; ``representational_geometry_paper_concepts.md`` section 5).  With
``U`` [d, k] an orthonormal basis of the subspace to deny at site ``L{LL}.{hook}`` of the AdaLN predictor, every
forward pass through that module during fine-tuning replaces its output ``h`` by ``(I - U U^T) h`` for ALL tokens of
all frames (a hook on the vendor module through ``predictor_hooks._module_for``; the frozen DINOv3 encoder is never
touched -- training runs on the cached features).  The projection is a differentiable operation in the graph, so
autograd applies ``(I - U U^T)`` to the backward pass as well: the weights that write into ``U`` receive no gradient
from any downstream loss (``--grad-check`` verifies ``max |U^T g| / ||g||`` at the site is at float precision).  The
fine-tuning is ``train_feature_predictor.train`` UNCHANGED (same loss, optimiser family, split rule, clip order and seed
as the original training; only the schedule is short: ``--epochs`` <= 25 % of the original 30, ``--ref-lr`` reduced) on
the SAME cached training features and clips.  The saved checkpoint is a plain predictor: evaluating it through the
frozen loader IS the test-time removal; ``run-with-hook`` re-installs the projection for the with-projection evaluation
through the unchanged ``latent_cache.py`` / gate scripts and ``closed_loop_rollout.py``.

Subspaces (``fit-subspaces``, from the two seed-0 localization dumps, arm A coordinates, ``arm_diff_pca.py`` helpers):
``conceptor`` = the top-k eigenvectors (mu-descending) of the arm's relational conceptor ``C_DiD(ped) AND NOT C_action
AND NOT C_null`` at the site / group / step (exactly ``geometry_cross_arm.py``'s ``rel_ped`` fit; the registered choice
at ``L03.mlp_out`` corridor step 0, k in {1, 4, 16}); ``armdiff`` = the top-k PCs of the transported all-cell
``h_A - h_B`` (second choice); ``random`` = a rank-matched random orthonormal subspace (hard projector, so rank is the
matched quantity); ``none`` = continued-training control.  The wrong-layer control fits the same conceptor rule at a site
where the patch stage recovered nothing (default ``L05.mlp_out``).

Readouts per variant (``summarize``): captured interaction fraction ``1 - median NMSE`` (B-gate T1b), Rec specificity,
CEM flip rate, T1c' solid-vs-ghost, ghost-identity captured fraction, hazard-free val unroll L2 relative to the parent
(equivalence margin 5 %, as ``drive_models/equivalence.json``), each WITHOUT and WITH the projection, plus the parent
model WITH the projection (pure inference-time ablation) -- and the closed-loop native success / failure rates when
present.  Interpretation (descriptive flags): consequence present WITH the projection after denial fine-tuning ->
re-routed (the subspace was sufficient, not necessary); absent -> not re-routable within the frozen budget (necessary
relative to this budget; the random-subspace and continued-training controls calibrate the budget); hazard-free loss
outside the margin -> the fine-tuning changed generic prediction and the variant is not interpretable.

``--self-test`` (CPU, seconds): a tiny AdaLN-style predictor (same module names, so the same hook code runs) is trained
on synthetic tokens whose next state carries a planted hazard x action interaction along a direction ``v``; the
predictor's output projection is frozen orthogonal, so the interaction can reach the output only through the planted
internal direction ``u = P^T v`` at the last block's residual output.  Checks: the trained model captures the
interaction; the projection hook with ``u`` removes it and a random direction does not; denial fine-tuning with the hook
keeps it removed WITH the projection while the continued-training control keeps it; the random-subspace denial keeps
it; hazard-free loss stays within margin; the gradient at the site has no ``u`` component; an empty subspace is a
bit-exact identity; the DiD-discovered direction at the site recovers ``u``; and denial at the FIRST block (where the
second block can re-route) is reported as the re-routing example.

References: arXiv:2507.16795 (CAFT), arXiv:2605.17144 (conceptors), arXiv:2309.16042 (patching practice).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_feature_predictor as tfp  # noqa: E402
from predictor_hooks import Site, _module_for  # noqa: E402

PROTOCOL = "cgs-denial-finetune-v0.1"
KEYS = ("conceptor", "armdiff", "random", "none")
DEFAULT_SITE, DEFAULT_GROUP, DEFAULT_STEP = "L03.mlp_out", "corridor", 0
WRONG_SITE_DEFAULT = "L05.mlp_out"
EQUIV_MARGIN = 0.05
CAPTURED_THRESHOLD = 0.25  # B-gate T1b: interaction NMSE median <= 0.75
C3_DISCLAIMER = ("Engineering / necessity follow-up (design doc Registered additions (e)); NOT C3 evidence on its own. Open-loop gates are "
                 "development readouts; the closed-loop native label (Label-first principle) is the outcome label when present.")


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The denial hook
# --------------------------------------------------------------------------- #


class SubspaceDenial:
    """Forward hook that replaces the site's activation by ``(I - U U^T) h`` (all tokens, all frames).  Differentiable, so
    the backward pass is projected too.  ``U`` [d, k] orthonormal columns; ``k = 0`` is the identity (bit-exact).  The
    projection is computed in float32 and returned in float32 (an autocast bf16 output would leak ~1e-3 of the removed
    component back through rounding); the residual stream is float32 under autocast anyway."""

    def __init__(self, predictor: torch.nn.Module, site: str, U: np.ndarray | torch.Tensor | None, grad_check: bool = False) -> None:
        self.predictor = predictor
        self.site = Site.parse(site)
        if U is None or (hasattr(U, "shape") and U.shape[-1] == 0):
            self.U = None
        else:
            U = torch.as_tensor(np.asarray(U), dtype=torch.float32)
            if U.ndim != 2:
                raise ValueError("U must be [d, k]")
            G = U.T @ U
            if not torch.allclose(G, torch.eye(U.shape[1]), atol=1e-4):
                raise ValueError("U columns are not orthonormal")
            self.U = U
        self.grad_check = grad_check
        self.grad_stats: list[dict[str, float]] = []
        self.n_calls = 0
        self._handles: list[Any] = []
        self._U_dev: dict[torch.device, torch.Tensor] = {}

    @property
    def k(self) -> int:
        return 0 if self.U is None else int(self.U.shape[1])

    def _u(self, device: torch.device) -> torch.Tensor:
        if device not in self._U_dev:
            self._U_dev[device] = self.U.to(device)
        return self._U_dev[device]

    def project(self, x: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        if self.U is None:
            return x
        if x.shape[-1] != self.U.shape[0]:
            raise ValueError(f"site width {x.shape[-1]} != U rows {self.U.shape[0]} at {self.site.site_id}")
        U = self._u(x.device)
        xf = x.float()
        if self.grad_check and xf.requires_grad:
            xf.register_hook(lambda g, U=U: self.grad_stats.append(self._grad_stat(g, U)))
        return xf - (xf @ U) @ U.T

    @staticmethod
    def _grad_stat(g: torch.Tensor, U: torch.Tensor) -> dict[str, float]:
        gf = g.float().reshape(-1, g.shape[-1])
        along = (gf @ U).abs().max().item()
        return {"max_abs_grad_along_U": float(along), "grad_norm": float(gf.norm().item()), "ratio": float(along / max(gf.norm().item(), 1e-30))}

    def __enter__(self) -> "SubspaceDenial":
        module, kind = _module_for(self.predictor, self.site)
        if kind == "pre":
            self._handles.append(module.register_forward_pre_hook(lambda m, args, kwargs: ((self.project(args[0]), *args[1:]), kwargs), with_kwargs=True))
        else:
            self._handles.append(module.register_forward_hook(lambda m, args, out: self.project(out)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def install(self) -> "SubspaceDenial":
        """Permanent installation (no context): used by ``run-with-hook``."""
        return self.__enter__()


# --------------------------------------------------------------------------- #
# Subspace files
# --------------------------------------------------------------------------- #


def orthonormalize(V: np.ndarray) -> np.ndarray:
    """Columns of V [d, k] -> orthonormal columns spanning the same space (SVD; drops dependent columns)."""
    V = np.asarray(V, dtype=np.float64)
    if V.size == 0:
        return V.reshape(V.shape[0], 0)
    u, s, _ = np.linalg.svd(V, full_matrices=False)
    keep = s > 1e-8 * max(float(s.max()), 1e-300)
    return u[:, keep]


def load_subspace(path: Path | None, key: str, k: int, d: int | None = None, random_seed: int = 0) -> tuple[np.ndarray | None, dict[str, Any]]:
    """U [d, k] (orthonormal columns) for ``key``; ``none`` -> None; ``random`` -> rank-matched random subspace."""
    if key == "none" or k == 0:
        return None, {"key": key, "k": 0}
    if key == "random":
        if d is None:
            if path is None:
                raise ValueError("random subspace needs --subspace (for d) or --d")
            with np.load(path) as z:
                d = int(z["d"])
        rng = np.random.default_rng(random_seed)
        Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
        U = Q[:, :k]
        return U.astype(np.float32), {"key": key, "k": k, "random_seed": random_seed, "d": d, "sha256": sha256_array(U.astype(np.float32))}
    if path is None:
        raise ValueError(f"--subspace is required for key {key}")
    with np.load(path) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if key == "conceptor":
            V, mu = np.asarray(z["conceptor_eigvecs"], dtype=np.float64), np.asarray(z["conceptor_mu"], dtype=np.float64)
            order = np.argsort(mu)[::-1]
            V = V[order]
            avail = int(len(mu))
            U = orthonormalize(V[:k].T)
            info = {"key": key, "k_requested": k, "k": int(U.shape[1]), "conceptor_rank": avail, "mu_top": [float(x) for x in mu[order][:k]], "mu_mass_fraction": float(np.sum(mu[order][:k]) / max(float(mu.sum()), 1e-300))}
        elif key == "armdiff":
            P = np.asarray(z["armdiff_pcs"], dtype=np.float64)
            U = orthonormalize(P[:k].T)
            info = {"key": key, "k_requested": k, "k": int(U.shape[1]), "variance_ratio_top": [float(x) for x in np.asarray(z["armdiff_variance_ratio"])[:k]]}
        else:
            raise ValueError(f"unknown key {key}")
    U = U.astype(np.float32)
    info.update({"sha256": sha256_array(U), "d": int(U.shape[0]), "source": str(path), "source_meta": meta})
    return U, info


def fit_site_subspaces(dump_a: Path, dump_b: Path, stimulus: Path, discovery_seeds: Path | None, site: str, group: str, step: int, out: Path, k_action: int = 4, target_dims: float = 4.0, seed: int = 0, n_pcs: int = 32) -> dict[str, Any]:
    """Relational conceptor of arm A, transported arm-diff PCs, identity and donor directions at one site (arm A coords)."""
    import arm_diff_pca as adp
    from geometry_cross_arm import align_arms, load_arm, load_site, matched_token_rows, scene_rows, token_rows, token_sets_for
    from geometry_localize import pool_site
    from token_groups import set_domain

    set_domain("driving")
    arm_a, arm_b = load_arm(dump_a, discovery_seeds), load_arm(dump_b, discovery_seeds)
    cells, scenes, status = align_arms(arm_a, arm_b)
    n_steps = min(arm_a["n_steps"], arm_b["n_steps"])
    if step >= n_steps:
        raise SystemExit(f"step {step} not in the dumps (n_steps {n_steps})")
    token_sets = token_sets_for(stimulus, cells, (group,), n_steps, arm_a["offset"], "driving")
    rows_a, rows_b = np.asarray([c["row_a"] for c in cells]), np.asarray([c["row_b"] for c in cells])
    actsA, tixA = load_site(dump_a, site, rows_a)
    actsB, tixB = load_site(dump_b, site, rows_b)
    d = int(actsA.shape[-1])
    XA, XB, scene_idx = matched_token_rows(token_rows(actsA, tixA, cells, step, tuple(adp.ACTION_LEVELS)), token_rows(actsB, tixB, cells, step, tuple(adp.ACTION_LEVELS)))
    folds, (W_full, _, _), tlog = adp.transport_maps(XA, XB, scene_idx, scenes)
    PA, _ = pool_site(actsA, tixA, cells, token_sets, step, group)
    PB, _ = pool_site(actsB, tixB, cells, token_sets, step, group)
    ok = np.all(np.isfinite(PA), axis=1) & np.all(np.isfinite(PB), axis=1)
    bad = {c["pair_id"] for c in cells if not ok[c["row"]]}
    keep = [s for s in scenes if s not in bad]
    sub_cells = [c for c in cells if c["pair_id"] in set(keep)]
    PBt = adp.transport_b_to_a(PB, sub_cells, folds)
    rowsA = adp.add_a1_rows(scene_rows(PA, sub_cells, keep), PA, sub_cells, keep)
    rowsB = adp.add_a1_rows(scene_rows(PBt, sub_cells, keep), PBt, sub_cells, keep)

    class _A:  # minimal args view for reference_objects
        pass

    a = _A()
    a.k_action, a.target_dims = k_action, target_dims
    refs = adp.reference_objects(rowsA, rowsB, W_full, d, a, seed)
    X, _ = adp.delta_rows(PA, PBt, sub_cells, keep, "all")
    p = adp.pca(X, n_pcs)
    c = refs["conceptor_A_rel_ped"]["conceptor"]
    order = np.argsort(np.asarray(c["mu"]))[::-1]
    meta = {"protocol": PROTOCOL, "site": site, "group": group, "step": step, "dump_a": str(dump_a), "dump_b": str(dump_b), "n_scenes": len(keep), "d": d, "alignment": status, "transport": tlog,
            "conceptor": {k: v for k, v in refs["conceptor_A_rel_ped"].items() if k in ("alpha", "form", "rank", "quota")}, "armdiff": {"pc1_ratio": p["pc1_ratio"], "participation_ratio": p["participation_ratio"], "n_rows": p["n_rows"]},
            "coordinates": "arm A predictor coordinates at the module output (hook site); U columns are unit vectors in that space", "k_action": k_action, "target_dims": target_dims, "seed": seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), d=np.int64(d), conceptor_eigvecs=np.asarray(c["eigvecs"])[order].astype(np.float32), conceptor_mu=np.asarray(c["mu"])[order].astype(np.float32),
                        armdiff_pcs=p["pcs"].astype(np.float32), armdiff_variance_ratio=np.asarray(p["ratio"], dtype=np.float32), armdiff_mean_shift=p["mean"].astype(np.float32),
                        identity_direction=refs["identity_direction"].astype(np.float32), donor_direction=refs["donor_direction"].astype(np.float32), W_full=W_full.astype(np.float32))
    # overlap bookkeeping between the candidate subspaces (what is being denied relative to what)
    Qc = orthonormalize(np.asarray(c["eigvecs"])[order][:16].T)
    Qa = orthonormalize(p["pcs"][:16].T)
    meta["overlaps"] = {"conceptor_k4_vs_armdiff_k4_mean_sq_cos": float(np.mean(np.linalg.svd(Qc[:, :4].T @ Qa[:, :4], compute_uv=False) ** 2)) if Qc.shape[1] >= 4 and Qa.shape[1] >= 4 else None,
                        "identity_in_conceptor_k4": float(np.linalg.norm(Qc[:, :4].T @ refs["identity_direction"])) if Qc.shape[1] >= 4 else None,
                        "identity_in_armdiff_k4": float(np.linalg.norm(Qa[:, :4].T @ refs["identity_direction"])) if Qa.shape[1] >= 4 else None,
                        "donor_in_conceptor_k4": float(np.linalg.norm(Qc[:, :4].T @ refs["donor_direction"])) if Qc.shape[1] >= 4 else None}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    return meta


# --------------------------------------------------------------------------- #
# Fine-tuning (real models and the self-test share this)
# --------------------------------------------------------------------------- #


def trainer_args(out: Path, features: Path | None, repo: Path | None, epochs: int, ref_lr: float, warmup_epochs: float, final_lr: float, seed: int, batch_size: int, label: str, extra: list[str] | None = None) -> argparse.Namespace:
    argv = ["--out", str(out), "--epochs", str(epochs), "--ref-lr", str(ref_lr), "--warmup-epochs", str(warmup_epochs), "--final-lr", str(final_lr), "--start-lr", str(min(1e-6, ref_lr)),
            "--seed", str(seed), "--batch-size", str(batch_size), "--arm", label, "--no-save-opt"]
    if features is not None:
        argv += ["--features", str(features)]
    if repo is not None:
        argv += ["--repo", str(repo)]
    return tfp.build_parser().parse_args(argv + (extra or []))


def val_losses(predictor: torch.nn.Module, store: tfp.FeatureStore, device: torch.device, batch_size: int, autocast_ctx, val_idx: np.ndarray) -> dict[str, Any]:
    hf = np.asarray([i for i in val_idx if not store.clips[i].get("hazard", False)])
    hz = np.asarray([i for i in val_idx if store.clips[i].get("hazard", False)])
    out = {"n_val": int(len(val_idx)), "n_hazard_free": int(len(hf)), "n_hazard": int(len(hz))}
    with torch.no_grad():
        out["hazard_free"] = tfp.evaluate(predictor, store, hf, batch_size, device, 2, autocast_ctx) if len(hf) else None
        out["hazard"] = tfp.evaluate(predictor, store, hz, batch_size, device, 2, autocast_ctx) if len(hz) else None
    return out


def grad_check(predictor: torch.nn.Module, store: tfp.FeatureStore, device: torch.device, args: argparse.Namespace, denial: SubspaceDenial, idx: np.ndarray) -> dict[str, Any]:
    """One loss.backward() through the hook: the gradient arriving at the site's raw output must have no U component."""
    if denial.U is None:
        return {"skipped": "no subspace"}
    denial.grad_check, denial.grad_stats = True, []
    predictor.train()
    feats, acts = tfp.fetch_batch(store, idx[: args.batch_size], device)
    gen = torch.Generator().manual_seed(args.seed)
    autocast_ctx = tfp.autocast_factory(device, args.dtype)
    with autocast_ctx():
        loss, _ = tfp.jepa_wm_loss(predictor, feats, acts, args.rollout_steps, args.rollout_prefixes, args.ctxt_window_train_rollout, not args.no_rollout_stop_gradient, gen)
    predictor.zero_grad(set_to_none=True)
    loss.backward()
    predictor.zero_grad(set_to_none=True)
    denial.grad_check = False
    stats = denial.grad_stats
    denial.grad_stats = []
    return {"n_backward_hooks": len(stats), "max_ratio_grad_along_U": max((s["ratio"] for s in stats), default=float("nan")), "max_abs_grad_along_U": max((s["max_abs_grad_along_U"] for s in stats), default=float("nan")),
            "loss": float(loss.detach()), "ok": bool(stats) and max(s["ratio"] for s in stats) < 1e-5}


def finetune(predictor: torch.nn.Module, store: tfp.FeatureStore, device: torch.device, cfg: tfp.ModelCfg, args: argparse.Namespace, site: str, U: np.ndarray | None, info: dict[str, Any], parent_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Val losses without / with the projection before training, gradient check, fine-tune with the hook, val losses
    with / without the projection after training.  Returns the denial metadata (also written to <out>/denial_meta.json)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    autocast_ctx = tfp.autocast_factory(device, args.dtype)
    train_idx, val_idx, split_rule = tfp.split_indices(store, args.val_fraction, args.seed)
    t0 = time.time()
    meta: dict[str, Any] = {"protocol": PROTOCOL, "disclaimer": C3_DISCLAIMER, "site": site, "subspace": info, "epochs": args.epochs, "ref_lr": args.ref_lr, "warmup_epochs": args.warmup_epochs, "final_lr": args.final_lr,
                            "seed": args.seed, "batch_size": args.batch_size, "split_rule": split_rule, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "parent": parent_meta or {}, "val": {}}
    predictor.eval()
    meta["val"]["parent_no_projection"] = val_losses(predictor, store, device, args.batch_size, autocast_ctx, val_idx)
    denial = SubspaceDenial(predictor, site, U)
    with denial:
        meta["val"]["parent_with_projection"] = val_losses(predictor, store, device, args.batch_size, autocast_ctx, val_idx)
        meta["grad_check"] = grad_check(predictor, store, device, args, denial, train_idx)
        if args.epochs > 0:
            predictor.train()
            meta["train"] = {k: v for k, v in tfp.train(args, predictor, store, device, cfg).items() if k in ("history", "best", "timing", "schedule", "data")}
            predictor.eval()
            meta["val"]["finetuned_with_projection"] = val_losses(predictor, store, device, args.batch_size, autocast_ctx, val_idx)
        meta["hook_calls"] = denial.n_calls
    if args.epochs > 0:
        meta["val"]["finetuned_no_projection"] = val_losses(predictor, store, device, args.batch_size, autocast_ctx, val_idx)
        sd = torch.load(out_dir / "jepa-latest.pth.tar", map_location="cpu", weights_only=True)
        meta["checkpoint"] = {"file": "jepa-latest.pth.tar", "epoch": int(sd["epoch"]), "sha256": tfp.sha256_path(out_dir / "jepa-latest.pth.tar")}
    ref = (meta["val"]["parent_no_projection"].get("hazard_free") or {}).get("unroll_l2")
    meta["hazard_free_equivalence"] = {}
    for name, v in meta["val"].items():
        x = (v.get("hazard_free") or {}).get("unroll_l2")
        if ref and x:
            rel = abs(x - ref) / (0.5 * (x + ref))
            meta["hazard_free_equivalence"][name] = {"unroll_l2": x, "parent_unroll_l2": ref, "rel_diff": rel, "rel_increase": (x - ref) / ref, "within_margin": bool(rel <= EQUIV_MARGIN),
                                                     "rule": f"two-sided relative difference <= {EQUIV_MARGIN} (drive_models/equivalence.json P2 margin); rel_increase is the signed degradation"}
    meta["runtime_s"] = time.time() - t0
    (out_dir / "denial_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


# --------------------------------------------------------------------------- #
# Real-model plumbing
# --------------------------------------------------------------------------- #


def build_from_model_dir(model_dir: Path, repo: Path, device: torch.device, checkpoint: str = "jepa-latest.pth.tar") -> tuple[torch.nn.Module, tfp.ModelCfg, dict[str, Any]]:
    meta = json.loads((model_dir / f"{checkpoint.replace('.pth.tar', '.meta.json')}").read_text())
    mcfg = {k: v for k, v in meta["model"].items() if k in tfp.ModelCfg.__dataclass_fields__}
    cfg = tfp.ModelCfg(**mcfg)
    sys.path.insert(0, str(repo.resolve()))
    predictor, _ = tfp.build_vendor_predictor(cfg, device)
    sd = torch.load(model_dir / checkpoint, map_location="cpu", weights_only=True)
    predictor.load_state_dict(sd["predictor"])
    parent = {"model_dir": str(model_dir), "checkpoint": checkpoint, "checkpoint_sha256": tfp.sha256_path(model_dir / checkpoint), "epoch": int(sd["epoch"]), "arm": meta.get("arm"), "epochs_original": meta["hyperparameters"]["epochs"],
              "ref_lr_original": meta["hyperparameters"]["ref_lr"], "history_last": meta["history"][-1] if meta.get("history") else None}
    return predictor, cfg, parent


def cmd_finetune(a: argparse.Namespace) -> None:
    tfp.set_determinism(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    predictor, cfg, parent = build_from_model_dir(a.model_dir, a.repo, device, a.checkpoint)
    store = tfp.load_store(a.features, a.meta)
    if a.epochs > 0.25 * parent["epochs_original"] + 1e-9:
        raise SystemExit(f"--epochs {a.epochs} exceeds the frozen 25 % budget of the original {parent['epochs_original']} epochs")
    U, info = load_subspace(a.subspace, a.key, a.k, d=a.d, random_seed=a.random_seed)
    args = trainer_args(a.out, a.features, a.repo, a.epochs, a.ref_lr, a.warmup_epochs, a.final_lr, a.seed, a.batch_size, a.label, ["--eval-ctxt-window", "2"])
    args.meta = a.meta
    meta = finetune(predictor, store, device, cfg, args, a.site, U, info, parent)
    print(json.dumps({k: meta[k] for k in ("site", "subspace", "grad_check", "hazard_free_equivalence")}, default=str))


def _patch_loader_with_hook(site: str, U: np.ndarray | None) -> None:
    """Make every ``model_action_sensitivity.load_model`` call install the projection on the loaded predictor."""
    import model_action_sensitivity as mas
    from predictor_hooks import predictor_of

    orig = mas.load_model

    def patched(*args, **kwargs):
        model, pre = orig(*args, **kwargs)
        SubspaceDenial(predictor_of(model), site, U).install()
        print(f"[denial] projection hook installed at {site} (k={0 if U is None else U.shape[1]})", file=sys.stderr, flush=True)
        return model, pre

    mas.load_model = patched
    for name in ("latent_cache", "closed_loop_rollout", "steered_planner_ranking", "patch_site", "localize_interaction"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "load_model"):
            mod.load_model = patched


def cmd_run_with_hook(a: argparse.Namespace) -> None:
    """Run an existing script (``latent_cache`` or ``closed_loop_rollout``) with the projection installed at load time."""
    U, info = load_subspace(a.subspace, a.key, a.k, d=a.d, random_seed=a.random_seed)
    if a.script == "latent_cache":
        import latent_cache as lc
        import token_groups as tg

        lc.load_token_mask = tg.cache_token_mask_driving
        _patch_loader_with_hook(a.site, U)
        sys.argv = ["latent_cache.py"] + a.script_args
        lc.main()
    elif a.script == "closed_loop_rollout":
        import closed_loop_rollout as clr

        _patch_loader_with_hook(a.site, U)
        sys.argv = ["closed_loop_rollout.py"] + a.script_args
        clr.main()
    else:
        raise SystemExit(f"unknown script {a.script}")
    print(json.dumps({"hook": {"site": a.site, **{k: v for k, v in info.items() if k != "source_meta"}}}))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _gate_readouts(ev: Path) -> dict[str, Any] | None:
    b = ev / "behavior_gate_discovery.json"
    if not b.exists():
        return None
    g = json.loads(b.read_text())
    out = {"verdict": g.get("verdict"), "n_scenes": g.get("n_scenes"), "captured": g.get("model_captured_interaction_fraction"), "nmse_median": g.get("T1b_interaction_nmse_median"),
           "spec": g.get("T1c_rec_h1_minus_h0"), "spec_p": g.get("T1c_p"), "flip": g.get("T2a_flip_rate_h1_vs_h0"), "t1c_prime": g.get("T1c_prime")}
    x = ev / "behavior_gate_discovery_level3.json"
    if x.exists():
        gx = json.loads(x.read_text())
        out["ghost"] = {"verdict": gx.get("verdict"), "captured": gx.get("model_captured_interaction_fraction"), "spec": gx.get("T1c_rec_h1_minus_h0"), "flip": gx.get("T2a_flip_rate_h1_vs_h0")}
    cf = ev / "cf_gate_discovery.json"
    if cf.exists():
        c = json.loads(cf.read_text())
        out["cf"] = {"verdict": c.get("verdict"), "rec_h1": c["pooled"]["rec_h1"].get("median"), "interaction_cosine": c["pooled"]["interaction_cosine"].get("mean"), "arc": {h: v.get("mean") for h, v in c["pooled"]["arc"].items()}, "drift": {h: v.get("mean") for h, v in c["pooled"]["drift_energy"].items()}}
    return out


def _closed_loop_readouts(cl: Path) -> dict[str, Any] | None:
    s = cl / "summary.json"
    if not s.exists():
        return None
    d = json.loads(s.read_text())
    return {k: d.get(k) for k in ("verdict", "n_rollouts", "rates", "c0", "label", "levels", "hazard_free") if k in d}


def cmd_summarize(a: argparse.Namespace) -> None:
    root = Path(a.root)
    parent = _gate_readouts(root / "parent" / "eval" / "nohook")
    rows = []
    for vd in sorted((root / "variants").glob("*")):
        if not vd.is_dir():
            continue
        dm = vd / "denial_meta.json"
        meta = json.loads(dm.read_text()) if dm.exists() else {}
        row: dict[str, Any] = {"variant": vd.name, "site": meta.get("site"), "key": (meta.get("subspace") or {}).get("key"), "k": (meta.get("subspace") or {}).get("k"), "epochs": meta.get("epochs"),
                               "grad_check_ok": (meta.get("grad_check") or {}).get("ok"), "hazard_free_equivalence": meta.get("hazard_free_equivalence"),
                               "eval": {m: _gate_readouts(vd / "eval" / m) for m in ("nohook", "hook")}, "closed_loop": {m: _closed_loop_readouts(vd / "closed_loop" / m) for m in ("nohook", "hook")}}
        e_no, e_hk = row["eval"]["nohook"], row["eval"]["hook"]
        hf = (meta.get("hazard_free_equivalence") or {}).get("finetuned_no_projection") or {}
        row["flags"] = {
            "consequence_without_projection": (e_no or {}).get("captured", float("nan")) >= CAPTURED_THRESHOLD if e_no else None,
            "consequence_with_projection": (e_hk or {}).get("captured", float("nan")) >= CAPTURED_THRESHOLD if e_hk else None,
            "hazard_free_preserved": hf.get("within_margin"),
            "reading": None,
        }
        f = row["flags"]
        if row["key"] in ("conceptor", "armdiff", "random") and e_hk is not None and (meta.get("epochs") or 0) > 0:
            f["reading"] = ("re-routed: consequence survives WITH the projection after denial fine-tuning (subspace sufficient, not necessary)" if f["consequence_with_projection"]
                            else "collapsed: consequence absent WITH the projection after denial fine-tuning (not re-routable within the frozen budget)") + ("" if f["hazard_free_preserved"] else " -- hazard-free loss outside the margin: not interpretable")
        rows.append(row)
    ablations = {}
    for ad in sorted((root / "parent" / "eval").glob("hook_*")):
        ablations[ad.name] = _gate_readouts(ad)
    summ = {"protocol": PROTOCOL, "disclaimer": C3_DISCLAIMER, "captured_threshold": CAPTURED_THRESHOLD, "equivalence_margin": EQUIV_MARGIN, "parent": parent, "parent_with_projection": ablations, "variants": rows}
    (root / "denial_summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    lines = ["# Denial retraining summary", "", C3_DISCLAIMER, "", f"Parent (no projection): captured {_fmt((parent or {}).get('captured'))}, flip {_fmt((parent or {}).get('flip'))}, spec {_fmt((parent or {}).get('spec'))}, ghost captured {_fmt(((parent or {}).get('ghost') or {}).get('captured'))}", "",
             "## Parent WITH the projection (inference-time ablation, no fine-tuning)", "", "| ablation | captured | ghost captured | flip | spec | verdict |", "|---|---|---|---|---|---|"]
    for n, g in ablations.items():
        lines.append(f"| {n} | {_fmt((g or {}).get('captured'))} | {_fmt(((g or {}).get('ghost') or {}).get('captured'))} | {_fmt((g or {}).get('flip'))} | {_fmt((g or {}).get('spec'))} | {(g or {}).get('verdict')} |")
    lines += ["", "## Fine-tuned variants", "", "| variant | site | key | k | grad ok | HF loss rel diff (no proj) | captured no-proj | captured WITH proj | ghost no-proj / with | flip no / with | T1c' p no / with | closed-loop (no / with) | reading |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e_no, e_hk = r["eval"]["nohook"] or {}, r["eval"]["hook"] or {}
        hf = (r["hazard_free_equivalence"] or {}).get("finetuned_no_projection") or {}
        cl = r["closed_loop"]
        cls = " / ".join(_fmt(((cl.get(m) or {}).get("rates") or {}).get("summary", (cl.get(m) or {}).get("verdict"))) if cl.get(m) else "-" for m in ("nohook", "hook"))
        lines.append(f"| {r['variant']} | {r['site']} | {r['key']} | {r['k']} | {r['grad_check_ok']} | {_fmt(hf.get('rel_diff'))} | {_fmt(e_no.get('captured'))} | {_fmt(e_hk.get('captured'))} | {_fmt((e_no.get('ghost') or {}).get('captured'))} / {_fmt((e_hk.get('ghost') or {}).get('captured'))} | "
                     f"{_fmt(e_no.get('flip'))} / {_fmt(e_hk.get('flip'))} | {_fmt((e_no.get('t1c_prime') or {}).get('p'))} / {_fmt((e_hk.get('t1c_prime') or {}).get('p'))} | {cls} | {r['flags'].get('reading') or ''} |")
    (root / "denial_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    return str(v)


# --------------------------------------------------------------------------- #
# Self-test: tiny AdaLN-style predictor with a planted interaction direction
# --------------------------------------------------------------------------- #


class TinyBlock(torch.nn.Module):
    """Residual block with the AdaLN entry of the vendor block (shift / scale / gate on the MLP) and no attention."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.Identity()
        self.attn = torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 4 * width), torch.nn.GELU(), torch.nn.Linear(4 * width, width))
        self.adaLN_modulation = torch.nn.Linear(width, 3 * width)
        torch.nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # x [B, N, W], z [B, N, W] (modulation per token position)
        shift, scale, gate = self.adaLN_modulation(z).chunk(3, dim=-1)
        return x + (1.0 + gate) * self.mlp(self.norm2(x) * (1.0 + scale) + shift)


class TinyPredictor(torch.nn.Module):
    """[B, T, 1, G, G, D] + actions [B, T, A] -> (out [B, T, N, D], None, None), per frame; frozen orthogonal output projection."""

    def __init__(self, dim: int, width: int, depth: int, action_dim: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.predictor_embed = torch.nn.Linear(dim, width)
        self.action_encoder = torch.nn.Linear(action_dim, width)
        self.predictor_blocks = torch.nn.ModuleList([TinyBlock(width) for _ in range(depth)])
        self.predictor_norm = torch.nn.Identity()
        self.predictor_proj = torch.nn.Linear(width, dim, bias=False)
        Q, _ = torch.linalg.qr(torch.randn(dim, width, generator=g))
        with torch.no_grad():
            self.predictor_proj.weight.copy_(Q[:, :width])
        self.predictor_proj.weight.requires_grad_(False)

    def forward(self, feats: torch.Tensor, acts: torch.Tensor):
        b, t, v, g1, g2, d = feats.shape
        n = v * g1 * g2
        x = self.predictor_embed(feats.reshape(b, t * n, d))
        z = self.action_encoder(acts).unsqueeze(2).expand(b, t, n, -1).reshape(b, t * n, -1)
        for blk in self.predictor_blocks:
            x = blk(x, z)
        out = self.predictor_proj(self.predictor_norm(x))
        return out.reshape(b, t, n, d), None, None


def tiny_synthetic(n: int, t: int, grid: int, dim: int, seed: int, cue: float = 1.5, drift: float = 1.0, inter: float = 1.0, noise: float = 0.05, hazard_fraction: float = 0.5, rank: int = 8):
    """Tokens in a rank-``rank`` content subspace; hazard clips carry the cue ``w``; z_{t+1} = z_t + a d_a + hazard * a * v + noise."""
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.normal(size=(dim, rank + 3)))[0]
    content, w, d_a, v = B[:, :rank], B[:, rank], B[:, rank + 1], B[:, rank + 2]
    N = grid * grid
    hazard = rng.uniform(size=n) < hazard_fraction
    actions = rng.uniform(-1.0, 1.0, size=(n, t - 1, 1)).astype(np.float32)
    base = (rng.normal(size=(n, N, rank)) @ content.T).astype(np.float32)
    feats = np.empty((n, t, grid, grid, dim), dtype=np.float16)
    z = base + cue * hazard[:, None, None] * w[None, None, :]
    feats[:, 0] = z.reshape(n, grid, grid, dim)
    for s in range(t - 1):
        a = actions[:, s, 0][:, None, None]
        z = z + drift * a * d_a[None, None, :] + inter * (hazard[:, None, None] * a) * v[None, None, :] + noise * rng.normal(size=z.shape)
        feats[:, s + 1] = z.reshape(n, grid, grid, dim)
    n_val = max(16, n // 8)
    clips = [{"clip_id": f"tiny_{i:05d}", "hazard": bool(hazard[i]), "split": "val" if i >= n - n_val else "train"} for i in range(n)]
    store = tfp.FeatureStore(feats, actions, clips, {}, source=f"tiny-synthetic(seed={seed})")
    store.validate()
    return store, {"w": w, "d_a": d_a, "v": v, "base": base, "hazard": hazard, "cue": cue, "drift": drift, "inter": inter, "grid": grid}


@torch.no_grad()
def tiny_captured(predictor: torch.nn.Module, syn: dict[str, Any], idx: np.ndarray, site: str | None = None) -> dict[str, float]:
    """Factorial hazard x action on held-out clips: per-clip NMSE of the predicted vs true next-state DiD (B-gate T1b
    style), captured = 1 - median NMSE (floored at 0); optionally the DiD rows at ``site`` (last-frame tokens pooled)."""
    predictor.eval()
    g, N = syn["grid"], syn["grid"] ** 2
    base = torch.from_numpy(syn["base"][idx])  # [n, N, D] without cue
    w, v = torch.from_numpy(syn["w"]).float(), torch.from_numpy(syn["v"]).float()
    preds, site_acts = {}, {}
    rec = None
    if site is not None:
        from predictor_hooks import PredictorRecorder
        rec = PredictorRecorder(predictor, [Site.parse(site)], N)
    for h in (0, 1):
        for a in (-1.0, 1.0):
            z0 = base + syn["cue"] * h * w
            feats = z0.reshape(len(idx), 1, 1, g, g, -1)
            acts = torch.full((len(idx), 1, 1), a)
            if rec is not None:
                with rec:
                    out, _, _ = predictor(feats, acts)
                site_acts[(h, a)] = rec.acts[site][rec.step].mean(1)  # [n, d] pooled over tokens
                rec.step = -1
            else:
                out, _, _ = predictor(feats, acts)
            preds[(h, a)] = out[:, 0]  # [n, N, D] predicted next state
    i_pred = preds[(1, 1.0)] - preds[(1, -1.0)] - preds[(0, 1.0)] + preds[(0, -1.0)]
    i_true = 2.0 * syn["inter"] * v[None, None, :].expand_as(i_pred)
    nmse = ((i_pred - i_true) ** 2).sum((1, 2)) / (i_true**2).sum((1, 2))
    out = {"nmse_median": float(nmse.median()), "captured": float(max(0.0, 1.0 - nmse.median().item())), "n": int(len(idx))}
    if rec is not None:
        did = site_acts[(1, 1.0)] - site_acts[(1, -1.0)] - site_acts[(0, 1.0)] + site_acts[(0, -1.0)]
        _, _, vt = torch.linalg.svd(did, full_matrices=False)
        out["_did_top1"] = vt[0].numpy()
    return out


def self_test(out: Path, seed: int = 0) -> dict[str, Any]:
    try:
        import yaml  # noqa: F401
    except ImportError:  # the trainer writes two yaml files; a json shim keeps the self-test dependency-free
        import types

        sys.modules["yaml"] = types.SimpleNamespace(safe_dump=lambda obj, **kw: json.dumps(obj, indent=1, default=str), safe_load=json.loads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dim, width, grid, t, depth = 32, 32, 2, 3, 2
    device = torch.device("cpu")
    store, syn = tiny_synthetic(n=512, t=t, grid=grid, dim=dim, seed=seed)
    cfg = tfp.ModelCfg(pred_depth=depth, pred_embed_dim=width, pred_num_heads=1, embed_dim=dim, grid_size=grid, img_size=grid * 16, num_frames_pred=t, action_dim=1)
    out.mkdir(parents=True, exist_ok=True)
    pre_epochs = 12
    ft_epochs = 3  # 25 % of the pre-training budget
    lr = 3e-3
    # 1. pre-train the tiny model (both "arms" of the analogy collapse to one: the parent model)
    parent = TinyPredictor(dim, width, depth, 1, seed).to(device)
    a0 = trainer_args(out / "parent", None, None, pre_epochs, lr, 1.0, 1e-4, seed, 32, "tiny_parent", ["--dtype", "float32", "--eval-ctxt-window", "2"])
    tfp.train(a0, parent, store, device, cfg)
    train_idx, val_idx, _ = tfp.split_indices(store, 0.1, seed)
    last_site, first_site = f"L{depth - 1:02d}.resid_post", "L00.resid_post"
    u = (parent.predictor_proj.weight.detach().T @ torch.from_numpy(syn["v"]).float()).numpy()  # planted internal direction at the last residual output
    u = u / np.linalg.norm(u)
    U = u[:, None].astype(np.float32)
    Ur = load_subspace(None, "random", 1, d=width, random_seed=seed + 7)[0]
    res: dict[str, Any] = {"parent": tiny_captured(parent, syn, val_idx, site=last_site)}
    disc = res["parent"].pop("_did_top1")
    res["discovery_cos_with_planted"] = float(abs(disc @ u))
    with SubspaceDenial(parent, last_site, U):
        res["parent_hook_planted"] = tiny_captured(parent, syn, val_idx)
    with SubspaceDenial(parent, last_site, Ur):
        res["parent_hook_random"] = tiny_captured(parent, syn, val_idx)
    # identity hook (k = 0): bit-exact
    feats, acts = tfp.fetch_batch(store, val_idx[:8], device)
    with torch.no_grad():
        ref = parent(feats, acts)[0]
        with SubspaceDenial(parent, last_site, None):
            same = parent(feats, acts)[0]
    res["identity_hook_bit_exact"] = bool(torch.equal(ref, same))
    parent_sd = {k: v.clone() for k, v in parent.state_dict().items()}

    def variant(name: str, site: str, Uv: np.ndarray | None, key: str) -> dict[str, Any]:
        m = TinyPredictor(dim, width, depth, 1, seed).to(device)
        m.load_state_dict(parent_sd)
        a = trainer_args(out / "variants" / name, None, None, ft_epochs, lr / 3, 1.0, 1e-4, seed, 32, f"tiny_{name}", ["--dtype", "float32", "--eval-ctxt-window", "2"])
        info = {"key": key, "k": 0 if Uv is None else int(Uv.shape[1])}
        meta = finetune(m, store, device, cfg, a, site, Uv, info, {"model": "tiny_parent"})
        r = {"grad_check": meta["grad_check"], "hazard_free_equivalence": meta["hazard_free_equivalence"], "captured_no_projection": tiny_captured(m, syn, val_idx)}
        with SubspaceDenial(m, site, Uv):
            r["captured_with_projection"] = tiny_captured(m, syn, val_idx)
        # reload the saved checkpoint through a fresh module: the saved weights are plain (no hook baked in)
        m2 = TinyPredictor(dim, width, depth, 1, seed)
        m2.load_state_dict(torch.load(out / "variants" / name / "jepa-latest.pth.tar", map_location="cpu", weights_only=True)["predictor"])
        r["reloaded_captured_no_projection"] = tiny_captured(m2, syn, val_idx)
        return r

    res["denial_planted_last"] = variant("denial_planted_last", last_site, U, "conceptor")
    res["continued"] = variant("continued", last_site, None, "none")
    res["denial_random_last"] = variant("denial_random_last", last_site, Ur, "random")
    # re-routing example: deny the planted direction at the FIRST block, where the second block can take over
    res["denial_planted_first"] = variant("denial_planted_first", first_site, U, "conceptor")
    cap = lambda r: r["captured"]  # noqa: E731
    # the tiny parent is not converged, so continued training keeps LOWERING the hazard-free loss; the self-test therefore
    # checks the signed degradation (rel_increase <= 25 %), the real run reports the frozen two-sided 5 % P2 margin
    hf_ok = lambda r: bool((r["hazard_free_equivalence"].get("finetuned_no_projection") or {}).get("rel_increase", 9.0) <= 0.25)  # noqa: E731
    checks = {
        "parent_captures_ge_0.5": cap(res["parent"]) >= 0.5,
        "parent_hook_planted_le_0.05": cap(res["parent_hook_planted"]) <= 0.05,
        "parent_hook_random_ge_0.8x_parent": cap(res["parent_hook_random"]) >= 0.8 * cap(res["parent"]),
        "denial_with_projection_le_0.05": cap(res["denial_planted_last"]["captured_with_projection"]) <= 0.05,
        "continued_keeps_ge_0.5": cap(res["continued"]["captured_no_projection"]) >= 0.5,
        "random_denial_with_projection_ge_0.4": cap(res["denial_random_last"]["captured_with_projection"]) >= 0.4,
        "grad_along_U_at_float_precision": bool(res["denial_planted_last"]["grad_check"]["ok"]) and bool(res["denial_random_last"]["grad_check"]["ok"]),
        "identity_hook_bit_exact": res["identity_hook_bit_exact"],
        "discovery_recovers_planted_cos_ge_0.5": res["discovery_cos_with_planted"] >= 0.5,
        "hazard_free_not_degraded_gt_25pct": all(hf_ok(res[k]) for k in ("denial_planted_last", "continued", "denial_random_last")),
        "reloaded_checkpoint_matches_no_projection": abs(cap(res["denial_planted_last"]["reloaded_captured_no_projection"]) - cap(res["denial_planted_last"]["captured_no_projection"])) < 1e-6,
    }
    res["checks"] = checks
    res["passed"] = all(checks.values())
    res["notes"] = {"rerouting_example_first_block_with_projection": cap(res["denial_planted_first"]["captured_with_projection"]),
                    "test_time_removal_after_denial_last": cap(res["denial_planted_last"]["captured_no_projection"]),
                    "reading": "denial at the last residual output cannot be re-routed by construction (frozen orthogonal output projection); denial at the first block can be re-routed by the second block -- reported, not asserted"}
    (out / "self_test_verdict.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit-subspaces", help="conceptor / arm-diff subspaces at sites from the two localization dumps")
    f.add_argument("--dump-a", type=Path, required=True)
    f.add_argument("--dump-b", type=Path, required=True)
    f.add_argument("--stimulus", type=Path, required=True)
    f.add_argument("--discovery-seeds", type=Path, default=None)
    f.add_argument("--sites", nargs="+", default=[DEFAULT_SITE, WRONG_SITE_DEFAULT])
    f.add_argument("--group", default=DEFAULT_GROUP)
    f.add_argument("--step", type=int, default=DEFAULT_STEP)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--k-action", type=int, default=4)
    f.add_argument("--target-dims", type=float, default=4.0)
    f.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("finetune", help="denial fine-tuning of a trained model (or --epochs 0 for the val-loss ablation only)")
    t.add_argument("--model-dir", type=Path, required=True)
    t.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    t.add_argument("--repo", type=Path, required=True)
    t.add_argument("--features", type=Path, required=True)
    t.add_argument("--meta", type=Path, default=None)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--site", default=DEFAULT_SITE)
    t.add_argument("--subspace", type=Path, default=None)
    t.add_argument("--key", choices=KEYS, default="conceptor")
    t.add_argument("--k", type=int, default=4)
    t.add_argument("--d", type=int, default=None)
    t.add_argument("--random-seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=7, help="frozen: <= 25 %% of the original 30 epochs")
    t.add_argument("--ref-lr", type=float, default=2e-4)
    t.add_argument("--warmup-epochs", type=float, default=1.0)
    t.add_argument("--final-lr", type=float, default=1e-5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--label", default="denial")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run-with-hook", help="run latent_cache / closed_loop_rollout with the projection installed at load time")
    r.add_argument("--script", choices=["latent_cache", "closed_loop_rollout"], required=True)
    r.add_argument("--site", default=DEFAULT_SITE)
    r.add_argument("--subspace", type=Path, default=None)
    r.add_argument("--key", choices=KEYS, default="conceptor")
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--d", type=int, default=None)
    r.add_argument("--random-seed", type=int, default=0)
    r.add_argument("script_args", nargs=argparse.REMAINDER, help="arguments for the wrapped script (after --)")
    s = sub.add_parser("summarize")
    s.add_argument("--root", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.self_test:
        out = a.out or Path("denial_self_test")
        t0 = time.time()
        res = self_test(out, a.seed)
        print(json.dumps({k: res[k] for k in ("checks", "passed", "notes", "discovery_cos_with_planted")}, indent=1, default=str))
        print(json.dumps({"parent": res["parent"]["captured"], "parent_hook_planted": res["parent_hook_planted"]["captured"], "parent_hook_random": res["parent_hook_random"]["captured"],
                          "denial_last_with": res["denial_planted_last"]["captured_with_projection"]["captured"], "denial_last_without": res["denial_planted_last"]["captured_no_projection"]["captured"],
                          "continued": res["continued"]["captured_no_projection"]["captured"], "random_with": res["denial_random_last"]["captured_with_projection"]["captured"],
                          "first_block_with": res["denial_planted_first"]["captured_with_projection"]["captured"], "runtime_s": time.time() - t0}))
        print("SELF_TEST_PASSED" if res["passed"] else "SELF_TEST_FAILED")
        if not res["passed"]:
            raise SystemExit(1)
        return
    if a.cmd == "fit-subspaces":
        for site in a.sites:
            m = fit_site_subspaces(a.dump_a, a.dump_b, a.stimulus, a.discovery_seeds, site, a.group, a.step, a.out_dir / f"subspace_{site}__s{a.step}__{a.group}.npz", a.k_action, a.target_dims, a.seed)
            print(json.dumps({k: m[k] for k in ("site", "n_scenes", "d", "conceptor", "armdiff", "overlaps")}, default=float))
    elif a.cmd == "finetune":
        if "script_args" in a:
            del a.script_args
        cmd_finetune(a)
    elif a.cmd == "run-with-hook":
        a.script_args = [x for x in a.script_args if x != "--"]
        cmd_run_with_hook(a)
    elif a.cmd == "summarize":
        cmd_summarize(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
