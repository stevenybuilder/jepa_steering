"""Published offline context/clip evaluation, with explicit protected-row exclusions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from .author_validation import CONFIGS, audit, official_partition, require_disjoint_fit
from .backends import JepaBackend
from .operator_fit import FIT_SEED, select_fit_rows
from .protocol import sha256, write_json
from .vendor import use_vendor


def build_cohort(rows, task, config, registry, manifest_hash):
    """Keep historical protection; new validation roles never overwrite old records."""
    if registry.get("manifest_sha256") != manifest_hash:
        raise ValueError("Registry does not bind the source manifest")
    kind = "pusht" if task == "pusht" else "metaworld"
    preflight = audit(rows, kind, config, registry)
    train, val = official_partition(rows, kind, config["data"]["seed"],
                                    config["data"]["custom"]["split_ratio"])
    entries = registry["trajectories"]
    protected = {row["lineage_group"] for row in rows if row["split"] == "holdout" or
                 entries.get(row["trajectory_id"], {}).get("use") == "protected"}
    val_groups = {row["lineage_group"] for row in val}
    eligible_fit = [row for row in train if row["split"] == "fit" and
                    row["lineage_group"] not in protected | val_groups]
    fit = select_fit_rows(eligible_fit, task, 128, FIT_SEED)
    evaluation = [row for row in val if row["task"] == task and row["lineage_group"] not in protected]
    excluded = [row for row in val if row["task"] == task and row["lineage_group"] in protected]
    if len(fit) != 128:
        raise ValueError("Expected 128 unprotected fit families")
    require_disjoint_fit(fit, val)
    return {"schema_version": 1, "task": task, "dataset": kind,
            "source_manifest_sha256": manifest_hash, "fit": fit, "evaluation": evaluation,
            "protected_official_validation": excluded,
            "all_author_validation_groups": sorted(val_groups),
            "all_protected_groups": sorted(protected),
            "coverage": {"evaluated_rows": len(evaluation), "official_rows": len(evaluation) + len(excluded),
                         "complete_author_split": not excluded},
            "reference_config": config,
            "role": "protected_excluded_author_validation_development_replication",
            "evaluation_permission": "unprotected_rows_only" if evaluation else "none_fit_only",
            "fresh_confirmation": False,
            "access_review": "User requested protected-preserving correction; no protected groups opened. "
                "Unprotected official validation is development/replication, never fresh confirmation. "
                "Original fit/development/external-reserve labels and exposure registry remain preserved.",
            "fit_sampling": "128 seed-fixed old-fit representatives; four evenly spaced official clip/prefix pairs per family",
            "evaluation_sampling": "all eligible author clips and all H6 prefixes; reordered by row only for decode reuse",
            "primary_precision": "bfloat16", "secondary_precision": "float32",
            "precision_policy": "Both fixed before outcomes; report separately, never select the favorable precision",
            "preflight_counts": {k: preflight[k] for k in ("validation_rows", "validation_lineage_groups", "validation_clips")}}


def validate_cohort(cohort):
    from .author_access import validate_access
    fit, evaluation = cohort["fit"], cohort["evaluation"]
    protected = set(cohort["all_protected_groups"])
    all_val = set(cohort["all_author_validation_groups"])
    authorized_replication = validate_access(cohort)
    if any(row["lineage_group"] in protected for row in fit) or (
            not authorized_replication and any(row["lineage_group"] in protected for row in evaluation)):
        raise ValueError("Protected family cannot enter fit or development")
    if any(row["lineage_group"] in all_val or row["split"] != "fit" for row in fit):
        raise ValueError("Fitting population is not disjoint from official validation")
    if any(row["lineage_group"] not in all_val for row in evaluation):
        raise ValueError("Evaluation is outside official validation")
    require_disjoint_fit(fit, evaluation)
    if len({row["lineage_group"] for row in fit}) != 128:
        raise ValueError("Fit family count differs from frozen selection")
    for role, rows in (("fit", fit), ("evaluation", evaluation)):
        if len({row["trajectory_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate trajectory identities in a cohort role")
        if any(row["task"] != cohort["task"] or
               (row["split"] == "holdout" and not (role == "evaluation" and authorized_replication)) for row in rows):
            raise ValueError("Unexpected task or protected original split")
    coverage = cohort["coverage"]
    reused = len(cohort.get("reused_evaluation", [])) if authorized_replication else 0
    if coverage["evaluated_rows"] != len(evaluation) or coverage["official_rows"] != len(evaluation) + len(cohort["protected_official_validation"]) + reused:
        raise ValueError("Coverage receipt differs from actual cohort")


def examples(rows, config, fitting=False, role="development"):
    frames = config["data"]["validation"]["num_frames_val"]
    stride = config["data"]["custom"]["frameskip"]
    prefixes = frames - 6
    result = []
    for row in rows:
        pairs = [(start, prefix) for start in range(row["length"] - frames * stride + 1)
                 for prefix in range(prefixes)]
        if fitting:
            pairs = [pairs[i] for i in sorted({round(j * (len(pairs) - 1) / 3) for j in range(4)})]
        for start, prefix in pairs:
            result.append({"trajectory_id": row["trajectory_id"], "lineage_group": row["lineage_group"],
                "task": row["task"], "clip_start": start, "prefix": prefix,
                # Collision-free key, NOT a physical frame number.
                "start": start * prefixes + prefix, "physical_cut_frame": start + stride * prefix,
                "split": "fit" if fitting else role})
    return result


class AuthorBackend(JepaBackend):
    """Use the official core, not the two-context planning wrapper."""
    def __init__(self, vendor, checkpoint, checkpoint_hash, kind, device, precision):
        super().__init__(vendor, checkpoint, checkpoint_hash, kind, device, precision)
        self.core = self.model.model
        if self.core.proprio_rollout_mode != "predict_proprio":
            raise ValueError("Refusing any rollout that supplies future ground-truth proprioception")
        self.provenance.update(offline_context=3, wrapper_unroll_used=False,
                               metrics="visual/proprio embedding losses; NOT decoded physical state")

    def encode_clip(self, obs, actions):
        obs = {key: value.to(self.device) for key, value in obs.items()}
        with self.autocast():
            visual, proprio, action = self.core.encode(obs, actions.to(self.device))
        return {"visual": visual, "proprio": proprio, "action": action}

    def context_at(self, encoded, prefix):
        start = max(0, prefix - 2)
        return {"visual": encoded["visual"][:, start:prefix + 1],
                "proprio": encoded["proprio"][:, start:prefix + 1],
                "past_actions": encoded["action"][:, start:prefix]}

    def expand_context(self, context, repeats):
        return {key: value.repeat_interleave(repeats, dim=0) for key, value in context.items()}

    def predict(self, context, actions, instrument=False):
        if instrument or actions.shape[0] != 6:
            raise ValueError("Author adapter expects an H6 recorded-action rollout")
        visual, proprio, past = context["visual"], context["proprio"], context["past_actions"]
        if visual.shape[1] != past.shape[1] + 1:
            raise ValueError("Misaligned observed context and past actions")
        vs, ps = [visual[:, -1]], [proprio[:, -1]]
        with self.autocast():
            for action in actions:
                past = torch.cat([past, action[:, None]], 1)
                v, _, p = self.core.forward_pred(visual[:, -3:].detach(), past[:, -3:],
                                                 proprio[:, -3:].detach(), debug=True)
                visual = torch.cat([visual.detach(), v[:, -1:]], 1)
                proprio = torch.cat([proprio.detach(), p[:, -1:]], 1)
                vs.append(v[:, -1])
                ps.append(p[:, -1])
        return {"visual": torch.stack(vs), "proprio": torch.stack(ps)}

    def verify_reference(self, encoded):
        receipts = []
        for prefix in sorted({0, 1, encoded["visual"].shape[1] - 7}):
            context = self.context_at(encoded, prefix)
            pred = self.predict(context, encoded["action"][:, prefix:prefix + 6].transpose(0, 1))
            with self.autocast():
                losses, _, visual, proprio = self.core.rollout(
                    video_features=encoded["visual"], proprio_features=encoded["proprio"],
                    action_features=encoded["action"], pred_video_features=None, pred_proprio_features=None,
                    action_noise=0., rollout_steps=6, rollout_stop_gradient=True,
                    ctxt_window=3, mode="sequential", t=prefix, debug=True)
            for key, expected in (("visual", visual[:, -7:].transpose(0, 1)),
                                  ("proprio", proprio[:, -7:].transpose(0, 1))):
                if not torch.equal(pred[key], expected):
                    raise RuntimeError(f"Author rollout parity failed: {key}, prefix {prefix}")
            scored = self.metrics(pred, {k: encoded[k][:, prefix:prefix + 7] for k in ("visual", "proprio")})
            for modality in ("visual", "proprio"):
                for loss_type, label in (("l1", "l1"), ("l2", "mse")):
                    for h in range(1, 7):
                        value = torch.tensor([row[f"{modality}_{label}_h{h}"] for row in scored]).mean()
                        expected = losses[f"{modality}_{loss_type}_loss"][h - 1].float().cpu()
                        if not torch.allclose(value, expected, atol=1e-7, rtol=2e-5):
                            raise RuntimeError(f"Author metric parity failed: {modality}/{label}/H{h}")
            receipts.append({"prefix": prefix, "predictions_bitwise_equal": True, "losses_match": True})
        return receipts

    def metrics(self, pred, target):
        columns, labels = [], []
        with self.autocast():
            for h in range(1, 7):
                losses = self.core.compute_loss(pred["visual"][h:h + 1].transpose(0, 1),
                    pred["proprio"][h:h + 1].transpose(0, 1), target["visual"][:, h:h + 1],
                    target["proprio"][:, h:h + 1], shift=0, reduce_mean=False)
                for modality in ("visual", "proprio"):
                    for source, label in (("l1", "l1"), ("l2", "mse")):
                        labels.append(f"{modality}_{label}_h{h}")
                        columns.append(losses[f"{modality}_{source}_loss"].float().flatten(1).mean(1))
        values = torch.stack(columns, 1)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite author metric")
        return [dict(zip(labels, row)) for row in values.cpu().tolist()]


def open_normalized_dataset(kind, root, config, fitting):
    from app.plan_common.datasets.transforms import make_transforms
    transform = make_transforms(img_size=config["data"]["img_size"], **config["data_aug"])
    if kind == "pusht":
        from app.plan_common.datasets.pusht_dset import PushTDataset
        return PushTDataset(data_path=str(root / ("train" if fitting else "val")),
                            n_rollout=None, transform=transform, normalize_action=True, with_velocity=True)
    if kind in ("wall", "pointmaze"):
        from app.plan_common.datasets.wall_dset import WallDataset
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        cls = WallDataset if kind == "wall" else PointMazeDataset
        return cls(data_path=str(root), n_rollout=None, transform=transform, normalize_action=True)
    if kind != "metaworld":
        raise ValueError("Unaudited offline dataset kind")
    from app.plan_common.datasets.metaworld_hf_dset import MetaworldHFDataset
    return MetaworldHFDataset(data_path=str(root), n_rollout=None, transform=transform,
                             normalize_action=True, filter_tasks=None, with_reward=False)


def encoded_batches(backend, dataset, rows, config, fitting=False, batch_size=4, role="development"):
    """Decode each selected row once; encode clips once for all their prefixes."""
    frames = config["data"]["validation"]["num_frames_val"]
    stride = config["data"]["custom"]["frameskip"]
    metadata = examples(rows, config, fitting, role=role)
    by_row_clip = defaultdict(lambda: defaultdict(list))
    for item in metadata:
        by_row_clip[item["trajectory_id"]][item["clip_start"]].append(item)
    pending = []

    def emit(items):
        obs = {k: torch.stack([item[1][k] for item in items]) for k in ("visual", "proprio")}
        actions = torch.stack([item[2] for item in items])
        return [item[0] for item in items], backend.encode_clip(obs, actions)

    for row in rows:
        obs, actions, _, _, _ = dataset[row["index"]]
        if len(actions) != row["length"] or obs["visual"].shape[0] != row["length"]:
            raise ValueError("Actual trajectory length does not match inventory")
        for start, meta in by_row_clip[row["trajectory_id"]].items():
            end = start + frames * stride
            clip = {k: v[start:end:stride] for k, v in obs.items()}
            # Exactly the official TrajSlicer action_skip=1 / concat operation.
            act = actions[start:end].reshape(frames, -1)
            pending.append((meta, clip, act))
            if len(pending) == batch_size:
                yield emit(pending)
                pending = []
    if pending:
        yield emit(pending)


def prefix_batches(backend, clip_metadata, encoded):
    for prefix in range(encoded["visual"].shape[1] - 6):
        selected = [(i, next((meta for meta in group if meta["prefix"] == prefix), None))
                    for i, group in enumerate(clip_metadata)]
        selected = [(i, meta) for i, meta in selected if meta is not None]
        if not selected:
            continue
        indices, meta = zip(*selected)
        subset = {k: v[list(indices)] for k, v in encoded.items()}
        yield (list(meta), backend.context_at(subset, prefix),
               subset["action"][:, prefix:prefix + 6].transpose(0, 1).contiguous(),
               {k: subset[k][:, prefix:prefix + 7] for k in ("visual", "proprio")})


def prepare_main():
    parser = argparse.ArgumentParser()
    for flag in ("vendor", "manifest", "registry", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["mw-reach", "mw-reach-wall", "pusht"])
    args = parser.parse_args()
    vendor = use_vendor(args.vendor)
    kind = "pusht" if args.task == "pusht" else "metaworld"
    config = yaml.safe_load((vendor / CONFIGS[kind]).read_text())
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    cohort = build_cohort(rows, args.task, config, json.loads(args.registry.read_text()), sha256(args.manifest))
    cohort["source_registry_sha256"] = sha256(args.registry)
    cohort["upstream_config_sha256"] = sha256(vendor / CONFIGS[kind])
    validate_cohort(cohort)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cohort.json", cohort)
    print(json.dumps(cohort["coverage"]))


if __name__ == "__main__":
    prepare_main()
