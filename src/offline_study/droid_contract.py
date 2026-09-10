"""Audited DROID planning/score contract; preparation grants no outcome access."""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
from pathlib import Path

import torch
import yaml

from . import VENDOR_COMMIT
from .droid_assets import CHECKPOINT_SHA256, validate_manifest
from .planning_contract import seed_schedule
from .protocol import sha256, write_json


TRAIN_CONFIG = "configs/vjepa_wm/droid_final_sweep/droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml"
EVAL_CONFIG = "configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml"
SOURCE_FILES = (TRAIN_CONFIG, EVAL_CONFIG, "app/plan_common/datasets/droid_dset.py",
    "evals/simu_env_planning/planning/plan_evaluator.py", "evals/simu_env_planning/eval.py",
    "evals/simu_env_planning/planning/utils.py", "app/vjepa_wm/train.py",
    "evals/simu_env_planning/envs/droid_dset_dummy_env.py",
    "app/plan_common/plot/logs_plan_joint_per_design_choice.py")


def action_metrics(planned, recorded):
    """Match upstream Act_err_xyz: abs AFTER summing the three planned deltas.

    Not mean per-step absolute error, and never the dummy environment's success.
    Preserve input precision/promotion, as in the upstream tensor subtraction.
    """
    planned, recorded = torch.as_tensor(planned), torch.as_tensor(recorded)
    if planned.shape != recorded.shape or planned.ndim not in (2, 3) or planned.shape[-2:] != (3, 7):
        raise ValueError("Require paired three-step, seven-dimensional DROID actions")
    if not planned.is_floating_point() or not recorded.is_floating_point():
        raise ValueError("DROID action measurements must be floating point")
    if not torch.isfinite(planned).all() or not torch.isfinite(recorded).all():
        raise ValueError("Nonfinite DROID actions; do not discard observations")
    delta = (planned.sum(-2) - recorded.sum(-2)).abs()
    xyz = delta[..., :3].sum(-1)
    return {"action_error_xyz": xyz,
            "action_error_orientation": delta[..., 3:6].sum(-1),
            "action_error_gripper": delta[..., 6:].sum(-1)}


def checkpoint_score(episode_errors, expected_episodes=64):
    """Official order: mean episode XYZ error, then clip/scale once per checkpoint.

    eval.aggregate_results writes the episode mean to eval.csv; the plotting
    script transforms that single value. Mean of individually clipped scores
    would be a different, optimistic statistic when any episode error exceeds .1.
    """
    errors = torch.as_tensor(episode_errors)
    if (errors.ndim != 1 or errors.numel() != expected_episodes or
            not errors.is_floating_point() or not torch.isfinite(errors).all() or
            (errors < 0).any()):
        raise ValueError("Require every finite nonnegative episode error exactly once")
    # Upstream episode tensors become Python floats, then NumPy aggregates them.
    return (800 * (.1 - errors.double().mean())).clamp_min(0)


def late_checkpoints(epochs=315, cadence=6, start_from_epoch=215):
    """One-based checkpoint metadata, including native final/latest evaluation.

    train.py tests cadence with zero-based loop indices, but passes epoch+1 to
    checkpoint metadata and the epoch-N evaluation directory/plotting code.
    """
    return [index + 1 for index in range(epochs)
            if (index % cadence == 0 or index == epochs - 1) and index + 1 >= start_from_epoch]


def prepare(vendor, manifest):
    validate_manifest(manifest)
    train = yaml.safe_load((vendor / TRAIN_CONFIG).read_text())
    source = yaml.safe_load((vendor / EVAL_CONFIG).read_text())
    cfg = copy.deepcopy(source)
    planner, task = cfg["planner"], cfg["task_specification"]
    expected = {"planner_name": "cem", "iterations": 15, "num_samples": 300,
        "num_elites": 10, "horizon": 3, "num_act_stepped": 3, "var_scale": .1,
        "max_norms": [.1, .75], "max_norm_dims": [[0, 1, 2, 3, 4, 5], [6]],
        "momentum_mean": 0., "momentum_std": 0., "repeat_actskip": False,
        "distribute_planner": False,
        "planning_objective": {"objective_type": "L2", "sum_all_diffs": False, "alpha": 0}}
    if any(planner.get(key) != value for key, value in expected.items()):
        raise ValueError("DROID CEM or action-norm groups changed")
    if (cfg["meta"] != {"quick_debug": False, "seed": 1, "eval_episodes": 64} or
            task["task"] != "droid-base" or task["goal_source"] != "dset" or
            task["goal_H"] != 3 or task["img_size"] != 256 or task["obs"] != "rgb"):
        raise ValueError("DROID population, goal or evaluation budget changed")
    model = cfg["model_kwargs"]
    data = model["data"]
    pretrain = model["pretrain_kwargs"]
    if (pretrain["visual_encoder"]["enc_version"] != "dinov3_vitl16" or
            pretrain["predictor"]["pred_depth"] != 12 or
            pretrain["predictor"]["pred_embed_dim"] != 1024 or
            pretrain["proprio_encoding"] != "none" or
            model["wrapper_kwargs"]["ctxt_window"] != 2 or
            data["custom"]["frameskip"] != 1 or data["custom"]["normalize_action"] is not False or
            data["droid"]["fps"] != 4 or
            data["validation"]["val_dataset_camera_views"] != ["exterior_image_2_left"] or
            data["validation"]["val_dataset_fpcs"] != [5]):
        raise ValueError("DROID architecture or recorded-stimulus preprocessing changed")
    opt = train["optimization"]["transition_model"]
    if (opt["num_epochs"] != 315 or opt["iterations_per_epoch"] != 300 or
            opt["betas"] != [.9, .995] or train["meta"]["save_every_freq"] != 6 or
            train["meta"]["eval_freq"] != 6):
        raise ValueError("DROID training/cadence differs from audited release")
    patterns = data["droid"]["mpk_manifest_patterns"]
    if patterns != train["data"]["droid"]["mpk_manifest_patterns"]:
        raise ValueError("Train-monitoring and evaluation recording selections disagree")
    files = [entry["filename"] for entry in manifest["assets"] if entry["kind"] == "evaluation_recording"]
    matched = []
    for pattern in patterns:
        matches = [name for name in files if fnmatch.fnmatchcase(name, pattern)]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous source recording pattern")
        matched.extend(matches)
    if len(set(matched)) != 15 or len(matched) != 15:
        raise ValueError("Released config's 15-recording selection changed")
    # Only remove optional visualization, not planner budgets or observations.
    cfg["logging"].update(optional_plots=False, tqdm_silent=True)
    cfg["planner"]["decode_each_iteration"] = False
    cfg["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] = {}
    cfg.update(frameskip=1, tasks=["droid-base"])
    cfg["task_specification"]["multitask"] = False
    return {"schema_version": 1, "task": "droid", "vendor_commit": VENDOR_COMMIT,
        "status": "prepared_not_executed_or_confirmation_frozen",
        "source_files_sha256": {name: sha256(vendor / name) for name in SOURCE_FILES},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "endpoint": "score_of_checkpoint_mean_episode_xyz_error_not_robot_task_success",
        "action_error_formula": "sum_xyz(abs(sum_time(planned)-sum_time(recorded)))",
        "score_formula": "max(0,800*(0.1-mean_episode_action_error_xyz)); then average checkpoint scores across late epochs/seeds",
        "dummy_environment_success_must_not_be_reported": True,
        "planning_replication_episodes_per_checkpoint_condition": 64,
        "released_config_recordings": matched,
        "paper_recording_count": 16,
        "downloaded_manifest_recording_count": len(files),
        "unselected_released_recordings": sorted(set(files) - set(matched)),
        "recording_discrepancy_resolved": False,
        "paper_population_exactly_matched": False,
        "stimulus": {"camera": "exterior_image_2_left", "data_rng_seed": 234,
            "source_fps_assumed_by_loader": 30, "configured_fps": 4,
            "raw_frame_stride": 8, "effective_sampling_fps": 3.75,
            "sampled_frames_before_goal_segment": 5, "goal_segment_frames": 4,
            "actions": "native poses_to_diffs of measured sampled poses; no action normalization",
            "preserve_dataset_initialization_rng_consumption": True,
            "record_source_file_and_raw_frame_indices": True,
            "silent_replacement_on_decode_error_allowed": False},
        "seed_policy": {"base_seed": 1, "logical_ranks": 8,
            "local_seed": "1 + logical_rank*3000",
            "dataset_rng": "separate numpy RandomState(234), including native constructor consumption",
            "cpu_scenario_and_cuda_cem_rng": "separate generators with local_seed; keep stream order",
            "paired_condition_rng_and_stimulus": True,
            "authors_exact_realized_stimuli_reconstructed": False},
        "episodes": seed_schedule(1, 64, 8, 3),
        "training": {"source_seed": train["meta"]["seed"],
            "reproduction_seeds": [234, 235, 236], "authors_exact_seed_triplet_available": False,
            "global_batch": train["nodes"] * train["tasks_per_node"] * data["loader"]["batch_size"],
            "epochs": 315, "updates_per_epoch": 300, "updates_per_seed": 94500,
            "source_save_eval_every_epochs": 6, "late_window_start_inclusive": 215,
            "required_late_epoch_metadata": late_checkpoints(),
            "late_epoch_count": len(late_checkpoints()), "training_histories_available": False,
            "aggregation_basis": "released rebuttal start_from_epoch logic explicitly permits sparse checkpoints in [215,315]",
            "required_change": "preserve final latest checkpoint immutably; no dense 100-checkpoint evaluation required",
            "training_dataset_manifest_status": "paper_8000_subset_identity_not_yet_reconciled"},
        "fresh_confirmation": False, "model_runs": 0, "config": cfg,
        "required_launch_gates": ["verified raw files and DINOv3 encoder",
            "native source/frames/actions/metric parity", "recording population decision",
            "DROID-specific fit population separate from evaluation recordings",
            "frozen applicable intervention panel; no silent H6 adapter no-op at H3",
            "verified exposure and paired stimuli", "US-only aggregate hourly budget"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    from .vendor import use_vendor
    use_vendor(args.vendor)
    contract = prepare(args.vendor, json.loads(args.manifest.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "contract.json", contract)
    write_json(args.output / "PREPARED.json", {"contract_sha256": sha256(args.output / "contract.json"),
        "jobs_launched": False, "confirmation_authorized": False})


if __name__ == "__main__":
    main()
