"""Paper-scale planning preparation using pinned upstream task configurations.

Creating this contract does not authorize or launch confirmation. Candidate, access,
native-runtime, and (for confirmation) prospective-family gates remain separate.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from . import VENDOR_COMMIT
from .protocol import sha256, write_json


CONFIGS = {
    "pusht": "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
    "reach": "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
    "reach-wall": "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
}


def seed_schedule(base_seed=1, episodes=96, logical_ranks=8, horizon=6):
    """Upstream rank offsets and episode formula, independent of physical GPU count.

    Each logical stream must be processed in order. CPU dataset and CUDA CEM
    generators are distinct and seeded with local_seed, as in GC_Agent.
    """
    if episodes < logical_ranks or logical_ranks < 1 or horizon < 1 or base_seed < 1:
        raise ValueError("Invalid seed schedule")
    rows = []
    per_rank, extras = divmod(episodes, logical_ranks)
    for rank in range(logical_ranks):
        start = rank * per_rank + min(rank, extras)
        end = start + per_rank + (rank < extras)
        local_seed = base_seed + rank * horizon * 1000
        for episode in range(start, end):
            rows.append({"episode": episode, "logical_rank": rank, "local_seed": local_seed,
                         "environment_seed": (local_seed * local_seed + episode * local_seed) % (2**32 - 2)})
    if len(rows) != episodes or len({row["episode"] for row in rows}) != episodes:
        raise ValueError("Duplicate or missing episodes")
    if len({row["environment_seed"] for row in rows}) != episodes:
        raise ValueError("Colliding environment seeds")
    return rows


def prepare(vendor, task):
    if task not in CONFIGS:
        raise ValueError("Unknown primary task")
    path = vendor / CONFIGS[task]
    original = yaml.safe_load(path.read_text())
    cfg = copy.deepcopy(original)
    planner, spec = cfg["planner"], cfg["task_specification"]
    pusht = task == "pusht"
    task_template = None
    if task == "reach":
        # There is no full JEPA-WM Reach example; the released MW model is shared.
        # Use its full model/data settings plus the official Reach task template.
        task_template = vendor / "configs/online_plan_evals/mw/reach_L2_cem_sourcexp_H6_nas3_ctxt2.yaml"
        template = yaml.safe_load(task_template.read_text())
        spec["task"] = template["task_specification"]["task"]
        for key in ("iterations", "horizon", "num_samples", "num_elites", "num_act_stepped"):
            if planner[key] != template["planner"][key]:
                raise ValueError("Reach template and shared MW planner disagree")
    expected = {"planner_name": "cem", "iterations": 30 if pusht else 15,
                "num_samples": 300, "num_elites": 10, "horizon": 6,
                "var_scale": 1., "num_act_stepped": 6 if pusht else 3,
                "repeat_actskip": False, "distribute_planner": False}
    if any(planner[key] != value for key, value in expected.items()):
        raise ValueError("Upstream CEM settings differ from audited paper recipe")
    if planner["planning_objective"] != {"objective_type": "L2", "sum_all_diffs": False, "alpha": .1}:
        raise ValueError("Wrong official planning objective")
    if spec["goal_source"] != ("dset" if pusht else "expert") or spec["img_size"] != 224:
        raise ValueError("Unexpected task sampling or image size")
    model = cfg["model_kwargs"]
    if model["wrapper_kwargs"]["ctxt_window"] != 2 or model["data"]["custom"]["frameskip"] != 5:
        raise ValueError("Do not substitute offline context or action spacing for planning")
    # The paper uses 96 for these environments, although MW release templates say 48.
    cfg["meta"]["eval_episodes"] = 96
    cfg["meta"]["quick_debug"] = False
    cfg["logging"].update(optional_plots=False, tqdm_silent=True)
    cfg["planner"]["decode_each_iteration"] = False
    cfg["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] = {}
    cfg["frameskip"] = 5
    cfg["tasks"] = [spec["task"]]
    spec["multitask"] = False
    contract = {
        "schema_version": 1, "status": "prepared_not_executed_or_confirmation_frozen",
        "task": task, "vendor_commit": VENDOR_COMMIT,
        "source_config": CONFIGS[task], "source_config_sha256": sha256(path),
        "task_template_sha256": sha256(task_template) if task_template else None,
        "source_config_episodes": original["meta"]["eval_episodes"],
        "planned_replication_episodes_per_condition": 96,
        "paper_reference": "https://arxiv.org/html/2512.24497v4#A7.SS2",
        "simulation": "official PushTEnv / Pymunk 6.8.0" if pusht else "official MetaWorldWrapper / MetaWorld V3 / MuJoCo",
        "goal_source": spec["goal_source"], "planning_context": 2, "offline_context": 3,
        "max_elementary_steps": 30 if pusht else 100,
        "replanning": "one H6 plan, execute all 30 elementary actions" if pusht else "replan after each 15 elementary actions until episode end",
        "reported_unit": "paired initial/goal scenario; cluster reused Push-T source families",
        "fresh_family_confirmation": False,
        "trained_checkpoints_per_task_available_to_this_run": 1,
        "matches_three_training_seeds_and_last_ten_epochs": False,
        "seed_policy": {
            "base_seed": cfg["meta"]["seed"], "logical_ranks": 8,
            "local_seed": "base_seed + logical_rank * horizon * 1000",
            "environment_seed": "(local_seed**2 + episode * local_seed) % (2**32 - 2)",
            "dataset_and_planner_rng": "separate CPU/CUDA generators seeded once per logical stream; retain stream state across episodes",
            "conditions": "same initial/goal schedule and initial RNG states for native, candidate and matched control",
            "physical_gpu_reassignment_changes_seeds": False,
            "actual_authors_episode_identities_reconstructed": False,
        },
        "explicit_changes": ["96 episodes following paper; original config count retained above",
                             "omit unused decoder heads and visualization outputs",
                             "logical rank streams remain fixed when scheduling across physical workers"]
                            + (["shared released MW model/config with task set by official Reach template"] if task_template else []),
        "required_launch_gates": ["frozen candidate and controls", "verified cohort access",
                                  "native simulator/planner smoke and intervention parity", "budget and geography"],
        "config": cfg,
        "episodes": seed_schedule(cfg["meta"]["seed"]),
    }
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .vendor import use_vendor
    use_vendor(args.vendor)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for task in CONFIGS:
        path = args.output / (task + ".json")
        write_json(path, prepare(args.vendor, task))
        hashes[task] = sha256(path)
    write_json(args.output / "PREPARED.json", {"contracts_sha256": hashes, "jobs_launched": False,
                                             "confirmation_authorized_by_this_receipt": False})


if __name__ == "__main__":
    main()
