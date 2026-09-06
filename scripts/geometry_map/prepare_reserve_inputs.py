#!/usr/bin/env python3
"""Reserve50 additional stimuli per task; no learnedmodel or planner execution."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from prepare_pusht_independent import sha, write_json


def validate_reserve(manifest, task):
    expected=list(range(62,112)) if task=="reach_wall" else list(range(50,100))
    if manifest["task"]!=task or manifest["episode_ids"]!=expected:
        raise RuntimeError("Wrong frozen reserve IDs")
    rows=manifest["starts"]
    if [r["episode"] for r in rows]!=expected or len({r["environment_seed"] for r in rows})!=50:
        raise RuntimeError("Missing/duplicate reserve starts")
    old=set(manifest["existing_same_task_environment_seeds"])
    if old & {r["environment_seed"] for r in rows}:raise RuntimeError("Reserve seeds overlap existingstudy")
    envbase,planbase=(2026090500,90500) if task=="reach_wall" else (2026090700,92600)
    if any(r["environment_seed"]!=envbase+r["episode"] or r["planner_seed"]!=planbase+r["episode"] for r in rows):
        raise RuntimeError("Original task seed rule changed")
    if manifest["full_model_rollouts_authorized"]:raise RuntimeError("Reserve preparation cannot authorize modelrollouts")
    return rows


def reach(args,manifest):
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    from collect_on_policy_bank import initialize_episode,physics_snapshot
    sys.path.insert(0,str(args.repo.resolve()))
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed
    configpath=args.repo/manifest["native_config"]
    if sha(configpath)!=manifest["native_config_sha256"]:raise RuntimeError("Nativeconfig differs")
    raw=yaml.safe_load(configpath.read_text()); raw["logging"]["optional_plots"]=False; raw["logging"]["tqdm_silent"]=True
    raw["planner"]["decode_each_iteration"]=False; raw["work_dir"]=str(args.output_dir)
    cfg=parse_cfg(OmegaConf.create(raw)); cfg.device="cpu";cfg.rank=0;cfg.world_size=1;cfg.num_active_gpus=1;cfg.active_ranks=[0]
    cfg.local_seed=int(raw["meta"]["seed"]);cfg.frameskip=int(raw["model_kwargs"]["data"]["custom"]["frameskip"]);cfg.action_ratio=cfg.frameskip
    cfg.work_dir=args.output_dir;cfg.planner.distribute_planner=False
    class ClosedExpertEvaluator(PlanEvaluator):
        def unroll_expert(self,env,*a,**kw):
            try:return super().unroll_expert(env,*a,**kw)
            finally:env.close()
    def generate(row):
        set_seed(cfg.local_seed)
        env=make_env(cfg)
        agent=SimpleNamespace(model=SimpleNamespace(tubelet_size_enc=1),set_goal=lambda _goal:None)
        evaluator=ClosedExpertEvaluator(cfg,agent)
        try:
            initial,_stale,goal,_expert_success,_expertframes=initialize_episode(cfg,agent,env,evaluator,row["episode"],row["environment_seed"])
            return {"initial_visual":initial["visual"].cpu().to(torch.uint8),"initial_proprio":initial["proprio"].cpu(),
                    "initial_physics":physics_snapshot(env),"goal_visual":goal["visual"].cpu().to(torch.uint8),"goal_proprio":goal["proprio"].cpu(),
                    "goal_state":np.asarray(evaluator.state_g).copy(),"expert_reference_actions":evaluator.expert_actions.cpu(),
                    "actual_initial_hand_xyz":initial["proprio"].cpu().reshape(-1)[:3]}
        finally:env.close()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    outputs=[];started=time.monotonic()
    for row in manifest["starts"]:
        tick=time.monotonic();value=generate(row);repeat=generate(row)
        checks={key:torch.equal(value[key],repeat[key]) for key in ("initial_visual","initial_proprio","goal_visual","goal_proprio","expert_reference_actions")}
        checks["goal_state"]=bool(np.array_equal(value["goal_state"],repeat["goal_state"]))
        checks["initial_physics"]=all(np.array_equal(value["initial_physics"]["physics"][k],repeat["initial_physics"]["physics"][k]) for k in value["initial_physics"]["physics"])
        if not all(checks.values()):raise RuntimeError(f"Reserve exact stimulus regeneration failed episode{row['episode']}: {checks}")
        value.update({**row,"panel":manifest["panel"],"split":"reserve_unallocated_to_current100startstudy","model_execution":False,
                      "planner_execution":False,"encoded_features_present":False,"goal_origin":"native expertpolicy endpoint via unchanged initialize_episode/PlanEvaluator",
                      "initial_generic_info_stale":True,"raw_action_prefix":0,"elapsed_steps_includes_warmup":1,"repeat_exact_checks":checks})
        target=args.output_dir/f"input-{row['episode']:03d}.pt";torch.save(value,target)
        outputs.append({**row,"path":target.name,"sha256":sha(target),"bytes":target.stat().st_size,"seconds":time.monotonic()-tick,"repeat_exact":True})
        print(json.dumps({"event":"reserve_reach_input_prepared","episode":row["episode"],"seconds":outputs[-1]["seconds"],"sha256":outputs[-1]["sha256"]}),flush=True)
    write_json(args.output_dir/"DONE.json",{"complete":True,"task":"reach_wall","panel":manifest["panel"],"outputs":outputs,"seconds":time.monotonic()-started,
                "model_execution":False,"planner_execution":False,"rendering":"native MuJoCo EGL images mayuse GPUgraphics; nolearnedmodelCUDAcompute",
                "manifest_sha256":sha(args.manifest),"script_sha256":sha(Path(__file__)),"initializer_source_sha256":sha(Path(inspect.getfile(initialize_episode))),
                "native_evaluator_source_sha256":sha(Path(inspect.getfile(PlanEvaluator))),"goal_inputs_fully_prepared":True,"current100startmatrix_unchanged":True,"no_outcome_screening":True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","repo","output-dir"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());rows=validate_reserve(manifest,manifest["task"])
    if manifest["task"]=="reach_wall":return reach(args,manifest)
    import prepare_pusht_scripted_goals as frozen
    if sha(Path(frozen.__file__))!=manifest["frozen_generator_sha256"]:raise RuntimeError("FrozenPushgenerator changed")
    expected={"steps":30,"through_block_offset":30,"aim_bounds":[20,492],"relative_scale":100,"raw_bounds":[-.25,.25]}
    if manifest["controller"]!=expected:raise RuntimeError("Reserve controller differs from frozenv2")
    frozen.validate=lambda value:validate_reserve(value,"pusht")
    # Only the ID/seed validator is adapted; actual generator/replay/angle/contact
    # code is byte-identical to the certified scriptedv2 implementation.
    frozen.main()


if __name__=="__main__":main()
