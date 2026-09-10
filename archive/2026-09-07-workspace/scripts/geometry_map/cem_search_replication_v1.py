#!/usr/bin/env python3
"""Frozen native CEM audit on four metadata-preallocated NEW TRAIN initial groups.

Only the stimulus adapter is new. NativeTrace/run_state are the previously
identity-tested, unmodified CEM audit core. No new policy or latent intervention.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import cem_search_audit_v1 as core

SOURCES=(2222,2323,2424,2525)
SHARDS=((2222,2323),(2424,2525))
MANIFEST_SHA='9fb993bfb632746649f725f088a424d81c0ab1fc5b2ac035e518a306031f1f8e'
RAW_DONE_SHA='d0feef5ae9e4063e0f51b9bc81965e49f9a89519aa7d6a81efaf2c925de721c0'


def raw_inventory(root,episodes):
    """Validate frozen identities/split and all required hashes BEFORE torch.load."""
    if tuple(episodes) not in SHARDS:raise ValueError('Exactly one frozen two-state shard required')
    if core.sha(root/'manifest.json')!=MANIFEST_SHA or core.sha(root/'DONE.json')!=RAW_DONE_SHA:
        raise ValueError('Immutable raw source receipts')
    manifest=json.loads((root/'manifest.json').read_text());done=json.loads((root/'DONE.json').read_text())
    if [r['source_id'] for r in manifest['rows']]!=list(SOURCES) or manifest['split']!='new_development_not_confirmation':
        raise ValueError('Frozen new DEVELOPMENT source identities')
    if not done['complete'] or done['stage']!='raw_source_staging_only':raise ValueError('Raw stage incomplete')
    entries={r['path']:r for r in done['outputs']}
    for e in episodes:
        if core.sha(root/f'input-{e}.pt')!=entries[f'input-{e}.pt']['sha256']:raise ValueError('Raw input SHA')
    return manifest,entries


def reset_payload(raw,truth,raw_sha):
    """Use observed expert endpoints, preserving the distinct RAW initializer."""
    e=raw['source_id']
    if e not in SOURCES or raw['split']!='new_development_not_confirmation' or raw['source_offset']!=0:
        raise ValueError('Tensor source/split/offset changed')
    if raw['environment_seed']!=2026090600+e or raw['planner_seed']!=91600+e:raise ValueError('Frozen seed rule')
    if raw['initial_state'].shape!=(7,) or raw['expert_actions_raw'].shape!=(30,2):raise ValueError('Raw stimulus shapes')
    initial_sha=hashlib.sha256(raw['initial_state'].contiguous().numpy().tobytes()).hexdigest()
    if initial_sha!=raw['initial_state_sha256'] or raw['source_manifest_sha256']!=MANIFEST_SHA:raise ValueError('Raw initializer provenance')
    if len(truth['states'])!=31 or truth['proprios'].numel()!=31*4:raise ValueError('Native expert replay length/proprio')
    core.exact(truth['states'][:,[0,1,5,6]],truth['proprios'].reshape(31,4),'native true initial/future proprio')
    return dict(episode=e,source_trajectory=e,split='development',source_split=raw['split'],source_offset=0,
        environment_seed=raw['environment_seed'],planner_seed=raw['planner_seed'],
        initial_state=raw['initial_state'].clone(),env_info=raw['env_info'],expert_actions_raw=raw['expert_actions_raw'].clone(),
        expert_observations=truth['frames'],expert_states=truth['states'],expert_proprios=truth['proprios'],
        expert_physics=truth['physics'],goal_state=truth['states'][-1].copy(),raw_source_sha256=raw_sha,
        source_manifest_sha256=MANIFEST_SHA,goal_construction='Actual native30 expert-command replay endpoint; NOT dataset frame30',
        initial_state_semantics='Original raw7D initializer, not already integrated observed reset state',exact_expert_replays=True)


def new_source_selector(row,prepared):
    path=Path(row['bank']);receipt=json.loads(Path(row['receipt']).read_text());e=row['episode']
    if e not in SOURCES or not receipt['complete'] or receipt['source_manifest_sha256']!=MANIFEST_SHA:
        raise ValueError('New source receipt before load')
    source=next(r for r in receipt['outputs'] if r['path']==path.name)
    rr=next(r for r in prepared['episodes'] if r['episode']==e)
    if source['source_id']!=e or source['split']!='new_development_not_confirmation' or not prepared['complete'] or rr['split']!='development':
        raise ValueError('New source identity/split before load')
    if not receipt['expert_replay_exact'] or core.sha(path)!=source['sha256']:raise ValueError('Prepared stimulus SHA/replay')
    return path,source,rr


@contextmanager
def source_adapter():
    """Isolated process-local input selector; restore even if native core fails."""
    original=core.select_source
    core.select_source=new_source_selector
    try:yield
    finally:core.select_source=original


@torch.no_grad()
def prepare_source(args,source_id,entry,wm,cfg):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.utils import make_td
    from capture_pusht_horizon import physical_fork,compare_physics
    start=time.monotonic();raw=torch.load(args.raw_inputs/entry['path'],map_location='cpu',weights_only=False)
    if raw['source_id']!=source_id or raw['split']!='new_development_not_confirmation':raise ValueError('Source tensor split')
    env=make_env(cfg)
    def initialize():
        env.update_env(raw['env_info'])
        return env.prepare(raw['environment_seed'],raw['initial_state'].numpy().copy(),raw['env_info'])
    try:
        obs,info=initialize();initial_obs=obs.clone();initial_info={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in info.items()}
        truth=physical_fork(env,obs,info,raw['expert_actions_raw'])
        obs,info=initialize();repeat=physical_fork(env,obs,info,raw['expert_actions_raw'])
        for k in ('frames','states','proprios','native_applied_targets_xy'):core.exact(truth[k],repeat[k],'expert replay '+k)
        compare_physics(truth['physics'],repeat['physics'])
        if truth['contacts']!=repeat['contacts']:raise ValueError('Expert contacts replay')
    finally:env.close()
    reset=reset_payload(raw,truth,entry['sha256'])
    core.exact(initial_obs,reset['expert_observations'][0],'initial pixels canary')
    core.exact(initial_info['proprio'],reset['expert_proprios'][0],'nonzero actual initial proprio canary')
    raw_goal=make_td(reset['expert_observations'][-1],{'proprio':reset['expert_proprios'][-1]})
    encoded_goal=wm.encode(raw_goal.to('cuda:0').unsqueeze(0),act=False)
    context=wm.encode(make_td(initial_obs,initial_info).to('cuda:0').unsqueeze(0),act=True)
    context2=wm.encode(make_td(initial_obs,initial_info).to('cuda:0').unsqueeze(0),act=True)
    goal2=wm.encode(raw_goal.to('cuda:0').unsqueeze(0),act=False)
    for k in ('visual','proprio'):
        core.exact(context[k],context2[k],'fresh initial encode repeat '+k)
        core.exact(encoded_goal[k],goal2[k],'fresh goal encode repeat '+k)
        if encoded_goal[k].dtype!=torch.float32:raise ValueError('Fullprecision goal required')
    bank=dict(row={'source_id':source_id,'split':raw['split']},context={k:v.cpu().clone() for k,v in context.items()},
        goal_encoded={k:v.cpu().clone() for k,v in encoded_goal.items()},raw_goal={k:v.cpu().clone() for k,v in raw_goal.items()},
        goal_state=reset['goal_state'],candidates=[{'states':truth['states']}],source_raw_sha256=entry['sha256'])
    reset_path=args.prepared_inputs/f'input-{source_id}.pt';bank_path=args.prepared_inputs/f'stimulus-{source_id}.pt'
    torch.save(reset,reset_path);torch.save(bank,bank_path)
    rr=dict(episode=source_id,path=reset_path.name,sha256=core.sha(reset_path),split='development')
    br=dict(path=bank_path.name,sha256=core.sha(bank_path),source_id=source_id,split=raw['split'])
    receipt=args.prepared_inputs/f'stimulus-{source_id}.DONE.json'
    core.write(receipt,dict(complete=True,outputs=[br],source_manifest_sha256=MANIFEST_SHA,raw_source_sha256=entry['sha256'],expert_replay_exact=True,
        initial_and_goal_encode_repeat_exact=True,seconds=time.monotonic()-start))
    print(json.dumps(dict(event='new_source_canary_complete',source_id=source_id,seconds=time.monotonic()-start,
        exact_pixels_states_proprio_physics=True,goal_from_actual_expert_endpoint=True)),flush=True)
    return dict(episode=source_id,bank=str(bank_path),receipt=str(receipt)),rr


def primary_contrast(report):
    rows={r['stage']:r for r in report['metric_rows'] if r['kind']=='mean'}
    if set(rows)!=set(core.STAGES):raise ValueError('All frozen stages required')
    return dict(source_id=report['episode'],stage30_minus15_mean_requested_coverage=rows[30]['requested_coverage_final']-rows[15]['requested_coverage_final'],
        stage30_minus15_mean_model_cost=rows[30]['predicted_cost_by_horizon'][-1]-rows[15]['predicted_cost_by_horizon'][-1],
        stage30_minus15_mean_xy_px=rows[30]['goal_xy_distance_by_horizon'][-1]-rows[15]['goal_xy_distance_by_horizon'][-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw-inputs','repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True);p.add_argument('--max-seconds',type=float,default=375)
    a=p.parse_args();start=time.monotonic();manifest,entries=raw_inventory(a.raw_inputs,a.episodes)
    if core.sha(a.checkpoint)!=core.CHECKPOINT:raise ValueError('Pinned checkpoint')
    a.output.mkdir(parents=True,exist_ok=False);a.prepared_inputs=a.output/'prepared-stimuli';a.prepared_inputs.mkdir()
    a.identity=False;a.resume_cem=None;a.resume_cem_sha256=None
    try:
        core.write(a.output/'PROTOCOL.json',dict(all_sources=SOURCES,shard=a.episodes,source_manifest_sha256=MANIFEST_SHA,raw_done_sha256=RAW_DONE_SHA,
            horizon=6,samples=300,iterations=30,elites=10,stages=core.STAGES,stage_plans=['current model-cost best','updated elite mean'],
            primary='Stage30 minus stage15 MEAN-plan requested-goal coverage and model cost, equal weight per new initial group',
            secondary='Stage30 mean minus best; stage1 context, forecast errors, actual native encoded cost,4DXY,7Dstate,native success',
            goal_construction='Exact native30 expert actions replayed twice from original raw7D; observed last RGB/proprio/state, NOT dataset rawframe30',
            native_num_act_stepped=6,raw_steps=30,native_first_execution_equals_H6=True,closed_loop_efficacy=False,
            new_full_cem_identity_run=False,prior_identity_source='cem_search_audit_v1/VERIFIED.json; all9000costs,30meanforecasts,RNG/returnedactions exact',
            core_sha256=core.sha(core.__file__),script_sha256=core.sha(__file__),checkpoint_sha256=core.CHECKPOINT,
            all_sources_development=True,confirmation_access=False,fit_or_selection_by_outcomes=False,process_cap_seconds=a.max_seconds,aggregate_cap_seconds=750))
        from collect_pusht_bank import config
        from model_loader import load_headless
        from capture_pusht_calibration import parameter_sha
        torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(a.repo,a.output)
        wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint);wm.eval().requires_grad_(False);before=parameter_sha(wm)
        rows=[];resets=[]
        for e in a.episodes:
            row,rr=prepare_source(a,e,entries[f'input-{e}.pt'],wm,cfg);rows.append(row);resets.append(rr)
        core.write(a.prepared_inputs/'DONE.json',dict(complete=True,episodes=resets,source_manifest_sha256=MANIFEST_SHA))
        reports=[]
        with source_adapter():
            for row in rows:
                if time.monotonic()-start>a.max_seconds:raise ValueError('Process cap before next state')
                reports.append(core.run_state(a,row,wm,prep,cfg))
        if parameter_sha(wm)!=before:raise ValueError('Model parameters changed')
        core.write(a.output/'PRIMARY.json',dict(complete=True,rows=[primary_contrast(r) for r in reports],independent_initial_groups=len(reports)))
        outputs=[dict(path=str(q.relative_to(a.output)),sha256=core.sha(q),bytes=q.stat().st_size) for q in sorted(a.output.rglob('*')) if q.is_file()]
        core.write(a.output/'DONE.json',dict(complete=True,episodes=a.episodes,seconds=time.monotonic()-start,outputs=outputs,parameter_sha256=before,
            provenance=provenance,full_cem_runs=len(reports),all_physics_replays_exact=True,held_access=False))
        print(json.dumps(dict(event='shard_complete',episodes=a.episodes,seconds=time.monotonic()-start)),flush=True)
    except Exception:
        core.write(a.output/'FAILED.json',dict(complete=False,seconds=time.monotonic()-start,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
