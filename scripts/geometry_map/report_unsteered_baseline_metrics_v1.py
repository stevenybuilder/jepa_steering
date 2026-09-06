#!/usr/bin/env python3
"""SHA-first CPU outcome reporting of149 registered unsteered baselines only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def build(project,output):
    base=project/'artifacts/geometry_map'; inv=json.loads((base/'steering_preparation_v1/source_inventory.json').read_text()); rows=[]
    for r in inv['banks']:
        rows.append(dict(panel='reach_wall' if r['task']=='reach_wall' else 'pusht_official',episode=r['episode'],
            split=r['split'],originally_sealed=r['sealed'],sha256=r['sha256'],local_path=r['path'],
            receipt_path=r['source_receipt'],receipt_sha256=r['source_receipt_sha256'],
            host=49982193,remote_path='/root/geometry-map-jepawm-reach-wall-v1/steering-bank-prep-v1/project/'+r['path'],
            harness='collect_on_policy_bank' if r['task']=='reach_wall' else 'collect_pusht_bank'))
    for name in ['steering_expansion_v1','eight_full_baselines_v1']:
        for job in json.loads((base/name/'monitor_manifest.json').read_text())['jobs']:
            p=project/job['local_directory']/'SEALED_DONE.json';receipt=json.loads(p.read_text())
            assert receipt['complete']
            for r in receipt['outputs']:
                panel='reach_wall' if job['task']=='reach_wall' else 'pusht_scripted'
                rows.append(dict(panel=panel,episode=r['episode'],split='evaluation_preparation',originally_sealed=True,
                    sha256=r['sha256'],local_path=str(Path(job['local_directory'])/r['path']),receipt_path=str(p.relative_to(project)),
                    receipt_sha256=sha(p),host=job['instance_id'],remote_path=job['remote_directory']+'/'+r['path'],
                    harness='collect_on_policy_bank' if panel=='reach_wall' else 'collect_pusht_bank',
                    source_panel_version='scripted_goal_v2' if r['episode']<50 else 'scripted_goal_reserve_v1'))
    for worker in [49902461,49982193]:
        root=base/f'reach_development_expansion_v1/worker-{worker}/results-v2'
        for p in sorted(root.glob('episode-*-unsteered.DONE.json')):
            d=json.loads(p.read_text());assert d['complete'];r=d['outputs'][0];episode=int(re.search(r'episode-(\d+)',r['path'])[1])
            rows.append(dict(panel='reach_wall',episode=episode,split='development_expansion_66_73',originally_sealed=False,
                sha256=r['sha256'],local_path=str((root/r['path']).relative_to(project)),receipt_path=str(p.relative_to(project)),
                receipt_sha256=sha(p),host=worker,
                remote_path=f'/root/geometry-map-jepawm-reach-wall-v1/development-expansion-66-73-v1/worker-{worker}/results-v2/'+r['path'],
                harness='collect_reach_development_expansion_full99'))
    # Offline original source195: stage its seven byte-identical registered tensors from the verified local mirror.
    for r in rows:
        if r['host']==49982195:
            r['offline_original_source']=49982195;r['host']=49987405
            r['remote_path']='/root/geometry-map-jepawm-pusht-v1/unsteered-baseline-metrics-v1/offline195-inputs/'+Path(r['local_path']).name
    assert len(rows)==149 and len({(r['panel'],r['episode']) for r in rows})==149
    assert Counter(r['panel'] for r in rows)=={'reach_wall':74,'pusht_official':21,'pusht_scripted':54}
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema_version=1,authorization='2026-09-06 user explicitly authorized ALL registered baseline outcomes for reporting only',
        no_tuning_or_steered_confirmation=True,registered_counts={'reach_wall':74,'pusht_official':21,'pusht_scripted':54},
        repeated_unsteered_arms_excluded=True,model_calls=0,simulator_calls=0,gpu_calls=0,
        aggregate_analysis_cpu_limit_seconds=180,grouping='Exact canonical little-endian float64 full initializer/goal fingerprints; does not assert independent physical conditions',
        source=rows)
    dump(output/'manifest.json',protocol)
    for host in sorted({r['host'] for r in rows}):dump(output/f'host-{host}.json',dict(protocol,source=[r for r in rows if r['host']==host]))
    print(json.dumps({'manifest_rows':len(rows),'hosts':Counter(str(r['host']) for r in rows)}))


def asarray(value):
    import numpy as np
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value)


def fingerprint(value):
    import numpy as np
    a=asarray(value).astype('<f8',copy=True);a[a==0]=0
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def push_metrics(value):
    import numpy as np
    from diagnose_pusht_goal_coverage import coverage
    states=asarray(value['states']).astype(float);goal=asarray(value['goal_state']).astype(float)
    if not np.isfinite(states).all() or not np.isfinite(goal).all():raise ValueError('Nonfinite states/goal')
    xy=np.linalg.norm(states[:,:4]-goal[:4],axis=1)
    diff=np.abs(states[:,4]-goal[4]);native_angle=np.minimum(diff,2*np.pi-diff)
    native=(xy<20)&(native_angle<np.pi/9);saved=asarray(value['step_goal_success']).astype(bool)
    if not np.array_equal(native,saved) or bool(native[-1])!=bool(value['final_success']):raise ValueError('Saved/native requestedgoal success mismatch')
    requested=np.asarray([coverage(s[2:5],goal[2:5]) for s in states])
    default=np.asarray([coverage(s[2:5],[256.,256.,np.pi/4]) for s in states])
    rewards=asarray(value['rewards']).astype(float)
    if not np.isfinite(rewards).all():raise ValueError('Nonfinite rewards')
    expected=np.minimum(default[1:]/.95,1.)
    discrepancy=float(np.max(np.abs(rewards-expected)))
    if discrepancy>3e-6:raise ValueError('Painted-default reward reconstruction exceeds frozen3e-6 tolerance')
    robust=np.abs(np.arctan2(np.sin(states[:,4]-goal[4]),np.cos(states[:,4]-goal[4])))
    return dict(native_ever_success=bool(native[1:].any()),native_final_success=bool(native[-1]),
        native_ever_including_initial=bool(native.any()),initial_success=bool(native[0]),
        native_success_definition='Native PushTWrapper.eval_state: jointagent/blockXYnorm<20pixels AND min(absangle,2pi-absangle)<pi/9; excludes initial for ever primary',
        raw_steps=int(len(value['actions_raw'])),state_count=len(states),replan_count=len(value['replans']),
        final_goal_xy_distance_px=float(xy[-1]),min_goal_xy_distance_px=float(xy[1:].min()),initial_goal_xy_distance_px=float(xy[0]),
        goal_xy_progress_px=float(xy[0]-xy[-1]),final_requested_goal_coverage=float(requested[-1]),max_requested_goal_coverage=float(requested[1:].max()),
        initial_requested_goal_coverage=float(requested[0]),requested_coverage_ge_095_ever=bool((requested[1:]>=.95).any()),
        requested_coverage_ge_095_final=bool(requested[-1]>=.95),requested_coverage_threshold_scope='Descriptive requestedpose overlap, NOT native positional-angle success',
        final_painted_default_coverage=float(default[-1]),max_painted_default_coverage=float(default[1:].max()),
        painted_default_reward_sum=float(rewards.sum()),painted_default_reward_max=float(rewards.max()),
        painted_default_reward_final=float(rewards[-1]),painted_reward_reconstruction_maxabs=discrepancy,
        painted_coverage_gt_095_ever=bool((default[1:]>.95).any()),painted_coverage_gt_095_final=bool(default[-1]>.95),
        painted_default_is_requested_goal=bool(np.allclose(goal[2:5],[256.,256.,np.pi/4],rtol=0,atol=1e-9)),
        native_robust_angle_success_disagreements=int(np.count_nonzero(native!=((xy<20)&(robust<np.pi/9)))),
        final_native_state7_distance=float(value['final_state_distance']),native_state7_distance_units='Mixed pixels/radians/velocities; not homogeneous geometry',
        goal_contract='Actual30-command replay endpoint; goal-state requestedpose differs from painted fixedgoal unless explicitly flagged',
        initial_state=fingerprint(value['initial_state']),requested_goal=fingerprint(goal),
        initial_state_values=asarray(value['initial_state']).tolist(),requested_goal_values=goal.tolist(),
        source_trajectory=value.get('source_trajectory'),environment_seed=value.get('environment_seed'),planner_seed=value.get('planner_seed'))


def reach_metrics(value):
    import numpy as np
    if 'metrics' in value:
        m=value['metrics'];prepared=value['prepared_initial'];states=asarray(value['states'])
        # Native reference initializer is recorded in the prepared snapshot/state, not future state0.
        initial=prepared.get('initial_state',prepared.get('initial_proprio'))
        goal=prepared.get('goal_state',prepared.get('goal_proprio'))
        if initial is None or goal is None:raise ValueError('New harness initial/goal fields missing')
        result=dict(native_ever_success=bool(m['native_ever_success']),native_final_success=bool(m['native_final_success']),
            raw_steps=len(value['actions_raw']),environment_elapsed_counter=100,replan_count=len(m['replans']),
            native_reward_sum=float(m['native_reward_sum']),native_reward_max=float(m['native_reward_max']),
            final_hand_goal_distance_m=float(m['final_hand_goal_distance_m']),
            hand_goal_progress_m=float(m['hand_goal_progress_m']),
            stored_final_expert_state_distance=None,final_expert_state_distance_status='Not stored by new full99 harness',
            initial_fingerprint_field='prepared_initial initial_state else initial_proprio',
            goal_fingerprint_field='prepared_initial goal_state else goal_proprio')
    else:
        initial=value['replans'][0]['state'];goal=value['goal']['state']
        result=dict(native_ever_success=bool(value['ever_success']),native_final_success=bool(value['final_success']),
            raw_steps=len(value['actions_raw']),environment_elapsed_counter=int(value['environment_steps']),replan_count=value['replan_count'],
            native_reward_sum=float(value['total_reward']),native_reward_max=None,
            final_hand_goal_distance_m=None,hand_goal_progress_m=None,
            stored_final_expert_state_distance=float(value['final_state_distance_to_expert_goal']),
            final_expert_state_distance_status='Stored fullstate distance toexpert endpoint, not hand-only target distance; mixedstate dimensions',
            initial_fingerprint_field='first replan full observation state',goal_fingerprint_field='goal.state full expert observation state')
    for key in ['native_reward_sum','stored_final_expert_state_distance','final_hand_goal_distance_m']:
        x=result.get(key)
        if x is not None and not np.isfinite(x):raise ValueError('Nonfinite '+key)
    result.update(native_success_definition='Native MetaWorld Reach-Wall info.success over executed rawsteps; originalcollector info versus newharness evaluate_state explicitly distinguished',
        goal_contract='Planner goal is native expert final observation; native task success is environment Reach-Wall target, not an invented hand threshold',
        initial_state=fingerprint(initial),requested_goal=fingerprint(goal),initial_state_values=asarray(initial).tolist(),
        requested_goal_values=asarray(goal).tolist(),environment_seed=value.get('environment_seed'),planner_seed=value.get('planner_seed'))
    return result


def extract(manifest,output):
    import torch
    torch.set_num_threads(1)
    d=json.loads(manifest.read_text());output.mkdir(parents=True,exist_ok=False);started=time.process_time();rows=[]
    for source in d['source']:
        row=dict(source)
        try:
            path=Path(source['remote_path'])
            if sha(path)!=source['sha256']:raise ValueError('ArtifactSHA mismatch BEFORE torch.load')
            row['artifact_sha_verified_before_deserialization']=True
            value=torch.load(path,map_location='cpu',weights_only=False)
            episode=value.get('episode',value.get('metrics',{}).get('episode'))
            if episode!=source['episode']:raise ValueError('Episode identity mismatch')
            row.update(reach_metrics(value) if source['panel']=='reach_wall' else push_metrics(value))
            row['initial_goal_pair']=hashlib.sha256((row['initial_state']+row['requested_goal']).encode()).hexdigest()
            row['status']='ok';del value
        except Exception as exc:row.update(status='failed',error=str(exc))
        rows.append(row)
        print(json.dumps({k:row.get(k) for k in ['panel','episode','status','error','native_ever_success','native_final_success']}),flush=True)
        if time.process_time()-started>20:raise RuntimeError('Perhost20CPU-second bound exceeded')
    result=dict(complete=all(r['status']=='ok' for r in rows),rows=rows,analysis_cpu_seconds=time.process_time()-started,
        manifest_sha256=sha(manifest),script_sha256=sha(__file__),held_baseline_outcomes_authorized=True,
        no_tuning_or_steered_confirmation=True,model_calls=0,gpu_calls=0,simulator_calls=0)
    dump(output/'rows.json',result)
    dump(output/'DONE.json',dict(complete=result['complete'],rows=len(rows),outputs=[dict(path='rows.json',sha256=sha(output/'rows.json'))],analysis_cpu_seconds=result['analysis_cpu_seconds']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','extract']);p.add_argument('--project',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='build':build(a.project,a.output)
    else:extract(a.manifest,a.output)
