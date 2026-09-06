#!/usr/bin/env python3
"""CPU-only checked compact readback; no causal or multimodality inference."""
import argparse
import json
from pathlib import Path
from statistics import mean
import torch
from cem_search_audit_v1 import sha,write,STAGES


def aggregate(reports):
    reports=sorted(reports,key=lambda r:r['episode'])
    if [r['episode'] for r in reports]!=[0,1,2,3] or not all(r['complete'] and r['all_physics_replays_exact'] for r in reports):raise ValueError('Four complete states required')
    stages={}
    for stage in STAGES:
        selected=[next(x for x in r['stage_differences'] if x['stage']==stage) for r in reports]
        stages[str(stage)]={k:dict(by_state=[x[k] for x in selected],equal_state_mean=mean(x[k] for x in selected)) for k in selected[0] if k!='stage'}
    return dict(complete=True,independent_seen_development_states=4,stages=stages,primary='stage30 requestedcoverage MEAN minus CURRENT lowest-model-cost candidate',
        native_executed_prefix_raw_steps=30,H6_raw_steps=30,closed_loop_full_episode=False,
        limitations='One fixed planner RNG per already-seen state; selected stages and plans dependent, not additional sample size. No hidden representation edit or physical-oracle selection.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    done=json.loads((a.source/'DONE.json').read_text());index={r['path']:r for r in done['outputs']}
    if not done['complete'] or sorted(done['episodes'])!=[0,1,2,3]:raise ValueError('Completed corrective batch required')
    reports=[];rows=[]
    for e in range(4):
        jp=a.source/f'episode-{e:03d}.json';pt=a.source/f'episode-{e:03d}-CEM.pt'
        for q in (jp,pt):
            if sha(q)!=index[q.name]['sha256']:raise ValueError('Source hash before read')
        report=json.loads(jp.read_text());reports.append(report);value=torch.load(pt,map_location='cpu',weights_only=False)
        for row in value['rows']:
            elites=row['actions'][:,row['elite_indices']].permute(1,0,2).reshape(10,60)
            rows.append(dict(episode=e,iteration=row['iteration'],elites_flat=elites.tolist(),delivered_mean_flat=row['mean_after'].flatten().tolist(),
                previous_mean_flat=row['mean_before'].flatten().tolist(),std_after_flat=row['std_after'].flatten().tolist(),elite_indices=row['elite_indices'].tolist(),
                candidate_costs=row['costs'].tolist(),best_index=row['best_index'],best_cost=row['best_cost'],mean_cost=row['mean_cost']))
    a.output.mkdir(parents=True,exist_ok=False);result=aggregate(reports)
    result['corrective_process_seconds']=done['seconds'];result['including_original_timeout_process_seconds_upper_bound']=270+done['seconds']
    result['source_done_sha256']=sha(a.source/'DONE.json')
    write(a.output/'AGGREGATE.json',result);write(a.output/'ELITES.json',dict(complete=True,states=[0,1,2,3],rows=rows,source_done_sha256=result['source_done_sha256'],action_coordinates='normalized native6x10,flattened in time-major order'))
    outputs=[dict(path=q.name,sha256=sha(q),bytes=q.stat().st_size) for q in sorted(a.output.iterdir())]
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,model_calls=0,simulator_calls=0))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
