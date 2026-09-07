#!/usr/bin/env python3
"""Compact, source-checked readback of the frozen NEW-four CEM replication."""
import argparse
import json
from pathlib import Path
from statistics import mean
import torch
from cem_search_audit_v1 import sha,write,STAGES
from cem_search_replication_v1 import SOURCES,SHARDS,primary_contrast


def aggregate(reports):
    reports=sorted(reports,key=lambda x:x['episode'])
    if [r['episode'] for r in reports]!=list(SOURCES) or not all(r['complete'] and r['all_physics_replays_exact'] for r in reports):
        raise ValueError('All four NEW preallocated initial groups required; no exclusions')
    primaries=[primary_contrast(r) for r in reports]
    primary={k:dict(by_source=[r[k] for r in primaries],equal_source_mean=mean(r[k] for r in primaries)) for k in primaries[0] if k!='source_id'}
    stages={}
    for stage in STAGES:
        values=[next(x for x in r['stage_differences'] if x['stage']==stage) for r in reports]
        stages[str(stage)]={k:dict(by_source=[x[k] for x in values],equal_source_mean=mean(x[k] for x in values)) for k in values[0] if k!='stage'}
    success={}
    for stage in STAGES:
        for kind in ('mean','best'):
            selected=[next(x for x in r['metric_rows'] if x['stage']==stage and x['kind']==kind) for r in reports]
            success[f'{stage}_{kind}']=dict(final=sum(x['native_goal_success_final'] for x in selected),ever=sum(x['native_goal_ever_success'] for x in selected),
                coverage_by_source=[x['requested_coverage_final'] for x in selected],model_cost_by_source=[x['predicted_cost_by_horizon'][-1] for x in selected])
    return dict(complete=True,source_ids=list(SOURCES),independent_new_development_initial_groups=4,primary=primary,secondary_mean_minus_best=stages,
        success_and_level_values=success,per_source=[dict(source_id=r['episode'],metrics=r['metric_rows'],primary=primary_contrast(r)) for r in reports],
        selected_stage_plans=24,physical_replays=48,expert_goal_replays=8,raw_steps_per_plan=30,native_executed_prefix_equals_H6=True,
        model_fit_or_hidden_intervention=False,held_confirmation=False,closed_loop_efficacy=False,
        limitations='Four new metadata-preallocated TRAIN initial groups, DEVELOPMENT replication. One planner RNG/group. Stages and plans are dependent, not extra sample size. No outcome-driven stage or plan selection.')


def export_shard(source,output):
    done=json.loads((source/'DONE.json').read_text());index={r['path']:r for r in done['outputs']}
    if not done['complete'] or tuple(done['episodes']) not in SHARDS:raise ValueError('Completed frozen shard required')
    for entry in done['outputs']:
        if sha(source/entry['path'])!=entry['sha256']:raise ValueError('Source output SHA '+entry['path'])
    rows=[]
    for e in done['episodes']:
        value=torch.load(source/f'episode-{e}-CEM.pt',map_location='cpu',weights_only=False)
        for r in value['rows']:
            elites=r['actions'][:,r['elite_indices']].permute(1,0,2).reshape(10,60)
            rows.append(dict(episode=e,iteration=r['iteration'],elites_flat=elites.tolist(),delivered_mean_flat=r['mean_after'].flatten().tolist(),
                previous_mean_flat=r['mean_before'].flatten().tolist(),std_after_flat=r['std_after'].flatten().tolist(),elite_indices=r['elite_indices'].tolist(),
                candidate_costs=r['costs'].tolist(),best_index=r['best_index'],best_cost=r['best_cost'],mean_cost=r['mean_cost']))
    write(output/'ELITES.json',dict(complete=True,source_ids=done['episodes'],rows=rows,action_coordinates='Native normalized6x10, time-major',source_done_sha256=sha(source/'DONE.json')))
    write(output/'SOURCE_VERIFIED.json',dict(complete=True,source=str(source),source_done_sha256=sha(source/'DONE.json'),files=len(done['outputs']),
        bytes=sum(r['bytes'] for r in done['outputs']),process_seconds=done['seconds'],parameter_sha256=done['parameter_sha256']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--reports',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if bool(a.source)==bool(a.reports):raise ValueError('Choose source shard or four compact reports')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    if a.source:export_shard(a.source,a.output)
    else:write(a.output/'AGGREGATE.json',aggregate([json.loads(q.read_text()) for q in a.reports]))
    outputs=[dict(path=q.name,sha256=sha(q),bytes=q.stat().st_size) for q in sorted(a.output.iterdir())]
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,model_calls=0,simulator_calls=0))


if __name__=='__main__':main()
