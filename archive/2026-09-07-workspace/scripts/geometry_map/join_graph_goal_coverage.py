#!/usr/bin/env python3
"""Descriptive, no-fit expert-goal coverage join for original nine candidates."""
import argparse
import json
from pathlib import Path
from run_chart_behavior_pullback import verified_load
from run_behavior_pullback import sha,write
from diagnose_pusht_goal_coverage import coverage


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    reference_receipt=json.loads((a.reference/'DONE.json').read_text())
    reference_row=next(r for r in reference_receipt['outputs'] if r['path']=='summary.json')
    if not reference_receipt['complete'] or sha(a.reference/'summary.json')!=reference_row['sha256']:raise ValueError('Coverage reference SHA')
    reference=json.loads((a.reference/'summary.json').read_text())
    reference=next(r for r in reference['rows'] if r['episode']==a.episode)
    receipt=json.loads((a.results/'DONE.json').read_text())
    row=next(r for r in receipt['outputs'] if r['path']=='summary.json')
    if not receipt['complete'] or receipt['episode']!=a.episode or sha(a.results/'summary.json')!=row['sha256']:raise ValueError('Experiment report SHA')
    report=json.loads((a.results/'summary.json').read_text())
    b,source=verified_load(a.inputs,f'near-dev-{a.episode:03}.pt',a.episode,'development_external')
    states=b['physical_states'].numpy();goal=b['goal_state']
    if hasattr(goal,'numpy'):goal=goal.numpy()
    values=[coverage(p,goal[2:5]) for p in states[:,-1,2:5]]
    error=max(abs(x-y) for x,y in zip(values,reference['goal_aligned_coverage_by_candidate'][:9]))
    if error>1e-12:raise ValueError('Original nine physical outcomes differ from verified coverage source')
    native=next(r for r in report['rows'] if r['method']=='native');native_value=values[native['selected_action_index']]
    rows=[dict(control_parent=r['control_parent'],method=r['method'],selected_index=r['selected_action_index'],
        goal_aligned_coverage=values[r['selected_action_index']],delta_vs_native=values[r['selected_action_index']]-native_value)
        for r in report['rows'] if r['waypoint']=='central']
    result=dict(complete=True,episode=a.episode,source_input_sha256=source['sha256'],experiment_summary_sha256=row['sha256'],
        verified_goal_coverage_source_sha256=reference_row['sha256'],computed_vs_reference_maxabs=error,
        goal_block_pose=list(map(float,goal[2:5])),goal_aligned_coverage_original9=values,native_selected_coverage=native_value,rows=rows,
        scope='Descriptive expert-goal polygon coverage; not original painted-default native reward, no refit or primary endpoint replacement',
        simulator_calls=0,model_calls=0,held_accessed=False)
    write(a.output/'coverage.json',result)
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,outputs=[dict(path='coverage.json',sha256=sha(a.output/'coverage.json'),bytes=(a.output/'coverage.json').stat().st_size)]))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=range(4),required=True)
    for key in ('inputs','results','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
