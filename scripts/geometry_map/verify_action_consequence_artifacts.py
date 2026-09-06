#!/usr/bin/env python3
"""Checksum-scoped completion receipt for the action-consequence diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    roots=[]
    for worker in ('49987402','49987413','49987414'):
        for name in ('bank-v2','intervention-v1'):roots.append(a.root/f'worker-{worker}'/name)
    for worker in ('49987402','49987413'):
        for name in ('finite-response-v1','causal-response-evaluation-v1'):roots.append(a.root/f'worker-{worker}'/name)
    roots.extend([a.root/'worker-49987402/causal-response-train-v2',a.root/'worker-49987413/causal-response-train-v1',
        a.root/'causal-response-models-v1',a.root/'calibration-v1/output-v1',a.root/'nearplan-v1/bank-v2',
        a.root/'nearplan-v1/intervention-v2',a.root/'nearplan-v1/causal-response-evaluation-v1'])
    remote=json.loads((a.root/'nearplan-v1/REMOTE_BACKUP_VERIFIED.json').read_text())
    if not remote['complete']:raise ValueError('Second remote backup incomplete')
    backup={r['sha256']:r for r in remote['files']};outputs=[];stages=[]
    for directory in roots:
        receipt=json.loads((directory/'DONE.json').read_text())
        if not receipt['complete']:raise ValueError('Incomplete stage')
        stages.append(dict(path=str(directory),done_sha256=sha(directory/'DONE.json'),seconds=receipt.get('seconds')))
        for r in receipt['outputs']:
            path=directory/r['path'];entry=dict(path=str(path),expected_sha256=r['sha256'])
            if path.exists() and sha(path)==r['sha256']:
                entry.update(local_verified=True,bytes=path.stat().st_size)
            elif r['sha256'] in backup and path.suffix=='.pt':
                entry.update(local_verified=False,second_remote_verified=backup[r['sha256']],bytes=backup[r['sha256']]['bytes'])
            else:raise ValueError('Missing or mismatching unbacked output: '+str(path))
            outputs.append(entry)
    reports=[]
    for name in ('intervention_summary_v1.json','finite_response_summary_v1.json','nearplan_summary_v2.json',
        'causal_response_transfer_summary_v1.json','nearplan_causal_response_summary_v1.json','nearplan-v1/ranking_analysis_v1.json'):
        path=a.root/name;value=json.loads(path.read_text())
        if not value['complete']:raise ValueError('Incomplete report')
        reports.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    result=dict(complete=True,scope='Coarse, near-plan, finite-response and TRAIN causal-response transfer; dense-chart extension separately tracked',
        unique_initialstate_groups=24,training_groups=16,development_groups=8,coarse_actions=168,near_actions=36,
        exact_paired_physical_replays=408,physical_raw_steps=12240,real_modelstep_future_points=1224,
        stage_receipts=stages,verified_outputs=outputs,verified_reports=reports,
        local_output_count=sum(r.get('local_verified',False) for r in outputs),remote_only_output_count=sum(not r.get('local_verified',False) for r in outputs),
        successful_stage_seconds=sum(r['seconds'] or 0 for r in stages),timing_semantics='Receipt stage wall seconds, not measured GPU utilization integral; excludes some loading/failed-startup overhead',
        confirmation_or_54_held_panel_accessed=False,weights_updated=False,native_cem_calls=0,
        failed_attempts_preserved=['action-bank v1 TensorDict axis adapter failure','near-plan v1 observed-state double-integration reset failure',
            'TRAIN-response402v1 strict fresh encoding mismatch1.192e-7; v2 original float32 goal fixed stimulus'],
        storage='Three near-plan PTs remote-only with verified second-host backup; local disk-space incident, no original data deletion',
        scientific_status='No demonstrated semantic action-ranking or task-performance improvement; improved response-Jacobian generalization is a separate result',
        script_sha256=sha(Path(__file__)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('complete','local_output_count','remote_only_output_count','successful_stage_seconds')}))

if __name__=='__main__':main()
