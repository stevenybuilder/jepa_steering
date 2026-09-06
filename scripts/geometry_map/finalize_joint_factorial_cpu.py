#!/usr/bin/env python3
"""Recover complete scientific tensors after hard GPU-process timer interrupts CPU receipts."""
import argparse
import json
import time
from pathlib import Path
import torch
from run_patch_policy_action_spatial import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);started=time.monotonic()
    path=a.source/f'expanded-dev-{a.episode:03d}.pt';digest=sha(path)
    d=torch.load(path,map_location='cpu',weights_only=False)
    if not d['complete'] or len(d['predictions'])!=11 or d['raw_actions'].shape!=(64,30,2):raise ValueError('Incomplete scientific tensor')
    for k,v in d['predictions'].items():
        for field in ('visual','proprio'):
            if not torch.isfinite(v[field]).all():raise ValueError('Nonfinite prediction')
            if k!='native':
                if not torch.equal(v[field][:2],d['predictions']['native'][field][:2]):raise ValueError('Early horizon drift')
                if not torch.equal(v[field][:,0],d['predictions']['native'][field][:,0]):raise ValueError('Central drift')
    report=dict(complete=True,key=f'expanded-dev-{a.episode:03d}',source_sha256=d['source_sha256'],score=d['score'],interactions=d['interactions'],
        candidate_metadata=d['metadata'],crossbatch=d['crossbatch'],full_tensor_sha256=digest,full_tensor_bytes=path.stat().st_size,
        original9_actions_goal_physics_pixels_exact=True,all_forward_identity_guards_passed_before_save=True,
        central_and_early_guards_rechecked_cpu=True,final_parameter_hash_guard_reached=False,
        interrupted_gpu_process_cap_seconds=75,recovery_model_calls=0,recovery_simulator_calls=0,
        visual_delta_l2=d['records']['native']['delta_visual'].double().flatten(1).norm(dim=-1).tolist(),
        condition_delta_l2=d['records']['native']['delta_z'].double().norm(dim=-1).tolist(),
        recovery_scope='Complete tensor exists only after all eleven forwards, range/zero/native/central/past/action guards and output/metric decompositions passed; process timer stopped later CPU hashing/receipt work. No final live parameter hash comparison available.',
        goal_coverage_headroom=None)
    old=a.source/(report['key']+'.json')
    if old.exists():
        original=json.loads(old.read_text())
        if original['full_tensor_sha256']!=digest:raise ValueError('Existing report tensor SHA mismatch')
        report.update(original);report['final_parameter_hash_guard_reached']=False
    a.output.mkdir(parents=True,exist_ok=False)
    output=a.output/(report['key']+'.json');write(output,report)
    write(a.output/'CPU_FINALIZED.json',dict(complete=True,source_tensor=str(path),source_tensor_sha256=digest,output_sha256=sha(output),
        interrupted_source_preserved=True,original_gpu_done_missing=True,final_parameter_hash_guard_reached=False,
        cpu_seconds=time.monotonic()-started,model_calls=0,simulator_calls=0,held_access=False))
    print(json.dumps(dict(complete=True,episode=a.episode,cpu_seconds=time.monotonic()-started,choices={x['arm']:x['choice'] for x in report['score']})),flush=True)


if __name__=='__main__':main()
