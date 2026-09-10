#!/usr/bin/env python3
"""Compare fixed central-action recipients to donor-prefix/central-suffix physics."""
import argparse
import json
from pathlib import Path
import time
import torch
from summarize_action_path_curvature import average,episode_interval


def source_rows(root):
    d=json.loads((root/'DONE.json').read_text())
    if not d['complete']:raise ValueError('Incomplete physical/model source')
    return {r['path']:r for r in d['outputs']}


def checked(root,row):
    from complete_cached_geometry import sha256
    p=root/row['path']
    if sha256(p)!=row['sha256']:raise ValueError('Source hash failed before tensor load')
    return torch.load(p,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser()
    for key in ('interior','hybrid','donor','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    from complete_cached_geometry import sha256,write_json
    ri,rh,rd=[source_rows(p) for p in (a.interior,a.hybrid,a.donor)]
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.monotonic();rows=[]
    for e in range(4):
        for pair in range(4):
            for radius in (1,4):
                for sign,t in (('minus',-.25),('plus',.25)):
                    key=f'near-dev-{e:03}';stem=f'{key}-pair{pair}-radius{radius}'
                    model=checked(a.interior,ri[stem+f'-{sign}025.pt'])
                    donor=checked(a.donor,rd[stem+f'-t{sign}025.pt'])
                    for j,h in enumerate((1,3,6)):
                        truth=donor if h==6 else checked(a.hybrid,rh[stem+f'-t{sign}025-H{h}.pt'])
                        if not torch.equal(model['raw_target_actions'][:5*h],truth['raw_actions'][:5*h]):raise ValueError('Hybrid donor prefix action mismatch')
                        # Collector separately verifies original central suffix and exact paired physical states.
                        actual=truth['actual_visual'].float();pred=model['patched_visual'][j].float()
                        error=(pred-actual[:,None]).square().flatten(2).mean(-1)
                        conditions={name:dict(hybrid_actual_future_mse_by_horizon=error[:,i].tolist(),
                            hybrid_actual_future_mse_after_patch=float(error[h-1:,i].mean())) for i,name in enumerate(model['conditions'])}
                        rows.append(dict(key=key,pair=pair,radius=radius,t=t,patch_horizon=h,conditions=conditions,
                            physics_source_sha256=(rd[stem+f'-t{sign}025.pt'] if h==6 else rh[stem+f'-t{sign}025-H{h}.pt'])['sha256']))
    views=[]
    for radius in (1,4):
        for h in (1,3,6):
            panel=[r for r in rows if r['radius']==radius and r['patch_horizon']==h];keys=sorted({r['key'] for r in panel})
            def grouped(fn):return [average(fn(r) for r in panel if r['key']==key) for key in keys]
            condition_names=list(panel[0]['conditions'])
            def value(row,name):return row['conditions'][name]['hybrid_actual_future_mse_after_patch']
            conditions={name:episode_interval(grouped(lambda r,name=name:value(r,name))) for name in condition_names}
            contrasts={}
            for other in ('linear_equal_data','near_chord','reparameterized_chord','actual_donor_patch','exact_self_patch'):
                contrasts['cubic_minus_'+other]=episode_interval(grouped(lambda r,other=other:value(r,'cubic_equal_data')-value(r,other)))
            views.append(dict(radius=radius,horizon=h,conditions=conditions,contrasts=contrasts))
    write_json(a.output/'JOIN.json',dict(complete=True,rows=rows,views=views,targets=64,hybrid_cases=192,new_physical_hybrids=128,
        limitations=['Ground truth uses donor controls through raw5H and original CENTRAL controls afterwards; H6 reuses full donor truth',
            'Only one P3 residual is transplanted; previous visual/proprio context and remaining native action conditioning may not import the full donor physical state',
            'These are offline state-transfer/model-fidelity diagnostics, not executed edited policies or success outcomes',
            'Four initial states, correlated directions/interior/radius/horizons; descriptive intervals only',
            'Targets withheld from each curve fit;4x targets are previous1x endpoints, not globally novel']))
    outputs=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in a.output.glob('*.json')]
    write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
        source_model_done_sha256=sha256(a.interior/'DONE.json'),source_hybrid_done_sha256=sha256(a.hybrid/'DONE.json'),
        source_donor_done_sha256=sha256(a.donor/'DONE.json'),gpu_calls=0,simulator_calls=0))
    print(json.dumps(dict(complete=True,seconds=time.monotonic()-start,targets=64,hybrid_cases=192)),flush=True)


if __name__=='__main__':main()
