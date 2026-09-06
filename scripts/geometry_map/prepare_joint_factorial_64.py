#!/usr/bin/env python3
"""CPU-only compact, receipt-linked adapter; never creates new model or physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_patch_policy_action_spatial import sha, write, exact


def same(a,b,label):
    if isinstance(a,torch.Tensor): exact(a,torch.as_tensor(b),label)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b): raise ValueError(label)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise ValueError(label+' keys')
        for k in a: same(a[k],b[k],label+'/'+k)
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise ValueError(label+' lengths')
        for i,(x,y) in enumerate(zip(a,b)): same(x,y,label+'/'+str(i))
    elif a!=b: raise ValueError(label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','near','output'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    receipt=json.loads((a.source/'DONE.json').read_text())
    if not receipt['complete'] or receipt['plans']!=64 or receipt.get('held_access',True): raise ValueError('Wrong completed bank')
    index={r['path']:r for r in receipt['outputs']}
    def checked(name):
        path=a.source/name
        if sha(path)!=index[name]['sha256']: raise ValueError('Source hash before load '+name)
        return torch.load(path,map_location='cpu',weights_only=False)
    f=checked('FROZEN_INPUTS.pt');e=f['episode']
    if e not in range(4) or f['split']!='development_external': raise ValueError('Only four original seen states')
    nearpath=a.near/f['source']['path']
    if sha(nearpath)!=f['source']['sha256']: raise ValueError('Original9 source hash')
    near=torch.load(nearpath,map_location='cpu',weights_only=False)
    for k,old in [('original_context','context'),('raw_goal','raw_goal'),('goal_encoded','goal_encoded'),('goal_state','goal_state')]: same(f[k],near[old],k)
    for k in ('raw_actions','normalized_actions'): same(f[k][:9],near[k],k)
    candidates=[];source_rows=[]
    for i in range(64):
        name=f'candidate-{i:03d}.pt';c=checked(name)
        if c['candidate']!=i or c['episode']!=e or c['split']!='development_external': raise ValueError('Candidate identity')
        if c['frozen_inputs_sha256']!=index['FROZEN_INPUTS.pt']['sha256']: raise ValueError('Candidate frozen input link')
        for k in ('raw_actions','normalized_actions'): same(c[k],f[k][i],k)
        if i<9:
            old=near['candidates'][i]
            for k in ('states','physics','native_applied_targets_xy'): same(c['truth'][k],old[k],'Original9 '+k)
            same(c['truth']['frames'][::5],old['frames_modelsteps'],'Original9 observed pixels')
        candidates.append(dict(actual_visual=c['actual_visual'],states=c['truth']['states'],
            predicted_visual=c['predicted_visual'],predicted_proprio=c['predicted_proprio'],native_cost_by_horizon=c['native_cost_by_horizon']))
        source_rows.append(index[name])
    a.output.mkdir(parents=True,exist_ok=False);name=f'expanded-dev-{e:03d}'
    row=dict(key=name,source_id=e,split='development_external')
    bank=dict(complete=True,row=row,context=f['original_context'],raw_goal=f['raw_goal'],goal_encoded=f['goal_encoded'],goal_state=f['goal_state'],
        raw_actions=f['raw_actions'],normalized_actions=f['normalized_actions'],candidates=candidates,metadata=f['metadata'],
        original9_actions_goal_physics_pixels_exact=True,source_done_sha256=sha(a.source/'DONE.json'),source_frozen_sha256=index['FROZEN_INPUTS.pt']['sha256'],
        original9_source=f['source'],source_root=str(a.source),source_rows=source_rows,held_access=False)
    path=a.output/(name+'.pt');torch.save(bank,path)
    row.update(path=path.name,sha256=sha(path),bytes=path.stat().st_size)
    write(a.output/'DONE.json',dict(complete=True,outputs=[row],original9_actions_goal_physics_pixels_exact=True,
        source_done_sha256=bank['source_done_sha256'],source_frozen_sha256=bank['source_frozen_sha256'],source_rows=source_rows,
        metadata=f['metadata'],episode=e,held_access=False,model_calls=0,simulator_calls=0))
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
