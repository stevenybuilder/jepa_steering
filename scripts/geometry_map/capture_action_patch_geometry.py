#!/usr/bin/env python3
"""Read-only full-patch and actual gated-head geometry on immutable action banks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from capture_pusht_calibration import first_tensor,parameter_sha
from capture_horizon_coordinates import exact
from complete_cached_geometry import sha256,write_json

BLOCKS=(1,3,5)

def head_geometry(concatenated,weight,bias,gate,heads):
    b,n,d=concatenated.shape
    if d%heads or weight.shape!=(d,d) or gate.shape!=(b,n,d):raise ValueError('Native head dimensions disagree')
    x=concatenated.float().reshape(b,n,heads,d//heads)
    w=weight.float().reshape(d,heads,d//heads).permute(1,0,2)
    contributions=torch.einsum('bnhd,hod->bhno',x,w)*gate.float()[:,None]
    flat=contributions.flatten(2);norm=flat.norm(dim=-1)
    cos=(flat@flat.transpose(1,2))/(norm[:,:,None]*norm[:,None,:]).clamp_min(1e-20)
    pooled=contributions.mean(2);pn=pooled.norm(dim=-1)
    pc=(pooled@pooled.transpose(1,2))/(pn[:,:,None]*pn[:,None,:]).clamp_min(1e-20)
    shared=torch.zeros_like(gate) if bias is None else bias.float()[None,None]*gate.float()
    return dict(head_norm=norm,head_cosine=cos,head_pooled_norm=pn,head_pooled_cosine=pc,
        shared_bias_norm=shared.flatten(1).norm(dim=-1)),contributions.sum(1)+shared

class CaptureGeometry:
    def __init__(self,blocks):
        self.blocks=blocks;self.handles=[];self.calls=[0]*len(blocks);self.inputs={};self.gates={}
        self.residuals={i:[] for i in BLOCKS};self.heads={i:[] for i in range(len(blocks))};self.pending={}
    def __enter__(self):
        for i,block in enumerate(self.blocks):
            if block.training:raise ValueError('Read-only frozen eval model required')
            self.handles.append(block.register_forward_pre_hook(lambda module,args,i=i:self.pre(i,args)))
            self.handles.append(block.adaLN_modulation.register_forward_hook(lambda module,args,out,i=i:self.mod(i,out)))
            self.handles.append(block.attn.proj.register_forward_pre_hook(lambda module,args,i=i:self.projpre(i,args)))
            self.handles.append(block.attn.proj.register_forward_hook(lambda module,args,out,i=i:self.projpost(i,out)))
            self.handles.append(block.register_forward_hook(lambda module,args,out,i=i:self.post(i,out)))
        return self
    def pre(self,i,args):
        self.calls[i]+=1;self.inputs[i]=args[0]
        if self.calls[i]>6 or args[0].shape[1]%256:raise ValueError('Wrong native H6 residual axes')
    def mod(self,i,out):
        n=self.inputs[i].shape[1]
        if out.shape[-1]!=2400 or n%out.shape[1]:raise ValueError('Wrong actual AdaLN axes')
        self.gates[i]=out.chunk(6,dim=-1)[2].repeat_interleave(n//out.shape[1],dim=1)[:,-256:]
    def projpre(self,i,args):
        block=self.blocks[i];x=args[0][:,-256:]
        if int(block.attn.num_heads)!=16:raise ValueError('Pinned native16head checkpoint required')
        stats,total=head_geometry(x,block.attn.proj.weight,block.attn.proj.bias,self.gates[i],16)
        self.pending[i]=(stats,total)
    def projpost(self,i,out):
        stats,total=self.pending.pop(i);native=out[:,-256:].float()*self.gates[i].float()
        stats['head_sum_plus_shared_bias_maxabs']=(total-native).abs().flatten(1).max(1).values
        stats['native_gated_attention_norm']=native.flatten(1).norm(dim=-1)
        self.heads[i].append({k:v.detach().cpu() for k,v in stats.items()})
    def post(self,i,out):
        if i in BLOCKS:self.residuals[i].append(first_tensor(out)[:,-256:].detach().float().cpu().clone())
    def __exit__(self,*_):
        for h in self.handles:h.remove()
        self.handles=[]
    def arrays(self):
        if self.calls!=[6]*6:raise ValueError('Missing block/horizon capture')
        residual=torch.stack([torch.stack(self.residuals[i])[:,0] for i in BLOCKS],dim=1)
        # [imagined_horizon, block, head...] newest256 actual patch tokens.
        stats={k:torch.stack([torch.stack([v[k][0] for v in self.heads[i]]) for i in range(6)],dim=1)
               for k in self.heads[0][0]}
        return residual,stats

@torch.no_grad()
def run_source(args,row,wm):
    from tensordict import TensorDict
    p=args.bank/row['path']
    if sha256(p)!=row['sha256']:raise ValueError('Source SHA mismatch before loading')
    bank=torch.load(p,map_location='cpu',weights_only=False)
    if bank['row']['split']!=row['split']:raise ValueError('Payload split mismatch')
    z=TensorDict(bank['context'],batch_size=[]).to(wm.device)
    start=time.monotonic();patches=[];heads=[]
    for index,truth in enumerate(bank['candidates']):
        actions=bank['normalized_actions'][index,:,None].to(wm.device)
        saved=actions.clone()
        with CaptureGeometry(wm.model.predictor.predictor_blocks) as cap:out=wm.unroll(z.clone(),act_suffix=actions)
        exact(out['visual'][1:,0].float().cpu(),truth['predicted_visual'],'native visual forecast unchanged')
        exact(out['proprio'][1:,0].float().cpu(),truth['predicted_proprio'],'native proprio forecast unchanged')
        exact(actions,saved,'fixed candidate actions unchanged')
        dense,stats=cap.arrays()
        exact(dense[:,1].mean(1),truth['p3_pooled'],'P3 full patches reproduce source mean')
        patches.append(dense);heads.append(stats)
    full=torch.stack(patches);stats={k:torch.stack([r[k] for r in heads]) for k in heads[0]}
    if full.shape!=(7,6,3,256,400) or full.dtype!=torch.float32:raise ValueError('Dense shape/dtype contract failed')
    out=args.output/(row['key']+'.pt')
    meta=dict(key=row['key'],split=row['split'],source_bank_sha256=row['sha256'],source_bank_path=str(p),
        block_indices=BLOCKS,candidate_names=[c['name'] for c in bank['candidates']],
        real_context_raw_step=0,imagined_steps=list(range(1,7)),physical_raw_action_offsets=[5,10,15,20,25,30],
        patch_layout='row-major16x16; newest256 only; no mean before float32save',
        head_semantics='Head contribution after native output projection and actual AdaLN gate; shared projection bias separate',
        no_simulator_calls=True,no_cem_calls=True,seconds=time.monotonic()-start)
    torch.save(dict(complete=True,residual_patches=full,head_geometry=stats,meta=meta),out)
    np.savez_compressed(out.with_suffix('.npz'),residual_pooled=full.mean(3).numpy(),**{k:v.numpy() for k,v in stats.items()})
    entries=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key'],split=row['split']) for p in (out,out.with_suffix('.npz'))]
    write_json(out.with_suffix('.DONE.json'),dict(complete=True,outputs=entries,meta=meta,
        maximum_head_decomposition_roundoff=float(stats['head_sum_plus_shared_bias_maxabs'].max())))
    print(json.dumps(dict(event='dense_patch_geometry_complete',key=row['key'],seconds=time.monotonic()-start)),flush=True)
    return entries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bank','repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();d=json.loads((args.bank/'DONE.json').read_text())
    rows=[r for r in d['outputs'] if r['path'].endswith('.pt')]
    if not d['complete'] or any(r['split'] not in ('fit','development_validation','development_external') for r in rows):raise ValueError('Held or incomplete source before loading')
    if sha256(args.checkpoint)!='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb':raise ValueError('Wrong checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',dict(rows=rows,blocks=BLOCKS,dense_shape_per_source=[7,6,3,256,400],dtype='float32',
        readonly_hooks=True,script_sha256=sha256(__file__),checkpoint_sha256=sha256(args.checkpoint),no_held_access=True))
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505)
    wm,_,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);start=time.monotonic();outputs=[]
    for row in rows:outputs.extend(run_source(args,row,wm))
    if parameter_sha(wm)!=before:raise ValueError('Parameters changed')
    write_json(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
        parameter_sha256=before,provenance=provenance,simulator_calls=0,cem_calls=0))

if __name__=='__main__':main()
