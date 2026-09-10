#!/usr/bin/env python3
"""Native Reach action interactions at fixed development scenes; no CEM or labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from capture_interface_response import finite_response
from protocol import file_sha256, write_json_atomic

SCALES = (.5, 1.)
SIGNS = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))


def factorial_actions(original, anchor_label='baseline_zero_valid'):
    """Primary all-zero valid commands; optional original future held exactly."""
    if original.shape != (30,4) or not torch.isfinite(original).all():
        raise ValueError("Finite native H6 raw actions [30,4] required")
    if anchor_label not in ('baseline_zero_valid','native_future_fixed'):
        raise ValueError('Unknown frozen anchor')
    anchor = torch.zeros_like(original) if anchor_label=='baseline_zero_valid' else original.clone()
    anchor[:5,:3] = 0
    actions, rows = [anchor], [{"scale":0., "x_sign":0,"z_sign":0,"raw_amplitude":0.}]
    for scale in SCALES:
        for xsign,zsign in SIGNS:
            value=anchor.clone(); value[:5,0]=.05*scale*xsign; value[:5,2]=.05*scale*zsign
            actions.append(value);rows.append({"scale":scale,"x_sign":xsign,"z_sign":zsign,"raw_amplitude":.05*scale})
    values=torch.stack(actions)
    if not torch.equal(values[:,:,3],anchor[:,3].expand(17,-1)) or not torch.equal(values[:,5:],anchor[5:].expand(17,-1,-1)):
        raise RuntimeError("Frozen gripper/future actions changed")
    if (values[:,:5,:3].abs()>1).any():
        raise RuntimeError("Perturbed coordinates leave raw bounds; no clipping allowed")
    return values,rows


def interaction_vectors(zero, x, z, both):
    if any(v.shape!=zero.shape for v in (x,z,both)):
        raise ValueError("Aligned factorial tensors required")
    return both.double()-x.double()-z.double()+zero.double()


class CaptureStages:
    def __init__(self, blocks):
        self.blocks=blocks;self.values={name:[] for name in blocks};self.handles=[]
    def save(self,name,value):
        if value.ndim!=3 or value.shape[-1]!=400 or value.shape[1]%256:
            raise RuntimeError("Expected actual predictor residual [B,T*256,400]")
        self.values[name].append(value[:,-256:].detach().float().cpu().clone())
    def __enter__(self):
        for name,block in self.blocks.items():
            self.handles.append(block.register_forward_hook(lambda m,a,v,name=name:self.save(name,v)))
        return self
    def __exit__(self,*_):
        for h in self.handles:h.remove()


def ratio(a,b):
    return [float(x/y) if float(y)>1e-15 else None for x,y in zip(a,b)]


def summarize_stages(stages, conditions):
    lookup={(r['scale'],r['x_sign'],r['z_sign']):i for i,r in enumerate(conditions)}
    rows=[];signed_pooled={}
    for stage, captured in stages.items():
        if captured.ndim!=4 or captured.shape[:2]!=(6,17):
            raise RuntimeError("Expected six horizons,17 conditions,spatial tokens,channels")
        for representation,values in (("full_tensor",captured),("spatial_mean",captured.mean(2))):
            zero=values[:,0]
            base=zero.double().reshape(6,-1).norm(dim=-1)
            for scale in SCALES:
                amplitude=.05*scale
                for axis, signs in (("x",((1,0),(-1,0))),("z",((0,1),(0,-1)))):
                    plus,minus=(values[:,lookup[(scale,*s)]] for s in signs)
                    metrics=finite_response(zero,plus,minus,amplitude)
                    for t in range(6):
                        rows.append({"stage":stage,"representation":representation,"scale":scale,"imagined_step":t+1,"kind":"axis_curvature","axis":axis,
                                     **{k:(v[t] if isinstance(v,list) else v) for k,v in metrics.items()}})
                    if representation=="spatial_mean":
                        signed_pooled[f"{stage}/{scale}/{axis}/first_derivative"]=(plus.double()-minus.double())/(2*amplitude)
                        signed_pooled[f"{stage}/{scale}/{axis}/curvature"]=(plus.double()+minus.double()-2*zero.double())
                for sx,sz in ((1,1),(1,-1),(-1,1),(-1,-1)):
                    x,z,both=(values[:,lookup[(scale,*s)]] for s in ((sx,0),(0,sz),(sx,sz)))
                    mixed=interaction_vectors(zero,x,z,both)
                    norm=mixed.reshape(6,-1).norm(dim=-1)
                    main=(x.double()-zero.double()).reshape(6,-1).norm(dim=-1)+(z.double()-zero.double()).reshape(6,-1).norm(dim=-1)
                    relative, relative_base = ratio(norm,main),ratio(norm,base)
                    for t in range(6):
                        rows.append({"stage":stage,"representation":representation,"scale":scale,"imagined_step":t+1,"kind":"xz_interaction","x_sign":sx,"z_sign":sz,
                                     "mixed_difference_l2":float(norm[t]),"mixed_derivative_l2":float(norm[t]/amplitude**2),
                                     "mixed_difference_signed_coordinate_mean":float(mixed[t].mean()),
                                     "interaction_relative_to_sum_single_effects":relative[t],"interaction_relative_to_baseline":relative_base[t]})
                    if representation=="spatial_mean":signed_pooled[f"{stage}/{scale}/mixed/{sx}/{sz}"]=mixed
    return rows,signed_pooled


def load_checked_plan(path,expected_sha):
    if file_sha256(path)!=expected_sha:
        raise RuntimeError("Verified completed development source SHA required before tensor load")
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['episode'] not in (0,4,10) or not value.get('complete'):
        raise RuntimeError("Only predetermined development starts0,4,10 at replan0")
    if 'initial_snapshot' in value:
        if value['arm']!='unsteered':raise RuntimeError('Only completed fresh unsteered initial scene')
        value={'episode':value['episode'],'raw_actions':torch.zeros(30,4),
               'raw_context_visual':value['initial_snapshot']['visual'],
               'raw_context_proprio':value['initial_snapshot']['proprio']}
    elif value.get('split')!='development' or value.get('replan')!=0:
        raise RuntimeError('Invalid development horizon source')
    return value,expected_sha


@torch.no_grad()
def capture_scene(source,source_sha,wm,preprocessor,output):
    from evals.simu_env_planning.planning.utils import make_td
    episode=source['episode'];started=time.monotonic()
    raw,conditions=factorial_actions(source['raw_actions'])
    normalized=preprocessor.normalize_actions(raw.reshape(-1,4)).reshape(17,6,20).permute(1,0,2).contiguous().cuda()
    td=make_td(source['raw_context_visual'].clone(),{'proprio':source['raw_context_proprio'].clone()})
    z=wm.encode(td.cuda().unsqueeze(0),act=True)
    original={k:z[k].clone() for k in z.keys()}
    unhooked=wm.unroll(z.clone(),act_suffix=normalized)
    blocks={f'P{i}':wm.model.predictor.predictor_blocks[i] for i in (0,3,5)}
    counts={k:len(v._forward_hooks) for k,v in blocks.items()}
    with CaptureStages(blocks) as capture:
        repeated=wm.unroll(z.clone(),act_suffix=normalized)
    for key in ('visual','proprio'):
        if not torch.equal(repeated[key],unhooked[key]):raise RuntimeError('Actual capture/no-hook fullH6 identity failed:'+key)
    for key,value in original.items():
        if not torch.equal(z[key],value):raise RuntimeError('Encoded observation mutated:'+key)
    if any(len(b._forward_hooks)!=counts[k] for k,b in blocks.items()):raise RuntimeError('Capture hooks not cleaned')
    stages={name:torch.stack(values) for name,values in capture.values.items()}
    stages['returned_visual']=repeated['visual'][1:].detach().float().cpu().reshape(6,17,256,384)
    if not all(torch.isfinite(x).all() for x in stages.values()):raise RuntimeError('Nonfinite actual model response')
    rows,signed=summarize_stages(stages,conditions)
    result={'complete':True,'episode':episode,'source_sha256':source_sha,'conditions':conditions,'raw_actions':raw,'normalized_actions':normalized.cpu(),
            'stages':stages,'signed_pooled_finite_differences':signed,'encoded_context':{k:v.cpu() for k,v in original.items()},
            'raw_context_visual':source['raw_context_visual'],'raw_context_proprio':source['raw_context_proprio'],
            'returned_proprio':repeated['proprio'].cpu(),'actual_capture_nohooks_identity_exact':True,'encoded_start_unchanged':True,
            'unchanged_original_future_out_of_bounds_entries':int((source['raw_actions'][5:].abs()>1).sum()),
            'unchanged_original_gripper_out_of_bounds_entries':int((source['raw_actions'][:,3].abs()>1).sum()),
            'action_clipping_performed':False,'anchor_label':'baseline_zero_valid','seconds':time.monotonic()-started}
    path=output/f'episode-{episode:03d}.pt';torch.save(result,path)
    report=output/f'episode-{episode:03d}.json'
    write_json_atomic(report,{k:v for k,v in result.items() if k in ('complete','episode','source_sha256','seconds','actual_capture_nohooks_identity_exact','encoded_start_unchanged','unchanged_original_future_out_of_bounds_entries','unchanged_original_gripper_out_of_bounds_entries','action_clipping_performed')}|{'rows':rows})
    entries=[{'path':p.name,'sha256':file_sha256(p),'bytes':p.stat().st_size,'episode':episode} for p in (path,report)]
    write_json_atomic(output/f'episode-{episode:03d}.DONE.json',{'complete':True,'outputs':entries})
    print(json.dumps({'event':'action_scene_complete','episode':episode,'seconds':result['seconds'],'outputs':entries}),flush=True)
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--sources',type=Path,nargs='+',required=True)
    parser.add_argument('--source-shas',nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    if len(args.sources)!=len(args.source_shas):raise RuntimeError('Each scene needs its verified source SHA')
    sources=[load_checked_plan(p,sha) for p,sha in zip(args.sources,args.source_shas)]
    if len({x['episode'] for x,_ in sources})!=len(sources):raise RuntimeError('Duplicate scene')
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    protocol={'episodes':[x['episode'] for x,_ in sources],'predetermined_complete_scene_set':[0,4,10],
              'native_horizon':6,'conditions':17,'anchor_label':'baseline_zero_valid',
              'translation_anchor':'All30 raw translations ANDgripper zero; first5 X/Z pulsed only; remaining25 commands held zero',
              'optional_secondary_anchor':'native_future_fixed not executed in primary test',
              'raw_axes':['X','Z'],'raw_amplitudes':[.025,.05],'normalizer':'Official checkpoint preprocessor, no clipping',
              'interpretation':'Local actual-network action interaction, not geometric manifold proof or simulator performance',
              'new_cem_optimizations':0,'new_physical_forks':0,'future_labels_used_by_model':False,'held_sources_loaded':0,
              'script_sha256':file_sha256(Path(__file__)),'budget_seconds':1800,
              'signed_vector_storage':'Full native stage tensors permit exact reconstruction; signed spatially pooled first/curvature/mixed vectors stored separately',
              'context_encoding':'Fresh frozen-model encoding of exact saved initial pixels/proprio; same encoding at all17 conditions'}
    write_json_atomic(args.output_dir/'protocol.json',protocol)
    outputs=[]
    try:
        wm,preprocessor,provenance=load_headless_metaworld(args.repo);wm.eval().requires_grad_(False)
        before=parameters_sha(wm)
        for source,sha in sources:
            if time.monotonic()-started>1800:raise RuntimeError('Action probe budget elapsed')
            print(json.dumps({'event':'action_scene_started','episode':source['episode'],'conditions':17}),flush=True)
            outputs.extend(capture_scene(source,sha,wm,preprocessor,args.output_dir))
        if parameters_sha(wm)!=before:raise RuntimeError('Frozen weights changed')
        write_json_atomic(args.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'seconds':time.monotonic()-started,'model':provenance,'parameters_sha256':before})
    except Exception as exc:
        write_json_atomic(args.output_dir/'FAILED.json',{'complete':False,'error':str(exc),'outputs':outputs});raise


if __name__=='__main__':main()
