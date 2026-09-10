#!/usr/bin/env python3
"""CPU-only final LayerNorm/split/readout of saved native P5 residuals."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

CHECKPOINT='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
SOURCE_DONE='13c9fd04fd41d0dc00312348d0167a318e3e0a31cd49a590921ee2f12de141fa'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def decomposition(x,delta):
    x=x.double();delta=delta.double();center=x-x.mean(-1,keepdim=True)
    mean=delta.mean(-1,keepdim=True).expand_as(delta)
    radial=((delta-mean)*center).sum(-1,keepdim=True)/center.square().sum(-1,keepdim=True).clamp_min(1e-30)*center
    tangent=delta-mean-radial
    return mean,radial,tangent


def layernorm_jvp(x,delta,weight,eps=1e-6):
    x=x.double();delta=delta.double();center=x-x.mean(-1,keepdim=True)
    d=delta-delta.mean(-1,keepdim=True);var=center.square().mean(-1,keepdim=True);scale=(var+eps).rsqrt()
    return weight.double()*scale*(d-center*(center*d).mean(-1,keepdim=True)/(var+eps))


def suffix(x,params):
    norm=F.layer_norm(x,(400,),params['norm_weight'].to(x),params['norm_bias'].to(x),eps=1e-6)
    visual=F.linear(norm[...,:384],params['projection_weight'].to(x),params['projection_bias'].to(x))
    return norm,visual,norm[...,384:]


def norms(value):return value.double().flatten(1).norm(dim=-1)


def energy_fraction(component,delta):
    numerator=component.double().square().sum();denominator=delta.double().square().sum()
    return None if denominator==0 else float(numerator/denominator)


def ratio(a,b):
    x=float(a);y=float(b)
    return None if y<=1e-30 else x/y


def contrast_rank(value):
    value=value.double().flatten(1);value=value-value[:1]
    s=torch.linalg.svdvals(value)
    return dict(singular_values=s.tolist(),rank_relative1e_5=int((s>s[0]*1e-5).sum()) if s[0]>0 else 0,
                rank_relative1e_6=int((s>s[0]*1e-6).sum()) if s[0]>0 else 0,
                maximum_rank_by_nine_plan_design=8)


def run(a):
    if torch.cuda.is_initialized():raise ValueError('CPU-only job')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.process_time();wall=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=False)
    source_receipt=json.loads((a.source/'DONE.json').read_text())
    if not source_receipt['complete'] or sha(a.source/'DONE.json')!=SOURCE_DONE:raise ValueError('Completed frozen P5 source receipt required')
    protocol_entry=next(v for v in source_receipt['outputs'] if v['path']=='protocol.json')
    if sha(a.source/'protocol.json')!=protocol_entry['sha256']:raise ValueError('Source protocol SHA')
    protocol_source=json.loads((a.source/'protocol.json').read_text())
    if protocol_source['block']!=5 or protocol_source['imagined_horizon']!=3:raise ValueError('P5/H3 only')
    if sha(a.checkpoint)!=CHECKPOINT:raise ValueError('Checkpoint SHA mismatch')
    predictor=a.repo/'app/plan_common/models/AdaLN_vit.py';video=a.repo/'app/vjepa_wm/video_wm.py'
    config=a.repo/'configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml'
    code=predictor.read_text();cfg=config.read_text();vid=video.read_text()
    required=['self.predictor_norm = norm_layer(self.predictor_total_embed_dim)','x = self.predictor_norm(x)','x = self.predictor_proj(x)','partial(nn.LayerNorm, eps=1e-6)']
    if not all(s in code for s in required) or 'normalize_reps: false' not in cfg or 'if self.normalize_reps:' not in vid:raise ValueError('Native suffix/config differs; no guessed reconstruction')
    write(a.output/'protocol.json',dict(complete=True,source_done_sha256=SOURCE_DONE,checkpoint_sha256=CHECKPOINT,
        source_code_sha256={str(p):sha(p) for p in (predictor,video,config)},script_sha256=sha(__file__),
        operation='Only saved P5 output -> native LayerNorm400 eps1e-6 -> visual384/proprio16 split -> visual learned384x384 projection',
        model_unroll_calls=0,simulator_calls=0,optimizer_steps=0,gpu_calls=0,held_access=False,max_CPU_process_seconds=120,
        arms=['native','dense_minus','dense_plus'],doses=[0,-.1,.1],imagined_horizon=3,
        parity='CPU-vs-savedGPU float32 maxabs and response difference reported; absolute parity tolerance1e-4, not bitexact claim',
        null_definition='Per-token uniform-channel shift exact LayerNorm null; radial centered-x direction approximate null for eps1e-6; remaining tangent separate',
        no_inference='Do not identify all attenuation with null directions: also measure inverse input scale, affine gain and projection gain'))
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);weights=checkpoint['predictor']
    names=dict(norm_weight='module.predictor_norm.weight',norm_bias='module.predictor_norm.bias',projection_weight='module.predictor_proj.weight',projection_bias='module.predictor_proj.bias')
    params={k:weights[v].detach().cpu().clone() for k,v in names.items()}
    action_weight=weights.get('module.action_encoder.weight')
    if action_weight is not None:action_weight=action_weight.detach().cpu().clone()
    del checkpoint,weights
    if params['norm_weight'].shape!=(400,) or params['projection_weight'].shape!=(384,384):raise ValueError('Native suffix parameter shapes differ')
    torch.save(params,a.output/'suffix_parameters.pt')
    _,singular,vh=torch.linalg.svd(params['projection_weight'].double(),full_matrices=False)
    projection_spectrum=dict(singular_values=singular.tolist(),rank_relative1e_6=int((singular>singular[0]*1e-6).sum()),
        isotropic_RMS_gain=float(singular.square().mean().sqrt()),minimum=float(singular.min()),maximum=float(singular.max()))
    summaries=[];tensor_outputs=[]
    for episode in range(4):
        if time.process_time()-started>105:raise RuntimeError('CPU budget insufficient for next state')
        name=f'near-dev-{episode:03}.pt';entry=next(v for v in source_receipt['outputs'] if v['path']==name)
        if sha(a.source/name)!=entry['sha256']:raise ValueError('State SHA before load')
        bank=torch.load(a.source/name,map_location='cpu',weights_only=False);original=bank['response']['original'];delta=bank['response']['delta']
        if original.shape!=(9,512,400) or delta.shape!=(9,256,400):raise ValueError('Frozen residual shape differs')
        if not torch.equal(delta[0],torch.zeros_like(delta[0])):raise ValueError('Central current-action null differs')
        x=original[:,-256:];components=decomposition(x,delta)
        action_excitation=dict(available=False)
        if action_weight is not None and 'z' in bank['response'] and 'counter_z' in bank['response']:
            condition_delta=(bank['response']['z'][:,-1]-bank['response']['counter_z'][:,-1]).double()
            u,ss,_=torch.linalg.svd(action_weight.double(),full_matrices=False)
            image=u[:,ss>ss[0]*1e-10];outside=condition_delta-(condition_delta@image)@image.T
            action_excitation=dict(available=True,condition_difference=contrast_rank(condition_delta),
                action_encoder_shape=list(action_weight.shape),action_encoder_singular_values=ss.tolist(),
                condition_outside_linear_action_image_energy_fraction=energy_fraction(outside,condition_delta),
                limitation='Nine dependent plan contrasts have rank at most eight by design; not an identification theorem')
            if 'normalized_actions' in bank:
                actions=bank['normalized_actions']
                if actions.shape!=(9,6,10):raise ValueError('Action time layout differs')
                expected=(actions[:,2].double()-actions[:1,2].double())@action_weight.double().T
                action_excitation.update(current_normalized_10D=contrast_rank(actions[:,2]),full_normalized_60D=contrast_rank(actions),
                    condition_vs_current_action_linear_map_maxabs=float((condition_delta-expected).abs().max()))
            if 'raw_actions' in bank:action_excitation['full_raw_60D']=contrast_rank(bank['raw_actions'])
        fractions={k:energy_fraction(v,delta) for k,v in zip(('channel_mean','centered_radial','remaining_tangent'),components)}
        by_candidate=[]
        for candidate in range(9):by_candidate.append(dict(candidate=candidate,raw_norm=float(norms(delta)[candidate]),
            fractions={k:energy_fraction(v[candidate],delta[candidate]) for k,v in zip(('channel_mean','centered_radial','remaining_tangent'),components)}))
        pre_var=x.double().var(-1,unbiased=False);std=(pre_var+1e-6).sqrt()
        base_norm,base_visual,base_prop=suffix(original,params)
        base_norm=base_norm[:,-256:];base_visual=base_visual[:,-256:];base_prop=base_prop[:,-256:]
        rows=[];saved={}
        for arm,dose in (('native',0.),('dense_minus',-.1),('dense_plus',.1)):
            changed=original.clone();requested=delta*dose;changed[:,-256:]+=requested
            delivered=changed[:,-256:]-x
            post,visual,prop=suffix(changed,params);post=post[:,-256:];visual=visual[:,-256:];prop=prop[:,-256:]
            before_affine=F.layer_norm(changed[:,-256:],(400,),eps=1e-6)-F.layer_norm(x,(400,),eps=1e-6)
            post_change=post-base_norm;visual_change=visual-base_visual;prop_change=prop-base_prop
            expected_visual=bank['predictions'][arm]['visual'][2].reshape(9,256,384)
            expected_prop=bank['predictions'][arm]['proprio'][2].reshape(9,256,16)
            saved_native_visual=bank['predictions']['native']['visual'][2].reshape(9,256,384)
            saved_native_prop=bank['predictions']['native']['proprio'][2].reshape(9,256,16)
            reference_v_change=expected_visual-saved_native_visual;reference_p_change=expected_prop-saved_native_prop
            parity=dict(visual_maxabs=float((visual-expected_visual).abs().max()),proprio_maxabs=float((prop-expected_prop).abs().max()),
                visual_MSE=float((visual.double()-expected_visual.double()).square().mean()),proprio_MSE=float((prop.double()-expected_prop.double()).square().mean()),
                visual_response_difference_norm=float((visual_change.double()-reference_v_change.double()).norm()),
                saved_visual_response_norm=float(reference_v_change.double().norm()),
                visual_response_relative_error=ratio((visual_change.double()-reference_v_change.double()).norm(),reference_v_change.double().norm()),
                proprio_response_relative_error=ratio((prop_change.double()-reference_p_change.double()).norm(),reference_p_change.double().norm()))
            if max(parity['visual_maxabs'],parity['proprio_maxabs'])>1e-4:raise ValueError('Saved GPU forecast does not match reconstructed CPU suffix within declared floor')
            jvp=layernorm_jvp(x,delivered,params['norm_weight'])
            projected_components=[layernorm_jvp(x,v*dose,params['norm_weight']) for v in components]
            coefficients=post_change[...,:384].double()@vh.T
            energy=coefficients.square().sum()
            low=singular<.1*singular[0]
            row=dict(arm=arm,dose=dose,requested_raw_norm=norms(requested).tolist(),delivered_raw_norm=norms(delivered).tolist(),
                pre_affine_LN_change_norm=norms(before_affine).tolist(),post_affine_LN_change_norm=norms(post_change).tolist(),
                visual_after_projection_change_norm=norms(visual_change).tolist(),proprio_output_change_norm=norms(prop_change).tolist(),
                normalization_no_affine_gain=ratio(before_affine.double().norm(),delivered.double().norm()),
                normalization_affine_gain=ratio(post_change.double().norm(),delivered.double().norm()),
                affine_gain=ratio(post_change.double().norm(),before_affine.double().norm()),
                visual_readout_gain=ratio(visual_change.double().norm(),post_change[...,:384].double().norm()),
                visual_low_singular_direction_energy_fraction=None if energy==0 else float(coefficients[...,low].square().sum()/energy),
                visual_input_fraction_of_postnorm_energy=energy_fraction(post_change[...,:384],post_change),
                LN_jvp_relative_error=ratio((post_change.double()-jvp).norm(),post_change.double().norm()),
                linearized_component_response_norm={k:float(v.norm()) for k,v in zip(('channel_mean','centered_radial','remaining_tangent'),projected_components)},
                saved_forecast_parity=parity)
            rows.append(row);saved[arm]=dict(delivered_delta=delivered,postnorm_change=post_change,visual_change=visual_change,proprio_change=prop_change)
        state=dict(episode=episode,source_sha256=entry['sha256'],raw_delta_energy_fractions=fractions,by_candidate=by_candidate,action_excitation=action_excitation,
            input_token_std_quantiles=torch.quantile(std,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),
            raw_delta_visual_channel_energy_fraction=energy_fraction(delta[...,:384],delta),rows=rows,
            CPU_process_seconds_so_far=time.process_time()-started)
        path=a.output/f'episode-{episode:03}-suffix.pt';torch.save(dict(state=state,responses=saved),path)
        tensor_outputs.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size));summaries.append(state)
        print(json.dumps(dict(event='suffix_state_complete',episode=episode,null_energy=fractions,normalization_gain=rows[-1]['normalization_affine_gain'],
            projection_gain=rows[-1]['visual_readout_gain'],seconds=time.process_time()-started)),flush=True)
        del bank,original,delta
    seconds=time.process_time()-started
    if seconds>120:raise RuntimeError('CPU process cap exceeded')
    result=dict(complete=True,states=summaries,CPU_process_seconds=seconds,wall_seconds=time.monotonic()-wall,
        projection_spectrum=projection_spectrum,norm_weight_abs_quantiles=torch.quantile(params['norm_weight'].double().abs(),torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),
        native_suffix_reconstructed=True,checkpoint_sha256=CHECKPOINT,source_done_sha256=SOURCE_DONE,
        model_unroll_calls=0,simulator_calls=0,gpu_calls=0,held_access=False,
        limitations=['LayerNorm radial direction is near-null, not exactly null with finite epsilon; uniform channel mean is algebraically null',
        'CPU suffix versus saved GPU forecasts has an explicitly measured float32 numerical floor; not claimed bitexact',
        'Suffix is tokenwise, so H3 newest-token forecast can be reconstructed; H4-H6 recurrence is not rerun or explained by this suffix alone',
        'Four repeated development states; dependent tokens/candidates are not independent evidence',
        'Normalization/readout attenuation is not a physical-utility or safety metric'])
    write(a.output/'summary.json',result)
    files=[a.output/'protocol.json',a.output/'suffix_parameters.pt',a.output/'summary.json']
    outputs=tensor_outputs+[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in files]
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,CPU_process_seconds=seconds,checkpoint_sha256=CHECKPOINT,source_done_sha256=SOURCE_DONE))
    print(json.dumps(dict(event='suffix_complete',seconds=seconds,wall_seconds=time.monotonic()-wall)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','checkpoint','repo','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except Exception:
        import traceback
        if a.output.exists():write(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()))
        raise
