#!/usr/bin/env python3
"""Propagated batch-floor reference and separate actual donor-future comparisons."""
import argparse
import json
from pathlib import Path
import time
import torch
from run_action_path_curvature import ResidualCapture,ResidualReplacement,HORIZONS


def checked_load(root,row):
    from complete_cached_geometry import sha256
    path=root/row['path']
    if sha256(path)!=row['sha256']:raise ValueError('SHA before loading '+str(path))
    return torch.load(path,map_location='cpu',weights_only=False)


def rows(root):
    done=json.loads((root/'DONE.json').read_text())
    if not done['complete']:raise ValueError('Incomplete source receipt')
    return {r['path']:r for r in done['outputs']}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for key in ('bank','curvature','interior','truth','repo','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    from complete_cached_geometry import sha256,write_json
    from model_loader import load_headless
    from capture_horizon_coordinates import exact
    from capture_pusht_calibration import parameter_sha
    from intervene_head_spatial_mean import batch_context
    rb,rc,ri,rt=[rows(p) for p in (a.bank,a.curvature,a.interior,a.truth)]
    if any(r.get('split')!='development_external' for r in rb.values() if r['path'].endswith('.pt')):raise ValueError('Wrong bank split')
    if sha256(a.checkpoint)!='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb':raise ValueError('Wrong checkpoint')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(floor_cases=12,states=[0,1,2,3],horizons=HORIZONS,batch=8,
        old_midpoint_source='fixed first path pair0/radius1',physics_join_targets=64,methods_refit=False,
        script_sha256=sha256(__file__),simulator_calls=0,cem_calls=0))
    torch.set_num_threads(2);torch.manual_seed(90505)
    wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);start=time.monotonic();floor=[];signatures=[]
    for e in range(4):
        key=f'near-dev-{e:03}';bank=checked_load(a.bank,rb[key+'.pt'])
        old=checked_load(a.curvature,rc[key+'-pair0-radius1.pt'])
        z=batch_context(bank['context'],8).to(wm.device)
        plan=bank['normalized_actions'][0,:,None].repeat(1,8,1).contiguous().to(wm.device)
        capture=[]
        def sig(module,args):
            capture.append([dict(shape=list(v.shape),dtype=str(v.dtype),stride=list(v.stride())) if isinstance(v,torch.Tensor) else None for v in args])
        handle=wm.model.predictor.register_forward_pre_hook(sig)
        with ResidualCapture(wm.model.predictor.predictor_blocks) as mid:base=wm.unroll(z.clone(),act_suffix=plan)
        handle.remove();signatures.append(dict(key=key,ctxt_window=wm.ctxt_window,proprio_mode=wm.proprio_mode,
            action_encoder_inpred=wm.model.action_encoder_inpred,proprio_encoder_inpred=wm.model.proprio_encoder_inpred,
            predictor_inputs_per_horizon=capture))
        for j,h in enumerate(HORIZONS):
            old_mid=old['residual_samples'][3+j,2]
            replacement=torch.stack([mid.values[3,h][0]]+[old_mid]*7).to(wm.device)
            with ResidualReplacement(wm.model.predictor.predictor_blocks[3],h,replacement):changed=wm.unroll(z.clone(),act_suffix=plan)
            for k in ('visual','proprio'):
                exact(changed[k][:,0],base[k][:,0],'floor exact self '+k)
                exact(changed[k][:h],base[k][:h],'floor preceding horizon '+k)
            delta=changed['visual'][1:,1].float().cpu()-base['visual'][1:,1].float().cpu()
            rd=old_mid.double()-mid.values[3,h][0].double()
            floor.append(dict(key=key,horizon=h,residual_l2=float(rd.norm()),residual_maxabs=float(rd.abs().max()),
                propagated_fullspatial_mse_by_horizon=delta.square().flatten(1).mean(-1).tolist(),
                propagated_fullspatial_mse_after_patch=float(delta[h-1:].square().mean()),
                propagated_fullspatial_maxabs_after_patch=float(delta[h-1:].abs().max()),self_exact=True))
    if parameter_sha(wm)!=before:raise ValueError('Parameters changed')
    gpu_seconds=time.monotonic()-start
    write_json(a.output/'PROPAGATED_BATCH_FLOOR.json',dict(complete=True,rows=floor,runtime_signatures=signatures,
        seconds=gpu_seconds,parameter_sha256=before,provenance=provenance,scope='Frozen midpoint batch7 placed in native batch8 central recipient; no operator fit'))
    del wm;torch.cuda.empty_cache();joined=[]
    for e in range(4):
        key=f'near-dev-{e:03}'
        for pair in range(4):
            for radius in (1,4):
                for sign,t in (('minus',-.25),('plus',.25)):
                    stem=f'{key}-pair{pair}-radius{radius}'
                    predicted=checked_load(a.interior,ri[stem+f'-{sign}025.pt'])
                    physical=checked_load(a.truth,rt[stem+f'-t{sign}025.pt'])
                    exact(predicted['raw_target_actions'],physical['raw_actions'],'interior target physical actions')
                    exact(predicted['normalized_target_actions'],physical['normalized_actions'],'interior target normalized actions')
                    truth=physical['actual_visual'].float()
                    donor_error=(predicted['natural_donor_visual'].float()-truth).square().flatten(1).mean(-1)
                    native_error=(predicted['native_central_visual'].float()-truth).square().flatten(1).mean(-1)
                    values=[]
                    for j,h in enumerate(HORIZONS):
                        # Predicted rows are central-action recipients: do not relabel oracle patch as executed donor dynamics.
                        error=(predicted['patched_visual'][j]-truth[:,None]).square().flatten(2).mean(-1)
                        values.append(dict(horizon=h,conditions={name:dict(donor_action_actual_future_mse_by_horizon=error[:,i].tolist(),
                            donor_action_actual_future_mse_after_patch=float(error[h-1:,i].mean())) for i,name in enumerate(predicted['conditions'])}))
                    joined.append(dict(key=key,pair=pair,radius=radius,t=t,native_donor_action_mse_by_horizon=donor_error.tolist(),
                        native_central_action_compared_to_donor_truth_mse_by_horizon=native_error.tolist(),sites=values,
                        original_physics_truth_sha256=rt[stem+f'-t{sign}025.pt']['sha256'],
                        model_interchange_sha256=ri[stem+f'-{sign}025.pt']['sha256']))
    write_json(a.output/'ACTUAL_DONOR_FUTURE_JOIN.json',dict(complete=True,targets=64,rows=joined,
        limitations=['Only native donor-action forecast is directly action-aligned with actual donor simulator futures',
            'All patched estimates and oracle residual use CENTRAL-action recipients; their donor-future errors measure transfer, not native policy prediction',
            'Oracle donor residual is a privileged diagnostic reference, not deployable steering information',
            'No physical edited policy executed, no CEM, no held panel or confirmation opened']))
    outputs=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in a.output.glob('*.json')]
    write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,gpu_stage_seconds=gpu_seconds,total_seconds=time.monotonic()-start,
        floor_cases=12,physics_join_targets=64,simulator_calls=0,cem_calls=0))
    print(json.dumps(dict(complete=True,gpu_stage_seconds=gpu_seconds,total_seconds=time.monotonic()-start,floor_cases=12,physics_join_targets=64)),flush=True)


if __name__=='__main__':main()
