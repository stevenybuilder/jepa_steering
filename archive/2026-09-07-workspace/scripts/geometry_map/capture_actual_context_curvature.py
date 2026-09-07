#!/usr/bin/env python3
"""Natural versus actual-past observation contexts, without target-frame leakage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from action_path_geometry import sampled_path_geometry
from run_action_path_curvature import ResidualCapture,BLOCKS,HORIZONS,fixed_action_path


def allowed_history_indices(horizon,window=2):
    if horizon not in range(1,7) or window!=2:raise ValueError('Pinned H6/window2 temporal contract')
    # Frame0 is the initial observation, frameh is the target of predictionh.
    return list(range(max(0,horizon-window),horizon))


class ActualPastInputs:
    def __init__(self,predictor,visual,proprio,window=2):
        if visual.shape[:2]!=(7,7) or proprio.shape[:2]!=(7,7):raise ValueError('Expected batch7,time7 actual histories')
        self.module=predictor;self.visual=visual;self.proprio=proprio;self.window=window;self.calls=0;self.records=[]
    def __enter__(self):self.handle=self.module.register_forward_pre_hook(self.patch);return self
    def patch(self,module,args):
        self.calls+=1;indices=allowed_history_indices(self.calls,self.window)
        if len(args)!=3:raise ValueError('Pinned predictor must consume visual/action/proprio')
        visual,action,proprio=args
        if visual.shape[1]!=len(indices) or action.shape[1]!=len(indices) or proprio.shape[1]!=len(indices):raise ValueError('Actual/native history window length mismatch')
        self.records.append(dict(horizon=self.calls,actual_frame_indices=indices,largest_actual_frame=max(indices),
                                 target_frame=self.calls,actions_unchanged=True))
        if self.calls==1:return None
        first,last=indices[0],indices[-1]+1
        actual_visual=self.visual[:,first:last].to(visual);actual_proprio=self.proprio[:,first:last].to(proprio)
        if actual_visual.shape!=visual.shape or actual_proprio.shape!=proprio.shape:raise ValueError('Actual/native feature layout mismatch')
        # Both modes use unchanged native action features, including previous action slot.
        return actual_visual,action,actual_proprio
    def __exit__(self,*exc):
        self.handle.remove()
        if exc[0] is None and self.calls!=6:raise ValueError('Missing native horizon')


def load_actual_history(entry,bank):
    from complete_cached_geometry import sha256
    if entry.get('split') not in ('development','development_external') or entry.get('episode') not in range(4):
        raise ValueError('Actual-history split must be original development0..3 before loading')
    path=Path(entry['path'])
    if sha256(path)!=entry['sha256']:raise ValueError('Physical-input checksum failed before loading')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if not value.get('complete',True):raise ValueError('Incomplete physics payload')
    if 'encoded_actual_with_initial' in value:
        full=value['encoded_actual_with_initial'];visual=full['visual'];proprio=full['proprio']
    elif 'actual_visual_with_initial' in value:
        visual=value['actual_visual_with_initial'];proprio=value['actual_proprio_with_initial']
    else:
        visual=torch.cat([bank['context']['visual'][0],value['actual_visual']])
        proprio=torch.cat([bank['context']['proprio'][0],value['actual_proprio']])
    if visual.shape!=(7,1,16,16,384) or proprio.shape!=(7,256,16):raise ValueError('Require both actual modalities at all7model steps')
    initial_difference=dict(visual=float((visual[0]-bank['context']['visual'][0,0]).abs().max()),
        proprio=float((proprio[0]-bank['context']['proprio'][0,0]).abs().max()))
    # Initial source encoding is the identical frozen model stimulus in BOTH modes.
    visual=visual.clone().float();proprio=proprio.clone().float()
    visual[0]=bank['context']['visual'][0,0];proprio[0]=bank['context']['proprio'][0,0]
    raw=value.get('raw_actions',value.get('actions_raw'))
    if raw is None:raise ValueError('Actual history requires matching full raw action commands')
    return visual,proprio,torch.as_tensor(raw),initial_difference


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for key in ('bank','truth_index','repo','checkpoint','output'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    from complete_cached_geometry import sha256,write_json
    from model_loader import load_headless
    from capture_horizon_coordinates import exact
    from capture_pusht_calibration import parameter_sha
    from intervene_head_spatial_mean import batch_context
    done=json.loads((a.bank/'DONE.json').read_text());index=json.loads(a.truth_index.read_text())
    rows=sorted([r for r in done['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not done['complete'] or not index['complete'] or [r['key'] for r in rows]!=[f'near-dev-{e:03}' for e in range(4)] or any(r['split']!='development_external' for r in rows):raise ValueError('Only frozen original development0..3')
    lookup={(r['episode'],r['pair'],float(r['coefficient'])):r for r in index['rows']}
    expected={(e,p,c) for e in range(4) for p in range(4) for c in (-4.,-2.,-1.,-.5,0.,.5,1.,2.,4.)}
    if set(lookup)!=expected:raise ValueError('Missing/extra predetermined actual action histories')
    if sha256(a.checkpoint)!='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb':raise ValueError('Wrong native checkpoint')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(bank_done_sha256=sha256(a.bank/'DONE.json'),truth_index_sha256=sha256(a.truth_index),
        paths=32,t=[-1,-.5,0,.5,1],radii=[1,4],blocks=BLOCKS,horizons=HORIZONS,batch=7,window=2,
        modes=['native','actual_past_context'],actual_frame_index_rule='[max(0,H-2),...,H-1]; never targetH',
        H1_identity_required=True,source_initial_encoding_shared=True,current_and_past_actions_unchanged=True,
        script_sha256=sha256(__file__),timing_only_gate='firstpath projected32≤900seconds',operator_fit=False,
        held_data_access=False,simulator_calls=0,cem_calls=0))
    torch.set_num_threads(2);torch.manual_seed(90505)
    wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint)
    wm.eval().requires_grad_(False)
    if wm.ctxt_window!=2 or wm.proprio_mode!='predict_proprio':raise ValueError('Native context semantics changed')
    before=parameter_sha(wm);start=time.monotonic();outputs=[];first_canary=True
    for episode,row in enumerate(rows):
        path=a.bank/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Near-bank SHA before loading')
        bank=torch.load(path,map_location='cpu',weights_only=False);z=batch_context(bank['context'],7).to(wm.device)
        for pair in range(4):
            histories={c:load_actual_history(lookup[episode,pair,c],bank) for c in (-4.,-2.,-1.,-.5,0.,.5,1.,2.,4.)}
            for radius in (1,4):
                tick=time.monotonic();raw,meta=fixed_action_path(bank['raw_actions'],pair,radius)
                coefficients=[radius*t for t in (-1,-.5,0,.5,1)]
                for i,c in enumerate(coefficients):exact(raw[i],histories[c][2],'physics/model raw actions same')
                visual=torch.stack([histories[c][0] for c in coefficients]+[histories[0.][0]]*2).to(wm.device)
                proprio=torch.stack([histories[c][1] for c in coefficients]+[histories[0.][1]]*2).to(wm.device)
                normalized=prep.normalize_actions(raw.reshape(5,6,5,2)).reshape(5,6,10)
                plan=torch.cat([normalized,normalized[2:3].repeat(2,1,1)]).transpose(0,1).contiguous().to(wm.device);fixed=plan.clone()
                base=wm.unroll(z.clone(),act_suffix=plan)
                with ResidualCapture(wm.model.predictor.predictor_blocks) as native:
                    repeat=wm.unroll(z.clone(),act_suffix=plan)
                for k in ('visual','proprio'):exact(base[k],repeat[k],'native capture parity '+k)
                # Same hook and stored-history tensor layout, but replay the native
                # encoded history instead of actual observations to measure a layout floor.
                with ActualPastInputs(wm.model.predictor,base['visual'].transpose(0,1),base['proprio'].transpose(0,1)):
                    with ResidualCapture(wm.model.predictor.predictor_blocks) as layout_control:
                        layout_replay=wm.unroll(z.clone(),act_suffix=plan)
                for k in ('visual','proprio'):exact(layout_replay[k][:2],base[k][:2],'H1 native-history replay identity '+k)
                with ActualPastInputs(wm.model.predictor,visual,proprio) as teacher:
                    with ResidualCapture(wm.model.predictor.predictor_blocks) as conditioned:
                        changed=wm.unroll(z.clone(),act_suffix=plan)
                for k in ('visual','proprio'):exact(changed[k][:2],base[k][:2],'initial+H1 teacher-force identity '+k)
                for b in BLOCKS:exact(conditioned.values[b,1],native.values[b,1],'H1 residual identity')
                exact(plan,fixed,'Native actionhistory unchanged')
                truth=visual[:,1:].transpose(0,1).cpu();site_rows=[]
                for b in BLOCKS:
                    for h in HORIZONS:
                        site_rows.append(dict(block=b,horizon=h,native=sampled_path_geometry(native.values[b,h][:5]),
                            actual_past_context=sampled_path_geometry(conditioned.values[b,h][:5]),
                            native_history_replay_residual_maxabs=float((layout_control.values[b,h][:5]-native.values[b,h][:5]).abs().max()),
                            native_history_replay=sampled_path_geometry(layout_control.values[b,h][:5])))
                base_visual=base['visual'][1:].float().cpu();new_visual=changed['visual'][1:].float().cpu()
                native_mse=(base_visual-truth).square().flatten(2).mean(-1)[:,:5]
                teacher_mse=(new_visual-truth).square().flatten(2).mean(-1)[:,:5]
                name=f'{row["key"]}-pair{pair}-radius{radius}';pt=a.output/(name+'.pt')
                torch.save(dict(complete=True,raw_actions=raw,normalized_actions=normalized,
                    native_residuals=torch.stack([native.values[b,h][:5] for b in BLOCKS for h in HORIZONS]),
                    actual_past_residuals=torch.stack([conditioned.values[b,h][:5] for b in BLOCKS for h in HORIZONS]),
                    native_history_replay_residuals=torch.stack([layout_control.values[b,h][:5] for b in BLOCKS for h in HORIZONS]),
                    native_predicted_visual=base_visual[:,:5],actual_past_predicted_visual=new_visual[:,:5],actual_visual=truth[:,:5]),pt)
                seconds=time.monotonic()-tick;report=pt.with_suffix('.json')
                write_json(report,dict(complete=True,key=row['key'],episode=episode,pair=pair,radius=radius,rows=site_rows,
                    temporal_records=teacher.records,H1_exact=True,actions_exact=True,
                    native_actual_future_mse_by_horizon=native_mse.mean(1).tolist(),
                    actual_past_actual_future_mse_by_horizon=teacher_mse.mean(1).tolist(),
                    native_history_replay_visual_maxabs=float((layout_replay['visual']-base['visual']).abs().max()),
                    native_history_replay_visual_mse_by_horizon=(layout_replay['visual'][1:]-base['visual'][1:]).square().flatten(2).mean(-1)[:,:5].mean(1).cpu().tolist(),
                    per_pathpoint_native_mse=native_mse.T.tolist(),per_pathpoint_actual_past_mse=teacher_mse.T.tolist(),
                    initial_fresh_encoder_differences={str(c):histories[c][3] for c in coefficients},
                    seconds=seconds,full_tensor_sha256=sha256(pt),full_tensor_bytes=pt.stat().st_size,
                    limitations=['Actual past context is privileged offline information, not available inside prospective imagination',
                        'Teacher-forced curvature can reflect physical-state/encoder/dynamics variation; reduction does not prove all native curvature erroneous',
                        'Unchanged current and previous action conditioning, no operator fit, no edited policy rollout',
                        'Four development states, correlated directions/horizons/radii, no confirmation or density/manifold claim']))
                outputs.extend(dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in (pt,report))
                print(json.dumps(dict(event='actual_context_path_complete',key=row['key'],pair=pair,radius=radius,seconds=seconds)),flush=True)
                if first_canary:
                    write_json(a.output/'CANARY.json',dict(complete=True,seconds=seconds,projected32_seconds=seconds*32,H1_exact=True))
                    if seconds*32>900:raise RuntimeError('Timing-only15minute projection exceeded')
                    first_canary=False
    if parameter_sha(wm)!=before:raise ValueError('Native weights changed')
    write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,paths=32,
        parameter_sha256=before,provenance=provenance,simulator_calls=0,cem_calls=0))


if __name__=='__main__':main()
