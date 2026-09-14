"""Finite eight-scenario posthoc attention/CEM/cached-correction diagnostics.

Inputs are the first four canonical archived DEVELOPMENT scenarios per task.
All physical outcomes remain historical; this module never creates a simulator.
"""
from __future__ import annotations
from offline_study._paths import source_path

import argparse
import json
import time
from pathlib import Path

import torch

from offline_study.experiments.candidate_replay_pilot import CaptureFull, ReplayField, components, score_summary
from offline_study.experiments.candidate_score_trace import CandidateScoreTrace
from offline_study.validation.fixed_combined_check import assert_bytes, rng_signature
from offline_study.interventions.fixed_response import FixedResponseIntervention, load_fitted_bank
from offline_study.experiments.layer_attention_pilot import PredictorObserver, file_hash


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def run_case(backend, bank, cfg, data, binding, output, execution_manifest_sha256=None):
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    if set(data) != {'initial','goal','actions'} or data['actions'].shape != (6,300,20):
        raise ValueError('Unexpected archived scenario input schema')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    initial=TensorDict(data['initial'],batch_size=[])
    goal=TensorDict(data['goal'],batch_size=[])
    obs_type=cfg['task_specification']['obs']
    context=backend.model.encode(prepare_obs(obs_type,initial).to(backend.device).unsqueeze(0),act=True)
    target=backend.model.encode(prepare_obs(obs_type,goal).to(backend.device).unsqueeze(0),act=False).detach()
    objective=ReprTargetDistMPCObjective(cfg,target,**cfg['planner']['planning_objective'])
    actions=data['actions'].to(backend.device)
    initial_rng=rng_signature('cuda:0')
    fixed=FixedResponseIntervention(backend,bank,'fixed_rank4')

    # The archived sampler deterministically inserted this zero-mean action;
    # no candidate selection is based on scores or physical outcomes.
    attention_actions=actions[:,:1].contiguous()
    if torch.count_nonzero(attention_actions):
        raise ValueError('The registered original candidate0 is not zero-mean')
    unobserved=backend.predict(context,attention_actions)
    with PredictorObserver(backend.predictor) as observer:
        observed=backend.predict(context,attention_actions)
    for key in ('visual','proprio'):
        assert_bytes(unobserved[key],observed[key],'development attention output '+key)
    if len(observer.rows)!=36 or any(sum(r['layer']==layer for r in observer.rows)!=6 for layer in range(6)):
        raise ValueError('Missing predictor block or H1-H6 attention')
    for index,row in enumerate(observer.rows):
        row['forecast_horizon']=index//6+1
    dump(output/'attention.json',{'input_binding':binding,'attention':observer.rows,
        'action_selection':'original archived candidate0 zero-mean action',
        'output_byte_parity':True,'same_input_rng_unchanged':rng_signature('cuda:0')==initial_rng})
    del unobserved,observed

    def planner():
        generator=torch.Generator(device='cuda:0').manual_seed(binding['candidate_seed'])
        result=CEMPlanner(unroll=backend.model.unroll,action_dim=backend.model.action_dim,
                          local_generator=generator,**cfg['planner'])
        result.set_objective(objective)
        return result
    native_planner,traced_planner=planner(),planner()
    cem_start=time.monotonic()
    reference=native_planner.plan(context.clone(),steps_left=6)
    with CandidateScoreTrace(traced_planner) as trace:
        measured=trace.run(context.clone(),steps_left=6)
    assert_bytes(reference.actions,measured.actions,'development traced CEM plan')
    assert_bytes(native_planner.local_generator.get_state(),traced_planner.local_generator.get_state(),'development traced CEM generator')
    if any('proposal_std' not in row for row in trace.iterations):
        raise ValueError('Missing actual native Gaussian proposal observation')
    with (output/'native-cem-trace.pt').open('xb') as stream:
        torch.save(trace.payload(),stream)
    cem_seconds=time.monotonic()-cem_start
    del reference,measured,native_planner,traced_planner,trace

    replay_start=time.monotonic()
    native=backend.predict(context,actions)
    native_cost=objective(native,actions)
    native_elites=torch.topk(-native_cost,10,dim=0).indices.cpu()
    summaries=[]
    score_records={}
    files={name:file_hash(output/name) for name in ['attention.json','native-cem-trace.pt']}

    def save(name,forecast,cost,coefficients=None):
        cost_cpu=cost.detach().cpu().clone()
        elites=torch.topk(-cost,10,dim=0).indices.cpu()
        payload={'arm':name,'input_binding':binding,'candidate_actions':actions.cpu(),
            'objective_costs':cost_cpu,'elite_indices':elites,
            'forecast_h6_archived':binding['episode']<4,
            'forecast_scope':'actual finalH6 embeddings archived for IDs0..3; later IDs computed and parity-checked onGPU with scalar summaries only'}
        if binding['episode']<4:
            payload['forecast_h6']={k:forecast[k][-1].detach().cpu().clone() for k in ('visual','proprio')}
        if coefficients is not None:payload['full_arm_cached_coefficients']=coefficients.cpu()
        path=output/(name+'.pt')
        with path.open('xb') as stream:torch.save(payload,stream)
        files[path.name]=file_hash(path)
        score_records[name]={'objective_costs':cost_cpu.tolist(),'elite_indices':elites.tolist()}
        summary={'arm':name,**score_summary(native_cost.cpu(),cost_cpu,native_elites,elites),
            'h6_change_l2_rms':{k:float((forecast[k][-1]-native[k][-1]).flatten(1).square().sum(1).mean().sqrt()) for k in ('visual','proprio')}}
        summaries.append(summary)
        print(json.dumps({'task':binding['task'],'episode':binding['episode'],**summary}),flush=True)
        return cost_cpu

    save('native',native,native_cost)
    for arm in ['fixed_rank4','matched_random_fixed_rank4']:
        with CaptureFull(backend.predictor,fixed.bank,arm) as capture:
            full=backend.predict(context,actions)
        full_cost=objective(full,actions)
        save(arm+'-full',full,full_cost,capture.record['coefficients'])
        common,centered=components(capture.full_delta)
        scores={}
        for name,delta in [('cached_full',capture.full_delta),('zero',None),
                           ('mean_only',common),('centered_only',centered)]:
            with ReplayField(backend.predictor,capture.native_field,delta):
                forecast=backend.predict(context,actions)
            cost=objective(forecast,actions)
            if name in ('cached_full','zero'):
                expected=full if name=='cached_full' else native
                expected_cost=full_cost if name=='cached_full' else native_cost
                for key in ('visual','proprio'):
                    assert_bytes(forecast[key],expected[key],name+' forecast '+key)
                assert_bytes(cost,expected_cost,name+' objective')
            scores[name]=save(arm+'-'+name,forecast,cost)
            del forecast
        def centered_cost(value):
            diff=value.cpu().double()-native_cost.cpu().double()
            return diff-diff.mean()
        full_change=centered_cost(full_cost)
        mean_change=centered_cost(scores['mean_only'])
        center_change=centered_cost(scores['centered_only'])
        denominator=full_change.square().sum()
        total=capture.full_delta.double().square().sum()
        summaries.append({'arm':arm,'component_audit':True,
            'cached_full_exact_forecast_and_score_parity':True,'zero_exact_forecast_and_score_parity':True,
            'centered_field_energy_fraction':float(centered.double().square().sum()/total),
            'common_field_energy_fraction':float(300*common.double().square().sum()/total),
            'common_centered_cost_reconstruction':float(1-(full_change-mean_change).square().sum()/denominator) if denominator>0 else None,
            'centered_centered_cost_reconstruction':float(1-(full_change-center_change).square().sum()/denominator) if denominator>0 else None,
            'no_component_renormalization':True})
        del capture,full,full_cost,common,centered
    if rng_signature('cuda:0')!=initial_rng:
        raise ValueError('Diagnostic changed global RNG')
    dump(output/'scores.json',score_records)
    files['scores.json']=file_hash(output/'scores.json')
    report={'status':'canonical_development_mechanism_case_complete','input_binding':binding,
        'execution_manifest_sha256':execution_manifest_sha256,'global_rng_unchanged':True,
        'fresh_confirmation':False,'physical_outcomes_measured':False,
        'attention_candidate_count':1,'attention_blocks':6,'attention_horizons':6,
        'cem_iterations':15,'cem_candidates_per_iteration':300,'cem_output_and_rng_parity':True,
        'shared_candidate_count':300,'shared_candidate_banks':1,'base_arms':2,'components_per_arm':5,
        'files':files,'summaries':summaries,'cem_reference_plus_trace_seconds':cem_seconds,
        'component_replay_seconds':time.monotonic()-replay_start,'total_seconds':time.monotonic()-started}
    dump(output/'report.json',report)
    dump(output/'DONE.json',{'report_sha256':file_hash(output/'report.json'),'files':files})
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','summaries')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('vendor','checkpoint','inputs','fits','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--task',choices=['reach','reach-wall'],required=True)
    parser.add_argument('--episode',type=int,choices=range(32),required=True)
    args=parser.parse_args()
    if file_hash(args.manifest)!=args.manifest_sha256:raise ValueError('Execution manifest changed')
    manifest=json.loads(args.manifest.read_text())
    input_manifest=json.loads((args.inputs/'INPUT_MANIFEST.json').read_text())
    if file_hash(args.inputs/'INPUT_MANIFEST.json')!=manifest['input_manifest_sha256']:
        raise ValueError('Finite input registry changed')
    for name,sha in manifest['source_sha256'].items():
        if file_hash(source_path(name))!=sha:raise ValueError('Frozen diagnostic source changed:'+name)
    if file_hash(args.checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if manifest['scenarios']!={'reach':list(range(32)),'reach-wall':list(range(32))}:
        raise ValueError('Finite scenario registry changed')
    binding=next(r for r in input_manifest['records'] if r['task']==args.task and r['episode']==args.episode)
    source=args.inputs/args.task/f'episode-{args.episode:03d}'/'inputs.pt'
    if file_hash(source)!=binding['inputs_sha256']:raise ValueError('Archived input bytes changed')
    fit=args.fits/args.task
    if file_hash(fit/'operator_bank.pt')!=manifest['fit_bank_sha256'][args.task]:raise ValueError('Frozen fit changed')
    from offline_study.models.backends import JepaBackend
    from offline_study.planning.planning_contract import prepare
    if torch.cuda.device_count()!=1:raise ValueError('Expose exactly one assigned GPU')
    torch.set_num_threads(1)
    backend=JepaBackend(args.vendor,args.checkpoint,manifest['checkpoint_sha256'],'metaworld','cuda:0','float32')
    bank=load_fitted_bank(fit,task='mw-'+args.task,checkpoint_sha256=manifest['checkpoint_sha256'])
    cfg=prepare(args.vendor,args.task)['config']
    if cfg['planner']['planning_objective']['objective_type']!='L2':raise ValueError('Registered nativeL2 objective changed')
    data=torch.load(source,map_location='cpu',weights_only=True)
    with torch.no_grad():
        run_case(backend,bank,cfg,data,binding,args.output,args.manifest_sha256)


if __name__=='__main__':main()
