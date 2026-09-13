"""Prospectively fixed eight-development-scenario steering/CEM diagnostic.

No simulator, refit, method search, or protected outcomes. Each arm uses the
original native seed and exact traced/untraced selected-plan/RNG parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .candidate_score_trace import CandidateScoreTrace
from .development_mechanism_replay import dump
from .fixed_combined_check import assert_bytes, rng_signature
from .fixed_response import FixedResponseIntervention, load_fitted_bank
from .layer_attention_pilot import file_hash


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compact_trace(payload):
    return {'selected_plan':payload['selected_plan'].tolist(),
        'final_mean':payload['final_mean'].tolist(),
        'iterations':[{key:(row[key].tolist() if isinstance(row[key],torch.Tensor) else row[key])
            for key in ('iteration','proposal_mean','proposal_std','proposal_entropy_nats',
                        'proposal_entropy_semantics','objective_costs','elite_indices',
                        'best_runnerup_margin','elite_boundary_margin')}
            | {'candidate_actions_sha256':tensor_sha(row['candidate_actions']),
               'candidate_actions_shape':list(row['candidate_actions'].shape),
               'candidate_actions_dtype':str(row['candidate_actions'].dtype).removeprefix('torch.')}
            for row in payload['iterations']]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['vendor','checkpoint','inputs','fits','native','manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--task',choices=['reach','reach-wall'],required=True)
    parser.add_argument('--episode',type=int,choices=range(4),required=True)
    args=parser.parse_args()
    if file_hash(args.manifest)!=args.manifest_sha256:raise ValueError('Add-on manifest changed')
    manifest=json.loads(args.manifest.read_text())
    for name,sha in manifest['source_sha256'].items():
        if file_hash(Path(__file__).parent/name)!=sha:raise ValueError('Add-on source changed')
    if manifest['scenarios']!={'reach':list(range(4)),'reach-wall':list(range(4))}:
        raise ValueError('Prospective eight-scenario registry changed')
    if manifest['arms']!=['fixed_rank4','matched_random_fixed_rank4']:
        raise ValueError('Arm registry changed')
    if file_hash(args.checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if file_hash(args.inputs/'INPUT_MANIFEST.json')!=manifest['input_manifest_sha256']:
        raise ValueError('Original input registry changed')
    input_manifest=json.loads((args.inputs/'INPUT_MANIFEST.json').read_text())
    binding=next(r for r in input_manifest['records'] if r['task']==args.task and r['episode']==args.episode)
    input_path=args.inputs/args.task/f'episode-{args.episode:03d}'/'inputs.pt'
    native_path=args.native/f'{args.task}-{args.episode}-native-cem-trace.pt'
    if file_hash(input_path)!=binding['inputs_sha256']:raise ValueError('Original input changed')
    if file_hash(native_path)!=manifest['native_trace_sha256'][f'{args.task}/{args.episode}']:
        raise ValueError('Original native trace changed')
    fit=args.fits/args.task
    if file_hash(fit/'operator_bank.pt')!=manifest['fit_bank_sha256'][args.task]:raise ValueError('Original bank changed')
    if torch.cuda.device_count()!=1:raise ValueError('Require one assigned GPU')
    torch.set_num_threads(1)
    from .backends import JepaBackend
    from .planning_contract import prepare
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    backend=JepaBackend(args.vendor,args.checkpoint,manifest['checkpoint_sha256'],'metaworld','cuda:0','float32')
    cfg=prepare(args.vendor,args.task)['config']
    bank=load_fitted_bank(fit,task='mw-'+args.task,checkpoint_sha256=manifest['checkpoint_sha256'])
    data=torch.load(input_path,map_location='cpu',weights_only=True)
    native=torch.load(native_path,map_location='cpu',weights_only=True)
    if len(native['iterations'])!=15:raise ValueError('Incomplete native comparator')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); files={}; traces={'native':compact_trace(native)}; parities={}
    traces['native']['source_trace_sha256']=file_hash(native_path)
    with torch.no_grad():
        obs=cfg['task_specification']['obs']
        context=backend.model.encode(prepare_obs(obs,TensorDict(data['initial'],batch_size=[])).to(backend.device).unsqueeze(0),act=True)
        target=backend.model.encode(prepare_obs(obs,TensorDict(data['goal'],batch_size=[])).to(backend.device).unsqueeze(0),act=False).detach()
        objective=ReprTargetDistMPCObjective(cfg,target,**cfg['planner']['planning_objective'])
        rng=rng_signature('cuda:0')
        for arm in manifest['arms']:
            def planner():
                intervention=FixedResponseIntervention(backend,bank,arm)
                generator=torch.Generator(device='cuda:0').manual_seed(binding['candidate_seed'])
                result=CEMPlanner(unroll=intervention,action_dim=backend.model.action_dim,
                                  local_generator=generator,**cfg['planner'])
                result.set_objective(objective)
                return result,intervention
            plain,plain_edit=planner(); traced,traced_edit=planner()
            reference=plain.plan(context.clone(),steps_left=6)
            with CandidateScoreTrace(traced) as trace:measured=trace.run(context.clone(),steps_left=6)
            assert_bytes(reference.actions,measured.actions,'steered CEM traced/untraced plan')
            assert_bytes(plain.local_generator.get_state(),traced.local_generator.get_state(),'steered CEM local RNG')
            assert_bytes(native['iterations'][0]['candidate_actions'],trace.iterations[0]['candidate_actions'],'native/steered first population')
            # Official CEM unrolls both the300-candidate population and its
            # updated1-candidate mean on every iteration (even without decoding).
            if plain_edit.calls!=30 or traced_edit.calls!=30:raise ValueError('Unexpected CEM forecast count')
            if rng_signature('cuda:0')!=rng:raise ValueError('Steered trace changed global RNG')
            payload=trace.payload()
            with (args.output/(arm+'-cem-trace.pt')).open('xb') as stream:torch.save(payload,stream)
            files[arm+'-cem-trace.pt']=file_hash(args.output/(arm+'-cem-trace.pt'))
            traces[arm]=compact_trace(payload)
            traces[arm]['source_trace_sha256']=files[arm+'-cem-trace.pt']
            parities[arm]={'selected_plan_byte_equal':True,'local_generator_byte_equal':True,
                'global_rng_unchanged':True,'native_iteration0_actions_byte_equal':True,
                'untraced_forecast_calls':plain_edit.calls,'traced_forecast_calls':traced_edit.calls}
    dump(args.output/'steered-cem-summary.json',{'input_binding':binding,'traces':traces,
        'native_trace_sha256':file_hash(native_path),'trace_sha256':files.copy(),
        'parities':parities,'candidate_id_comparability':'iteration0 only; later populations differ'})
    files['steered-cem-summary.json']=file_hash(args.output/'steered-cem-summary.json')
    report={'status':'fixed_eight_development_steered_cem_complete','input_binding':binding,
        'execution_manifest_sha256':args.manifest_sha256,'files':files,'parities':parities,
        'native_trace_sha256':file_hash(native_path),'fresh_confirmation':False,
        'physical_outcomes_measured':False,'global_rng_unchanged':True,'total_seconds':time.monotonic()-start}
    dump(args.output/'report.json',report)
    dump(args.output/'DONE.json',{'report_sha256':file_hash(args.output/'report.json'),'files':files})
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
