"""Fixed 56-development-context extension with a concurrent same-GPU native CEM.

The original eight-case source is unchanged. All three arms execute the pinned
planner twice for exact traced/untraced identity; no physical actions are run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from .candidate_score_trace import CandidateScoreTrace
from .fixed_combined_check import assert_bytes, rng_signature
from .fixed_response import FixedResponseIntervention, load_fitted_bank
from .steered_cem_pilot import compact_trace, tensor_sha
from .development_mechanism_replay import dump
from .layer_attention_pilot import file_hash

ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4')
TASKS = ('reach', 'reach-wall')
PROTOCOL_SHA = '4766488e4907dff62f588894845f5b5aecf084e2c47a5ab4a1605cc2f07463db'
INPUT_SHA = '7eaaec460a9057078a36cb348e859222cdf426842620107bb0d2cdabe7df70ef'
CHECKPOINT_SHA = 'c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'


def validate_registry(manifest, protocol):
    if manifest['scenarios'] != {t:list(range(4,32)) for t in TASKS}:
        raise ValueError('Exact extension56 registry required')
    if manifest['arms'] != list(ARMS) or protocol['arms'] != list(ARMS):
        raise ValueError('All three concurrent arms required')
    if protocol['new_episode_ids'] != list(range(4,32)) or protocol['new_cases'] != 56:
        raise ValueError('Protocol population changed')
    if protocol['counts'] != {'searches_per_case':6,'population300_forecasts_per_case':90,
        'mean1_forecasts_per_case':90,'callbacks_per_case':180,'new_cases':56}:
        raise ValueError('Population/mean forecast counts changed')


def validate_payload(payload):
    if payload['selected_plan'].shape != (3,20) or payload['final_mean'].shape != (6,20):
        raise ValueError('Native returned prefix/full proposal layout changed')
    if len(payload['iterations']) != 15:
        raise ValueError('Incomplete native CEM trace')
    for index,row in enumerate(payload['iterations']):
        if row['iteration'] != index:
            raise ValueError('Nonsequential CEM trace')
        for key,shape in [('candidate_actions',(6,300,20)),('objective_costs',(300,)),
            ('proposal_mean',(6,20)),('proposal_std',(6,20)),('elite_indices',(10,))]:
            value=row[key]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid CEM field '+key)
            if key!='elite_indices' and value.dtype != torch.float32:
                raise ValueError('Strict FP32 CEM field required')
        if row['elite_indices'].dtype != torch.int64 or len(row['elite_indices'].unique()) != 10:
            raise ValueError('Invalid actual native elite IDs')
        if torch.any(row['elite_indices']<0) or torch.any(row['elite_indices']>=300):
            raise ValueError('Elite index out of bounds')
        if torch.any(row['proposal_std']<=0):
            raise ValueError('Nonpositive Gaussian proposal scale')


class CountedForecast:
    def __init__(self, intervention):
        self.intervention=intervention
        self.population_calls=0
        self.mean_calls=0

    def __call__(self, context, act_suffix=None, **kwargs):
        if act_suffix is None or tuple(act_suffix.shape) not in ((6,300,20),(6,1,20)):
            raise ValueError('Unexpected native population/mean forecast shape')
        if act_suffix.shape[1]==300:self.population_calls+=1
        else:self.mean_calls+=1
        return self.intervention(context,act_suffix=act_suffix,**kwargs)

    def validate(self):
        if (self.population_calls,self.mean_calls,self.intervention.calls)!=(15,15,30):
            raise ValueError('Require actual15 population plus15 updated-mean forecasts')


def model_versions(model):
    return {(kind,name):(id(value),value._version) for kind,iterator in
        [('parameters',model.named_parameters()),('buffers',model.named_buffers())]
        for name,value in iterator}


def input_hashes(context,target,data):
    result={label:{k:tensor_sha(value[k]) for k in value.keys()}
        for label,value in [('context',context),('target',target)]}
    result['raw']={label:{k:tensor_sha(v) for k,v in value.items()}
        for label,value in data.items() if isinstance(value,dict)}
    result['actions']=tensor_sha(data['actions'])
    return result


def run_case(backend,cfg,data,binding,bank,manifest,args,runtime):
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    # All vendor/dependency initialization precedes the strict experimental RNG snapshot.
    obs=cfg['task_specification']['obs']
    context=backend.model.encode(prepare_obs(obs,TensorDict(data['initial'],batch_size=[])).to(backend.device).unsqueeze(0),act=True)
    target=backend.model.encode(prepare_obs(obs,TensorDict(data['goal'],batch_size=[])).to(backend.device).unsqueeze(0),act=False).detach()
    objective=ReprTargetDistMPCObjective(cfg,target,**cfg['planner']['planning_objective'])
    before_inputs=input_hashes(context,target,data);before_model=model_versions(backend.model)
    before_rng=rng_signature(backend.device)
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();files={};traces={};parities={};native_first=None
    archive_binding=manifest['archived_native'][f'{args.task}/{args.episode}']
    dump(args.output/'STARTED.json',{'execution_manifest_sha256':args.manifest_sha256,
        'protocol_sha256':args.protocol_sha256,'input_binding':binding,**runtime,
        'archived_native':archive_binding,'arms':list(ARMS)})
    for arm in ARMS:
        arm_start=time.monotonic()
        def planner():
            unroll=CountedForecast(FixedResponseIntervention(backend,bank,arm))
            generator=torch.Generator(device=backend.device).manual_seed(binding['candidate_seed'])
            planner=CEMPlanner(unroll=unroll,action_dim=backend.model.action_dim,
                local_generator=generator,**cfg['planner'])
            planner.set_objective(objective)
            return planner,unroll
        plain,plain_calls=planner();traced,traced_calls=planner()
        reference=plain.plan(context.clone(),steps_left=6)
        with CandidateScoreTrace(traced) as observer:measured=observer.run(context.clone(),steps_left=6)
        assert_bytes(reference.actions,measured.actions,arm+' traced/untraced prefix')
        assert_bytes(plain._prev_mean,traced._prev_mean,arm+' traced/untraced full mean')
        assert_bytes(plain.local_generator.get_state(),traced.local_generator.get_state(),arm+' private RNG')
        plain_calls.validate();traced_calls.validate()
        payload=observer.payload();validate_payload(payload)
        if native_first is None:native_first=payload['iterations'][0]['candidate_actions'].clone()
        assert_bytes(payload['iterations'][0]['candidate_actions'],native_first,arm+' concurrent native first population')
        if rng_signature(backend.device)!=before_rng:raise ValueError('Global RNG changed')
        filename=arm+'-cem-trace.pt'
        with (args.output/filename).open('xb') as stream:torch.save(payload,stream)
        files[filename]=file_hash(args.output/filename)
        traces[arm]={**compact_trace(payload),'source_trace_sha256':files[filename]}
        parities[arm]={'selected_plan_byte_equal':True,'final_mean_byte_equal':True,
            'local_generator_byte_equal':True,'global_rng_unchanged':True,
            'native_iteration0_actions_byte_equal':True,'untraced_forecast_calls':30,
            'traced_forecast_calls':30,'untraced_population_calls':15,'traced_population_calls':15,
            'untraced_mean_calls':15,'traced_mean_calls':15,'seconds':time.monotonic()-arm_start}
        print(json.dumps({'task':args.task,'episode':args.episode,'arm':arm,
            'seconds':parities[arm]['seconds'],'parities_passed':True,'forecast_calls':60}),flush=True)
    if input_hashes(context,target,data)!=before_inputs:raise ValueError('Inputs changed')
    if model_versions(backend.model)!=before_model:raise ValueError('Frozen model changed')
    if rng_signature(backend.device)!=before_rng:raise ValueError('Global RNG changed at closeout')
    summary={'input_binding':binding,'execution_manifest_sha256':args.manifest_sha256,
        'protocol_sha256':args.protocol_sha256,'traces':traces,'parities':parities,
        'archived_native':archive_binding,'candidate_id_comparability':'iteration0 only; later populations differ'}
    dump(args.output/'steered-cem-summary.json',summary)
    files.update({n:file_hash(args.output/n) for n in ('STARTED.json','steered-cem-summary.json')})
    report={'status':'complete_cem_extension_development_case','input_binding':binding,
        'execution_manifest_sha256':args.manifest_sha256,'protocol_sha256':args.protocol_sha256,
        'fit_bank_sha256':manifest['fit_bank_sha256'][args.task],'files':files,'parities':parities,
        'archived_native':archive_binding,'parameters_unchanged':True,'inputs_unchanged':True,
        'global_rng_unchanged':True,'physical_outcomes_measured':False,'fresh_confirmation':False,
        'total_forward_count':180,'population_forecast_count':90,'mean_forecast_count':90,
        'total_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**runtime}
    dump(args.output/'report.json',report)
    dump(args.output/'DONE.json',{'report_sha256':file_hash(args.output/'report.json'),'files':files})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('vendor','checkpoint','inputs','fits','manifest','protocol','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('manifest-sha256','protocol-sha256','gpu-uuid'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--episode',type=int,choices=range(4,32),required=True)
    args=parser.parse_args()
    if file_hash(args.manifest)!=args.manifest_sha256 or file_hash(args.protocol)!=args.protocol_sha256 or args.protocol_sha256!=PROTOCOL_SHA:
        raise ValueError('Protocol/execution manifest changed')
    manifest=json.loads(args.manifest.read_text());validate_registry(manifest,json.loads(args.protocol.read_text()))
    if manifest['protocol_sha256']!=PROTOCOL_SHA or manifest['input_manifest_sha256']!=INPUT_SHA or manifest['checkpoint_sha256']!=CHECKPOINT_SHA:
        raise ValueError('Frozen parent bindings changed')
    if file_hash(args.checkpoint)!=CHECKPOINT_SHA or file_hash(args.inputs/'INPUT_MANIFEST.json')!=INPUT_SHA:
        raise ValueError('Original asset bytes changed')
    required={'cem_expansion_pilot.py','steered_cem_pilot.py','candidate_score_trace.py','fixed_response.py','backends.py','model_loader.py','fixed_combined_check.py','planning_contract.py'}
    if not required<=manifest['source_sha256'].keys():raise ValueError('Required execution source missing')
    for name,digest in manifest['source_sha256'].items():
        if Path(name).name!=name or file_hash(Path(__file__).parent/name)!=digest:raise ValueError('Source changed '+name)
    for name,digest in manifest['vendor_source_sha256'].items():
        path=(args.vendor/name).resolve()
        if not path.is_relative_to(args.vendor.resolve()) or file_hash(path)!=digest:raise ValueError('Vendor source changed')
    registry=json.loads((args.inputs/'INPUT_MANIFEST.json').read_text())['records']
    selected=[r for r in registry if (r['task'],r['episode'])==(args.task,args.episode)]
    if len(selected)!=1:raise ValueError('Nonunique input binding')
    binding=selected[0];input_path=args.inputs/args.task/f'episode-{args.episode:03d}'/'inputs.pt'
    if file_hash(input_path)!=binding['inputs_sha256']:raise ValueError('Input bytes changed')
    fit=args.fits/args.task
    if file_hash(fit/'operator_bank.pt')!=manifest['fit_bank_sha256'][args.task]:raise ValueError('Original fit changed')
    if torch.cuda.device_count()!=1 or os.environ.get('JEPA_VERIFIED_LOCAL_DINO')!='1':raise ValueError('One assigned GPU and verified local DINO required')
    uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))
    if uuid.removeprefix('GPU-').lower()!=args.gpu_uuid.removeprefix('GPU-').lower():raise ValueError('Physical GPU mismatch')
    torch.set_num_threads(1)
    from .backends import JepaBackend
    from .planning_contract import prepare
    backend=JepaBackend(args.vendor,args.checkpoint,CHECKPOINT_SHA,'metaworld','cuda:0','float32',allow_tf32=False)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('TF32 forbidden')
    bank=load_fitted_bank(fit,task='mw-'+args.task,checkpoint_sha256=CHECKPOINT_SHA)
    runtime={'gpu_uuid':uuid,'backend_provenance':backend.provenance,'torch_version':torch.__version__,'tf32_matmul':False,'tf32_cudnn':False}
    with torch.no_grad():
        report=run_case(backend,prepare(args.vendor,args.task)['config'],torch.load(input_path,map_location='cpu',weights_only=True),binding,bank,manifest,args,runtime)
    print(json.dumps({k:report[k] for k in ('status','total_seconds','total_forward_count','peak_allocated_bytes')}),flush=True)


if __name__=='__main__':main()
