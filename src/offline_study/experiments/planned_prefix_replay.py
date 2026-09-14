"""Hash-bound development-only physical replay of three saved CEM prefixes.

The frozen H6 intervention remains unchanged. Only its H3 forecast is scored
against the actual endpoint after fifteen elementary simulator actions.
"""
from __future__ import annotations
from offline_study._paths import source_path

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import types

import numpy as np
import torch

from offline_study.experiments.cem_expansion_pilot import ARMS, TASKS, CHECKPOINT_SHA, INPUT_SHA, model_versions
from offline_study.planning.decision_runtime import physics, reset_initial, tensor_hash, jsonable
from offline_study.experiments.development_mechanism_replay import dump
from offline_study.validation.fixed_combined_check import assert_bytes, rng_signature
from offline_study.interventions.fixed_response import FixedResponseIntervention, load_fitted_bank
from offline_study.experiments.layer_attention_pilot import file_hash
from offline_study.planning.planning_env_smoke import observation_digest

PROTOCOL_SHA = '8d80f98a557c79b6dedad27bb3ef63c2ede4979150ba8373c893da8882a232b3'
ORIGINAL_PROTOCOL_SHA = 'ca0061a6861a9f0ddb2694d9277cffff142909db69eb8b75a6316ab85bb328dd'
CEM_MANIFEST_SHA = 'da2a512a4d70bee10051ad1c152467d95d397ed8829cd4d9eb1357d861a8ca92'
PACKAGES = {'torch':'2.7.1+cu128','numpy':'2.2.6','mujoco':'3.3.0',
    'metaworld':'3.1.1','gym':'0.23.1','gymnasium':'1.3.0','tensordict':'0.9.1'}
MODALITIES = ('visual', 'proprio')


class PrivateEvalCrop:
    """Exact upstream degenerate crop, with private RNG globals, never restore RNG.

    The upstream crop draws random numbers even at scale=ratio=(1,1). Its
    output is nevertheless exactly the full square image. Every invocation is
    checked bytewise against that literal reference before normalization.
    """
    def __init__(self, transform):
        self.transform=transform;self.calls=0;self.original=transform.spatial_transform
        if (tuple(transform.random_resize_scale)!=(1.,1.) or
                tuple(transform.random_resize_aspect_ratio)!=(1.,1.) or
                transform.random_horizontal_flip or transform.auto_augment or
                transform.motion_shift or transform.reprob!=0 or transform.hwc):
            raise ValueError('Only frozen deterministic evaluation preprocessing is supported')
        if self.original.__name__!='random_resized_crop':raise ValueError('Unexpected upstream spatial transform')
        helper=self.original.__globals__['_get_param_spatial_crop']
        private_globals=dict(helper.__globals__)
        private_globals.update(random=random.Random(20260913),np=types.SimpleNamespace(random=np.random.RandomState(20260913)))
        isolated_helper=types.FunctionType(helper.__code__,private_globals,helper.__name__,helper.__defaults__,helper.__closure__)
        spatial_globals=dict(self.original.__globals__);spatial_globals['_get_param_spatial_crop']=isolated_helper
        self.isolated=types.FunctionType(self.original.__code__,spatial_globals,self.original.__name__,
            self.original.__defaults__,self.original.__closure__)

    def crop(self,images,target_height,target_width,scale,ratio):
        if (images.ndim!=4 or images.shape[-2:]!=(224,224) or tuple(scale)!=(1.,1.) or
                tuple(ratio)!=(1.,1.) or target_height!=224 or target_width!=224):
            raise ValueError('Degenerate full-frame reference contract changed')
        actual=self.isolated(images,target_height,target_width,scale,ratio)
        reference=torch.nn.functional.interpolate(images[:,:,0:224,0:224],size=(224,224),mode='bilinear',align_corners=False)
        assert_bytes(actual,reference,'private upstream evaluation crop/full-frame reference')
        self.calls+=1
        return actual

    def __enter__(self):
        self.transform.spatial_transform=self.crop
        return self

    def __exit__(self,kind,value,traceback):
        self.transform.spatial_transform=self.original


def validate_registry(manifest, protocol):
    if manifest['scenarios'] != {t:list(range(4,32)) for t in TASKS}:
        raise ValueError('Require every registered extension56 case')
    for key in ('model_arms','plan_sources'):
        if manifest[key] != list(ARMS) or protocol[key] != list(ARMS):
            raise ValueError('Require all three models and all three prefixes')
    if protocol['episode_ids'] != list(range(4,32)) or protocol['cases'] != 56:
        raise ValueError('Physical population changed')
    if manifest['package_versions'] != PACKAGES:
        raise ValueError('Original simulator/runtime versions required')
    cases = manifest['tail_qualification_cases']
    if not cases or len({(x['task'],x['episode']) for x in cases}) != len(cases):
        raise ValueError('Explicit unique receiving tail-qualification cases required')
    if any(x['task'] not in TASKS or x['episode'] not in range(4,32) for x in cases):
        raise ValueError('Qualification is outside the registered population')


def selected_prefix(value):
    result = torch.as_tensor(value, dtype=torch.float32).clone()
    if result.shape != (3,20) or not torch.isfinite(result).all():
        raise ValueError('Require all sixty finite selected-prefix coordinates')
    return result


def padded_actions(prefix, nonzero_tail=False):
    if prefix.shape != (3,20) or prefix.dtype != torch.float32 or not torch.isfinite(prefix).all():
        raise ValueError('Invalid exact selected prefix')
    result = torch.zeros(6,1,20, dtype=prefix.dtype, device=prefix.device)
    result[:3,0] = prefix
    if nonzero_tail:
        # Literal deterministic engineering stimulus; no generator, fit or search.
        result[3:,0] = torch.where(torch.arange(60,device=prefix.device).reshape(3,20)%2 == 0,
                                  .125, -.125)
    return result


def elementary_actions(prefix, preprocessor):
    prefix = selected_prefix(prefix)
    elementary = prefix.reshape(15,4)
    independent = torch.stack([prefix[t,4*f:4*f+4] for t in range(3) for f in range(5)])
    assert_bytes(elementary, independent, 'packed five-action ordering')
    result = preprocessor.denormalize_actions(elementary).detach().cpu().contiguous()
    if result.shape != (15,4) or result.dtype != torch.float32 or not torch.isfinite(result).all():
        raise ValueError('Invalid denormalized elementary actions')
    return result


def tree_hash(value):
    """Exact dtype/shape/byte digest, including every trajectory scalar."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            array=item.detach().cpu().contiguous().numpy()
            digest.update(b'tensor'+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
        elif isinstance(item,np.ndarray):
            digest.update(b'array'+str(item.dtype).encode()+str(item.shape).encode()+item.tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict')
            for key in sorted(item):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode()+str(len(item)).encode())
            for child in item:visit(child)
        else:
            digest.update(type(item).__name__.encode()+json.dumps(item,allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def physical_prefix(env, record, actions):
    initial, info = reset_initial(env, record['environment_seed'])
    initial_physics, initial_physics_sha = physics(env)
    if observation_digest(initial) != record['initial_sha256'] or initial_physics_sha != record['physics_sha256']:
        raise ValueError('Exact original observation/physics reset qualification failed')
    if actions.shape != (15,4) or actions.dtype != torch.float32 or not torch.isfinite(actions).all():
        raise ValueError('Exactly fifteen finite elementary actions required')
    before_actions = tensor_hash(actions)
    initial_state = torch.as_tensor(info['state']).clone()
    observations, rewards, dones, infos = env.step_multiple(actions)
    if any(len(x) != 15 for x in (observations,rewards,dones,infos)):
        raise ValueError('Incomplete fifteen-step physical prefix; no omission allowed')
    endpoint = {'visual':observations[-1].detach().cpu().clone(),
                'proprio':torch.as_tensor(infos[-1]['proprio']).detach().cpu().clone()}
    endpoint_physics, endpoint_physics_sha = physics(env)
    result = {'initial_state':initial_state,'initial_physics':initial_physics,
        'initial_observation_sha256':record['initial_sha256'],'initial_physics_sha256':initial_physics_sha,
        'states':torch.stack([torch.as_tensor(x['state']).clone() for x in infos]),
        'rewards':torch.stack([torch.as_tensor(x).clone() for x in rewards]),
        'dones':torch.stack([torch.as_tensor(x).clone() for x in dones]),
        'successes':torch.stack([torch.as_tensor(x['success']).clone() for x in infos]),
        'executed_actions':actions.clone(),'endpoint_observation':endpoint,
        'endpoint_observation_sha256':observation_digest(endpoint),
        'endpoint_physics':endpoint_physics,'endpoint_physics_sha256':endpoint_physics_sha}
    for key in ('initial_state','states','rewards','dones','successes','executed_actions'):
        if not torch.isfinite(result[key]).all():raise ValueError('Nonfinite physical trajectory '+key)
    if tensor_hash(actions) != before_actions:raise ValueError('Simulator mutated supplied action bytes')
    return result


def assert_native_repeat(first, repeat):
    if tree_hash(first) != tree_hash(repeat):
        raise ValueError('Native physical repeat differs in action/state/reward/done/success/endpoint bytes')


def encoded_endpoint(encoded):
    result = {}
    for key in MODALITIES:
        value = encoded[key]
        if value.shape[:2] != (1,1) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Expected single-frame single-batch strict FP32 encoding')
        result[key] = value[0,0].detach().clone()
    return result


def forecast_endpoint(forecast, context):
    result = {}
    for key in MODALITIES:
        value = forecast[key]
        if value.shape[:2] != (7,1) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Require H0 through H6, one candidate, strict FP32')
        assert_bytes(value[0,0],context[key][0,0],'forecast H0/context address')
        result[key] = value[3,0].detach().clone()
    return result


def distances(first, second):
    result = {}
    for key in MODALITIES:
        if first[key].shape != second[key].shape:
            raise ValueError('Embedding endpoint shape mismatch; broadcasting forbidden')
        value = (first[key]-second[key]).pow(2).mean()
        if not torch.isfinite(value):raise ValueError('Nonfinite embedding error')
        result[key] = float(value)
    result['weighted'] = result['visual'] + .1*result['proprio']
    return result


def archive_member(members, task, episode, name, digest):
    matches = [value for path,value in members.items() if path == name or
               path.endswith(f'/{task}/episode-{episode}/{name}')]
    if matches != [digest]:raise ValueError('Missing/ambiguous original archive member '+name)


def load_parent(cem_root, binding, manifest):
    root = cem_root/binding['task']/f"episode-{binding['episode']}"
    report = json.loads((root/'report.json').read_text())
    done = json.loads((root/'DONE.json').read_text())
    cloud = json.loads((root/'CLOUD_VERIFIED.json').read_text())
    if done['report_sha256'] != file_hash(root/'report.json') or done['files'] != report['files']:
        raise ValueError('CEM parent DONE mismatch')
    if report['execution_manifest_sha256'] != CEM_MANIFEST_SHA or report['input_binding'] != binding:
        raise ValueError('Not the concurrently rerun native extension parent')
    if report['status'] != 'complete_cem_extension_development_case' or report['total_forward_count'] != 180:
        raise ValueError('Incomplete concurrent parent CEM case')
    if report['fit_bank_sha256'] != manifest['fit_bank_sha256'][binding['task']]:
        raise ValueError('Physical forecast and parent CEM fit differ')
    if not cloud['gcs_download_sha256_verified'] or not cloud['all_report_files_hash_verified'] or cloud['raw_preservation_pending']:
        raise ValueError('Full CEM parent archive must be durable before physical replay')
    for name in ('report.json','DONE.json','steered-cem-summary.json'):
        if cloud['compact_sha256'][name] != file_hash(root/name):raise ValueError('CEM compact hash mismatch')
    for name,digest in report['files'].items():
        archive_member(cloud['verified_archive_members'],binding['task'],binding['episode'],name,digest)
    if file_hash(root/'steered-cem-summary.json') != report['files']['steered-cem-summary.json']:
        raise ValueError('Parent plans changed')
    summary = json.loads((root/'steered-cem-summary.json').read_text())
    if summary['input_binding'] != binding or summary['execution_manifest_sha256'] != CEM_MANIFEST_SHA or set(summary['traces']) != set(ARMS):
        raise ValueError('Parent compact identity changed')
    for arm in ARMS:
        if summary['traces'][arm]['source_trace_sha256'] != report['files'][arm+'-cem-trace.pt']:
            raise ValueError('Compact prefix source trace differs from full archived trace')
        parity = report['parities'][arm]
        if any(parity[key] is not True for key in ('selected_plan_byte_equal','final_mean_byte_equal',
                'local_generator_byte_equal','global_rng_unchanged','native_iteration0_actions_byte_equal')):
            raise ValueError('Parent CEM parity failed')
        if parity['traced_forecast_calls'] != 30 or parity['untraced_forecast_calls'] != 30:
            raise ValueError('Parent CEM search incomplete')
    return {arm:selected_prefix(summary['traces'][arm]['selected_plan']) for arm in ARMS}, {
        'execution_manifest_sha256':CEM_MANIFEST_SHA,'report_sha256':file_hash(root/'report.json'),
        'done_sha256':file_hash(root/'DONE.json'),'summary_sha256':file_hash(root/'steered-cem-summary.json'),
        'cloud_receipt_sha256':file_hash(root/'CLOUD_VERIFIED.json'),'cloud_receipt':cloud,
        'source_trace_sha256':{arm:summary['traces'][arm]['source_trace_sha256'] for arm in ARMS}}


def run_case(backend, cfg, env, data, record, binding, bank, plans, parent, manifest, args, runtime):
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    dump(args.output/'STARTED.json',{'execution_manifest_sha256':args.manifest_sha256,
        'protocol_sha256':PROTOCOL_SHA,'input_binding':binding,'cem_parent':parent,**runtime})
    before_model = model_versions(backend.model)
    before_data = tree_hash(data);before_plans = tree_hash(plans)
    physical = {}; actual = {}; forecasts = {}; audits = {}; files = {}
    # Simulator seed/reset intentionally changes global RNG. No state restoration
    # masks that: deterministic physical replay is its separate exact control.
    for arm in ARMS:
        actions = elementary_actions(plans[arm],backend.preprocessor)
        physical[arm] = physical_prefix(env,record,actions)
        if arm == 'native':
            physical['native_repeat'] = physical_prefix(env,record,actions.clone())
            assert_native_repeat(physical['native'],physical['native_repeat'])
    forecast_rng = rng_signature(backend.device)
    def encode(raw, act=False):
        return backend.model.encode(prepare_obs(cfg.task_specification.obs,
            TensorDict(raw,batch_size=[])).to(backend.device).unsqueeze(0),act=act)
    with PrivateEvalCrop(backend.preprocessor.transform) as crop:
        context = encode(data['initial'],True)
        goal = encoded_endpoint(encode(data['goal']))
        initial_encoded = encoded_endpoint(context)
        for arm in ARMS:actual[arm] = encoded_endpoint(encode(physical[arm]['endpoint_observation']))
    if crop.calls!=5:raise ValueError('Require five exact-reference preprocessing calls')
    if rng_signature(backend.device)!=forecast_rng:raise ValueError('Encoding with private crop changed global RNG')
    before_context = tree_hash(context.to_dict())
    qualified = {'task':args.task,'episode':args.episode} in manifest['tail_qualification_cases']
    for model_arm in ARMS:
        intervention = FixedResponseIntervention(backend,bank,model_arm)
        forecasts[model_arm] = {};audits[model_arm] = {}
        for plan_arm in ARMS:
            actions = padded_actions(plans[plan_arm].to(backend.device))
            action_sha = tensor_hash(actions)
            forecasts[model_arm][plan_arm] = forecast_endpoint(intervention(context,act_suffix=actions),context)
            audits[model_arm][plan_arm] = jsonable(intervention.last_record)
            if tensor_hash(actions) != action_sha:raise ValueError('Forecast mutated supplied actions')
            if model_arm != 'native' and ('realized_l2' not in audits[model_arm][plan_arm] or
                    intervention.last_record['horizon'] != 6):
                raise ValueError('Frozen intervention was not applied')
        if qualified:
            alternative = forecast_endpoint(intervention(context,act_suffix=padded_actions(
                plans['native'].to(backend.device),True)),context)
            for key in MODALITIES:
                assert_bytes(alternative[key],forecasts[model_arm]['native'][key],model_arm+' H3 future-tail causal address')
        if intervention.calls != 3+int(qualified):raise ValueError('Forecast count changed')
    if rng_signature(backend.device) != forecast_rng:raise ValueError('Encode/forecast phase changed global RNG')
    if before_context != tree_hash(context.to_dict()) or before_data != tree_hash(data) or before_plans != tree_hash(plans):
        raise ValueError('Original inputs/context/plans changed')
    if before_model != model_versions(backend.model):raise ValueError('Frozen model changed')
    goal_state = np.asarray(record['goal_state'])
    metrics = {'physical':{},'forecast':{},'preferences':{},'initial_goal_cost':distances(initial_encoded,goal)}
    for arm in ARMS:
        trajectory = physical[arm]
        metrics['physical'][arm] = {'terminal_ee_distance':float(np.linalg.norm(
            trajectory['states'][-1].numpy()[:3]-goal_state[:3])),
            'initial_ee_distance':float(np.linalg.norm(trajectory['initial_state'].numpy()[:3]-goal_state[:3])),
            'actual_goal_cost':distances(actual[arm],goal),'prefix_success':bool(trajectory['successes'][-1]),
            'elementary_steps':15,'selected_prefix_sha256':tensor_hash(plans[arm]),
            'executed_actions_sha256':tensor_hash(trajectory['executed_actions']),
            'endpoint_observation_sha256':trajectory['endpoint_observation_sha256'],
            'endpoint_physics_sha256':trajectory['endpoint_physics_sha256']}
    for model_arm in ARMS:
        metrics['forecast'][model_arm] = {arm:{'actual_endpoint_mse':distances(forecasts[model_arm][arm],actual[arm]),
            'predicted_goal_cost':distances(forecasts[model_arm][arm],goal)} for arm in ARMS}
    for arm in ARMS[1:]:
        predicted = metrics['forecast'][arm][arm]['predicted_goal_cost']['weighted']-metrics['forecast'][arm]['native']['predicted_goal_cost']['weighted']
        observed = metrics['physical'][arm]['actual_goal_cost']['weighted']-metrics['physical']['native']['actual_goal_cost']['weighted']
        metrics['preferences'][arm] = {'predicted_H3_own_minus_native':predicted,'actual_H3_own_minus_native':observed,
            'preference_reversal':predicted<0 and observed>0,'predicted_tie':predicted==0,'actual_tie':observed==0}
    parities = {'exact_original_reset_all_four_trajectories':True,'native_physical_repeat_byte_equal':True,
        'forecast_phase_global_rng_unchanged':True,'simulator_phase_global_rng_guarded':False,
        'inputs_unchanged':True,'parameters_unchanged':True,'future_tail_H3_byte_equal':True if qualified else None,
        'future_tail_qualification_case':qualified}
    bulk = {'physical':physical,'selected_prefixes':plans,'actual_H3_encoded':actual,
        'goal_encoded':goal,'H3_forecast_encoded':forecasts,'goal_state':record['goal_state']}
    with (args.output/'physical-prefix.pt').open('xb') as stream:torch.save(bulk,stream)
    dump(args.output/'prefix-summary.json',{'input_binding':binding,'execution_manifest_sha256':args.manifest_sha256,
        'protocol_sha256':PROTOCOL_SHA,'metrics':metrics,'parities':parities,'intervention_audits':audits,
        'forecast_horizon_scored':3,'CEM_optimized_horizon':6,'executed_elementary_steps_per_prefix':15})
    files = {name:file_hash(args.output/name) for name in ('STARTED.json','physical-prefix.pt','prefix-summary.json')}
    report = {'status':'complete_development_physical_prefix_case','input_binding':binding,
        'execution_manifest_sha256':args.manifest_sha256,'protocol_sha256':PROTOCOL_SHA,'files':files,
        'cem_parent':parent,'original_record_binding':manifest['original_records'][f'{args.task}/{args.episode}'],
        'fit_bank_sha256':manifest['fit_bank_sha256'][args.task],'parities':parities,
        'preprocessing_rng_audit':{'method':'exact_upstream_degenerate_crop_private_function_globals',
            'full_frame_reference_byte_equal':True,'global_rng_unchanged':True,'calls':crop.calls,
            'global_rng_restored':False,'original_spatial_transform_restored':True},
        'physical_trajectories':4,'total_elementary_actions':60,'scientific_forecast_calls':9,
        'engineering_forecast_calls':3*int(qualified),'forecast_horizon_scored':3,'CEM_optimized_horizon':6,
        'physical_outcomes_measured':True,'full_task_success_measured':False,'fresh_confirmation':False,
        'total_seconds':time.monotonic()-started,**runtime}
    dump(args.output/'report.json',report)
    dump(args.output/'DONE.json',{'report_sha256':file_hash(args.output/'report.json'),'files':files})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('vendor','checkpoint','inputs','fits','manifest','protocol','original-records','cem-root','cem-manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('manifest-sha256','protocol-sha256','gpu-uuid'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--episode',type=int,choices=range(4,32),required=True)
    args = parser.parse_args()
    if file_hash(args.manifest) != args.manifest_sha256 or file_hash(args.protocol) != PROTOCOL_SHA or args.protocol_sha256 != PROTOCOL_SHA:
        raise ValueError('Physical execution/protocol freeze changed')
    manifest = json.loads(args.manifest.read_text());validate_registry(manifest,json.loads(args.protocol.read_text()))
    expected = {'protocol_sha256':PROTOCOL_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
        'input_manifest_sha256':INPUT_SHA,'cem_execution_manifest_sha256':CEM_MANIFEST_SHA,
        'original_protocol_sha256':ORIGINAL_PROTOCOL_SHA}
    if any(manifest[key] != digest for key,digest in expected.items()):raise ValueError('Parent scientific binding changed')
    if file_hash(args.cem_manifest) != CEM_MANIFEST_SHA or file_hash(args.checkpoint) != CHECKPOINT_SHA or file_hash(args.inputs/'INPUT_MANIFEST.json') != INPUT_SHA:
        raise ValueError('Bound asset changed')
    required = {'planned_prefix_replay.py','cem_expansion_pilot.py','fixed_response.py','backends.py','model_loader.py',
        'fixed_combined_check.py','planning_contract.py','decision_runtime.py','planning_env_smoke.py'}
    if not required <= manifest['source_sha256'].keys():raise ValueError('Required local source missing')
    for name,digest in manifest['source_sha256'].items():
        if Path(name).name != name or file_hash(source_path(name)) != digest:raise ValueError('Local source changed '+name)
    required_vendor = {
        'app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py',
        'app/plan_common/datasets/preprocessor.py',
        'app/plan_common/datasets/transforms.py','src/datasets/utils/video/transforms.py',
        'evals/simu_env_planning/envs/init.py',
        'evals/simu_env_planning/envs/metaworld.py',
        'evals/simu_env_planning/envs/wrappers/pixels.py',
        'evals/simu_env_planning/envs/wrappers/tensor.py',
        'evals/simu_env_planning/planning/utils.py','evals/utils.py'}
    if not required_vendor <= manifest['vendor_source_sha256'].keys():
        raise ValueError('Required predictor/preprocessor/simulator source bindings missing')
    for name,digest in manifest['vendor_source_sha256'].items():
        path = (args.vendor/name).resolve()
        if not path.is_relative_to(args.vendor.resolve()) or file_hash(path) != digest:raise ValueError('Vendor source changed '+name)
    rows = json.loads((args.inputs/'INPUT_MANIFEST.json').read_text())['records']
    rows = [r for r in rows if (r['task'],r['episode']) == (args.task,args.episode)]
    if len(rows) != 1:raise ValueError('Nonunique original input identity')
    binding = rows[0];input_path = args.inputs/args.task/f'episode-{args.episode:03d}'/'inputs.pt'
    original = args.original_records/args.task/f'episode-{args.episode:03d}'
    rb = manifest['original_records'][f'{args.task}/{args.episode}']
    if file_hash(original/'record.json') != rb['record_sha256'] or file_hash(original/'DONE.json') != rb['done_sha256']:
        raise ValueError('Original reset/goal record changed')
    record = json.loads((original/'record.json').read_text());done = json.loads((original/'DONE.json').read_text())
    if done['files']['record.json'] != rb['record_sha256'] or done['files']['inputs.pt'] != binding['inputs_sha256'] or file_hash(input_path) != binding['inputs_sha256']:
        raise ValueError('Original record/input DONE binding failed')
    if any(record[key] != binding[key] for key in binding if key != 'inputs_sha256') or record['protocol_sha256'] != ORIGINAL_PROTOCOL_SHA:
        raise ValueError('Original reset and CEM input identities differ')
    fit = args.fits/args.task
    if file_hash(fit/'operator_bank.pt') != manifest['fit_bank_sha256'][args.task]:raise ValueError('Fit changed')
    plans,parent = load_parent(args.cem_root,binding,manifest)
    packages = {name:importlib.metadata.version(name) for name in PACKAGES}
    if packages != PACKAGES:raise ValueError('Original simulator package versions differ: '+repr(packages))
    if torch.cuda.device_count() != 1 or os.environ.get('JEPA_VERIFIED_LOCAL_DINO') != '1':raise ValueError('One assigned GPU and verified local DINO required')
    uuid = str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))
    if uuid.removeprefix('GPU-').lower() != args.gpu_uuid.removeprefix('GPU-').lower():raise ValueError('Physical GPU mismatch')
    torch.set_num_threads(1)
    from offline_study.models.backends import JepaBackend
    from offline_study.planning.planning_contract import prepare
    backend = JepaBackend(args.vendor,args.checkpoint,CHECKPOINT_SHA,'metaworld','cuda:0','float32',allow_tf32=False)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('TF32 forbidden')
    cfg = OmegaConf.create(prepare(args.vendor,args.task)['config']);cfg.local_seed = record['local_seed']
    if cfg.planner.repeat_actskip or cfg.frameskip != 5 or cfg.task_specification.num_frames != 1:raise ValueError('Physical action/observation contract changed')
    bank = load_fitted_bank(fit,task='mw-'+args.task,checkpoint_sha256=CHECKPOINT_SHA)
    data = torch.load(input_path,map_location='cpu',weights_only=True)
    if observation_digest(data['initial']) != record['initial_sha256'] or observation_digest(data['goal']) != record['goal_sha256'] or tensor_hash(data['actions']) != record['candidate_sha256']:
        raise ValueError('Canonical observation/goal/candidate tensor mismatch')
    runtime = {'gpu_uuid':uuid,'backend_provenance':backend.provenance,'package_versions':packages,
        'tf32_matmul':False,'tf32_cudnn':False,'mujoco_gl':os.environ.get('MUJOCO_GL')}
    env = make_env(cfg)
    try:
        with torch.no_grad():report = run_case(backend,cfg,env,data,record,binding,bank,plans,parent,manifest,args,runtime)
    except Exception as exc:
        if args.output.exists() and not (args.output/'FAILED.json').exists():
            dump(args.output/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),
                'execution_manifest_sha256':args.manifest_sha256,'protocol_sha256':PROTOCOL_SHA})
        raise
    finally:env.close()
    print(json.dumps({key:report[key] for key in ('status','total_seconds','scientific_forecast_calls','engineering_forecast_calls')}),flush=True)


if __name__ == '__main__':main()
