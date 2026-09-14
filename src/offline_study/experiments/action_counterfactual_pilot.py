"""Fixed development C: causal-route parity and action-range matched controls.

Changing H3 action persists at H4's older context position. Range membership is
only membership in the literal affine action encoder's image, not a manifold.
"""
from __future__ import annotations
from offline_study._paths import source_path

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from offline_study.experiments.action_condition_specificity import file_hash, tensor_hash, dump, initial_population, verify_native_sampler, score_vectors, agreement

TASKS = ("reach", "reach-wall")
BANKS = ("original", "fresh")
MODES = ("donor_h3", "donor_persistent", "random_range", "random_off_range", "random_isotropic")
ARMS = ("native", "zero_capture", "coherent_raw_h3", "all_blocks_h3_donor",
        "all_blocks_persistent_donor", *(f"{mode}_B{layer}" for layer in range(6) for mode in MODES))
OCCURRENCES = {3: 1, 4: 0}
SVD_TOL = 1e-6
PROTOCOL_SHA = "7c16653101ebc9a29bca82b88783e7ae44edac2a134733bc87dc0f6263141f72"
INPUT_SHA = "7eaaec460a9057078a36cb348e859222cdf426842620107bb0d2cdabe7df70ef"
CHECKPOINT_SHA = "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"


def private_seed(task, episode, bank, layer):
    label = f"action-counterfactual-v1|{task}|{episode}|{bank}|{layer}"
    return int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) & ((1 << 63)-1)


def swap_h3_actions(actions):
    if actions.shape != (6,300,20) or actions.dtype != torch.float32 or not torch.isfinite(actions).all():
        raise ValueError("Expected fixed FP32 H6/300/20 action bank")
    changed = actions.clone()
    changed[2] = torch.roll(actions[2], shifts=-1, dims=0)
    return changed


def action_range(weight):
    if weight.shape != (400,20) or weight.dtype != torch.float32 or not torch.isfinite(weight).all():
        raise ValueError("Require literal finite FP32 action encoder weight [400,20]")
    cpu = weight.detach().cpu().double()
    u,s,vh = torch.linalg.svd(cpu, full_matrices=False)
    if s[0] <= 0:
        raise ValueError("Degenerate action encoder")
    keep = s > SVD_TOL*s[0]
    basis = u[:,keep].contiguous()
    if not 0 < basis.shape[1] < 400 or not torch.allclose(basis.T@basis,torch.eye(basis.shape[1]).double(),atol=1e-12,rtol=1e-12):
        raise ValueError("Action range projector qualification failed")
    reconstruction = (u*s)@vh
    relative = (reconstruction-cpu).norm()/cpu.norm()
    if relative > 1e-12:
        raise ValueError("SVD reconstruction qualification failed")
    return basis, {"weight_sha256":tensor_hash(weight), "weight_shape":[400,20],
        "singular_values":s.tolist(), "relative_rank_threshold":SVD_TOL,
        "retained_rank":int(keep.sum()), "svd_reconstruction_relative_error":float(relative),
        "orthogonality_max_error":float((basis.T@basis-torch.eye(basis.shape[1]).double()).abs().max()),
        "projector_basis_sha256":tensor_hash(basis), "scope":"affine action encoder range, not physical or latent manifold"}


def random_directions(basis, seed, device):
    q = basis.to(device=device,dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(seed)
    epsilon = torch.randn((300,400),device=device,dtype=torch.float64,generator=generator)
    projected = (epsilon@q)@q.T
    return {"random_range":projected,"random_off_range":epsilon-projected,
            "random_isotropic":epsilon}, tensor_hash(epsilon)


def replacement(z, position, mode, basis, direction=None):
    if z.shape != (300,2,400) or z.dtype != torch.float32 or position not in (0,1):
        raise ValueError("Unexpected actual AdaLN condition layout")
    current = z[:,position]
    donor = torch.roll(current, shifts=-1, dims=0)
    target = (donor.double()-current.double()).norm(dim=1)
    if mode == "donor":
        value = donor
    else:
        if direction is None or direction.shape != current.shape or not torch.isfinite(direction).all():
            raise ValueError("Missing finite registered random direction")
        lengths = direction.norm(dim=1)
        if torch.any((target>0)&(lengths==0)):
            raise ValueError("Degenerate nonzero-norm random control")
        scaled = direction*(target/lengths.clamp_min(torch.finfo(torch.float64).tiny))[:,None]
        value = current+scaled.to(torch.float32)
    delta = value.double()-current.double()
    delivered = delta.norm(dim=1)
    relative = torch.where(target>0,(delivered-target).abs()/target.clamp_min(1e-300),torch.zeros_like(target))
    if torch.any(relative>1e-5) or torch.any(delivered[target==0]!=0):
        raise ValueError("Actually delivered donor/control norm mismatch")
    q = basis.to(delta.device)
    in_range = (delta@q)@q.T
    in_energy = in_range.square().sum(1)
    out_energy = (delta-in_range).square().sum(1)
    total = delta.square().sum(1)
    def fractions(part):
        return [float(a/b) if b>0 else None for a,b in zip(part.cpu(),total.cpu())]
    changed = z.clone();changed[:,position]=value
    untouched = 1-position
    if not torch.equal(changed[:,untouched],z[:,untouched]):
        raise ValueError("Untargeted action history changed")
    return changed, {"position":position,"mode":mode,"requested_delta_l2":target.cpu().tolist(),
        "delivered_delta_l2":delivered.cpu().tolist(),"relative_norm_error":relative.cpu().tolist(),
        "max_relative_norm_error":float(relative.max()),"zero_target_count":int((target==0).sum()),
        "in_range_energy_fraction":fractions(in_energy),"off_range_energy_fraction":fractions(out_energy),
        "native_condition_sha256":tensor_hash(z),"edited_condition_sha256":tensor_hash(changed)}


class CounterfactualHook:
    def __init__(self,predictor,*,basis,cache=None,layers=(),persistent=False,mode="capture",direction=None):
        self.predictor,self.basis,self.cache = predictor,basis,cache
        self.layers,self.persistent,self.mode,self.direction = set(layers),persistent,mode,direction
        self.counts=[0]*6;self.captured={};self.audits=[];self.handles=[]

    def __enter__(self):
        if len(self.predictor.predictor_blocks)!=6 or any(m._forward_hooks or m._forward_pre_hooks for m in self.predictor.modules()):
            raise ValueError("Six hook-free predictor blocks required")
        try:
            for layer,block in enumerate(self.predictor.predictor_blocks):
                if block.training:raise ValueError("Frozen evaluation mode required")
                def pre(module,args,kwargs,layer=layer):
                    self.counts[layer]+=1
                    horizon=self.counts[layer]
                    if horizon not in OCCURRENCES:return None
                    if len(args)<2:raise ValueError("Missing actual AdaLN x,z inputs")
                    z=args[1];key=(horizon,layer)
                    if self.mode=="capture":
                        self.captured[key]=z.detach().clone();changed=z.clone()
                    elif layer in self.layers and (horizon==3 or self.persistent):
                        expected=self.cache[key]
                        if z.shape!=expected.shape or z.dtype!=expected.dtype or tensor_hash(z)!=tensor_hash(expected):
                            raise ValueError("Action-only condition differs from frozen native cache")
                        changed,audit=replacement(z,OCCURRENCES[horizon],self.mode,self.basis,self.direction)
                        self.audits.append({"horizon":horizon,"layer":layer,**audit})
                    else:return None
                    return (args[0],changed,*args[2:]),kwargs
                self.handles.append(block.register_forward_pre_hook(pre,with_kwargs=True))
        except Exception:
            self.__exit__(Exception,None,None);raise
        return self

    def __exit__(self,typ,value,tb):
        for handle in self.handles:handle.remove()
        if typ is None:
            expected=12 if self.mode=="capture" else len(self.layers)*(2 if self.persistent else 1)
            actual=len(self.captured) if self.mode=="capture" else len(self.audits)
            if self.counts!=[6]*6 or actual!=expected:
                raise ValueError("Incomplete exact H3/H4 condition addressing")


def reconstruction(native,coherent,forecast):
    rows=[];distances={}
    for modality in ("visual","proprio"):
        for horizon in (3,4,6):
            baseline=(native[modality][horizon].double()-coherent[modality][horizon].double()).flatten(1).square().mean(1)
            error=(forecast[modality][horizon].double()-coherent[modality][horizon].double()).flatten(1).square().mean(1)
            if baseline.shape!=(300,) or not torch.isfinite(baseline).all() or not torch.isfinite(error).all():
                raise ValueError("Invalid same-context reconstruction fields")
            denominator,numerator=baseline.cpu().tolist(),error.cpu().tolist()
            distances[modality,horizon]=(error,baseline)
            values=[1-n/d if d>0 else None for n,d in zip(numerator,denominator)]
            rows.append({"modality":modality,"horizon":horizon,"numerator_mse":numerator,
                "denominator_mse":denominator,"reconstruction":values,
                "n_defined":sum(v is not None for v in values),
                "pooled_reconstruction":1-sum(numerator)/sum(denominator) if sum(denominator)>0 else None})
    for horizon in (3,4,6):
        numerator=(distances['visual',horizon][0]+.1*distances['proprio',horizon][0]).cpu().tolist()
        denominator=(distances['visual',horizon][1]+.1*distances['proprio',horizon][1]).cpu().tolist()
        values=[1-n/d if d>0 else None for n,d in zip(numerator,denominator)]
        rows.append(dict(modality='official',horizon=horizon,numerator_mse=numerator,denominator_mse=denominator,
            reconstruction=values,n_defined=sum(v is not None for v in values),
            pooled_reconstruction=1-sum(numerator)/sum(denominator) if sum(denominator)>0 else None))
    return rows


def validate_protocol(protocol):
    if (protocol.get('scenarios')!={t:list(range(8)) for t in TASKS} or protocol.get('banks')!=list(BANKS)
        or protocol.get('candidate_shape')!=[6,300,20] or protocol.get('layers')!=list(range(6))
        or protocol.get('arms_per_bank')!=list(ARMS) or protocol.get('counts')!=dict(forecasts_per_bank=35,banks_per_case=2,cases=16,total_h6_forecasts=1120)):
        raise ValueError('Frozen counterfactual branch/input registry changed')


def check_runtime(backend):
    predictor=backend.predictor
    if (backend.model.ctxt_window!=2 or backend.model.proprio_mode!="predict_proprio"
        or not predictor.action_encoder_inpred or not isinstance(predictor.action_encoder,torch.nn.Linear)
        or predictor.action_encoder.in_features!=20 or predictor.action_encoder.out_features!=400
        or getattr(backend.model.model,"pred_type",None)!="AdaLN"):
        raise ValueError("Unexpected action/history pathway: cannot claim coherent parity")
    return action_range(predictor.action_encoder.weight)


def run_case(backend,cfg,data,binding,output,manifest_sha,protocol_sha,runtime):
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from offline_study.validation.fixed_combined_check import assert_bytes, rng_signature
    if set(data)!={"initial","goal","actions"}:raise ValueError("Unexpected input payload")
    basis,range_audit=check_runtime(backend)
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    initial=TensorDict(data['initial'],batch_size=[]);goal=TensorDict(data['goal'],batch_size=[])
    obs_type=cfg['task_specification']['obs']
    context=backend.model.encode(prepare_obs(obs_type,initial).to(backend.device).unsqueeze(0),act=True)
    target=backend.model.encode(prepare_obs(obs_type,goal).to(backend.device).unsqueeze(0),act=False).detach()
    objective=ReprTargetDistMPCObjective(cfg,target,**cfg['planner']['planning_objective'])
    signatures={label:{k:tensor_hash(v[k]) for k in v.keys()} for label,v in [('context',context),('target',target)]}
    versions={(kind,k):v._version for kind,iterator in [('parameters',backend.model.named_parameters()),('buffers',backend.model.named_buffers())] for k,v in iterator}
    rng=rng_signature(backend.device)
    fresh,private_rng=initial_population(cfg,binding['candidate_seed']^0x40000000,backend.device)
    verify_native_sampler(cfg,binding['candidate_seed']^0x40000000,fresh,private_rng,CEMPlanner)
    banks={'original':data['actions'].to(backend.device),'fresh':fresh}
    if tensor_hash(banks['original'])==tensor_hash(fresh):raise ValueError('Fresh bank duplicates original')
    records,bank_receipts={},{}
    dump(output/'STARTED.json',dict(input_binding=binding,execution_manifest_sha256=manifest_sha,
        protocol_sha256=protocol_sha,runtime_provenance=runtime,range_audit=range_audit,arms=list(ARMS)))
    for bank,actions in banks.items():
        bank_start=time.monotonic();action_sha=tensor_hash(actions)
        if torch.count_nonzero(actions[:,0]):raise ValueError("Native bank candidate0 must be zero")
        native=backend.predict(context,actions)
        with CounterfactualHook(backend.predictor,basis=basis) as capture:
            zero=backend.predict(context,actions)
        for modality in ('visual','proprio'):assert_bytes(zero[modality],native[modality],'zero fullhorizon '+modality)
        swapped=swap_h3_actions(actions)
        swapped_sha=tensor_hash(swapped)
        coherent=backend.predict(context,swapped)
        for modality in ('visual','proprio'):assert_bytes(coherent[modality][:3],native[modality][:3],'raw donor H1/H2 '+modality)
        records[bank]={}
        native_scores=score_vectors(native,target,objective,actions)
        if score_vectors(zero,target,objective,actions)!=native_scores:raise ValueError('Zero full cost/elite parity failed')
        def save(name,forecast,action_input,audits=None,seed=None,epsilon_sha=None):
            for modality in ('visual','proprio'):assert_bytes(forecast[modality][:3],native[modality][:3],name+' H1/H2 '+modality)
            scores=score_vectors(forecast,target,objective,action_input)
            records[bank][name]=dict(scores=scores,reconstruction=reconstruction(native,coherent,forecast),
                condition_audits=audits or [],random_seed=seed,epsilon_sha256=epsilon_sha,
                agreement_native={m:agreement(native_scores[m]['costs'],scores[m]['costs'],native_scores[m]['elite_indices'],scores[m]['elite_indices']) for m in native_scores})
        save('native',native,actions);save('zero_capture',zero,actions);save('coherent_raw_h3',coherent,swapped)
        del zero
        for name,persistent in [('all_blocks_h3_donor',False),('all_blocks_persistent_donor',True)]:
            with CounterfactualHook(backend.predictor,basis=basis,cache=capture.captured,layers=range(6),persistent=persistent,mode='donor') as hook:
                forecast=backend.predict(context,actions)
            if persistent:
                for modality in ('visual','proprio'):assert_bytes(forecast[modality],coherent[modality],'allblock persistent versus raw '+modality)
                if score_vectors(forecast,target,objective,actions)!=score_vectors(coherent,target,objective,swapped):
                    raise ValueError("Coherent full cost/elite parity failed")
            save(name,forecast,actions,hook.audits);del forecast
        for layer in range(6):
            seed=private_seed(binding['task'],binding['episode'],bank,layer)
            directions,epsilon_sha=random_directions(basis,seed,backend.device)
            for mode in MODES:
                donor=mode.startswith('donor')
                with CounterfactualHook(backend.predictor,basis=basis,cache=capture.captured,layers=[layer],
                    persistent=mode=='donor_persistent',mode='donor' if donor else mode,
                    direction=None if donor else directions[mode]) as hook:
                    forecast=backend.predict(context,actions)
                save(f'{mode}_B{layer}',forecast,actions,hook.audits,None if donor else seed,None if donor else epsilon_sha)
                del forecast
        if set(records[bank])!=set(ARMS) or tensor_hash(actions)!=action_sha or tensor_hash(swapped)!=swapped_sha:
            raise ValueError("Incomplete35 branch registry or native action mutation")
        bank_receipts[bank]=dict(original_actions_sha256=action_sha,coherent_actions_sha256=tensor_hash(swapped),
            forecast_count=35,zero_full_horizon_byte_parity=True,persistent_allblock_raw_full_horizon_byte_parity=True,
            h1_h2_unchanged_all_arms=True,seconds=time.monotonic()-bank_start)
        del native,coherent,capture
        print(json.dumps(dict(task=binding['task'],episode=binding['episode'],bank=bank,seconds=bank_receipts[bank]['seconds'],forecast_count=35)),flush=True)
    if rng_signature(backend.device)!=rng:raise ValueError("Experimental global RNG changed")
    if signatures!={label:{k:tensor_hash(v[k]) for k in v.keys()} for label,v in [('context',context),('target',target)]}:
        raise ValueError("Encoded inputs changed")
    if versions!={(kind,k):v._version for kind,iterator in [('parameters',backend.model.named_parameters()),('buffers',backend.model.named_buffers())] for k,v in iterator}:
        raise ValueError("Frozen model changed")
    dump(output/'scores.json',dict(input_binding=binding,execution_manifest_sha256=manifest_sha,
        protocol_sha256=protocol_sha,banks=records,bank_receipts=bank_receipts,range_audit=range_audit))
    with (output/'actions.pt').open('xb') as stream:torch.save({k:v.cpu() for k,v in banks.items()},stream)
    files={name:file_hash(output/name) for name in ('STARTED.json','scores.json','actions.pt')}
    report=dict(status='complete_action_counterfactual_development_case',input_binding=binding,
        execution_manifest_sha256=manifest_sha,protocol_sha256=protocol_sha,runtime_provenance=runtime,
        range_audit=range_audit,bank_receipts=bank_receipts,files=files,global_rng_unchanged=True,
        physical_outcomes_measured=False,fresh_confirmation=False,total_forward_count=70,
        total_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(output/'report.json',report)
    dump(output/'DONE.json',dict(report_sha256=file_hash(output/'report.json'),files=files,status=report['status']))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('vendor','checkpoint','inputs','manifest','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','protocol-sha256','gpu-uuid'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--task',choices=TASKS,required=True);parser.add_argument('--episode',type=int,choices=range(8),required=True)
    args=parser.parse_args()
    if file_hash(args.manifest)!=args.manifest_sha256 or file_hash(args.protocol)!=args.protocol_sha256 or args.protocol_sha256!=PROTOCOL_SHA:
        raise ValueError('Manifest/protocol hash mismatch')
    manifest=json.loads(args.manifest.read_text());protocol=json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    expected={task:list(range(8)) for task in TASKS}
    if manifest['scenarios']!=expected or protocol['scenarios']!=expected or manifest['protocol_sha256']!=args.protocol_sha256:
        raise ValueError('Exact fixed16 registry required')
    if (file_hash(args.checkpoint)!=manifest['checkpoint_sha256'] or manifest['checkpoint_sha256']!=CHECKPOINT_SHA
        or file_hash(args.inputs/'INPUT_MANIFEST.json')!=manifest['input_manifest_sha256'] or manifest['input_manifest_sha256']!=INPUT_SHA):
        raise ValueError('Checkpoint/input manifest changed')
    sources=manifest['source_sha256']
    if not {'action_counterfactual_pilot.py','action_condition_specificity.py','backends.py','model_loader.py','fixed_combined_check.py'}<=sources.keys():
        raise ValueError('Required source bindings missing')
    for name,digest in sources.items():
        if Path(name).name!=name or file_hash(source_path(name))!=digest:raise ValueError('Source bytes changed: '+name)
    for name,digest in manifest['vendor_source_sha256'].items():
        path=(args.vendor/name).resolve()
        if not path.is_relative_to(args.vendor.resolve()) or file_hash(path)!=digest:raise ValueError('Vendor source changed')
    registry=json.loads((args.inputs/'INPUT_MANIFEST.json').read_text())['records']
    selected=[r for r in registry if (r['task'],r['episode'])==(args.task,args.episode)]
    if len(selected)!=1:raise ValueError('Nonunique original input binding')
    binding=selected[0];path=args.inputs/args.task/f'episode-{args.episode:03d}'/'inputs.pt'
    if file_hash(path)!=binding['inputs_sha256']:raise ValueError('Original inputs changed')
    if torch.cuda.device_count()!=1:raise ValueError('One visible assigned GPU required')
    uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))
    if uuid.removeprefix('GPU-').lower()!=args.gpu_uuid.removeprefix('GPU-').lower() or os.environ.get('JEPA_VERIFIED_LOCAL_DINO')!='1':
        raise ValueError('UUID/local DINO binding missing')
    from offline_study.models.backends import JepaBackend
    from offline_study.planning.planning_contract import prepare
    torch.set_num_threads(1)
    backend=JepaBackend(args.vendor,args.checkpoint,manifest['checkpoint_sha256'],'metaworld','cuda:0','float32',allow_tf32=False)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('TF32 forbidden')
    runtime=dict(gpu_uuid=uuid,backend_provenance=backend.provenance,torch_version=torch.__version__,tf32_matmul=False,tf32_cudnn=False)
    with torch.no_grad():
        report=run_case(backend,prepare(args.vendor,args.task)['config'],torch.load(path,map_location='cpu',weights_only=True),
            binding,args.output,args.manifest_sha256,args.protocol_sha256,runtime)
    print(json.dumps(dict(status=report['status'],seconds=report['total_seconds'],forward_count=70)),flush=True)


if __name__=='__main__':main()
