"""Independent complete16 CPU audit of frozen counterfactual C, no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'paper/data'
TASKS=('reach','reach-wall')
BANKS=('original','fresh')
MODALITIES=('visual','proprio','official')
MODES=('donor_h3','donor_persistent','random_range','random_off_range','random_isotropic')
ARMS=('native','zero_capture','coherent_raw_h3','all_blocks_h3_donor','all_blocks_persistent_donor',
      *(f'{m}_B{l}' for l in range(6) for m in MODES))
PROTOCOL_SHA='7c16653101ebc9a29bca82b88783e7ae44edac2a134733bc87dc0f6263141f72'
INPUT_SHA='7eaaec460a9057078a36cb348e859222cdf426842620107bb0d2cdabe7df70ef'
CHECKPOINT_SHA='c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'
EXPECTED={(t,e) for t in TASKS for e in range(8)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def finite(value,shape=None):
    a=np.asarray(value,dtype=float)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('Nonfinite or unexpected array shape')
    return a


def close(actual,expected,label,rtol=1e-8,atol=1e-12):
    if not np.isclose(actual,expected,rtol=rtol,atol=atol):raise ValueError('Identity failed: '+label)


def same_optional(actual,expected,label):
    if (actual is None)!=(expected is None):raise ValueError('Undefined value mismatch: '+label)
    if expected is not None:close(actual,expected,label)


def rank(values):
    order=np.argsort(values,kind='stable');result=np.empty(len(values),dtype=float);start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        result[order[start:end]]=(start+end-1)/2+1;start=end
    return result


def score_audit(native,current):
    a,b=finite(native['costs'],(300,)),finite(current['costs'],(300,))
    for item,costs in ((native,a),(current,b)):
        ids=np.asarray(item['elite_indices'])
        if ids.shape!=(10,) or ids.dtype.kind not in 'iu' or len(set(ids))!=10 or min(ids)<0 or max(ids)>=300:
            raise ValueError('Invalid top10 candidate IDs')
        if costs[ids].max()>np.delete(costs,ids).min():raise ValueError('Invalid actual top10 membership')
    ra,rb=rank(a),rank(b);ra-=ra.mean();rb-=rb.mean()
    denominator=np.linalg.norm(ra)*np.linalg.norm(rb)
    spearman=float(ra@rb/denominator) if denominator>0 else None
    delta=b-a
    return dict(native_spearman=spearman,spearman_loss=1-spearman if spearman is not None else None,
        native_top10_overlap=len(set(native['elite_indices'])&set(current['elite_indices'])),
        cost_change_mean=float(delta.mean()),cost_change_rms=float(np.sqrt(np.mean(delta**2))),
        centered_cost_change_rms=float(np.sqrt(np.mean((delta-delta.mean())**2))))


def reconstruction_audit(row):
    numerator,denominator=finite(row['numerator_mse'],(300,)),finite(row['denominator_mse'],(300,))
    if np.any(numerator<0) or np.any(denominator<0):raise ValueError('Negative squared forecast distance')
    expected=[1-float(n/d) if d>0 else None for n,d in zip(numerator,denominator)]
    if len(row['reconstruction'])!=300:raise ValueError('Missing candidate reconstruction')
    for actual,value in zip(row['reconstruction'],expected):same_optional(actual,value,'candidate R')
    if row['n_defined']!=sum(v is not None for v in expected):raise ValueError('Wrong defined candidate count')
    # The primary independent unit is a context: ratio of means, NOT mean of R_i.
    pooled=1-float(numerator.mean()/denominator.mean()) if denominator.mean()>0 else None
    same_optional(row['pooled_reconstruction'],pooled,'scenario ratio of means')
    return dict(numerator_mse_mean=float(numerator.mean()),denominator_mse_mean=float(denominator.mean()),
        donor_reconstruction=pooled,zero_denominator_candidates=int((denominator==0).sum()))


def norm_audit(audit):
    target=finite(audit['requested_delta_l2'],(300,));delivered=finite(audit['delivered_delta_l2'],(300,))
    relative=finite(audit['relative_norm_error'],(300,))
    if np.any(target<0) or np.any(delivered<0):raise ValueError('Negative edit norm')
    expected=np.divide(abs(delivered-target),target,out=np.zeros_like(target),where=target>0)
    if np.any(expected>1e-5) or np.any(delivered[target==0]!=0) or not np.allclose(relative,expected,rtol=1e-8,atol=1e-14):
        raise ValueError('Actual delivered norm audit failed')
    close(audit['max_relative_norm_error'],float(expected.max()),'maximum norm error')
    if audit['zero_target_count']!=int((target==0).sum()):raise ValueError('Zero norm count mismatch')
    ranges={}
    for name in ('in_range_energy_fraction','off_range_energy_fraction'):
        values=audit[name]
        if len(values)!=300:raise ValueError('Missing candidate subspace diagnostics')
        for value,norm in zip(values,delivered):
            if norm==0 and value is not None:raise ValueError('Undefined zero-energy subspace fraction')
            if norm>0 and (value is None or not math.isfinite(value) or value<0 or value>1+1e-10):
                raise ValueError('Invalid subspace fraction')
        ranges[name]=np.array([v if v is not None else np.nan for v in values])
    valid=delivered>0
    if not np.allclose(ranges['in_range_energy_fraction'][valid]+ranges['off_range_energy_fraction'][valid],1,atol=1e-10,rtol=1e-10):
        raise ValueError('Orthogonal energy decomposition failed')
    return dict(requested_l2_mean=float(target.mean()),delivered_l2_mean=float(delivered.mean()),
        max_relative_norm_error=float(expected.max()),zero_target_count=int((target==0).sum()),
        **{name:float(v.mean()) if np.isfinite(v).all() else None for name,v in ranges.items()})


def expected_addresses(arm):
    if arm in ('native','zero_capture','coherent_raw_h3'):return set()
    layers=range(6) if arm.startswith('all_blocks') else [int(arm.rsplit('_B',1)[1])]
    horizons=(3,4) if 'persistent' in arm else (3,)
    return {(h,l,1 if h==3 else 0) for h in horizons for l in layers}


def audit_range(info):
    singular=finite(info['singular_values'],(20,))
    if info['weight_shape']!=[400,20] or singular[0]<=0 or np.any(singular<0) or np.any(np.diff(singular)>0):
        raise ValueError('Literal action weight/SVD shape invalid')
    if info['relative_rank_threshold']!=1e-6 or info['retained_rank']!=int((singular>1e-6*singular[0]).sum()):
        raise ValueError('Frozen rank threshold mismatch')
    if info['retained_rank']<1 or info['svd_reconstruction_relative_error']>1e-12 or info['orthogonality_max_error']>1e-12:
        raise ValueError('SVD/projector qualification missing')
    for name in ('weight_sha256','projector_basis_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',info[name]):raise ValueError('Missing matrix hash')


def audit_scores(payload,key):
    rows,norm_rows=[],[];audit_range(payload['range_audit'])
    if set(payload['banks'])!=set(BANKS):raise ValueError('Missing original/fresh bank')
    for bank in BANKS:
        arms=payload['banks'][bank]
        if set(arms)!=set(ARMS):raise ValueError('Missing/extra frozen35 arms')
        reference=arms['native']['scores']
        if arms['zero_capture']['scores']!=reference or arms['all_blocks_persistent_donor']['scores']!=arms['coherent_raw_h3']['scores']:
            raise ValueError('Exact native/coherent cost and elite parity failed')
        epsilons={};targets={}
        for arm,entry in arms.items():
            if set(entry['scores'])!=set(MODALITIES):raise ValueError('Incomplete cost modalities')
            scores={m:finite(entry['scores'][m]['costs'],(300,)) for m in MODALITIES}
            # Producer's objective is FP32 arithmetic, not reassociated FP64.
            official=(scores['visual'].astype(np.float32)+np.float32(.1)*scores['proprio'].astype(np.float32)).astype(float)
            if not np.array_equal(official,scores['official']):raise ValueError('Official weighted goal-cost identity failed')
            audits=entry['condition_audits'];addresses={(a['horizon'],a['layer'],a['position']) for a in audits}
            if addresses!=expected_addresses(arm) or len(audits)!=len(addresses):raise ValueError('Condition occurrence addressing failed')
            for a in audits:
                norm_rows.append(dict(task=key[0],episode=key[1],bank=bank,arm=arm,horizon=a['horizon'],layer=a['layer'],**norm_audit(a)))
                if a['horizon']==3:
                    target=np.asarray(a['requested_delta_l2'])
                    if a['layer'] in targets and not np.array_equal(target,targets[a['layer']]):raise ValueError('Matched donor targets changed between controls')
                    targets[a['layer']]=target
            if arm.startswith('random_'):
                layer=int(arm.rsplit('_B',1)[1]);label=f'action-counterfactual-v1|{key[0]}|{key[1]}|{bank}|{layer}'
                expected_seed=int(hashlib.sha256(label.encode()).hexdigest()[:16],16)&((1<<63)-1)
                if entry['random_seed']!=expected_seed or not re.fullmatch('[0-9a-f]{64}',entry['epsilon_sha256']):raise ValueError('Private random draw binding failed')
                if layer in epsilons and epsilons[layer]!=entry['epsilon_sha256']:raise ValueError('Controls did not share Gaussian draw')
                epsilons[layer]=entry['epsilon_sha256']
            rec={(r['modality'],r['horizon']):r for r in entry['reconstruction']}
            if len(entry['reconstruction'])!=9 or set(rec)!={(m,h) for m in MODALITIES for h in (3,4,6)}:
                raise ValueError('Missing H3/H4/H6 reconstruction grid')
            for h in (3,4,6):
                for metric in ('numerator_mse','denominator_mse'):
                    expected=np.array(rec['visual',h][metric])+.1*np.array(rec['proprio',h][metric])
                    if not np.allclose(rec['official',h][metric],expected,rtol=1e-12,atol=0):raise ValueError('Weighted forecast-distance identity failed')
            for (modality,horizon),r in rec.items():
                agreement=score_audit(reference[modality],entry['scores'][modality])
                saved=entry['agreement_native'][modality]
                for actual_name,saved_name in (('native_spearman','spearman'),('native_top10_overlap','top10_overlap_count'),
                    ('cost_change_mean','cost_change_mean'),('cost_change_rms','cost_change_rms'),('centered_cost_change_rms','centered_cost_change_rms')):
                    same_optional(saved[saved_name],agreement[actual_name],'saved score/rank agreement')
                rows.append(dict(task=key[0],episode=key[1],bank=bank,arm=arm,modality=modality,horizon=horizon,
                    **reconstruction_audit(r),**agreement))
        for m in MODALITIES:
            for h in (3,4,6):
                native=next(r for r in arms['native']['reconstruction'] if (r['modality'],r['horizon'])==(m,h))
                for arm in ARMS:
                    current=next(r for r in arms[arm]['reconstruction'] if (r['modality'],r['horizon'])==(m,h))
                    if current['denominator_mse']!=native['denominator_mse']:raise ValueError('Native-to-coherent denominator changed across arms')
                    if arm in ('coherent_raw_h3','all_blocks_persistent_donor') and any(current['numerator_mse']):raise ValueError('Coherent positive control distance nonzero')
    return rows,norm_rows


def discover(source):
    cases={}
    for path in Path(source).rglob('report.json'):
        match=re.fullmatch(r'episode-(\d+)',path.parent.name)
        key=(path.parent.parent.name,int(match.group(1))) if match else None
        if key not in EXPECTED or key in cases:raise ValueError('Unexpected/duplicate case directory')
        cases[key]=path.parent
    if set(cases)!=EXPECTED:raise ValueError(f'Complete16 required before result access: {len(cases)}/16')
    if any(not (p/n).is_file() for p in cases.values() for n in ('STARTED.json','scores.json','actions.pt','DONE.json','CLOUD_VERIFIED.json')):
        raise ValueError('Complete16 case files/cloud proofs required')
    return cases


def validate_runtime(runtime,manifest):
    allowed={row['gpu_uuid'].removeprefix('GPU-') for row in manifest.get('receivers',{}).values()}
    if not allowed or str(runtime.get('gpu_uuid','')).removeprefix('GPU-') not in allowed:
        raise ValueError('GPU UUID is not a frozen manifest receiver')
    backend=runtime.get('backend_provenance',{})
    if (backend.get('checkpoint_sha256')!=CHECKPOINT_SHA or backend.get('precision')!='float32'
        or backend.get('allow_tf32') is not False or backend.get('autocast') is not False
        or runtime.get('tf32_matmul') is not False or runtime.get('tf32_cudnn') is not False):
        raise ValueError('Backend checkpoint or strict FP32 provenance mismatch')


def validate_case(directory,key,binding,manifest_sha,manifest):
    names=('STARTED.json','scores.json','actions.pt','report.json','DONE.json')
    hashes={n:sha(directory/n) for n in names};report=read(directory/'report.json');done=read(directory/'DONE.json');cloud=read(directory/'CLOUD_VERIFIED.json')
    files={n:hashes[n] for n in ('STARTED.json','scores.json','actions.pt')}
    if done.get('report_sha256')!=hashes['report.json'] or done.get('files')!=files or report.get('files')!=files:
        raise ValueError('Original case-file/DONE hashes differ')
    if (cloud.get('compact_sha256')!=hashes or cloud.get('gcs_download_sha256_verified') is not True
        or cloud.get('all_report_files_hash_verified') is not True or cloud.get('raw_preservation_pending',False)):
        raise ValueError('Complete cloud readback/member proof missing')
    if not str(cloud.get('cloud_uri','')).startswith('gs://') or not cloud.get('generation') or not re.fullmatch('[0-9a-f]{64}',str(cloud.get('sha256',''))):
        raise ValueError('Cloud archive identity missing')
    started,payload=read(directory/'STARTED.json'),read(directory/'scores.json')
    for item in (started,report,payload):
        if item.get('input_binding')!=binding or item.get('execution_manifest_sha256')!=manifest_sha or item.get('protocol_sha256')!=PROTOCOL_SHA:
            raise ValueError('Frozen source/protocol/input identity mismatch')
    if report.get('status')!='complete_action_counterfactual_development_case' or done.get('status')!=report['status']:
        raise ValueError('Case not scientifically complete')
    if report.get('global_rng_unchanged') is not True or report.get('physical_outcomes_measured') is not False or report.get('fresh_confirmation') is not False:
        raise ValueError('RNG or evidence-role check missing')
    runtime=report['runtime_provenance']
    if runtime!=started['runtime_provenance'] or not runtime.get('gpu_uuid') or runtime['gpu_uuid']=='unavailable':raise ValueError('GPU binding changed')
    validate_runtime(runtime,manifest)
    if report.get('total_forward_count')!=70 or started.get('arms')!=list(ARMS):raise ValueError('Fixed35×2 branches missing')
    if payload['range_audit']!=report['range_audit'] or payload['range_audit']!=started['range_audit']:raise ValueError('Action range binding changed')
    for bank in BANKS:
        rec=report['bank_receipts'][bank]
        if rec!=payload['bank_receipts'][bank] or rec.get('forecast_count')!=35 or any(rec.get(k) is not True for k in
            ('zero_full_horizon_byte_parity','persistent_allblock_raw_full_horizon_byte_parity','h1_h2_unchanged_all_arms')):
            raise ValueError('Full-horizon engineering parity missing')
    rows,norms=audit_scores(payload,key)
    provenance=dict(task=key[0],episode=key[1],hashes=hashes,cloud_receipt_sha256=sha(directory/'CLOUD_VERIFIED.json'),
        cloud_uri=cloud['cloud_uri'],generation=cloud['generation'],archive_sha256=cloud['sha256'],gpu_uuid=runtime['gpu_uuid'],range_audit=report['range_audit'])
    return rows,norms,provenance


def estimate(values,weights):
    valid=np.isfinite(values);result=dict(n=8,n_defined=int(valid.sum()),mean=None,marginal_95_low=None,marginal_95_high=None)
    if valid.all():
        draws=weights@values;result.update(mean=float(values.mean()),marginal_95_low=float(np.quantile(draws,.025)),marginal_95_high=float(np.quantile(draws,.975)))
    else:result['undefined_reason']='One or more fixed context denominators/ranks undefined; no selective mean'
    return result


PRIMARY=(('persistence_minus_h3','donor_persistent','donor_h3','donor_reconstruction'),
         ('h3_donor_minus_range_random','donor_h3','random_range','donor_reconstruction'),
         ('range_minus_offrange_rank_loss','random_range','random_off_range','spearman_loss'))


def aggregate(frame,draws=20000,seed=20260913):
    weights=np.random.default_rng(seed).multinomial(8,np.full(8,1/8),size=draws)/8
    summary=[]
    for key,group in frame.groupby(['task','bank','arm','modality','horizon'],sort=True):
        group=group.sort_values('episode')
        if list(group.episode)!=list(range(8)):raise ValueError('Eight whole contexts required per summary cell')
        for metric in ('donor_reconstruction','numerator_mse_mean','denominator_mse_mean','native_spearman','spearman_loss','native_top10_overlap','centered_cost_change_rms'):
            summary.append(dict(zip(('task','bank','arm','modality','horizon'),key),metric=metric,**estimate(group[metric].to_numpy(float),weights)))
    primary=[];source=frame[(frame.modality=='official')&(frame.horizon==6)]
    for task in TASKS:
        for name,left,right,metric in PRIMARY:
            values=[];cells=[]
            for bank in BANKS:
                for layer in range(6):
                    paired=[]
                    for arm in (left,right):
                        group=source[(source.task==task)&(source.bank==bank)&(source.arm==f'{arm}_B{layer}')].sort_values('episode')
                        if list(group.episode)!=list(range(8)):raise ValueError('Incomplete primary paired registry')
                        paired.append(group[metric].to_numpy(float))
                    values.append(paired[0]-paired[1]);cells.append((bank,layer))
            matrix=np.stack(values,axis=1);valid=np.isfinite(matrix).all();critical=None
            if valid:
                means=matrix.mean(0);resamples=weights@matrix
                # Frozen C uses unstudentized centered deviations; NOT A's standardized band.
                maxima=np.abs(resamples-means).max(1)
                critical=float(np.quantile(maxima,1-.05/6))
            for j,(bank,layer) in enumerate(cells):
                point=estimate(matrix[:,j],weights)
                primary.append(dict(task=task,contrast=name,bank=bank,layer=layer,metric=metric,
                    **point,simultaneous_72_primary_95_low=point['mean']-critical if critical is not None else None,
                    simultaneous_72_primary_95_high=point['mean']+critical if critical is not None else None,
                    family_complete=bool(valid),family_quantile=1-.05/6,critical_absolute_deviation=critical))
    return pd.DataFrame(summary),pd.DataFrame(primary)


def run(args):
    if sha(args.manifest)!=args.manifest_sha256 or sha(args.protocol)!=PROTOCOL_SHA:raise ValueError('Frozen manifest/protocol changed')
    manifest=read(args.manifest)
    if (manifest.get('protocol_sha256')!=PROTOCOL_SHA or manifest.get('scenarios')!={t:list(range(8)) for t in TASKS}
        or manifest.get('input_manifest_sha256')!=INPUT_SHA or manifest.get('checkpoint_sha256')!=CHECKPOINT_SHA or sha(args.inputs_manifest)!=INPUT_SHA):
        raise ValueError('Cohort/input/checkpoint registry mismatch')
    registry=read(args.inputs_manifest)['records'];bindings={(r['task'],r['episode']):r for r in registry}
    if len(bindings)!=len(registry) or not EXPECTED<=bindings.keys():raise ValueError('Input registry incomplete/duplicate')
    cases=discover(args.source);rows=[];norms=[];sources=[]
    for key,directory in sorted(cases.items()):
        case,norm,provenance=validate_case(directory,key,bindings[key],args.manifest_sha256,manifest)
        rows.extend(case);norms.extend(norm);sources.append(provenance)
    weights={s['range_audit']['weight_sha256'] for s in sources}
    if len(weights)!=1:raise ValueError('Cases used different literal action encoder weights')
    frame=pd.DataFrame(rows);summary,primary=aggregate(frame)
    tables={'case_metrics':frame,'norm_audits':pd.DataFrame(norms),'summary':summary,'primary_contrasts':primary}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    receipt_path=output/'action_counterfactual_summary.json'
    if receipt_path.exists() or any((output/f'action_counterfactual_{k}.csv').exists() for k in tables):raise ValueError('Existing analysis outputs must not be overwritten')
    outputs={}
    for name,table in tables.items():
        path=output/f'action_counterfactual_{name}.csv';table.to_csv(path,index=False,float_format='%.17g')
        outputs[path.name]=dict(sha256=sha(path),rows=len(table))
    receipt=dict(status='complete16_action_counterfactual_development',cohort=16,n_per_task=8,
        execution_manifest_sha256=args.manifest_sha256,protocol_sha256=PROTOCOL_SHA,analysis_source_sha256=sha(__file__),
        outputs=outputs,sources=sources,all_cloud_verified=True,physical_outcomes_measured=False,fresh_confirmation=False,
        bootstrap_draws=20000,bootstrap_seed=20260913,primary_cells=72,
        primary_interval='Max absolute centered bootstrap deviations over12bank-layer cells per task/contrast; six families Bonferroni quantile1-.05/6',
        reconstruction='Scenario ratio of mean squared errors, never mean candidate ratios',
        parity_basis='Full-horizon bytes/RNG execution-attested; archived costs/elites/scalar distance identities independently recomputed')
    with receipt_path.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','manifest','protocol','inputs-manifest'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--output',type=Path,default=DATA)
    args=parser.parse_args();receipt=run(args);print(json.dumps(dict(status=receipt['status'],cohort=16,primary_cells=72)))


if __name__=='__main__':main()
