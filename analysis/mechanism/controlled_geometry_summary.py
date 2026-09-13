"""Complete64, source-bound CPU geometry audit; never executes a model.

Direct field errors are execution records. Their independent float64 Gram
reconstructions and finite-difference RMS identities are recomputed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data"
TASKS = ("reach", "reach-wall")
CONDITIONS = ("float32", "float32_field_bfloat16_roundtrip", "bfloat16")
EXPECTED = {(t, e) for t in TASKS for e in range(32)}
INPUT_SHA = "7eaaec460a9057078a36cb348e859222cdf426842620107bb0d2cdabe7df70ef"
CHECKPOINT_SHA = "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
WEIGHTS = {"linear": np.array([.25]*4), "cubic": np.array([-1/6, 2/3, 2/3, -1/6])}
EPS = np.finfo(np.float64).eps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value, shape=None):
    array = np.asarray(value, dtype=float)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError("Nonfinite or wrong-shaped geometry metric")
    return array


def discover(root):
    cases = {}
    for path in Path(root).rglob("report.json"):
        match = re.fullmatch(r"episode-(\d+)", path.parent.name)
        if not match or path.parent.parent.name not in TASKS:
            raise ValueError("Unexpected report outside exact task/episode layout: " + str(path))
        key = path.parent.parent.name, int(match.group(1))
        if key not in EXPECTED or key in cases:
            raise ValueError("Unexpected or duplicate scientific context")
        cases[key] = path.parent
    if set(cases) != EXPECTED:
        raise ValueError(f"Complete64 required before reading results: {len(cases)}/64")
    if any(not (p / name).is_file() for p in cases.values()
           for name in ("STARTED.json", "DONE.json", "CLOUD_VERIFIED.json")):
        raise ValueError("Complete64 DONE/cloud preservation required before reading results")
    return cases


def ratio_record(linear, cubic):
    if not math.isfinite(linear) or not math.isfinite(cubic) or min(linear, cubic) < 0:
        raise ValueError("MSE must be finite and nonnegative")
    if linear == cubic == 0:
        return {"ratio_status": "both_zero", "ratio": None, "log_ratio": None}
    if linear == 0:
        return {"ratio_status": "linear_zero", "ratio": None, "log_ratio": None}
    if cubic == 0:
        return {"ratio_status": "cubic_zero", "ratio": 0., "log_ratio": None}
    return {"ratio_status": "both_positive", "ratio": cubic / linear,
            "log_ratio": math.log(cubic) - math.log(linear)}


def audit_row(item):
    gram = finite(item["centered_anchor_gram_mean_inner_product"], (4, 4))
    scale = max(float(np.abs(gram).max()), 1e-300)
    if not np.allclose(gram, gram.T, rtol=0, atol=64*EPS*scale):
        raise ValueError("Gram not symmetric")
    if np.linalg.eigvalsh(gram).min() < -128*EPS*scale:
        raise ValueError("Gram not positive semidefinite within float64 roundoff")
    numel = int(item["field_numel"])
    if numel <= 0 or math.prod(item["field_shape"]) != numel:
        raise ValueError("Field dimension identity failed")
    donor = finite(item["donor_response_rms"], (4,))
    unchanged = finite(item["exact_unchanged_fraction"], (4,))
    center_rms = float(item["center_rms"])
    if center_rms < 0 or not math.isfinite(center_rms) or np.any(donor < 0) or np.any((unchanged < 0)|(unchanged > 1)):
        raise ValueError("Invalid RMS or unchanged fraction")
    if not np.allclose(donor**2, np.diag(gram), rtol=1e-12, atol=128*EPS*scale):
        raise ValueError("Donor RMS/Gram identity failed")
    if np.any((unchanged == 1) & (donor != 0)):
        raise ValueError("Unchanged donor has nonzero response")
    result = {}
    for name, weights in WEIGHTS.items():
        direct, saved = float(item[name+"_center_mse"]), float(item[name+"_gram_mse"])
        recomputed = float(weights @ gram @ weights)
        if direct < 0 or not np.isfinite([direct,saved]).all():
            raise ValueError("Invalid reconstruction error")
        # Conservative bound from recorded RMS: max coordinate <= sqrt(N)*RMS.
        max_coordinate_bound = math.sqrt(numel)*(center_rms+float(donor.max()))
        roundoff = 32*EPS*max_coordinate_bound
        upper = 2*math.sqrt(max(direct,abs(recomputed)))*roundoff+roundoff**2+64*EPS*scale
        tolerance = float(item[name+"_gram_tolerance"])
        if not math.isfinite(tolerance) or tolerance < 0 or tolerance > upper*(1+1e-10):
            raise ValueError("Saved direct/Gram tolerance exceeds arithmetic bound")
        numerical = 128*EPS*scale
        if abs(saved-recomputed) > numerical or abs(direct-recomputed) > tolerance+numerical:
            raise ValueError("Direct/Gram reconstruction identity failed")
        if abs(float(item[name+"_gram_absolute_error"])-abs(direct-saved)) > numerical:
            raise ValueError("Saved direct/Gram absolute-error identity failed")
        result[name+"_center_mse"] = direct
        result[name+"_gram_recomputed_mse"] = recomputed
    differences = item["finite_differences"]
    if len(differences) != 2 or {d["radius"] for d in differences} != {.05,.1}:
        raise ValueError("Both fixed finite-difference radii required")
    for difference in differences:
        radius = difference["radius"]
        a,b = (1,2) if radius == .05 else (0,3)
        first,second = np.zeros(4),np.zeros(4)
        first[a],first[b] = -.5,.5
        second[a],second[b] = 1,1
        for label,weights,normalizer in (("first",first,radius),("second",second,radius**2)):
            expected = max(float(weights @ gram @ weights),0.)
            rms = float(difference[label+"_difference_rms"])
            derivative = float(difference["central_"+label+"_derivative_rms"])
            mean = float(difference[label+"_difference_mean"])
            if not np.isfinite([rms,derivative,mean]).all() or min(rms,derivative)<0:
                raise ValueError("Invalid finite difference")
            roundoff = 64*EPS*math.sqrt(numel)*(center_rms+float(donor.max()))
            tolerance = 2*math.sqrt(expected)*roundoff+roundoff**2+512*EPS*scale
            if abs(rms*rms-expected) > tolerance or not np.isclose(derivative,rms/normalizer,rtol=1e-10,atol=0):
                raise ValueError("Finite-difference RMS/Gram identity failed")
            if abs(mean)>rms+64*EPS*max(center_rms,1e-300):
                raise ValueError("Finite-difference signed mean exceeds RMS")
            result[f"r{radius}_{label}_derivative_rms"] = derivative
    result.update(ratio_record(result['linear_center_mse'],result['cubic_center_mse']))
    result.update(donor_response_rms_mean=float(donor.mean()),
                  exact_unchanged_fraction_mean=float(unchanged.mean()))
    return result


def validate_case(directory, key, manifest, manifest_sha, protocol_sha, input_binding):
    hashes = {name:sha(directory/name) for name in ('STARTED.json','report.json','DONE.json')}
    done, cloud = read(directory/'DONE.json'), read(directory/'CLOUD_VERIFIED.json')
    if (done.get('report_sha256')!=hashes['report.json'] or done.get('started_sha256')!=hashes['STARTED.json']
            or done.get('rows')!=18):
        raise ValueError('DONE hash/count mismatch')
    if (cloud.get('gcs_download_sha256_verified') is not True or cloud.get('all_report_files_hash_verified') is not True
            or cloud.get('raw_preservation_pending',False) or cloud.get('compact_sha256')!=hashes):
        raise ValueError('Complete cloud member/readback proof missing')
    if (not str(cloud.get('cloud_uri','')).startswith('gs://') or not cloud.get('generation')
            or not re.fullmatch('[0-9a-f]{64}',str(cloud.get('sha256','')))):
        raise ValueError('Cloud object identity missing')
    report, started = read(directory/'report.json'),read(directory/'STARTED.json')
    if report.get('status')!='controlled_geometry_development_complete' or done.get('status')!=report['status']:
        raise ValueError('Scientific completion status missing')
    for payload in (report,started):
        if payload.get('binding')!=input_binding:
            raise ValueError('Original input binding mismatch')
        for field,expected in {'execution_manifest_sha256':manifest_sha,'protocol_sha256':protocol_sha,
                'checkpoint_sha256':CHECKPOINT_SHA,'input_manifest_sha256':INPUT_SHA,
                'source_sha256':manifest['source_sha256'],'vendor_source_sha256':manifest['vendor_source_sha256']}.items():
            if payload.get(field)!=expected:
                raise ValueError('Source/protocol binding mismatch: '+field)
        if not payload.get('gpu_uuid') or payload['gpu_uuid']=='unavailable':
            raise ValueError('Missing physical UUID')
    if report['gpu_uuid']!=started['gpu_uuid'] or report.get('native_hook_byte_parity') is not True or report.get('rng_parity') is not True:
        raise ValueError('UUID or execution-attested byte/RNG parity failed')
    if report.get('physical_outcomes_measured') is not False or report.get('fresh_confirmation') is not False:
        raise ValueError('Incorrect scientific evidence role')
    rows,seen=[],set()
    for item in report['rows']:
        cell=item['condition'],item['layer']
        if cell in seen or cell not in {(c,l) for c in CONDITIONS for l in range(6)}:
            raise ValueError('Duplicate/unexpected layer-condition')
        seen.add(cell)
        rows.append(dict(task=key[0],episode=key[1],condition=cell[0],layer=cell[1],**audit_row(item)))
    if len(seen)!=18:
        raise ValueError('Incomplete layer-condition grid')
    return rows,dict(task=key[0],episode=key[1],**hashes,cloud_receipt_sha256=sha(directory/'CLOUD_VERIFIED.json'),
                    cloud_uri=cloud['cloud_uri'],generation=cloud['generation'],archive_sha256=cloud['sha256'])


def summarize(frame, replicates=20000, seed=20260913):
    """One joint context draw reused across all layers/conditions/contrasts."""
    weights=np.random.default_rng(seed).multinomial(32,np.full(32,1/32),size=replicates)/32
    summaries,contrasts=[],[]
    for task in TASKS:
        values={}
        for condition in CONDITIONS:
            for layer in range(6):
                group=frame[(frame.task==task)&(frame.condition==condition)&(frame.layer==layer)].sort_values('episode')
                if list(group.episode)!=list(range(32)):
                    raise ValueError('Exact 32 paired contexts required per cell')
                values[condition,layer]=group.log_ratio.to_numpy(dtype=float)
                for metric in ('linear_center_mse','cubic_center_mse','log_ratio','donor_response_rms_mean','exact_unchanged_fraction_mean'):
                    array=group[metric].to_numpy(dtype=float)
                    summaries.append(dict(task=task,condition=condition,layer=layer,metric=metric,
                        **estimate(array,weights),**{status+'_count':int((group.ratio_status==status).sum())
                        for status in ('both_positive','both_zero','linear_zero','cubic_zero')}))
        for left,right in ((CONDITIONS[1],CONDITIONS[0]),(CONDITIONS[2],CONDITIONS[0]),(CONDITIONS[2],CONDITIONS[1])):
            matrix=np.stack([values[left,l]-values[right,l] for l in range(6)],axis=1)
            family_defined=np.isfinite(matrix).all()
            simultaneous=[(None,None)]*6
            if family_defined:
                means=matrix.mean(0); draws=weights@matrix
                se=draws.std(axis=0,ddof=1)
                standardized=np.divide(abs(draws-means),se,out=np.zeros_like(draws),where=se>0)
                critical=float(np.quantile(standardized.max(1),.95))
                simultaneous=list(zip(means-critical*se,means+critical*se))
            for layer in range(6):
                lo,hi=simultaneous[layer]
                contrasts.append(dict(task=task,left_condition=left,right_condition=right,layer=layer,
                    **estimate(matrix[:,layer],weights),simultaneous_six_layer_95_low=lo,
                    simultaneous_six_layer_95_high=hi,practical_log_margin=.1,
                    equivalence_claimed=False,simultaneous_family_complete=bool(family_defined)))
    return pd.DataFrame(summaries),pd.DataFrame(contrasts)


def estimate(array,weights):
    n_defined=int(np.isfinite(array).sum())
    result=dict(n=32,n_defined=n_defined,mean=None,marginal_95_low=None,marginal_95_high=None)
    if n_defined==32:
        draws=weights@array
        result.update(mean=float(array.mean()),marginal_95_low=float(np.quantile(draws,.025)),
                      marginal_95_high=float(np.quantile(draws,.975)))
    else:
        result['undefined_reason']='At least one exact-zero ratio; no epsilon or selective-context mean'
    return result


def plots(receipt_path, output_dir):
    receipt=read(receipt_path)
    if receipt.get('status')!='complete64_controlled_geometry_development' or receipt.get('cohort')!=64:
        raise ValueError('Plots require complete64 public summary')
    tables={}
    for relative,identity in receipt['outputs'].items():
        path=Path(receipt_path).parent/Path(relative).name
        if sha(path)!=identity['sha256']:
            raise ValueError('Public plot CSV hash mismatch')
        tables[Path(relative).name]=pd.read_csv(path)
        if len(tables[Path(relative).name])!=identity['rows']:
            raise ValueError('Public plot CSV row count mismatch')
    table=tables['controlled_geometry_summary.csv']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10.5,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(6.8,4.8),sharex=True,sharey=True,layout='constrained')
    titles=('FP32','FP32 → BF16 → FP32','BF16 autocast')
    for i,task in enumerate(TASKS):
        for j,condition in enumerate(CONDITIONS):
            ax=axes[i,j]
            block=table[(table.task==task)&(table.condition==condition)&(table.metric=='log_ratio')].sort_values('layer')
            if list(block.layer)!=list(range(6)) or set(block.n)!={32}:
                raise ValueError('Public plot missing complete layer grid')
            valid=block['mean'].notna()
            good=block[valid]
            color=('#225e9b','#6f5295','#ab572a')[j]
            ax.vlines(good.layer,good.marginal_95_low,good.marginal_95_high,color=color)
            ax.plot(good.layer,good['mean'],marker='o',markersize=3,color=color)
            if not valid.all():
                ax.text(.03,.04,'Undefined log ratios\n(exact zeros retained)',transform=ax.transAxes,fontsize=8.5)
            ax.axhline(0,color='.6',linewidth=.7);ax.grid(axis='y',alpha=.15)
            ax.set_xticks(range(6),[str(l) for l in range(6)])
            ax.spines[['top','right']].set_visible(False)
            if i==0:ax.set_title(titles[j],fontsize=10.5)
            if i==1:ax.set_xlabel('Predictor block')
            if j==0:ax.set_ylabel(('Reach' if task=='reach' else 'Reach-Wall')+'\nlog(cubic / linear MSE)')
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg','pdf'):
        fig.savefig(destination/f'paper_controlled_geometry.{extension}',dpi=200)
    plt.close(fig)


def run(args):
    manifest,protocol=read(args.manifest),read(args.protocol)
    if sha(args.manifest)!=args.manifest_sha256 or sha(args.protocol)!=manifest.get('protocol_sha256'):
        raise ValueError('Externally pinned manifest/protocol hash mismatch')
    if manifest.get('scenarios')!={t:list(range(32)) for t in TASKS} or manifest.get('checkpoint_sha256')!=CHECKPOINT_SHA:
        raise ValueError('Execution manifest cohort/checkpoint mismatch')
    if manifest.get('input_manifest_sha256')!=INPUT_SHA or sha(args.inputs_manifest)!=INPUT_SHA:
        raise ValueError('Original input manifest hash mismatch')
    records=read(args.inputs_manifest)['records']
    bindings={(r['task'],r['episode']):r for r in records}
    if len(records)!=64 or set(bindings)!=EXPECTED:
        raise ValueError('Input registry duplicate/missing contexts')
    cases=discover(args.source)
    rows,sources=[],[]
    for key,directory in sorted(cases.items()):
        case,source=validate_case(directory,key,manifest,args.manifest_sha256,sha(args.protocol),bindings[key])
        rows.extend(case);sources.append(source)
    frame=pd.DataFrame(rows)
    summary,contrasts=summarize(frame)
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    receipt_path=destination/'controlled_geometry_summary.json'
    if receipt_path.exists():
        raise ValueError('Existing analysis receipt must not be overwritten')
    outputs={}
    for name,table in [('case_metrics',frame),('summary',summary),('contrasts',contrasts)]:
        path=destination/f'controlled_geometry_{name}.csv'
        if path.exists():raise ValueError('Existing analysis CSV must not be overwritten: '+str(path))
        table.to_csv(path,index=False,float_format='%.17g')
        outputs[path.name]={'sha256':sha(path),'rows':len(table)}
    receipt={'status':'complete64_controlled_geometry_development','cohort':64,'n_per_task':32,
        'execution_manifest_sha256':args.manifest_sha256,'protocol_sha256':sha(args.protocol),
        'analysis_source_sha256':sha(__file__),'analysis_protocol_sha256':sha(DATA/'controlled_geometry_analysis_protocol.json'),
        'outputs':outputs,'sources':sources,'source_count':64,'metric_rows':len(frame),
        'all_cloud_verified':True,'all_gram_audits_passed':True,'physical_outcomes_measured':False,
        'fresh_confirmation':False,'parity_basis':'Execution-attested exact bytes/RNG; CPU independently recomputes stored Gram and scalar identities',
        'bootstrap_replicates':20000,'bootstrap_seed':20260913,
        'interval_scope':'Marginal percentile95; max-standardized-deviation simultaneous95 across6layers within each task/condition-pair contrast',
        'equivalence_claimed':False}
    with receipt_path.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False)
    plots(receipt_path,args.figures)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',type=Path)
    for name in ('source','manifest','protocol','inputs-manifest'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output',type=Path,default=DATA)
    parser.add_argument('--figures',type=Path,default=ROOT/'docs/figures')
    args=parser.parse_args()
    if args.plots_only:plots(args.plots_only,args.figures)
    elif not all((args.source,args.manifest,args.protocol,args.inputs_manifest,args.manifest_sha256)):
        parser.error('Aggregation requires source, manifest/hash, protocol and input manifest')
    else:run(args)


if __name__=='__main__':main()
