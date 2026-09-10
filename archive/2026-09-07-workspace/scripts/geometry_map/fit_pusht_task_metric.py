#!/usr/bin/env python3
"""Cached leave-one-development-state-out PSD planner-metric diagnostics.

No model forward/backward or physical rollout. The extraction mode is outcome-
blind with respect to fitting and preserves exact original scalar-cost checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def weighted_difference(visual,proprio,goal_visual,goal_proprio,alpha=.1):
    v=np.asarray(visual,dtype=np.float64).reshape(-1)-np.asarray(goal_visual,dtype=np.float64).reshape(-1)
    p=np.asarray(proprio,dtype=np.float64).reshape(-1)-np.asarray(goal_proprio,dtype=np.float64).reshape(-1)
    if alpha<0 or not np.isfinite(v).all() or not np.isfinite(p).all():raise ValueError('Finite PSD native channels required')
    return np.concatenate([v/np.sqrt(v.size),p*np.sqrt(alpha/p.size)])


def extract(bank,output):
    import torch
    torch.set_num_threads(1)
    done=json.loads((bank/'DONE.json').read_text());entries={r['path']:r for r in done['outputs']}
    if not done['complete'] or done['plans']!=64 or done['episode'] not in range(4) or done['held_access']:
        raise ValueError('Only complete64plan originaldevelopment receipt beforeload')
    def load(name):
        if digest(bank/name)!=entries[name]['sha256']:raise ValueError('Source SHA failed beforeload')
        return torch.load(bank/name,map_location='cpu',weights_only=False)
    frozen=load('FROZEN_INPUTS.pt');goal=frozen['goal_encoded'];pred=[];actual=[];states=[]
    summary=json.loads((bank/'summary.json').read_text())
    if digest(bank/'summary.json')!=entries['summary.json']['sha256']:raise ValueError('Score receipt SHA failed')
    if summary['native_objective_sum_all_diffs'] or summary['native_objective_alpha']!=.1:raise ValueError('Terminal native alpha.1 contract required')
    for i in range(64):
        b=load(f'candidate-{i:03d}.pt')
        if b['episode']!=done['episode'] or b['candidate']!=i or b['split']!='development_external':raise ValueError('Candidate grouping differs')
        pred.append(weighted_difference(b['predicted_visual'][-1],b['predicted_proprio'][-1],goal['visual'],goal['proprio']))
        actual.append(weighted_difference(b['actual_visual'][-1],b['actual_proprio'][-1],goal['visual'],goal['proprio']))
        states.append(b['truth']['states'][-1])
    pred,actual=np.stack(pred),np.stack(actual)
    errors={}
    for name,array,key in [('predicted',pred,'native_full_objective_cost'),('actual',actual,'actual_encoded_full_objective_cost')]:
        old=np.asarray([r[key] for r in summary['rows']]);reconstructed=np.sum(array**2,axis=1)
        errors[name]=float(np.max(np.abs(reconstructed-old)))
        if not np.allclose(reconstructed,old,rtol=2e-6,atol=1e-8):raise ValueError('Native weighted scalar objective reconstruction failed')
    output.mkdir(parents=True,exist_ok=False);path=output/'terminal-differences.npz'
    np.savez_compressed(path,predicted=pred.astype(np.float32),actual=actual.astype(np.float32),states=np.asarray(states),
        goal_state=np.asarray(frozen['goal_state']),episode=done['episode'],candidate_ids=np.arange(64),
        native_predicted_cost=np.asarray([r['native_full_objective_cost'] for r in summary['rows']]),
        native_actual_cost=np.asarray([r['actual_encoded_full_objective_cost'] for r in summary['rows']]))
    receipt=dict(complete=True,episode=done['episode'],source_root=str(bank),source_done_sha256=digest(bank/'DONE.json'),
        native_objective_reconstruction_maxabs=errors,shape=list(pred.shape),model_calls=0,simulator_calls=0,
        outputs=[dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size)])
    (output/'DONE.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


def fit_pca_directions(x,rank=16):
    """Centered TRAIN covariance determines axes only; metric Δ stays uncentered."""
    x=np.asarray(x,dtype=np.float64);mean=x.mean(0);centered=x-mean
    eigen,vectors=np.linalg.eigh(centered@centered.T)
    order=np.argsort(eigen)[::-1];valid=order[eigen[order]>max(1e-12,eigen[order[0]]*1e-10)][:rank]
    if not len(valid):raise ValueError('No identifiable TRAIN covariance directions')
    basis=(vectors[:,valid].T@centered)/np.sqrt(eigen[valid])[:,None]
    basis=np.linalg.qr(basis.T,mode='reduced')[0].T
    return basis,mean,eigen[valid]


def fit_weights(x,label,basis,shrinkage=.1):
    from scipy.optimize import nnls
    x=np.asarray(x,dtype=float);label=np.asarray(label,dtype=float)
    if np.any(label<0) or not np.isfinite(label).all() or label.mean()<=0:raise ValueError('Nonnegative nonzero TRAIN labels required')
    native=np.sum(x*x,axis=1);coordinates=x@basis.T;features=coordinates**2
    outside=native-features.sum(1)
    if np.min(outside)<-1e-8*max(1.,float(native.max())):raise ValueError('PCA complement negativity')
    label_scale=float(native.mean()/label.mean())
    penalty=shrinkage*float(np.sum(features**2)/len(basis))
    matrix=np.vstack([features,np.sqrt(penalty)*np.eye(len(basis))])
    target=np.concatenate([label*label_scale-outside,np.sqrt(penalty)*np.ones(len(basis))])
    weights,residual=nnls(matrix,target,maxiter=10000)
    return weights,dict(label_scale_train_only=label_scale,shrinkage=shrinkage,augmented_penalty=penalty,
        nnls_augmented_residual=float(residual),training_native_mean=float(native.mean()),training_label_mean=float(label.mean()),
        train_fit_mse=float(np.mean((outside+features@weights-label*label_scale)**2)),rank=len(basis))


def metric_cost(x,basis,weights,beta=1.,rotation=None):
    if beta not in (0.,.25,.5,1.) or np.any(weights<0):raise ValueError('Only frozen PSD blends')
    x=np.asarray(x,dtype=float);native=np.sum(x*x,axis=1)
    coordinates=x@basis.T
    if rotation is not None:coordinates=coordinates@rotation
    return native+beta*((coordinates**2)@(weights-1.))


def score_cost(cost,states,goal,coverage,baseline):
    from run_expand_pusht_candidates import spearman
    xy=np.linalg.norm(states[:,:4]-goal[:4],axis=1);picked=int(np.argmin(cost));base=int(np.argmin(baseline))
    return dict(selected_index=picked,baseline_selected_index=base,choice_changed=picked!=base,
        selected_xy_distance_px=float(xy[picked]),xy_delta_vs_baseline_px=float(xy[picked]-xy[base]),
        xy_regret_px=float(xy[picked]-xy.min()),selected_goal_coverage=float(coverage[picked]),
        goal_coverage_delta_vs_baseline=float(coverage[picked]-coverage[base]),
        goal_coverage_regret=float(coverage.max()-coverage[picked]),
        cost_vs_xy_squared_spearman=spearman(cost,xy**2),negative_cost_vs_goal_coverage_spearman=spearman(-cost,coverage),
        cost_by_candidate=cost.tolist())


def fit(a):
    started=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    coverage_receipt=json.loads(a.coverage.read_text())
    if not coverage_receipt['complete'] or sorted(r['episode'] for r in coverage_receipt['rows'])!=list(range(4)):
        raise ValueError('Four verified goal-matched coverage groups required')
    cover={r['episode']:np.asarray(r['goal_aligned_coverage_by_candidate']) for r in coverage_receipt['rows']}
    banks={};sources=[]
    for directory in a.inputs:
        receipt=json.loads((directory/'DONE.json').read_text());row=receipt['outputs'][0];episode=receipt['episode']
        if not receipt['complete'] or episode not in range(4) or episode in banks or digest(directory/row['path'])!=row['sha256']:
            raise ValueError('Source grouping/checksum failure before cachedarray load')
        value=dict(np.load(directory/row['path'],allow_pickle=False));banks[episode]=value;sources.append(receipt)
        if value['predicted'].shape!=(64,102400) or value['actual'].shape!=(64,102400):raise ValueError('Native fullweighted support changed')
    if sorted(banks)!=list(range(4)):raise ValueError('Allfour developmentstates retained')
    protocol=dict(complete=True,states=list(range(4)),outer_split='Leave one whole development state out;3TRAIN,1TEST; allpreviouslyseen',
        fit_channel='ACTUAL encoded terminal visual/proprio minus original expert goal, TRAIN only',
        native_weighting='visual /sqrt98304; proprio sqrt(.1/4096); squarednorm equals native terminal objective',
        pca_rank_max=16,pca_centering='TRAIN covariance axes centered; PSD metric acts on UNCENTERED latent-minus-goal Δ',
        labels=['joint_xy_squared_proxy','one_minus_requested_goal_coverage'],
        label_scope='Two NEW exploratory metric surrogates, not a retrospective primary endpoint rewrite',
        regularization='.1*mean diagonal(A.T@A) weight penalty toward nativeones; A=uncentered PCA coordinates squared',
        weights='nonnegative NNLS, no intercept; outsidePCA nativeenergy preserved',
        scale='TRAIN labelmean mapped to TRAIN nativecostmean; native-only global scaling is rank-invariant',
        blends=[0,.25,.5,1],control='Fixed Haar orthogonal within same TRAIN PCA span; identical learned spectrum and complement',
        control_seed=2026090605,test_channels=['predicted','actual_oracle_encoding'],
        no_tuning_or_winner_selection=True,model_calls=0,simulator_calls=0,held_confirmation_access=False,
        sources=sources,coverage_sha256=digest(a.coverage),script_sha256=digest(__file__))
    protocol_path=a.output/'protocol.json';protocol_path.write_text(json.dumps(protocol,indent=2));outputs=[protocol_path];reports=[]
    for held in range(4):
        train=[i for i in range(4) if i!=held]
        x=np.concatenate([banks[i]['actual'] for i in train]).astype(float)
        basis,mean,eigen=fit_pca_directions(x,16)
        rng=np.random.default_rng(2026090605+held);rotation=np.linalg.qr(rng.normal(size=(len(basis),len(basis))))[0]
        labels=dict(joint_xy_squared_proxy=np.concatenate([np.sum((banks[i]['states'][:,:4]-banks[i]['goal_state'][:4])**2,axis=1) for i in train]),
            one_minus_requested_goal_coverage=np.concatenate([1-cover[i] for i in train]))
        learned={};metadata={}
        for label,y in labels.items():learned[label],metadata[label]=fit_weights(x,y,basis,.1)
        fitted=a.output/f'fold-{held}-FROZEN.npz'
        np.savez_compressed(fitted,basis=basis,train_feature_mean_for_covariance_only=mean,eigenvalues=eigen,
            random_rotation=rotation,train_states=np.asarray(train),held_state=held,**{k+'_weights':v for k,v in learned.items()})
        receipt=a.output/f'fold-{held}-FROZEN.json'
        receipt.write_text(json.dumps(dict(complete=True,fit_states=train,test_state=held,fit_before_test_scoring=True,
            path=fitted.name,sha256=digest(fitted),models=metadata,weights={k:v.tolist() for k,v in learned.items()}),indent=2))
        outputs.extend([fitted,receipt]);rows=[];baseline_guards={}
        for channel,key in [('predicted','native_predicted_cost'),('actual_oracle_encoding','native_actual_cost')]:
            test=banks[held]['predicted' if channel=='predicted' else 'actual'].astype(float)
            native=np.sum(test*test,axis=1);original=banks[held][key]
            discrepancy=float(np.max(np.abs(native-original)))
            if not np.allclose(native,original,rtol=2e-6,atol=1e-8) or np.argmin(native)!=np.argmin(original):
                raise ValueError('Native cost reconstruction/selectedplan guard failed')
            baseline_guards[channel]=dict(maxabs=discrepancy,argmin_exact=True,
                full_rank_order_exact=bool(np.array_equal(np.argsort(native,kind='stable'),np.argsort(original,kind='stable'))))
            for label,weights in learned.items():
                for family,q in [('learned_psd',None),('spectrum_matched_rotation',rotation)]:
                    for beta in (0.,.25,.5,1.):
                        costs=metric_cost(test,basis,weights,beta,q)
                        if beta==0 and not np.array_equal(costs,native):raise ValueError('Exact no-edit metric guard failed')
                        if float(costs.min()) < -1e-8:raise ValueError('PSD cost negativity')
                        rows.append(dict(episode=held,fit_label=label,family=family,blend=beta,channel=channel,
                            **score_cost(costs,banks[held]['states'],banks[held]['goal_state'],cover[held],native)))
        report=dict(complete=True,held_state=held,fit_states=train,rank=len(basis),rows=rows,baseline_guards=baseline_guards,
            uncentered_metric_goal_zero=True,off_subspace_native_preserved=True,training_models=metadata,
            requested_goal_coverage_used_only_for_fit_on_otherstates_or_offline_test=True)
        path=a.output/f'fold-{held}.json';path.write_text(json.dumps(report,indent=2));outputs.append(path);reports.append(report)
        print(json.dumps(dict(event='metric_fold_complete',held_state=held,rows=len(rows),seconds=time.monotonic()-started)),flush=True)
        if time.monotonic()-started>900:raise RuntimeError('Bounded900s CPU diagnostic exhausted')
    result=dict(complete=True,states=4,folds=reports,seconds=time.monotonic()-started,model_calls=0,simulator_calls=0,
        claim_scope='Exploratory cached fixed-bank metric intervention; actualencoding arm is oracle diagnosis, not online steering; no confirmation or efficacypromotion')
    path=a.output/'summary.json';path.write_text(json.dumps(result,indent=2));outputs.append(path)
    (a.output/'DONE.json').write_text(json.dumps(dict(complete=True,seconds=result['seconds'],states=4,outputs=[dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size) for p in outputs]),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extract',type=Path);p.add_argument('--inputs',type=Path,nargs='+')
    p.add_argument('--coverage',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.extract:extract(a.extract,a.output)
    elif a.inputs and a.coverage:fit(a)
    else:p.error('Require extract or inputs plus goalalignedcoverage')


if __name__=='__main__':main()
