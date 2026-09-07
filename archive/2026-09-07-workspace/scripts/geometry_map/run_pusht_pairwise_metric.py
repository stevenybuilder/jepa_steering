#!/usr/bin/env python3
"""Fixed-budget convex pairwise PSD ranking versus immutable pointwise fits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from fit_pusht_task_metric import digest,metric_cost,score_cost
from run_expand_pusht_candidates import spearman

REFERENCE='https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf'
BLENDS=(0.,.25,.5,1.)


def native_pair_temperature(native_by_state):
    values=[]
    for native in native_by_state:
        i,j=np.triu_indices(len(native),1);values.append(np.mean((native[j]-native[i])**2))
    tau=float(np.sqrt(np.mean(values)))
    if not np.isfinite(tau) or tau<=1e-12:raise ValueError('TRAIN native pair-cost scale is degenerate')
    return tau


def make_pair_groups(features,complements,scaled_labels):
    tolerance=1e-8*max(1.,float(np.max(np.abs(np.concatenate(scaled_labels)))))
    groups=[]
    for a,c,y in zip(features,complements,scaled_labels):
        i,j=np.triu_indices(len(y),1);difference=y[j]-y[i]
        preference=np.where(difference>tolerance,1.,np.where(difference < -tolerance,0.,.5))
        groups.append(dict(features=a[j]-a[i],offset=c[j]-c[i],preference=preference,
            pair_count=len(i),tie_count=int(np.sum(preference==.5)),tie_tolerance=tolerance))
    return groups


def pairwise_objective(weights,groups,tau,regularization):
    loss=0.;gradient=np.zeros_like(weights,dtype=float);n=len(groups)
    for group in groups:
        z=(group['offset']+group['features']@weights)/tau;p=group['preference']
        loss+=float(np.mean(np.logaddexp(0.,z)-p*z))/n
        gradient+=group['features'].T@(expit(z)-p)/(tau*len(z)*n)
    deviation=weights-1
    return loss+regularization*float(deviation@deviation),gradient+2*regularization*deviation


def pair_diagnostics(weights,groups,tau,regularization):
    objective,grad=pairwise_objective(weights,groups,tau,regularization)
    states=[]
    for g in groups:
        z=(g['offset']+g['features']@weights)/tau;p=g['preference'];strict=p!=.5
        agreement=np.where(z>0,p==1,np.where(z<0,p==0,.5))
        states.append(dict(pairs=g['pair_count'],ties=g['tie_count'],
            cross_entropy=float(np.mean(np.logaddexp(0.,z)-p*z)),
            strict_pair_accuracy=float(np.mean(agreement[strict])) if np.any(strict) else None))
    projected=grad.copy();projected[(weights<=1e-10)&(grad>0)]=0.
    return dict(objective=float(objective),prior_penalty=regularization*float(np.sum((weights-1)**2)),
        mean_pair_cross_entropy=float(np.mean([s['cross_entropy'] for s in states])),
        mean_strict_pair_accuracy=float(np.mean([s['strict_pair_accuracy'] for s in states if s['strict_pair_accuracy'] is not None])),
        projected_gradient_maxabs=float(np.max(np.abs(projected))),states=states)


def candidate_margins(cost,native):
    order=np.argsort(native,kind='stable');winner,runner=map(int,order[:2]);mask=np.arange(len(cost))!=winner
    challenger=int(np.arange(len(cost))[mask][np.argmin(cost[mask])])
    return dict(native_winner=winner,native_runner=runner,native_pair_margin=float(native[runner]-native[winner]),
        original_pair_margin=float(cost[runner]-cost[winner]),strongest_challenger=challenger,
        margin_to_strongest_challenger=float(cost[challenger]-cost[winner]))


def run(a):
    started=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    done=json.loads((a.pointwise/'DONE.json').read_text());entries={r['path']:r for r in done['outputs']}
    if not done['complete'] or done['states']!=4:raise ValueError('Require existing complete4fold pointwise fits')
    def prior_file(name):
        p=a.pointwise/name
        if digest(p)!=entries[name]['sha256']:raise ValueError('Prior immutablefit SHA mismatch')
        return p
    old_protocol=json.loads(prior_file('protocol.json').read_text())
    if digest(a.coverage)!=old_protocol['coverage_sha256']:raise ValueError('Requested goalcoverage labels changed')
    coverage=json.loads(a.coverage.read_text());cover={r['episode']:np.asarray(r['goal_aligned_coverage_by_candidate']) for r in coverage['rows']}
    banks={};sources=[]
    for directory in a.inputs:
        receipt=json.loads((directory/'DONE.json').read_text());row=receipt['outputs'][0]
        if not receipt['complete'] or receipt['episode'] not in range(4) or digest(directory/row['path'])!=row['sha256']:
            raise ValueError('Inputgroup/SHA failed before arrays loaded')
        if receipt['episode'] in banks:raise ValueError('Duplicate state')
        banks[receipt['episode']]=dict(np.load(directory/row['path'],allow_pickle=False));sources.append(receipt)
    if sorted(banks)!=list(range(4)):raise ValueError('Retain all4 seen developmentstates')
    protocol=dict(complete=True,labels=['joint_xy_squared_proxy','one_minus_requested_goal_coverage'],
        split='4 leave-one-whole-state-out folds, TRAIN3states only; allpreviouslyseen DEVELOPMENT',
        source_pointwise_done_sha256=digest(a.pointwise/'DONE.json'),sources=sources,coverage_sha256=digest(a.coverage),
        native_cost='terminal visualMSE+.1proprioMSE; saved weightedΔ; zeroatgoal; sameuncentered16D span andnativecomplement',
        objective='mean_TRAINstates mean_i<j [logaddexp(0,z)-p*z] + lambda*sum((w-1)^2), w>=0',
        z='(cost_j(w)-cost_i(w))/tau',target='p=1 ifscaledlabel_i<scaledlabel_j,0 ifreverse,.5 iftie',
        tie_tolerance='1e-8*max(1,TRAIN scaledlabelmaxabs)',
        temperature='sqrt(equal-state mean allwithinstate nativecostpair difference squared), TRAINonly, samebothlabels',
        regularization='original pointwise augmentedpenalty/(192*tau^2), preserves original quadratic prior in normalized costunits',
        solver=dict(method='L-BFGS-B',bounds='w>=0',initial='ones',maxiter=200,maxfun=10000,maxls=40,ftol=1e-12,gtol=1e-9),
        comparisons=['native','savedpointwise','newpairwiselogistic','same-original-Haar-rotation spectrumcontrols'],
        blends=BLENDS,channels=['predicted','actual_oracle_encoding'],no_hyperparameter_tuning=True,
        no_new_model_or_simulator_calls=True,held_confirmation_access=False,maximum_cpu_seconds=600,
        primary='predicted-channel fixedbank selected requestedgoalcoverage vsnative; XYproxy separate',
        statistical_units='Fourstates, notdependentpaircounts orplans; allblends retained, no confirmationpromotion',
        reference=dict(url=REFERENCE,method='Burges etal2005 Sec3 Eq1-3 pairwise crossentropy/logistic/ties; affinePSD adaptation not neural RankNet reproduction'),
        code_sha256=digest(__file__))
    path=a.output/'protocol.json';path.write_text(json.dumps(protocol,indent=2));files=[path];folds=[]
    for held in range(4):
        prior=dict(np.load(prior_file(f'fold-{held}-FROZEN.npz'),allow_pickle=False))
        meta=json.loads(prior_file(f'fold-{held}-FROZEN.json').read_text());old_report=json.loads(prior_file(f'fold-{held}.json').read_text())
        train=[i for i in range(4) if i!=held]
        if meta['fit_states']!=train or int(prior['held_state'])!=held:raise ValueError('Existing TRAINbasis grouping changed')
        basis=prior['basis'];rotation=prior['random_rotation'];features=[];complements=[];native=[]
        for episode in train:
            x=banks[episode]['actual'].astype(float);f=(x@basis.T)**2;e=np.sum(x*x,axis=1)
            features.append(f);complements.append(e-f.sum(1));native.append(e)
        tau=native_pair_temperature(native);learned={};fit_reports={};training={}
        for label in protocol['labels']:
            m=meta['models'][label];ys=[]
            for episode in train:
                y=np.sum((banks[episode]['states'][:,:4]-banks[episode]['goal_state'][:4])**2,axis=1) if label=='joint_xy_squared_proxy' else 1-cover[episode]
                ys.append(np.asarray(y,dtype=float)*m['label_scale_train_only'])
            groups=make_pair_groups(features,complements,ys)
            regularization=float(m['augmented_penalty']/(192*tau*tau))
            initial=np.ones(len(basis));trace=[]
            def objective(w):
                if time.monotonic()-started>600:raise RuntimeError('Frozen CPUtime limit reached')
                return pairwise_objective(w,groups,tau,regularization)
            def callback(w):trace.append(float(objective(w)[0]))
            fitted=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=[(0,None)]*len(initial),callback=callback,
                options=dict(maxiter=200,maxfun=10000,maxls=40,ftol=1e-12,gtol=1e-9))
            if not np.isfinite(fitted.x).all() or np.any(fitted.x<0):raise ValueError('Invalid PSDweights')
            learned[label]=fitted.x
            fit_reports[label]=dict(success=bool(fitted.success),message=str(fitted.message),iterations=int(fitted.nit),
                evaluations=int(fitted.nfev),lambda_numeric=regularization,tau_numeric=tau,
                original_label_scale=m['label_scale_train_only'],original_pointwise_augmentedpenalty=m['augmented_penalty'],
                state_pair_counts=[g['pair_count'] for g in groups],state_tie_counts=[g['tie_count'] for g in groups],
                tie_tolerance=groups[0]['tie_tolerance'],objective_trace=trace,
                weights_min=float(fitted.x.min()),weights_max=float(fitted.x.max()),weights=fitted.x.tolist())
            training[label]={name:pair_diagnostics(w,groups,tau,regularization) for name,w in
                [('native',initial),('saved_pointwise',prior[label+'_weights']),('pairwise_logistic',fitted.x)]}
            for name,w in [('native',initial),('saved_pointwise',prior[label+'_weights']),('pairwise_logistic',fitted.x)]:
                training[label][name]['within_state_spearman']=[spearman(c+f@w,y) for c,f,y in zip(complements,features,ys)]
        frozen=a.output/f'fold-{held}-FROZEN.npz'
        np.savez_compressed(frozen,**{k+'_weights':v for k,v in learned.items()},basis_source_sha256=entries[f'fold-{held}-FROZEN.npz']['sha256'],
            tau=tau,train_states=np.asarray(train),held_state=held)
        receipt=a.output/f'fold-{held}-FROZEN.json'
        receipt.write_text(json.dumps(dict(complete=True,held_state=held,train_states=train,fits=fit_reports,training=training,
            fit_saved_before_held_scoring=True,path=frozen.name,sha256=digest(frozen)),indent=2));files.extend([frozen,receipt]);rows=[]
        for channel,key,oldcostkey in [('predicted','predicted','native_predicted_cost'),('actual_oracle_encoding','actual','native_actual_cost')]:
            test=banks[held][key].astype(float);baseline=np.sum(test*test,axis=1)
            if np.argmin(baseline)!=np.argmin(banks[held][oldcostkey]):raise ValueError('Native choice guard failed')
            for label in protocol['labels']:
                for algorithm,w in [('saved_pointwise',prior[label+'_weights']),('pairwise_logistic',learned[label])]:
                    for family,q in [('learned_psd',None),('spectrum_matched_rotation',rotation)]:
                        for beta in BLENDS:
                            cost=metric_cost(test,basis,w,beta,q)
                            if beta==0 and not np.array_equal(cost,baseline):raise ValueError('Zero blendexact identity failed')
                            if float(cost.min()) < -1e-8:raise ValueError('PSDcost negative')
                            if algorithm=='saved_pointwise':
                                old=next(r for r in old_report['rows'] if r['channel']==channel and r['fit_label']==label and r['family']==family and r['blend']==beta)
                                if not np.allclose(cost,old['cost_by_candidate'],rtol=1e-12,atol=1e-12):raise ValueError('Immutablepointwise comparison changed')
                            rows.append(dict(episode=held,channel=channel,fit_label=label,algorithm=algorithm,family=family,blend=beta,
                                **score_cost(cost,banks[held]['states'],banks[held]['goal_state'],cover[held],baseline),
                                margins=candidate_margins(cost,baseline)))
        report=dict(complete=True,held_state=held,train_states=train,rank=len(basis),tau=tau,fits=fit_reports,training=training,rows=rows,
            pointwise_scores_unchanged=True,native_complement_preserved=True,metric_goalzero=True)
        path=a.output/f'fold-{held}.json';path.write_text(json.dumps(report,indent=2));files.append(path);folds.append(report)
        print(json.dumps(dict(event='pairwise_fold_complete',held_state=held,rows=len(rows),seconds=time.monotonic()-started,
            converged={k:v['success'] for k,v in fit_reports.items()})),flush=True)
    result=dict(complete=True,folds=folds,seconds=time.monotonic()-started,all_fits_converged=all(m['success'] for f in folds for m in f['fits'].values()),
        model_calls=0,simulator_calls=0,states=4,conditions=sum(len(f['rows']) for f in folds),
        scope='Exploratory fixed-bank metric re-ranking, not newphysical steering/CEM or untouched confirmation')
    path=a.output/'summary.json';path.write_text(json.dumps(result,indent=2));files.append(path)
    (a.output/'DONE.json').write_text(json.dumps(dict(complete=True,seconds=result['seconds'],states=4,
        outputs=[dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size) for p in files]),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pointwise',type=Path,required=True)
    p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--coverage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
