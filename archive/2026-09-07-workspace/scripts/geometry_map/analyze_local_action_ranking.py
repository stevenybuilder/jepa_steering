#!/usr/bin/env python3
"""Fixed coarse versus near-selected-plan rank diagnostics, N=initial states."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from complete_cached_geometry import sha256,write_json

def correlation(x,y):
    v=spearmanr(x,y).statistic
    return float(v) if np.isfinite(v) else None

def metrics(pred,truth,xy,block,angle):
    pred=np.asarray(pred).reshape(-1);truth=np.asarray(truth).reshape(-1)
    pairs=np.triu_indices(len(pred),1);pd=(pred[:,None]-pred[None])[pairs];td=(truth[:,None]-truth[None])[pairs]
    ties=(np.abs(pd)<=1e-6)|(np.abs(td)<=1e-6);use=~ties
    i=int(pred.argmin());oracle=int(truth.argmin())
    return dict(rank_actual_encoded=correlation(pred,truth),rank_physical_xy=correlation(pred,xy),
        rank_block_xy=correlation(pred,block),rank_wrapped_angle=correlation(pred,angle),
        candidate_count=len(pred),pairs=int(len(pd)),ties_at_abs_1e_minus6=int(ties.sum()),
        pair_order_accuracy_non_ties=float(np.mean(np.sign(pd[use])==np.sign(td[use]))) if use.any() else None,
        native_cost_mse=float(np.square(pred-truth).mean()),native_cost_mean_error=float(np.mean(pred-truth)),
        predicted_top2_margin=float(np.diff(np.sort(pred)[:2])[0]),actual_top2_margin=float(np.diff(np.sort(truth)[:2])[0]),
        selected_candidate=i,actual_encoded_oracle=oracle,encoded_regret=float(truth[i]-truth[oracle]),
        selected_physical_4coordinate_L2_px=float(xy[i]),selected_block_L2_px=float(block[i]),selected_angle_rad=float(angle[i]),
        physical_4coordinate_oracle=int(np.argmin(xy)),block_oracle=int(np.argmin(block)),angle_oracle=int(np.argmin(angle)))

def load(directory):
    done=json.loads((directory/'DONE.json').read_text());out={}
    if not done['complete']:raise ValueError('Incomplete bank')
    for row in done['outputs']:
        if not row['path'].endswith('.npz'):continue
        if row['split']!='development_external':raise ValueError('Only four declared original development sources')
        path=directory/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Bank SHA mismatch')
        out[row['source_id']]=(row,dict(np.load(path,allow_pickle=False)))
    if set(out)!=set(range(4)):raise ValueError('Require all four starts, no selection')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('coarse','near','intervention','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();coarse=load(args.coarse);near=load(args.near);rows=[]
    for source in range(4):
        cr,c=coarse[source];nr,n=near[source]
        states_match=bool(np.array_equal(c['states'][0,0],n['states'][0,0]));goal_diff=float(np.max(np.abs(c['goal_state']-n['goal_state'])))
        row=dict(source_id=source,same_observed_initial_state=states_match,goal_state_maxabs_difference=goal_diff,
            coarse_sha256=cr['sha256'],near_sha256=nr['sha256'],panels={})
        for name,bank in (('coarse',c),('near',n)):
            table=[]
            for h in range(6):
                table.append(dict(imagined_step=h+1,raw_action_offset=5*(h+1),**metrics(bank['native_cost_by_horizon'][:,h+1],
                    bank['actual_encoded_native_cost_by_horizon'][:,h+1],bank['goal_xy_distance'][:,h],
                    bank['goal_block_distance'][:,h],bank['wrapped_angle_error'][:,h])))
            row['panels'][name]=table
        ir=json.loads((args.intervention/'DONE.json').read_text());entry=next(r for r in ir['outputs'] if r['key']==nr['key'] and r['path'].endswith('.npz'))
        path=args.intervention/entry['path']
        if sha256(path)!=entry['sha256']:raise ValueError('Intervention SHA mismatch')
        a=dict(np.load(path,allow_pickle=False));conditions={}
        meta=json.loads(path.with_suffix('.json').read_text());alpha=meta['native_goal_objective']['alpha']
        for setting in meta['conditions']:
            arm=setting['name'];p=a[arm+'_native_cost'][:,-1].reshape(-1)
            conditions[arm]=metrics(p,n['actual_encoded_native_cost_by_horizon'][:,-1],n['goal_xy_distance'][:,-1],n['goal_block_distance'][:,-1],n['wrapped_angle_error'][:,-1])
        row['near_conditions']=conditions
        v=a['unsteered_visual_goal_cost'][:,-1];prop=a['unsteered_proprio_goal_cost'][:,-1]*alpha
        # Visual objective exactly equals visualMSE + alpha*proprioMSE in native wrapper.
        row['native_component_analysis']=dict(alpha=alpha,visual_candidate_sd=float(v.std()),weighted_proprio_candidate_sd=float(prop.std()),
            component_reconstruction_maxabs=float(np.abs(v+prop-a['unsteered_native_cost'][:,-1].reshape(-1)).max()),
            visual_rank_actual_encoded=correlation(v,n['actual_encoded_native_cost_by_horizon'][:,-1].reshape(-1)),
            weighted_proprio_rank_actual_encoded=correlation(prop,n['actual_encoded_native_cost_by_horizon'][:,-1].reshape(-1)))
        rows.append(row)
    aggregates={}
    for panel in ('coarse','near'):
        endpoints=[r['panels'][panel][-1] for r in rows]
        aggregates[panel]={k:float(np.mean([r[k] for r in endpoints if r[k] is not None])) for k in
            ('rank_actual_encoded','rank_physical_xy','rank_block_xy','rank_wrapped_angle','pair_order_accuracy_non_ties','encoded_regret','native_cost_mse')}
    write_json(args.output,dict(complete=True,n_initialstates=4,rows=rows,endpoint_aggregate=aggregates,
        script_sha256=sha256(__file__),limitations=['Four previously seen development initialconditions, not36 independent trials',
        'Native cost ground reference uses actual encoded futures, not simulator reward; physical block/angle scored separately',
        'Coarse reference candidate constructs goal, causing selection ceiling; near panel uses cached native-selected plan and fixed perturbations',
        'Rank comparisons across panels require same initialstate/goal; exact differences are reported',
        'No claimed policy success gain from decoded movement or surrogate prediction error alone']))

if __name__=='__main__':main()
