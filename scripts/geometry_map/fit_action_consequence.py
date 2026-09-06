#!/usr/bin/env python3
"""TRAIN-only matched-rank readable/action-effect subspaces and latent residual maps."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from complete_cached_geometry import sha256,write_json

def linear_map(x,y,alpha=100.):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1.
    model=Ridge(alpha=alpha).fit((x-mean)/scale,y)
    return {"mean":mean,"scale":scale,"coef":model.coef_,"intercept":model.intercept_}

def predict(m,x):return ((x-m["mean"])/m["scale"])@m["coef"].T+m["intercept"]

def subspaces(x,q,rank=2):
    # Shape [initialcondition,candidate,horizon,channel]. All fits use16trainingstarts only.
    xf=x.reshape(-1,400);mean=xf.mean(0);scale=xf.std(0);scale[scale<1e-6]=1.
    qf=q.reshape(-1,6);ym=qf.mean(0);ys=qf.std(0);ys[ys<1e-6]=1.
    xm=np.broadcast_to(x.mean(1,keepdims=True),x.shape);qm=np.broadcast_to(q.mean(1,keepdims=True),q.shape)
    inputs={"readable":(xm-mean)/scale,"action_effect":(x-xm)/scale}
    targets={"readable":(qm-ym)/ys,"action_effect":(q-qm)/ys};bases={}
    for name in inputs:
        model=Ridge(alpha=100.,fit_intercept=False).fit(inputs[name].reshape(-1,400),targets[name].reshape(-1,6))
        raw=model.coef_.T/scale[:,None]
        bases[name]=np.linalg.svd(raw,full_matrices=False)[0][:,:rank]
    joint=np.linalg.qr(np.concatenate(list(bases.values()),axis=1))[0]
    random=np.random.default_rng(2718).normal(size=(400,rank));random-=joint@(joint.T@random)
    bases["random"]=np.linalg.qr(random)[0][:,:rank]
    return bases,mean,scale

def load_banks(directories):
    rows=[];seen=set();fingerprints=set()
    for directory in directories:
        receipt=json.loads((directory/"DONE.json").read_text())
        if not receipt["complete"]:raise ValueError("Incomplete bank")
        for row in receipt["outputs"]:
            if not row["path"].endswith(".npz"):continue
            if row["split"] not in ("fit","development_validation","development_external"):raise ValueError("Held bank forbidden before load")
            if row["key"] in seen:raise ValueError("Duplicate initialcondition bank")
            path=directory/row["path"]
            if sha256(path)!=row["sha256"]:raise ValueError("Source SHA mismatch")
            bank=dict(np.load(path,allow_pickle=False));initial=bank["states"][0,0]
            if not np.array_equal(bank["states"][:,0],np.broadcast_to(initial,bank["states"][:,0].shape)):raise ValueError("Not same-state candidates")
            fingerprint=hashlib.sha256(initial.astype(np.float32).tobytes()).hexdigest()
            if fingerprint in fingerprints:raise ValueError("Overlapping initialcondition groups across panels")
            rows.append((row,bank));seen.add(row["key"]);fingerprints.add(fingerprint)
    return rows

def ranking(cost,truth,physical):
    chosen=int(np.argmin(cost));oracle=int(np.argmin(truth))
    def corr(a,b):
        value=spearmanr(a,b).statistic
        return float(value) if np.isfinite(value) else None
    return {"selected_candidate":chosen,"actual_encoded_oracle":oracle,"actual_encoded_regret":float(truth[chosen]-truth.min()),
        "selected_physical_xy_distance":float(physical[chosen]),"physical_xy_oracle":int(np.argmin(physical)),
        "rank_correlation_actual_encoded":corr(cost,truth),"rank_correlation_physical_xy":corr(cost,physical)}

def summarize_baseline(rows):
    result=[]
    for row,b in rows:
        result.append({"key":row["key"],"split":row["split"],"forecast_visual_mse":float(b["latent_mse_to_actual"].mean()),
            "horizon_visual_mse":b["latent_mse_to_actual"].mean(0).tolist(),
            "ranking":ranking(b["native_cost_by_horizon"][:,-1].reshape(7),b["actual_encoded_native_cost_by_horizon"][:,-1].reshape(7),b["goal_xy_distance"][:,-1]),
            "contact_candidates":int(b["contact"][:,-1].sum()),"reference_final_xy":float(b["goal_xy_distance"][0,-1])})
    return result

def fit(args):
    start=time.monotonic();rows=load_banks(args.bank_dirs)
    train=[v for r,v in rows if r["split"]=="fit"]
    if len(train)!=16:raise ValueError("Require fixed16fit initialgroups; no development for fitting")
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={"rank":2,"site":"predictor residual block3 newest256patches, pooled400",
        "train_keys":[r["key"] for r,_ in rows if r["split"]=="fit"],
        "development_keys":[r["key"] for r,_ in rows if r["split"]!="fit"],"alpha":100.,
        "readable_basis":"between-initialcondition candidate-mean nextphysical6D target",
        "action_effect_basis":"within-sameinitialcondition/horizon candidate-centered nextphysical6D target",
        "target":"agentXY,blockXY,sin(angle),cos(angle); training standardized equally",
        "matching":"same400inputs,6targets, ridgealpha100, raworthonormal rank2, frozenradius/timing",
        "residual_correction":"Horizon-specific P3→actual-minus-native pooledvisual residual; runtimeusesP3 only",
        "response_map":"Training linear P3→nativepooledvisual surrogate, not exact causalJacobian; nativeeditedoutputerror is decisive",
        "doses":[.001,.005],"gates":["first","all"],"no_development_fit":True,
        "native_cost_arrays_include_initial_context":True,"candidate_order_not_runtime_feature":True,
        "reference_generates_goal":"candidate0 is knownfeasible goalgenerator; not unbiasedrandomcandidatepopulation",
        "no_confirmation_or_heldstudy":True}
    write_json(args.output/"protocol.json",protocol)
    x=np.stack([v["p3_pooled"] for v in train]).astype(float);q=np.stack([v["q_next"] for v in train]).astype(float)
    pred=np.stack([v["predicted_visual_pooled"] for v in train]).astype(float)
    actual=np.stack([v["actual_visual_pooled"] for v in train]).astype(float)
    bases,mean,scale=subspaces(x,q);arrays={"basis_"+k:v for k,v in bases.items()}
    arrays.update(p3_mean=mean,p3_scale=scale,p3_norm_median=np.median(np.linalg.norm(x,axis=-1)))
    maps=[];response=[]
    for h in range(6):
        xx=x[:,:,h].reshape(-1,400);maps.append(linear_map(xx,(actual-pred)[:,:,h].reshape(-1,384)))
        fitted=linear_map(xx,pred[:,:,h].reshape(-1,384))
        response.append(fitted["coef"].T/fitted["scale"][:,None])
    for name in ("mean","scale","coef","intercept"):arrays["residual_"+name]=np.stack([m[name] for m in maps])
    arrays["response_raw"]=np.stack(response)
    observer=linear_map(actual.reshape(-1,384),q.reshape(-1,6))
    for key,value in observer.items():arrays["observer_"+key]=value
    np.savez_compressed(args.output/"calibration.npz",**arrays)
    frozen={"complete":True,"calibration_sha256":sha256(args.output/"calibration.npz"),"fit_initialgroups":16,
        "bases_orthonormal_error":{k:float(np.abs(b.T@b-np.eye(2)).max()) for k,b in bases.items()},
        "readable_action_principal_cosines":np.linalg.svd(bases["readable"].T@bases["action_effect"],compute_uv=False).tolist(),
        "protocol_sha256":sha256(args.output/"protocol.json"),"script_sha256":sha256(__file__)}
    write_json(args.output/"FROZEN.json",frozen) # before development scoring
    observer_rows=[]
    for row,b in rows:
        yy=b["q_next"].reshape(-1,6);aa=predict(observer,b["actual_visual_pooled"].reshape(-1,384));pp=predict(observer,b["predicted_visual_pooled"].reshape(-1,384))
        observer_rows.append({"key":row["key"],"split":row["split"],"actual_encoded_observer_rmse_px":float(np.sqrt(np.square(aa[:,:4]-yy[:,:4]).mean())),
            "native_prediction_decoded_rmse_px":float(np.sqrt(np.square(pp[:,:4]-yy[:,:4]).mean())),
            "physical_persistence_rmse_px":float(np.sqrt(np.square(b["q_current"].reshape(-1,6)[:,:4]-yy[:,:4]).mean()))})
    write_json(args.output/"baseline_diagnostics.json",{"complete":True,"rows":summarize_baseline(rows),"observer_rows":observer_rows,
        "causal_claim":"Action-supervised basis is not declared causal until matched native interventions improve actual held-developmentconsequences",
        "seconds":time.monotonic()-start})
    write_json(args.output/"DONE.json",{"complete":True,"outputs":[{"path":p.name,"sha256":sha256(p)} for p in sorted(args.output.iterdir()) if p.is_file()]})
    print(json.dumps({"event":"action_consequence_calibration_frozen","training_starts":16,"development_starts":len(rows)-16,"seconds":time.monotonic()-start}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bank-dirs",type=Path,nargs="+",required=True);p.add_argument("--output",type=Path,required=True)
    fit(p.parse_args())

if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
