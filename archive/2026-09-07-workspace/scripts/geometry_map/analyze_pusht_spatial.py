#!/usr/bin/env python3
"""Bounded permitted-source alignment/coverage diagnosis and equal-rank probes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from screen_coordinate_time import fit_models,predict_models,per_episode_ci
from capture_pusht_spatial import permitted
from complete_cached_geometry import sha256,write_json

VIEWS=("p3_global","p3_spatial","predicted_global","predicted_spatial")
ALPHAS=(.01,1.)


def load_permitted(directories,spatial=False):
    import torch
    rows=[];seen=set()
    for directory in directories:
        receipt=json.loads((directory/"DONE.json").read_text())
        if not receipt["complete"]:raise RuntimeError("Incomplete source")
        for row in receipt["outputs"]:
            if not permitted(row):continue # before hash/load, neverconfirmation
            source=row["source_id"]
            if source in seen:raise ValueError("Duplicate source")
            path=directory/row["path"]
            if not path.resolve().is_relative_to(directory.resolve()) or sha256(path)!=row["sha256"]:raise ValueError("Unsafe/mismatched source")
            value=torch.load(path,map_location="cpu",weights_only=True)
            if not permitted(value["meta"]) or value["meta"]["source_id"]!=source:raise ValueError("Metadata split/ID mismatch")
            rows.append((source,value,{"path":str(path),"sha256":row["sha256"],"split":row["split"]}));seen.add(source)
    if seen!=set(range(125)):raise ValueError("Require exactly100fit+25seenvalidation")
    return sorted(rows,key=lambda x:x[0])


def identity_key(state):
    return hashlib.sha256(np.asarray(state,dtype=np.float64).tobytes()).hexdigest()


def diagnostic(rows):
    groups={};initial_ids=[];records=[];max_errors={"q_current":0.,"q_next":0.,"native_proprio":0.}
    for source,value,provenance in rows:
        states=value["states"].numpy().astype(float);cur=value["q_current"].numpy();nxt=value["q_next"].numpy()
        expected=np.concatenate([states[:,:4],np.sin(states[:,4:5]),np.cos(states[:,4:5])],-1)
        max_errors["q_current"]=max(max_errors["q_current"],float(np.abs(cur-expected[:-1]).max()))
        max_errors["q_next"]=max(max_errors["q_next"],float(np.abs(nxt-expected[1:]).max()))
        prop=np.concatenate([states[:,:2],states[:,5:7]],-1)
        max_errors["native_proprio"]=max(max_errors["native_proprio"],float(np.abs(value["proprio"].numpy().reshape(7,4)-prop).max()))
        indices=value["meta"]["observed_raw_indices"]
        if indices!=list(range(0,31,5)) or value["raw_actions"].shape!=(30,2):raise ValueError("Wrong current-to-next5action alignment")
        key=identity_key(states[0]);groups.setdefault(key,len(groups));initial_ids.append(groups[key])
        records.append({"source_id":source,"split":value["meta"]["split"],"initial_group":groups[key],
            "initial_state":states[0].tolist(),"video_sha256":value["meta"]["video_sha256"],
            "state_sequence_sha256":identity_key(states),"action_sequence_sha256":identity_key(value["raw_actions"].numpy()),
            "observed_indices":indices})
    report={"complete":True,"scope":"100fit+25alreadyseenvalidation only; noconfirmation/heldstudy",
        "alignment_max_abs":max_errors,"source_records":records,"splits":{},
        "native_correspondence":{"state":"agentXY,blockXY,blocktheta,agentvelocityXY","proprio":"agentXY+agentvelocityXY, notblockposition",
            "actions":"native rel_actions/100; stored normalized chunks verified exactly during spatialcapture",
            "target":"actual state[t+5] for5rawactions[t:t+5]","video":"native episode_{source_id:03d}.mp4 andsame raw frame indices"},
        "confirmation_opened":False,"model_metadata_not_appended_to_features":True}
    for split in ("fit","validation"):
        selected=[r for r in records if r["split"]==split]
        selected_values=[v for _,v,_ in rows if v["meta"]["split"]==split]
        current=np.stack([v["states"][0].numpy().astype(float) for v in selected_values])
        following=np.concatenate([v["q_next"].numpy()[:,:4].astype(float) for v in selected_values])
        report["splits"][split]={"source_files":len(selected),"unique_initial_conditions":len({r["initial_group"] for r in selected}),
            "initial_group_counts":{str(g):sum(r["initial_group"]==g for r in selected) for g in sorted({r["initial_group"] for r in selected})},
            "unique_videos":len({r["video_sha256"] for r in selected}),"unique_action_sequences":len({r["action_sequence_sha256"] for r in selected}),
            "unique_state_sequences":len({r["state_sequence_sha256"] for r in selected}),
            "initial_state_min":current.min(0).tolist(),"initial_state_max":current.max(0).tolist(),
            "next_xy_mean":following.mean(0).tolist(),"next_xy_std":following.std(0).tolist(),
            "next_xy_min":following.min(0).tolist(),"next_xy_max":following.max(0).tolist()}
    fitgroups={r["initial_group"] for r in records if r["split"]=="fit"}
    report["seen_validation_initial_sources"]=sum(r["initial_group"] in fitgroups for r in records if r["split"]=="validation")
    report["unseen_validation_initial_sources"]=25-report["seen_validation_initial_sources"]
    report["within_fit_source_mod5_folds_share_initial_conditions"]=all(
        {r["initial_group"] for r in records if r["split"]=="fit" and r["source_id"]%5==fold} &
        {r["initial_group"] for r in records if r["split"]=="fit" and r["source_id"]%5!=fold} for fold in range(5))
    report["conclusion"]="Source-file grouping is not independent initial-condition grouping; pooled-vs-spatial effects are confounded with single-start fit coverage. No relabeling or validation-based reselection performed."
    report["correction_needed"]="Predeclare diverse distinct TRAIN initial-condition groups and refit/simple held-group validation; requires separate frozen sampling authorization, not swapping currentseenvalidation into fit."
    return report


def compress_fit(x):
    scale=StandardScaler().fit(x)
    pca=PCA(n_components=64,svd_solver="randomized",random_state=17).fit(scale.transform(x))
    return scale,pca,pca.transform(scale.transform(x))


def compress_predict(state,x):
    scale,pca=state
    return pca.transform(scale.transform(x))


def summarize(y,prediction,current,groups):
    err=np.square(y[:,:4]-prediction[:,:4]).mean(1)
    delta=np.arctan2(prediction[:,4],prediction[:,5])-np.arctan2(y[:,4],y[:,5])
    angular=np.arctan2(np.sin(delta),np.cos(delta))
    return {"world_endpoint_rmse_pixels":float(np.sqrt(err.mean())),
        "agent_rmse_pixels":float(np.sqrt(np.square(y[:,:2]-prediction[:,:2]).mean())),
        "block_rmse_pixels":float(np.sqrt(np.square(y[:,2:4]-prediction[:,2:4]).mean())),
        "angle_wrapped_rmse_radians":float(np.sqrt(np.square(angular).mean())),
        "physical_persistence_rmse_pixels":float(np.sqrt(np.square(y[:,:4]-current[:,:4]).mean())),
        "unique_source_files":len(np.unique(groups))}


def probe(rows,out):
    start=time.monotonic()
    y=np.concatenate([v["q_next"].numpy() for _,v,_ in rows]).astype(float)
    current=np.concatenate([v["q_current"].numpy() for _,v,_ in rows]).astype(float)
    groups=np.concatenate([np.full(6,i) for i,_,_ in rows]);train=groups<100;validation=~train
    x={key:np.concatenate([v[key].numpy() for _,v,_ in rows]).astype(float) for key in VIEWS}
    protocol={"features":VIEWS,"compression_rank":64,"compression":"training-fold-only feature standardization thenPCA64, no labels",
        "models":["linear","rbf"],"alphas_each":ALPHAS,"folds":"sourceIDmodulo5, preservedforcomparability butnotindependentinitialconditions",
        "seen_validation_sources":[100,124],"confirmation_opened":False,"selection":"fit-only CV worldendpointMSE; no validationtuning",
        "native_coordinates_claimed":False,"comparison_limitation":"Fit sources share one exactinitialcondition; this alone cannot isolate pooling inadequacy"}
    write_json(out/"probe_protocol.json",protocol)
    bundles={};chosen={};cv={};all_cv_predictions={}
    for view in VIEWS:
        predictions={f"{family}:{alpha}":np.empty((600,6)) for family in ("linear","rbf") for alpha in ALPHAS}
        for fold in range(5):
            a=train&(groups%5!=fold);b=train&(groups%5==fold)
            scale,pca,z=compress_fit(x[view][a]);bundle=fit_models(z,y[a])
            pred=predict_models(bundle,compress_predict((scale,pca),x[view][b]))
            for key,value in pred.items():predictions[key][np.flatnonzero(b)]=value
        chosen[view]={family:min((k for k in predictions if k.startswith(family+":")),key=lambda k:np.square(y[train,:4]-predictions[k][:,:4]).mean()) for family in ("linear","rbf")}
        cv[view]={family:summarize(y[train],predictions[key],current[train],groups[train]) for family,key in chosen[view].items()}
        scale,pca,z=compress_fit(x[view][train]);bundles[view]={"compression":(scale,pca),"model":fit_models(z,y[train])}
        all_cv_predictions[view]=predictions
        print(json.dumps({"event":"spatial_fit_view_complete","view":view}),flush=True)
    import joblib
    joblib.dump(bundles,out/"readouts.joblib")
    write_json(out/"FROZEN.json",{"chosen_models":chosen,"readouts_sha256":sha256(out/"readouts.joblib"),"fit_sources":100,"fit_initial_condition_groups":1,
        "validation_used_for_fitting":False,"confirmation_opened":False,"discovery":cv})
    results={};vp={}
    for view in VIEWS:
        bundle=bundles[view];pred=predict_models(bundle["model"],compress_predict(bundle["compression"],x[view][validation]));vp[view]=pred
        results[view]={family:{"model":key,**summarize(y[validation],pred[key],current[validation],groups[validation])} for family,key in chosen[view].items()}
    comparisons={}
    for site in ("p3","predicted"):
        comparisons[site]={}
        for family in ("linear","rbf"):
            global_pred=vp[site+"_global"][chosen[site+"_global"][family]]
            spatial_pred=vp[site+"_spatial"][chosen[site+"_spatial"][family]]
            diff=np.square(y[validation,:4]-spatial_pred[:,:4]).mean(1)-np.square(y[validation,:4]-global_pred[:,:4]).mean(1)
            comparisons[site][family]={"spatial_minus_global_mse":per_episode_ci(diff,groups[validation]),
                "CI_limit":"source-file bootstrap is not independent initial-condition bootstrap; only2validationinitialgroups"}
    means=np.broadcast_to(y[train].mean(0),y[validation].shape)
    report={"complete":True,"protocol":protocol,"discovery":cv,"validation":results,"comparisons":comparisons,
        "fit_mean_control":summarize(y[validation],means,current[validation],groups[validation]),"seconds":time.monotonic()-start,
        "confirmation_opened":False,"deployed_operator":False,
        "simplest_next_operator":"Do not deploycurrentpoorprobe. Refit simplelinearphysicalreadout on predeclared distinctTRAINinitialconditiongroups; retainleastcomplexpooling that beatsphysicalpersistence on disjointinitialgroups beforeanydevelopmentsteeringpilot."}
    write_json(out/"spatial_probe_results.json",report)
    np.savez_compressed(out/"validation_predictions.npz",**{view+"_"+family:vp[view][key] for view in VIEWS for family,key in chosen[view].items()},
        q_next=y[validation],q_current=current[validation],source_id=groups[validation])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--capture-dirs",nargs="+",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--diagnose-only",action="store_true")
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=load_permitted(args.capture_dirs,spatial=not args.diagnose_only)
    report=diagnostic(rows);write_json(args.output_dir/"alignment_coverage.json",report)
    if not args.diagnose_only:probe(rows,args.output_dir)
    write_json(args.output_dir/"DONE.json",{"complete":True,"confirmation_opened":False,"script_sha256":sha256(__file__),
        "outputs":[{"path":q.name,"sha256":sha256(q)} for q in sorted(args.output_dir.iterdir()) if q.is_file()]})
    print(json.dumps({"event":"spatial_analysis_complete","unique_fit_initial_states":report["splits"]["fit"]["unique_initial_conditions"]}),flush=True)


if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
