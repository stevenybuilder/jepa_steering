#!/usr/bin/env python3
"""Bounded frozen20 independent TRAIN-start correction; no CEM or held panel."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from capture_pusht_calibration import CHECKPOINT_SHA,INDICES,capture_one,parameter_sha
from capture_pusht_spatial import run_one
from complete_cached_geometry import sha256,write_json

SOURCES=tuple(range(202,2122,101))

def fingerprint(state):
    return hashlib.sha256(np.asarray(state,dtype=np.float32).tobytes()).hexdigest()

def rows():
    return [{"source_id":i,"split":"fit" if j<16 else "development_validation",
             "clip_offset":0,"observed_raw_indices":list(INDICES),"initial_group_ordinal":j}
            for j,i in enumerate(SOURCES)]

def prepare(args):
    import pickle
    import torch
    torch.set_num_threads(1)
    if args.dataset.name!="train":raise ValueError("Official TRAIN only")
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={"rows":rows(),"checkpoint_sha256":CHECKPOINT_SHA,"source_selection":"First20 distinct TRAIN initial-state groups after previously seen groups; initial-state metadata only, no outcomes",
        "sealed_original_source_ids":[125,149],"fit_groups":16,"development_validation_groups":4,
        "fold_rule":"4folds by actual initial fingerprint (ordinal modulo4), no shared initialstate",
        "feature_rank":64,"alphas_each":[.01,1.],"families":["linear","rbf"],"frames":list(INDICES),
        "no_confirmation_or_scripted_panel":True}
    write_json(args.output/"manifest.json",manifest) # before source tensor loading
    start=time.monotonic()
    with (args.dataset/"seq_lengths.pkl").open("rb") as f:lengths=pickle.load(f)
    if any(lengths[i]<31 for i in SOURCES):raise ValueError("Fixed clip unavailable; no replacement")
    raw=torch.load(args.dataset/"states.pth",mmap=True,map_location="cpu",weights_only=True)
    velocity=torch.load(args.dataset/"velocities.pth",mmap=True,map_location="cpu",weights_only=True)
    actions=torch.load(args.dataset/"rel_actions.pth",mmap=True,map_location="cpu",weights_only=True)
    known={fingerprint(torch.cat([raw[i,0],velocity[i,0]]).float().numpy()) for i in (0,101)}
    outputs=[];seen=set();(args.output/"videos").mkdir()
    for row in rows():
        i=row["source_id"];states=torch.cat([raw[i,:31].float(),velocity[i,:31].float()],-1).contiguous()
        key=fingerprint(states[0].numpy())
        if key in known or key in seen:raise ValueError("Fixed selection is not independent from known starts or itself")
        seen.add(key)
        source=args.dataset/"obses"/f"episode_{i:03d}.mp4";video=args.output/"videos"/source.name
        shutil.copyfile(source,video)
        path=args.output/f"input-{i:03d}.pt"
        torch.save({"states":states,"raw_actions":actions[i,:30].float().contiguous()/100.},path)
        outputs.append({**row,"initial_state_sha256":key,"path":path.name,"sha256":sha256(path),"video":str(video.relative_to(args.output)),"video_sha256":sha256(video)})
    write_json(args.output/"DONE.json",{"complete":True,"rows":outputs,"unique_initial_groups":len(seen),
        "manifest_sha256":sha256(args.output/"manifest.json"),"script_sha256":sha256(__file__),
        "source_root":str(args.dataset),"seconds":time.monotonic()-start,"confirmation_opened":False,"model_execution":False})
    print(json.dumps({"event":"diverse_inputs_complete","unique_initial_groups":len(seen)}),flush=True)

def capture(args):
    import torch
    import imageio.v2 as imageio
    from model_loader import load_headless
    receipt=json.loads((args.inputs/"DONE.json").read_text())
    if not receipt["complete"] or [r["source_id"] for r in receipt["rows"]]!=list(SOURCES):raise ValueError("Wrong frozen sources")
    if sha256(args.checkpoint)!=CHECKPOINT_SHA:raise ValueError("Wrong checkpoint")
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.inputs/"manifest.json",args.output/"manifest.json")
    torch.set_num_threads(2);torch.manual_seed(90505)
    start=time.monotonic();wm,_,provenance=load_headless(args.repo,model_name="jepa_wm_pusht",checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);outputs=[]
    for row in receipt["rows"]:
        path=args.inputs/row["path"];video=args.inputs/row["video"]
        if sha256(path)!=row["sha256"] or sha256(video)!=row["video_sha256"]:raise ValueError("Input hash mismatch before loading")
        bank=torch.load(path,map_location="cpu",weights_only=True)
        if fingerprint(bank["states"][0].numpy())!=row["initial_state_sha256"]:raise ValueError("Initial-state group mismatch")
        with imageio.get_reader(video,format="ffmpeg") as reader:frames=np.stack([reader.get_data(i) for i in INDICES])
        global_bank,_=capture_one(wm,frames,bank["states"][list(INDICES)],bank["raw_actions"],wm.model.predictor.predictor_blocks[3])
        result,difference=run_one(wm,frames,global_bank)
        if row["source_id"]==SOURCES[0]:
            repeat,_=run_one(wm,frames,global_bank)
            if not all(torch.equal(result[k],repeat[k]) for k in result):raise ValueError("Exact repeat failed")
        result["meta"]={**row,"global_repeat_difference":difference,"checkpoint_sha256":CHECKPOINT_SHA}
        out=args.output/f"source-{row['source_id']:03d}.pt";torch.save(result,out)
        outputs.append({**row,"path":out.name,"sha256":sha256(out),"bytes":out.stat().st_size})
        print(json.dumps({"event":"diverse_capture_complete","source_id":row["source_id"]}),flush=True)
    if parameter_sha(wm)!=before:raise ValueError("Weights changed")
    write_json(args.output/"DONE.json",{"complete":True,"outputs":outputs,"unique_initial_groups":20,"seconds":time.monotonic()-start,
        "input_done_sha256":sha256(args.inputs/"DONE.json"),"script_sha256":sha256(__file__),"parameter_sha256":before,
        "checkpoint_sha256":CHECKPOINT_SHA,"provenance":provenance,"no_cem":True,"confirmation_opened":False})

def fit(args):
    import torch
    import joblib
    from analyze_pusht_spatial import VIEWS,ALPHAS,compress_fit,compress_predict,summarize
    from screen_coordinate_time import fit_models,predict_models,per_episode_ci
    receipt=json.loads((args.inputs/"DONE.json").read_text())
    if not receipt["complete"] or [r["source_id"] for r in receipt["outputs"]]!=list(SOURCES):raise ValueError("Wrong fixed captures")
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();values=[]
    for row in receipt["outputs"]:
        path=args.inputs/row["path"]
        if sha256(path)!=row["sha256"]:raise ValueError("Capture hash mismatch")
        values.append(torch.load(path,map_location="cpu",weights_only=True))
    keys=[fingerprint(v["states"][0].numpy()) for v in values]
    if len(set(keys))!=20:raise ValueError("Initial-condition groups not disjoint")
    groups=np.repeat(np.arange(20),6);train=groups<16;val=~train
    y=np.concatenate([v["q_next"].numpy() for v in values]).astype(float)
    current=np.concatenate([v["q_current"].numpy() for v in values]).astype(float)
    views=(*VIEWS,"encoded_current_global","encoded_current_spatial","encoded_next_global","encoded_next_spatial")
    matrices={}
    for view in views:
        if view in VIEWS:
            arrays=[v[view].numpy() for v in values]
        elif "_current" in view:
            arrays=[v[view.replace("_current","")][:-1].numpy() for v in values]
        else:
            arrays=[v[view.replace("_next","")][1:].numpy() for v in values]
        matrices[view]=np.concatenate(arrays).astype(float)
    # Encoded-current probes predict next physical state; encoded-next is a diagnostic observation decoder, not a future predictor.
    protocol={"fit_groups":16,"development_validation_groups":4,"folds":4,"fold_key":"actualinitialstate fingerprint; disjoint groups, ordinalmod4",
        "rank":64,"alphas_each":ALPHAS,"features":views,"compression":"training-fold-only standardization and PCA64",
        "selection":"fit-only CV world-endpoint MSE","validation_source_ids":list(SOURCES[16:]),"confirmation_opened":False,"held_scripted_opened":False}
    write_json(args.output/"protocol.json",protocol)
    models={};chosen={};cv={}
    for view,x in matrices.items():
        predictions={f"{family}:{alpha}":np.empty((96,6)) for family in ("linear","rbf") for alpha in ALPHAS}
        for fold in range(4):
            a=train&(groups%4!=fold);b=train&(groups%4==fold)
            scale,pca,z=compress_fit(x[a]);bundle=fit_models(z,y[a]);pred=predict_models(bundle,compress_predict((scale,pca),x[b]))
            for key,value in pred.items():predictions[key][np.flatnonzero(b)]=value
        chosen[view]={family:min((k for k in predictions if k.startswith(family+":")),key=lambda k:np.square(y[train,:4]-predictions[k][:,:4]).mean()) for family in ("linear","rbf")}
        cv[view]={family:summarize(y[train],predictions[key],current[train],groups[train]) for family,key in chosen[view].items()}
        scale,pca,z=compress_fit(x[train]);models[view]={"compression":(scale,pca),"models":fit_models(z,y[train])}
    joblib.dump(models,args.output/"readouts.joblib")
    write_json(args.output/"FROZEN.json",{"chosen":chosen,"fit_initial_groups":keys[:16],"readout_sha256":sha256(args.output/"readouts.joblib"),"validation_used_for_fit":False,"cv":cv})
    results={};predictions={}
    for view,x in matrices.items():
        bundle=models[view];pred=predict_models(bundle["models"],compress_predict(bundle["compression"],x[val]))
        results[view]={}
        for family,key in chosen[view].items():
            prediction=pred[key];predictions[view+"_"+family]=prediction
            delta=np.square(y[val,:4]-prediction[:,:4]).mean(1)-np.square(y[val,:4]-current[val,:4]).mean(1)
            results[view][family]={"model":key,**summarize(y[val],prediction,current[val],groups[val]),"minus_persistence_mse_group_ci":per_episode_ci(delta,groups[val])}
    write_json(args.output/"results.json",{"complete":True,"protocol":protocol,"cv":cv,"development_validation":results,
        "mean_control":summarize(y[val],np.broadcast_to(y[train].mean(0),y[val].shape),current[val],groups[val]),
        "seconds":time.monotonic()-start,"limitation":"Only4 disjoint development validation initialstates/24transitions; no held-study efficacy or deployed operator claim",
        "encoded_next_semantics":"Observed next-frame decoder upper-bound diagnostic; not forecast or causal prediction"})
    np.savez_compressed(args.output/"predictions.npz",**predictions,q_next=y[val],q_current=current[val],initial_group=groups[val])
    write_json(args.output/"DONE.json",{"complete":True,"outputs":[{"path":p.name,"sha256":sha256(p)} for p in sorted(args.output.iterdir()) if p.is_file()],"script_sha256":sha256(__file__)})
    print(json.dumps({"event":"diverse_fit_complete","fit_initial_groups":16,"validation_initial_groups":4}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["prepare","capture","fit"])
    for k in ("output","dataset","inputs","repo","checkpoint"):p.add_argument("--"+k,type=Path,required=k=="output")
    args=p.parse_args();globals()[args.mode](args)

if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
