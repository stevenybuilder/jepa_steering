#!/usr/bin/env python3
"""Fixed global+quadrant summaries of the same125 permitted Push TRAIN clips."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from capture_pusht_calibration import CHECKPOINT_SHA, INDICES, first_tensor, parameter_sha
from complete_cached_geometry import sha256, write_json


def permitted(row):
    return (row["split"]=="fit" and 0<=row["source_id"]<100) or (row["split"]=="validation" and 100<=row["source_id"]<125)


def summaries(features):
    if features.ndim!=3 or features.shape[1]!=256:raise ValueError("Expected [time,256,channels] patch layout")
    n,_,d=features.shape
    grid=features.reshape(n,16,16,d).float()
    global_mean=grid.mean((1,2))
    regions=[global_mean]+[grid[:,y:y+8,x:x+8].mean((1,2)) for y,x in ((0,0),(0,8),(8,0),(8,8))]
    return global_mean.cpu(),torch.cat(regions,-1).cpu()


def run_one(wm, frames, bank):
    states=bank["states"].float()
    actions=bank["raw_actions"].float()
    proprio=torch.cat([states[:,:2],states[:,5:7]],-1)[None]
    if not torch.equal(proprio,bank["proprio"]):raise RuntimeError("Stored native4D proprio does not match agentXY+velocity")
    normalized=wm.preprocessor.normalize_actions(actions.reshape(6,5,2)).reshape(6,1,10).to(wm.device,dtype=torch.float32)
    if not torch.equal(normalized[:,0].cpu(),bank["normalized_action_chunks"]):raise RuntimeError("Stored normalized actions differ")
    captured=[]
    handle=wm.model.predictor.predictor_blocks[3].register_forward_hook(lambda _m,_i,out:captured.append(first_tensor(out).detach().clone()))
    try:
        with torch.inference_mode():
            encoded=wm.encode({"visual":torch.from_numpy(frames).permute(0,3,1,2)[None],"proprio":proprio})
            predicted,_,_=wm.model.forward_pred(encoded["visual"][0,:6,None],wm.model.encode_act(normalized),encoded["proprio"][0,:6,None])
    finally:handle.remove()
    if len(captured)!=1 or captured[0].shape!=(6,256,400):raise RuntimeError("Unexpected P3 layout/calls")
    result={}
    for name,value in {"p3":captured[0],"predicted":predicted.reshape(6,256,384),"encoded":encoded["visual"].reshape(7,256,384)}.items():
        result[name+"_global"],result[name+"_spatial"]=summaries(value)
    differences={name:float((result[name+"_global"]-bank[key]).abs().max()) for name,key in (("p3","predictor_p3_pooled"),("predicted","predicted_visual_pooled"),("encoded","encoded_visual_pooled"))}
    result.update({key:bank[key] for key in ("states","q_current","q_next","proprio","raw_actions","normalized_action_chunks")})
    if not all(torch.isfinite(x).all() for x in result.values()):raise RuntimeError("Nonfinite output")
    return result,differences


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("capture-dir","video-inputs","repo","checkpoint","output-dir"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--source-start",type=int,required=True);p.add_argument("--source-stop",type=int,required=True)
    args=p.parse_args()
    if not 0<=args.source_start<args.source_stop<=125:raise ValueError("Only first125 already-permitted sources")
    receipt=json.loads((args.capture_dir/"DONE.json").read_text())
    rows=[r for r in receipt["outputs"] if permitted(r) and args.source_start<=r["source_id"]<args.source_stop]
    if not receipt["complete"] or [r["source_id"] for r in rows]!=list(range(args.source_start,args.source_stop)):raise ValueError("Incomplete fixed shard")
    video_receipt=json.loads((args.video_inputs/"DONE.json").read_text())
    videos={r["source_id"]:r for r in video_receipt["rows"] if permitted(r)}
    if sha256(args.checkpoint)!=CHECKPOINT_SHA:raise RuntimeError("Wrong pinned checkpoint")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.output_dir/"protocol.json",{"source_rows":rows,"source_capture_done_sha256":sha256(args.capture_dir/"DONE.json"),
        "summary":"global plus fixed TL,TR,BL,BR2x2 quadrant means of actual16x16 patch grid",
        "source_indices":[args.source_start,args.source_stop],"confirmation_access":False,"held_panel_access":False,
        "script_sha256":sha256(__file__),"checkpoint_sha256":CHECKPOINT_SHA,"no_cem_or_simulator":True})
    import imageio.v2 as imageio
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505)
    start=time.monotonic()
    wm,_,provenance=load_headless(args.repo,model_name="jepa_wm_pusht",checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);outputs=[]
    for row in rows:
        source=row["source_id"];path=args.capture_dir/row["path"];video=videos[source]
        if sha256(path)!=row["sha256"] or sha256(args.video_inputs/video["video"])!=video["video_sha256"]:raise RuntimeError("Source hash mismatch before tensor/video load")
        bank=torch.load(path,map_location="cpu",weights_only=True)
        if not permitted(bank["meta"]) or bank["meta"]["observed_raw_indices"]!=list(INDICES):raise ValueError("Wrong permitted sample/alignment")
        with imageio.get_reader(args.video_inputs/video["video"],format="ffmpeg") as reader:
            frames=np.stack([reader.get_data(i) for i in INDICES])
        result,differences=run_one(wm,frames,bank)
        if source==args.source_start:
            repeat,_=run_one(wm,frames,bank)
            if not all(torch.equal(result[k],repeat[k]) for k in result):raise RuntimeError("Exact same-shard repeat failed")
        result["meta"]={**bank["meta"],"source_capture_sha256":row["sha256"],"new_global_vs_existing_max_abs":differences,
            "pooling_order":["global","top_left","top_right","bottom_left","bottom_right"]}
        out=args.output_dir/f"source-{source:03d}.pt";torch.save(result,out)
        outputs.append({"source_id":source,"split":row["split"],"path":out.name,"sha256":sha256(out),"bytes":out.stat().st_size,"old_global_max_abs":differences})
        print(json.dumps({"event":"spatial_source_complete","source_id":source,"old_global_max_abs":differences}),flush=True)
    if parameter_sha(wm)!=before:raise RuntimeError("Frozen weights changed")
    write_json(args.output_dir/"DONE.json",{"complete":True,"outputs":outputs,"seconds":time.monotonic()-start,
        "checkpoint_sha256":CHECKPOINT_SHA,"parameter_sha256":before,"provenance":provenance,
        "confirmation_opened":False,"held_scripted_outcomes_accessed":False})


if __name__=="__main__":main()
