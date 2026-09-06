#!/usr/bin/env python3
"""Checked adapter to unchanged native Push baseline collector, outcomes sealed."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
from collect_steering_expansion import ProgressOnly, file_hash, write_exclusive


def validate_inputs(manifest, receipt, episodes):
    if manifest["panel"] != "pusht_scripted_push_goal_v2" or not receipt.get("complete") or receipt.get("panel") != manifest["panel"]:
        raise RuntimeError("Only completed immutable scripted-goalv2 inputs are authorized")
    if sorted(r["episode"] for r in receipt["outputs"]) != list(range(50)):
        raise RuntimeError("All50inputs must be retained without selection")
    if len(set(episodes)) != len(episodes) or not episodes or any(e not in range(50) for e in episodes):
        raise RuntimeError("Invalid or duplicate fixed episode IDs")
    return [next(r for r in receipt["outputs"] if r["episode"]==e) for e in episodes]


class Progress(ProgressOnly):
    allowed=ProgressOnly.allowed|{"seconds","replay_state_error","split"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","inputs","repo","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--episodes",nargs="+",type=int,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    receipt=json.loads((args.inputs/"DONE.json").read_text())
    selected=validate_inputs(manifest,receipt,args.episodes)
    provenance=json.loads((args.inputs/"PROVENANCE.json").read_text())
    if file_hash(args.inputs/"PROVENANCE.json")!=receipt["provenance_sha256"] or file_hash(args.manifest)!=provenance["manifest_sha256"]:
        raise RuntimeError("Input provenance/manifest mismatch")
    for row in selected:
        if file_hash(args.inputs/row["path"])!=row["sha256"]:
            raise RuntimeError("Input SHA mismatch before native tensorload")
    if file_hash(args.repo/manifest["native_config"])!=manifest["native_config_sha256"]:
        raise RuntimeError("Nativeconfig mismatch")
    if file_hash(args.checkpoint)!=manifest["checkpoint"]["sha256"]:
        raise RuntimeError("Explicit local checkpoint SHA differs from frozen panel")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    import model_loader
    original=model_loader.load_headless
    def checked_loader(*a,**kw):
        model,preprocessor,source=original(*a,**kw)
        digest=file_hash(Path(source["checkpoint"]))
        if digest!=manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Native checkpoint differs from frozen panel")
        source["checkpoint_sha256"]=digest
        print(json.dumps({"event":"checkpoint_verified","checkpoint_sha256":digest}),flush=True)
        return model,preprocessor,source
    model_loader.load_headless=checked_loader
    import collect_pusht_bank as native
    write_exclusive(args.output_dir/"LAUNCH.json",{"panel":manifest["panel"],"episode_ids":args.episodes,
                    "manifest_sha256":file_hash(args.manifest),"input_done_sha256":file_hash(args.inputs/"DONE.json"),
                    "input_rows":selected,"adapter_sha256":file_hash(Path(__file__)),"native_collector_sha256":file_hash(Path(native.__file__)),
                    "stage":"native unsteered baseline only; no steering or successscreening","outcomes_sealed":True})
    started=time.monotonic()
    with contextlib.redirect_stdout(Progress(sys.stdout)):
        native.collect(args)
    done=json.loads((args.output_dir/"DONE.json").read_text())
    if not done.get("complete") or done["episodes"]!=args.episodes:
        raise RuntimeError("Native baseline completion mismatch")
    for row in done["outputs"]:
        if file_hash(args.output_dir/row["path"])!=row["sha256"]:
            raise RuntimeError("Baseline output SHA mismatch")
    write_exclusive(args.output_dir/"SEALED_DONE.json",{"complete":True,"panel":manifest["panel"],"episodes":args.episodes,
                    "outputs":done["outputs"],"native_done_sha256":file_hash(args.output_dir/"DONE.json"),
                    "seconds_including_model_init":time.monotonic()-started,"no_successscreening":True,"native_replay_guard_passed":True})
    print(json.dumps({"event":"scripted_panel_baseline_complete","episode_ids":args.episodes}),flush=True)


if __name__=="__main__":main()
