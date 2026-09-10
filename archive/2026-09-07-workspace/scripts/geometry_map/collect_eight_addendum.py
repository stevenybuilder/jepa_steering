#!/usr/bin/env python3
"""Fixed eight-full-baseline addendum; unchanged native collectors, sealed outcomes."""
import json
from pathlib import Path
import sys


def reach_shard(manifest,worker):
    if manifest["allowed_new_episode_ids"]!=list(range(62,66)) or manifest["addendum"]!="eight-full-baselines-v1":raise RuntimeError("WrongReachaddendum")
    shards=manifest["shards"]
    if sorted(e for r in shards for e in r["episode_ids"])!=list(range(62,66)):raise RuntimeError("Reachshardoverlap/missing")
    row=next(r for r in shards if r["instance_id"]==worker)
    if len(row["episode_ids"])!=2:raise RuntimeError("ExpectedtwoReachperworker")
    return row


def push_inputs(manifest,receipt,episodes):
    if manifest["panel"]!="pusht_scripted_push_goal_reserve_v1" or receipt.get("panel")!=manifest["panel"] or not receipt.get("complete"):
        raise RuntimeError("Unverified reservePushpanel")
    if sorted(r["episode"] for r in receipt["outputs"])!=list(range(50,100)):raise RuntimeError("All50reserveinputs mustremainretained")
    if episodes not in ([50,51],[52,53]):raise RuntimeError("OnlythefourexplicitlyauthorizedPushbaselines")
    return [next(r for r in receipt["outputs"] if r["episode"]==e) for e in episodes]


if __name__=="__main__":
    mode=sys.argv.pop(1)
    if mode=="reach":
        import collect_steering_expansion as native
        native.validate_request=reach_shard
    elif mode=="push":
        import collect_pusht_scripted_baselines as native
        native.validate_inputs=push_inputs
    else:raise RuntimeError("Unknownaddendumtask")
    native.main()
