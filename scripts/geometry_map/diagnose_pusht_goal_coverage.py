#!/usr/bin/env python3
"""Goal-matched T-polygon coverage of cached endpoints; no simulator reset/step."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,MultiPolygon

VERTICES = (np.array([[-60,30],[60,30],[60,0],[-60,0]],dtype=float),
            np.array([[-15,30],[-15,120],[15,120],[15,30]],dtype=float))


def polygon(pose):
    x,y,angle=np.asarray(pose,dtype=float)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return MultiPolygon([Polygon(v@rotation.T+[x,y]) for v in VERTICES])


def coverage(pose,goal):
    actual,target=polygon(pose),polygon(goal)
    return float(actual.intersection(target).area/target.area)


def native_helper(source):
    """Compile exactly the official pure polygon helper, not an environment."""
    import pymunk
    import shapely.geometry as sg
    tree=ast.parse(Path(source).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pymunk_to_shapely')
    namespace={'pymunk':pymunk,'sg':sg}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    def native_polygon(pose):
        body=pymunk.Body(1,1);shapes=[pymunk.Poly(body,v.tolist()) for v in VERTICES]
        body.angle=float(pose[2]);body.position=list(map(float,pose[:2]))
        return namespace['pymunk_to_shapely'](body,shapes)
    return native_polygon


def analyze(summary,native_polygon):
    from run_expand_pusht_candidates import spearman
    goal=np.asarray(summary['goal_state'])[2:5];rows=summary['rows'];states=np.asarray([r['final_state'] for r in rows])
    poses=states[:,2:5];default=np.array([256.,256.,np.pi/4])
    aligned=np.array([coverage(p,goal) for p in poses]);default_coverage=np.array([coverage(p,default) for p in poses])
    manual=np.array([native_polygon(p).intersection(native_polygon(goal)).area/native_polygon(goal).area for p in poses])
    helper_error=float(np.max(np.abs(manual-aligned)))
    if helper_error>1e-12: raise ValueError('Official pure polygon helper parity failed')
    default_reward=np.minimum(default_coverage/.95,1.)
    replay_reward=np.array([r['native_reward_final'] for r in rows])
    reward_error=float(np.max(np.abs(default_reward-replay_reward)))
    # Saved observed positions/angles and reward are float32; no float64 physics is reconstructed.
    if reward_error>3e-6: raise ValueError('Default reward parity exceeds float32 endpoint reconstruction tolerance')
    predicted=np.array([r['native_full_objective_cost'] for r in rows])
    encoded=np.array([r['actual_encoded_full_objective_cost'] for r in rows])
    xy=np.linalg.norm(states[:,:4]-np.asarray(summary['goal_state'])[:4],axis=1)
    picked=int(np.argmin(predicted));oracle=int(np.argmax(aligned));xyoracle=int(np.argmin(xy))
    return dict(episode=summary['episode'],goal_block_pose=goal.tolist(),native_default_goal_pose=default.tolist(),
        goal_aligned_coverage_by_candidate=aligned.tolist(),default_raw_coverage_by_candidate=default_coverage.tolist(),
        negative_predicted_cost_vs_goal_coverage_spearman=spearman(-predicted,aligned),
        negative_actual_encoded_cost_vs_goal_coverage_spearman=spearman(-encoded,aligned),
        negative_xy_distance_vs_goal_coverage_spearman=spearman(-xy,aligned),
        native_selected_index=picked,native_selected_goal_coverage=float(aligned[picked]),
        oracle_index=oracle,oracle_goal_coverage=float(aligned[oracle]),
        oracle_headroom=float(aligned[oracle]-aligned[picked]),
        original9_oracle_headroom=float(aligned[:9].max()-aligned[picked]),
        xy_oracle_index=xyoracle,xy_oracle_goal_coverage=float(aligned[xyoracle]),
        xy_oracle_coverage_delta_vs_native=float(aligned[xyoracle]-aligned[picked]),
        xy_oracle_xy_distance=float(xy[xyoracle]),native_selected_xy_distance=float(xy[picked]),
        official_helper_maxabs=helper_error,default_native_reward_reconstruction_maxabs=reward_error,
        goal_aligned_coverage_ge_095_count=int((aligned>=.95).sum()),
        label_scope='New descriptive goal-matched polygon coverage; prior4DXY primary unchanged; not a refit or episode success claim')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--native-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert len(a.inputs)==4;a.output.mkdir(parents=True,exist_ok=False)
    helper=native_helper(a.native_source)
    rows=[];sources=[]
    for path in a.inputs:
        s=json.loads(path.read_text());rows.append(analyze(s,helper));sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows.sort(key=lambda r:r['episode']);assert [r['episode'] for r in rows]==list(range(4))
    result=dict(complete=True,rows=rows,sources=sources,simulator_calls=0,model_calls=0,held_access=False,
        native_source_sha256=hashlib.sha256(a.native_source.read_bytes()).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target_contract='Original expert goal_state blockXYangle, exactly paired with rawgoalimage; native fixedpaintedgoal kept separately')
    out=a.output/'summary.json';out.write_text(json.dumps(result,indent=2))
    (a.output/'DONE.json').write_text(json.dumps(dict(complete=True,outputs=[dict(path=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size)]),indent=2))
    print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]}))


if __name__=='__main__': main()
