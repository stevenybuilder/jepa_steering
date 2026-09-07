"""Join all registered149 rows; retain path failures, never select by outcome."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path('artifacts/geometry_map/unsteered_baseline_metrics_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
manifest=json.loads((ROOT/'manifest.json').read_text());expected={(r['panel'],r['episode']):r for r in manifest['source']}
rows={};failures=[];receipts=[];cpu=0
for p in sorted(ROOT.glob('worker-*/*/rows.json')):
    done=json.loads(p.with_name('DONE.json').read_text())
    assert sha(p)==done['outputs'][0]['sha256']
    d=json.loads(p.read_text());cpu+=d['analysis_cpu_seconds']
    receipts.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'done_sha256':sha(p.with_name('DONE.json')),'analysis_cpu_seconds':d['analysis_cpu_seconds']})
    for r in d['rows']:
        key=(r['panel'],r['episode']);assert key in expected and r['sha256']==expected[key]['sha256']
        if r['status']!='ok':failures.append(r);continue
        assert r['artifact_sha_verified_before_deserialization'] and key not in rows
        rows[key]=r
assert len(rows)==149 and set(rows)==set(expected)
for r in rows.values():
    if r['harness']=='collect_reach_development_expansion_full99':
        # Do not call a4D proprio-only fingerprint a full39D initial-state group.
        r['initial_proprio_fingerprint']=r.pop('initial_state');r['initial_proprio_values']=r.pop('initial_state_values')
        r['initial_state']=None;r['initial_state_values']=None;r['initial_goal_pair']=None
        r['initial_state_group_status']='Full initial physics exists in source, but initial39D observation was not cached. Proprio-only groups are not counted as independent state groups.'
    else:r['initial_state_group_status']='Exact saved fullstate fingerprint, not proof of statistical independence'

FIELDS=['native_reward_sum','stored_final_expert_state_distance','final_hand_goal_distance_m','hand_goal_progress_m',
        'final_goal_xy_distance_px','min_goal_xy_distance_px','goal_xy_progress_px','final_requested_goal_coverage','max_requested_goal_coverage',
        'painted_default_reward_sum','painted_default_reward_max','final_painted_default_coverage']
def summarize(rs):
    result=dict(n=len(rs),episodes=[r['episode'] for r in rs],ever_success=sum(r['native_ever_success'] for r in rs),
        terminal_success=sum(r['native_final_success'] for r in rs),
        ever_success_rate=sum(r['native_ever_success'] for r in rs)/len(rs),
        terminal_success_rate=sum(r['native_final_success'] for r in rs)/len(rs),
        raw_step_counts=dict(Counter(str(r['raw_steps']) for r in rs)),
        n_available_full_initialstate_fingerprints=sum(r['initial_state'] is not None for r in rs),
        n_unique_full_initialstates=len({r['initial_state'] for r in rs if r['initial_state'] is not None}),
        n_unique_requested_goals=len({r['requested_goal'] for r in rs}),
        n_available_full_initialgoal_pairs=sum(r['initial_goal_pair'] is not None for r in rs),
        n_unique_full_initialgoal_pairs=len({r['initial_goal_pair'] for r in rs if r['initial_goal_pair'] is not None}),
        n_unique_environment_seeds=len({r['environment_seed'] for r in rs}),
        n_unique_planner_seeds=len({r['planner_seed'] for r in rs}),harness_counts=dict(Counter(r['harness'] for r in rs)))
    result['physical_metrics']={}
    for k in FIELDS:
        vals=[r[k] for r in rs if r.get(k) is not None]
        if vals:result['physical_metrics'][k]={'n':len(vals),'mean':statistics.fmean(vals),'median':statistics.median(vals),'min':min(vals),'max':max(vals)}
    if rs[0]['panel'].startswith('pusht'):
        result.update(requested_coverage_ge_095_ever=sum(r['requested_coverage_ge_095_ever'] for r in rs),
            requested_coverage_ge_095_terminal=sum(r['requested_coverage_ge_095_final'] for r in rs),
            n_painted_goals_matching_requested=sum(r['painted_default_is_requested_goal'] for r in rs),
            native_robust_angular_rule_disagreements=sum(r['native_robust_angle_success_disagreements'] for r in rs),
            max_painted_reward_reconstruction_error=max(r['painted_reward_reconstruction_maxabs'] for r in rs))
    return result

panels={}
for panel in ['reach_wall','pusht_official','pusht_scripted']:
    rs=[r for (p,e),r in sorted(rows.items()) if p==panel]
    panels[panel]=summarize(rs)
    panels[panel]['by_split']={s:summarize([r for r in rs if r['split']==s]) for s in sorted({r['split'] for r in rs})}
    panels[panel]['by_harness']={s:summarize([r for r in rs if r['harness']==s]) for s in sorted({r['harness'] for r in rs})}
result=dict(complete=True,registered_rows=149,verified_artifact_rows=149,missing_final_rows=0,nonfinite_final_rows=0,
    panels=panels,rows=[rows[k] for k in sorted(rows)],source_attempt_receipts=receipts,
    resolved_path_failures=[{'panel':r['panel'],'episode':r['episode'],'error':r['error']} for r in failures],
    timed_analysis_cpu_seconds=cpu,model_calls=0,simulator_calls=0,gpu_calls=0,
    exposed_held_baseline_outcomes_for_user_requested_reporting=True,no_tuning_or_steered_confirmation=True,
    scope='Descriptive registered unsteered baselines only; do not pool tasks or official/scripted Push panels; repeated full99 controls excluded',
    native_success_definitions={p:next(r['native_success_definition'] for r in rows.values() if r['panel']==p) for p in panels},
    uncertainty='No binomial intervals asserted here; exact fingerprints/seed counts are not evidence of iid sampling. Root separately handles interval assumptions.',
    requested_goal_coverage='New descriptive polygon overlap with exact requested blockpose, not painted native reward and not a redefinition of positional-angle native success',
    goal_and_initializer_contracts='Per-row fields and harness scope preserve native goal/reset contracts; no goals or states changed',
    manifest_sha256=sha(ROOT/'manifest.json'),script_sha256=sha(Path(__file__)))
dump(ROOT/'AGGREGATE.json',result)
dump(ROOT/'AGGREGATE_DONE.json',{'complete':True,'outputs':[{'path':'AGGREGATE.json','sha256':sha(ROOT/'AGGREGATE.json')}],
    'verified_artifact_count':149,'timed_analysis_cpu_seconds':cpu})
print(json.dumps({'counts':{p:{k:v[k] for k in ['n','ever_success','terminal_success','raw_step_counts','n_unique_full_initialstates','n_unique_full_initialgoal_pairs']} for p,v in panels.items()},'cpu':cpu,'sha256':sha(ROOT/'AGGREGATE.json')},indent=2))
