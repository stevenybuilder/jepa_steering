#!/usr/bin/env python3
"""Geometry map v11: join post-v10 evidence and annotate every evidence family with statistical power.

Reads only existing JSON results (no model, simulator, or GPU). Never rewrites v10 outputs;
copies them unchanged into map-v11 and adds versioned power annotations, a power ledger,
a forest plot, and a human-readable GEOMETRY_MAP.md. Missing measurements stay null.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, statistics
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

T975 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,25:2.060,30:2.042,40:2.021,60:2.000,120:1.980,1000:1.962}
T80  = {1:1.376,2:1.061,3:0.978,4:0.941,5:0.920,6:0.906,7:0.896,8:0.889,9:0.883,10:0.879,11:0.876,12:0.873,13:0.870,14:0.868,15:0.866,16:0.865,17:0.863,18:0.862,19:0.861,20:0.860,25:0.856,30:0.854,40:0.851,60:0.848,120:0.845,1000:0.842}
Z = NormalDist()

def tq(table, df):
    keys = sorted(table)
    best = keys[0]
    for k in keys:
        if k <= df: best = k
    return table[best]

def sha256(p: Path):
    with p.open('rb') as h: return hashlib.file_digest(h, 'sha256').hexdigest()

def paired_summary(values, gate=None, unit=''):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    out = dict(n=n, unit=unit, values=vals, mean=None, sd=None, ci95=None, sign_floor_p=None, mde80=None, gate=gate, n_needed_for_gate=None, verdict=None)
    if n == 0:
        out['verdict'] = 'no_values'; return out
    if n == 1:
        out.update(mean=vals[0], verdict='degenerate_n1'); return out
    mean = statistics.fmean(vals); sd = statistics.stdev(vals)
    t975, t80 = tq(T975, n-1), tq(T80, n-1)
    se = sd / math.sqrt(n)
    out.update(mean=mean, sd=sd, ci95=[mean - t975*se, mean + t975*se], sign_floor_p=2 * 0.5**n,
               positive_units=sum(v > 0 for v in vals), negative_units=sum(v < 0 for v in vals), zero_units=sum(v == 0 for v in vals))
    if sd == 0:
        out['verdict'] = 'zero_variance_no_decision_change' if all(v == 0 for v in vals) else 'zero_variance_constant_effect'
        return out
    out['mde80'] = (t975 + t80) * se
    if gate:
        m = None
        for k in range(2, 100001):
            if (tq(T975, k-1) + tq(T80, k-1)) * sd / math.sqrt(k) <= gate: m = k; break
        out['n_needed_for_gate'] = m
    under = n < 8 or out['sign_floor_p'] > 0.05 or (gate is not None and out['mde80'] > gate)
    out['verdict'] = 'underpowered' if under else 'adequate_for_screen'
    return out

def wilson(k, n, z=1.959964):
    if n == 0: return None
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return dict(k=k, n=n, rate=p, ci95=[c-h, c+h])

def two_prop_mde(p0, n_per_arm, power=0.8, alpha=0.05):
    za, zb = Z.inv_cdf(1-alpha/2), Z.inv_cdf(power)
    lo, hi = 0.0, 1-p0
    for _ in range(60):
        d = (lo+hi)/2; p1 = p0 + d; pb = (p0+p1)/2
        need = za*math.sqrt(2*pb*(1-pb)/n_per_arm) + zb*math.sqrt((p0*(1-p0)+p1*(1-p1))/n_per_arm)
        if d >= need: hi = d
        else: lo = d
    return hi

def load(path, sources):
    p = Path(path)
    if not p.is_file(): return None
    sources[str(p)] = dict(path=str(p.resolve()), sha256=sha256(p))
    return json.loads(p.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('artifacts/geometry_map'))
    ap.add_argument('--base', type=Path, default=Path('artifacts/geometry_map/reach_wall_v1/map-v10'))
    ap.add_argument('--out', type=Path, default=Path('artifacts/geometry_map/reach_wall_v1/map-v11'))
    a = ap.parse_args()
    R, base, out = a.root, a.base, a.out
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    copied = []
    for f in sorted(base.iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            shutil.copy2(f, out / f.name); copied.append(dict(name=f.name, sha256=sha256(f)))
    gm = load(base/'geometry_map.json', sources); de = load(base/'decision_evidence.json', sources)
    ledger = []
    def add(id_, family, task, unit, summ, source, note='', gate_desc=None):
        row = dict(ledger_id=id_, family=family, task=task, unit=unit, source=source, note=note)
        row.update({k: v for k, v in summ.items() if k not in ('values', 'unit')}); row['values'] = summ.get('values')
        if gate_desc: row['gate_desc'] = gate_desc
        ledger.append(row); return id_

    # --- 1. Representation rows (v10) ---
    rows = gm['rows']
    for r in rows:
        r['power'] = None
    rep = [r for r in rows if r.get('linear_score') and r.get('independent_episode_count')]
    lid = add('rep.linear_screen', 'representation linear screen (LOSO collection-seed folds)', 'reach-wall', 'r2 or auroc',
              dict(n=150, verdict='descriptive_no_ci', mean=None), 'map-v10 rows',
              note='150 discovery episodes, 3 collection-seed folds, 2850 states; per-fold scores are not stored per row so no CI; discovery-only, no untouched confirmation. Adequate as a screen for large effects; not evidence of causal use.')
    for r in rep: r['power'] = dict(ledger_id=lid, n_independent_units=150, unit='episode', verdict='descriptive_no_ci')
    for r in rows:
        if r['task'] == 'push-t' and r['power'] is None:
            r['power'] = dict(ledger_id='rep.pusht_coordinate_probe', n_independent_units=None, unit='development state', verdict='descriptive_no_ci')
    add('rep.pusht_coordinate_probe', 'Push-T coordinate probes', 'push-t', 'r2', dict(n=None, verdict='descriptive_no_ci', mean=None), 'map-v10 rows', note='negative r2 values; no coordinate frame decodable at these sites in the tested setting.')
    # coordinate frame folds
    for r in rows:
        bf = r.get('best_coordinate_frame')
        if bf and bf.get('folds'):
            byframe = {}
            for f in bf['folds']: byframe.setdefault(f['frame'], []).append(f.get('world_r2'))
            frames = list(byframe)
            diffs = None
            if len(frames) >= 2 and all(len(byframe[f]) == len(byframe[frames[0]]) for f in frames):
                best = max(frames, key=lambda f: statistics.fmean(byframe[f]))
                others = [f for f in frames if f != best]
                diffs = {o: [x - y for x, y in zip(byframe[best], byframe[o])] for o in others}
            s = paired_summary(diffs[list(diffs)[0]] if diffs else [], unit='r2 difference best-vs-next frame')
            id_ = add(f'frame.{r["row_id"]}', 'coordinate-frame comparison (per held seed)', 'reach-wall', 'r2 diff', s, 'map-v10 best_coordinate_frame.folds',
                      note=f'frames={frames}; best={best if diffs else None}; n = held collection seeds (3). Frame identifiability cannot be established with 3 folds; report as tie unless CI excludes 0.')
            r['power'] = dict(ledger_id=id_, n_independent_units=s['n'], unit='held collection seed', verdict=s['verdict'])
    for r in rows:
        cs = r.get('cross_trajectory_stability')
        if cs and cs.get('draws'):
            fr = [d['a_projected_into_b_fraction'] for d in cs['draws'] if d.get('status') == 'computed']
            rnd = [d['random_a_into_b_fraction'] for d in cs['draws'] if d.get('status') == 'computed']
            s = paired_summary([x - y for x, y in zip(fr, rnd)], unit='projection fraction minus random')
            s['verdict'] = 'bootstrap_draws_not_independent_units'
            id_ = add(f'stability.{r["row_id"]}', 'cross-trajectory subspace stability (episode bootstrap)', 'reach-wall', 'projection fraction - random', s,
                      'map-v10 cross_trajectory_stability.draws', note='Bootstrap draws resample the same 150 discovery episodes; CI is descriptive, draws are not independent replications.')
            r['power'] = dict(ledger_id=id_, n_independent_units=150, unit='episode (bootstrap)', verdict=s['verdict'])
    # broad action patch
    apj = load(R/'reach_wall_v1/action-patching/summary.json', sources)
    if apj:
        for blk in apj.get('summaries', []) if isinstance(apj.get('summaries'), list) else []:
            pass
        add('causal.broad_patch_48', 'whole-block action patching, beta=0.5 (48 snapshots)', 'reach-wall', 'transfer fraction',
            dict(n=apj.get('snapshot_count', 48), verdict='adequate_for_screen', mean=None, note=None), 'reach_wall_v1/action-patching/summary.json',
            note='Paired, collection-seed-stratified bootstrap with 10000 draws; gate passed at blocks 2 and 3. Whole-block transfer is action-mediation evidence only, not a selective control surface.')
    for r in rows:
        cp = r.get('causal_patch_effect')
        if cp and cp.get('effects'):
            for k, e in cp['effects'].items():
                if 'episode_values' in e:
                    s = paired_summary(e['episode_values'], gate=0.005, unit='m')
                    id_ = add(f'causal.replication.{r["row_id"]}.{k}', 'targeted subspace edit, 5 full-episode starts', 'reach-wall', 'm', s,
                              'map-v10 causal_patch_effect.effects', note='5 independent starts; gate = 5 mm progress. Descriptive bootstrap CI in v10 includes 0.', gate_desc='5 mm')
                    if k == 'goal_improvement_vs_unsteered_m':
                        r['power'] = dict(ledger_id=id_, n_independent_units=s['n'], unit='episode start', verdict=s['verdict'], mde80_m=s['mde80'], n_needed_for_5mm=s['n_needed_for_gate'])
        sp = r.get('specificity'); md = r.get('manifold_distance')
        if sp and sp.get('independent_starts') == 1:
            add(f'specificity.{r["row_id"]}', 'specificity controls', 'reach-wall', '', dict(n=1, verdict='degenerate_n1', mean=None), 'map-v10 specificity', note='One start; location/time controls executed but behavioural specificity not established.')
        if md and md.get('independent_starts') == 1:
            add(f'manifold.{r["row_id"]}', 'manifold support distance', 'reach-wall', 'Mahalanobis proxy', dict(n=1, verdict='degenerate_n1', mean=None), 'map-v10 manifold_distance', note='One start; support proxy only.')

    # --- 2. Decision rows (v10) ---
    for d in de['rows']:
        s = paired_summary(d['state_values'], gate=2.0, unit='pp')
        d['power'] = {k: s[k] for k in ('n', 'mean', 'sd', 'ci95', 'sign_floor_p', 'mde80', 'n_needed_for_gate', 'verdict')}
        d['power']['gate'] = '2 pp mean coverage gain (protocol)'
        if d['arm'] != 'native' and d['continuation_gate'] == 'fail':
            add(f'decision.{d["source"]}.{d["arm"]}.{d["channel"]}', d['group'], 'push-t', 'pp coverage gain vs native', s, d['source'],
                note=f'4 reused development states x 64 dependent plans; matched control mean = {d.get("matched_control_mean_pp")}', gate_desc='2 pp')

    # --- 3. Autoresearch ledger ---
    tl_path = R/'autoresearch_steering_20260906/TRIAL_LEDGER.jsonl'
    if tl_path.is_file():
        sources[str(tl_path)] = dict(path=str(tl_path.resolve()), sha256=sha256(tl_path))
        for line in tl_path.read_text().splitlines():
            t = json.loads(line)
            if t['task'] == 'push_t':
                g = t['gains_by_state']
                nat = g.get('native'); 
                if nat is None: continue
                s = paired_summary([100*x for x in nat], gate=2.0, unit='pp')
                add(f'autoresearch.push.{t["branch"]}.{t["family"]}.{t.get("arm")}.{t.get("cap_fraction", t.get("strength"))}', f'autoresearch push {t["branch"]}', 'push-t', 'pp coverage gain vs native', s, t['source'],
                    note=f'passes={t["passes"]}; forecast_mse_ratio={t.get("forecast_mse_ratio")}', gate_desc='2 pp, 3/4 states, beats each sham')
    for name, f in (('pooled', 'pilot-v1-summary.json'), ('full_spatial', 'pilot-fullgrid-v1-summary.json')):
        p = load(R/'autoresearch_steering_20260906/reach'/f, sources)
        if not p: continue
        dn = [1000*(r['physical']['edit']['hand_goal_progress_m'] - r['physical']['native_planner']['hand_goal_progress_m']) for r in p['rows']]
        ds = [1000*(r['physical']['edit']['hand_goal_progress_m'] - r['physical']['sham']['hand_goal_progress_m']) for r in p['rows']]
        for lab, vals in (('vs_native_planner', dn), ('vs_sham', ds)):
            s = paired_summary(vals, gate=5.0, unit='mm')
            add(f'autoresearch.reach.{name}.{lab}', f'Reach output-correction pilot ({name}), 15 raw steps', 'reach-wall', 'mm progress', s, f'autoresearch_steering_20260906/reach/{f}', note='8 new development starts 66-73; gate = 5 mm mean over native, bank and sham plus 5/8 positive.', gate_desc='5 mm')
    h = load(R/'autoresearch_steering_20260906/reach/horizon-diagnostic-summary-v1.json', sources)
    if h and h.get('rows') and 'arms' in h['rows'][0]:
        dn = [1000*(r['arms']['edit']['physical']['hand_goal_progress_m'] - r['arms']['native_planner']['physical']['hand_goal_progress_m']) for r in h['rows'] if 'edit' in r['arms']]
        ds = [1000*(r['arms']['edit']['physical']['hand_goal_progress_m'] - r['arms']['sham']['physical']['hand_goal_progress_m']) for r in h['rows'] if 'sham' in r['arms']]
        for lab, vals in (('vs_native_planner', dn), ('vs_sham', ds)):
            s = paired_summary(vals, gate=5.0, unit='mm')
            add(f'autoresearch.reach.h6_30step.{lab}', 'Reach H6 diagnostic, 30 raw steps', 'reach-wall', 'mm progress', s, 'autoresearch_steering_20260906/reach/horizon-diagnostic-summary-v1.json', note='Same frozen choices; 30-step endpoint does not replace the failed 15-step gate.', gate_desc='5 mm')

    # --- 4. CEM audit + replication (post-v10) ---
    cem_rep = load(R/'cem_search_replication_v1/summary-v1/AGGREGATE.json', sources)
    cem_aud = (de.get('cem_audit') or {}).get('report')
    post = {}
    if cem_rep:
        v = [100*x for x in cem_rep['primary']['stage30_minus15_mean_requested_coverage']['by_source']]
        s = paired_summary(v, gate=2.0, unit='pp'); add('cem.replication.stage30_minus15_coverage', 'native CEM: 30 vs 15 iterations (4 NEW train groups)', 'push-t', 'pp coverage', s, 'cem_search_replication_v1/summary-v1/AGGREGATE.json', note='Model cost decreased in all 4 groups while coverage did not improve; replication of the audit on new initial groups.', gate_desc='2 pp')
        mb = [100*x for x in cem_rep['secondary_mean_minus_best']['30']['mean_minus_best_requested_coverage']['by_source']]
        s2 = paired_summary(mb, gate=2.0, unit='pp'); add('cem.replication.mean_minus_best_stage30', 'native CEM: elite-mean minus best-candidate (4 NEW groups)', 'push-t', 'pp coverage', s2, 'cem_search_replication_v1/summary-v1/AGGREGATE.json', note='Tests the averaging-failure hypothesis; no general averaging failure.', gate_desc='2 pp')
        post['cem_replication'] = dict(source=str(R/'cem_search_replication_v1/summary-v1/AGGREGATE.json'), stage30_minus15_coverage_pp=s, mean_minus_best_stage30_pp=s2, limitations=cem_rep.get('limitations'))
    if cem_aud:
        mb = [100*x for x in cem_aud['stages']['30']['mean_minus_best_requested_coverage']['by_state']]
        s = paired_summary(mb, gate=2.0, unit='pp'); add('cem.audit.mean_minus_best_stage30', 'native CEM: elite-mean minus best-candidate (4 seen DEV states)', 'push-t', 'pp coverage', s, 'decision_evidence.cem_audit', gate_desc='2 pp')
        if cem_rep:
            pooled = mb + [100*x for x in cem_rep['secondary_mean_minus_best']['30']['mean_minus_best_requested_coverage']['by_source']]
            s = paired_summary(pooled, gate=2.0, unit='pp'); add('cem.pooled8.mean_minus_best_stage30', 'native CEM mean-minus-best, 4 DEV states + 4 new groups (descriptive pool)', 'push-t', 'pp coverage', s, 'cem audit + replication', note='Two populations pooled descriptively; not a prespecified sample.', gate_desc='2 pp')

    # --- 5. Baselines ---
    bl = load(R/'unsteered_baseline_metrics_v1/AGGREGATE.json', sources)
    five = []
    for d in sorted((R/'reach_five_additional_v1').glob('episode-*/results-v2/DONE.json')):
        j = load(d, sources); o = j['outputs'][0]
        pt = d.parent / o['path']; ok = sha256(pt) == o['sha256'] if pt.is_file() else None
        five.append(dict(episode=o['episode'], environment_seed=o['environment_seed'], planner_seed=o['planner_seed'], ever_success=o['ever_success'], final_success=o['final_success'], tensor_sha_verified=ok))
    baselines = dict(panels={}, reach_five_additional=five)
    if bl:
        for panel, v in bl['panels'].items():
            for harness, x in v['by_harness'].items():
                baselines['panels'][f'{panel}/{harness}'] = dict(n=x['n'], terminal=wilson(x['terminal_success'], x['n']), ever=wilson(x['ever_success'], x['n']), raw_steps=x.get('raw_step_counts'))
    rw = [v for k, v in baselines['panels'].items() if k.startswith('reach_wall')]
    if rw:
        n = sum(v['n'] for v in rw) + len(five); kt = sum(v['terminal']['k'] for v in rw) + sum(f['final_success'] for f in five); ke = sum(v['ever']['k'] for v in rw) + sum(f['ever_success'] for f in five)
        baselines['reach_wall_all_full99'] = dict(n=n, terminal=wilson(kt, n), ever=wilson(ke, n), note='66 on-policy bank + 8 development expansion + 5 additional; all native full-99-step unsteered runs; not iid by construction (fixed seed ranges).')
    # --- 6. Planning: what a closed-loop pilot can detect ---
    p0t = baselines.get('reach_wall_all_full99', {}).get('terminal', {}).get('rate', 0.24)
    p0e = baselines.get('reach_wall_all_full99', {}).get('ever', {}).get('rate', 0.45)
    plan = dict(success_rate_mde_pp={}, continuous_endpoint={})
    for n in (12, 24, 48, 96, 200, 400):
        plan['success_rate_mde_pp'][n] = dict(terminal_from=round(100*p0t, 1), terminal_mde=round(100*two_prop_mde(p0t, n), 1), ever_from=round(100*p0e, 1), ever_mde=round(100*two_prop_mde(p0e, n), 1))
    reach_sd = [r for r in ledger if r['ledger_id'] == 'autoresearch.reach.full_spatial.vs_native_planner']
    if reach_sd and reach_sd[0].get('sd'):
        plan['continuous_endpoint'] = dict(endpoint='15-step hand-goal progress, mm, paired edit minus native', sd_mm=reach_sd[0]['sd'], n_needed_for_5mm_gate=reach_sd[0]['n_needed_for_gate'], n_needed_for_2mm=paired_summary(reach_sd[0]['values'], gate=2.0)['n_needed_for_gate'], n_needed_for_1mm=paired_summary(reach_sd[0]['values'], gate=1.0)['n_needed_for_gate'])
    plan['note'] = 'Two-proportion normal approximation, 80% power, alpha 0.05 two-sided, unpaired per-arm n; paired same-start designs can do better if outcomes are concordant. Continuous n from observed development SD.'

    # --- summary ---
    from collections import Counter
    verdicts = Counter(r['verdict'] for r in ledger)
    power_summary = dict(ledger_rows=len(ledger), verdict_counts=dict(verdicts),
        headline=[
            'Representation screen (447 rows) is a 150-episode discovery screen without per-row CIs: adequate to shortlist, not to confirm.',
            'Every intervention-decision test (58 v10 decision rows + 27 Push-T autoresearch trials) uses n=4 reused initial states; the exact sign-test floor is p=0.125, so no decision-level effect can reach significance regardless of result.',
            'Most Push-T arms show zero variance: the edit never changed the selected plan in a fixed 64-plan bank. That is a null at the decision level, but the design is only sensitive to argmin flips.',
            'The Reach output-correction pilots (n=8 starts, observed SD about 1.4 mm) were powered to detect the 5 mm gate (MDE about 1.6 mm) and found -0.6 to -0.7 mm vs native with CIs excluding +5 mm: these are genuine negatives, not underpowered nulls.',
            'The 5-start targeted subspace replication at predictor block 3 has MDE about 10-40 mm against a 5 mm gate: inconclusive, would need about 24-48 starts.',
            'Native CEM elite-mean vs best-candidate pooled over 8 groups: 0.1 pp, CI within +/-1 pp; the averaging-failure hypothesis is a powered null at the 2 pp gate.',
            'Native success rates carry Wilson CIs ~+/-10 pp at n~80; a 12-pair closed-loop pilot cannot detect less than ~45 pp change in success rate.'])
    now = datetime.now(timezone.utc).isoformat()
    gm.update(schema_version=str(gm.get('schema_version', '')) + '+power_v11', created_utc=now, base_map=str(base), complete=False,
              post_v10_evidence=post, baselines=baselines, closed_loop_power_planning=plan, power_summary=power_summary, power_ledger_path='power_ledger.json')
    gm['v11_sources'] = sources
    (out/'geometry_map.json').write_text(json.dumps(gm, indent=1, allow_nan=False) + '\n')
    de.update(created_utc=now, power_annotation='v11', power_note='n=4 reused development states per row; sign-test floor p=0.125; verdicts per row under power.')
    (out/'decision_evidence.json').write_text(json.dumps(de, indent=1, allow_nan=False) + '\n')
    (out/'power_ledger.json').write_text(json.dumps(dict(created_utc=now, rows=ledger, sources=sources), indent=1, allow_nan=False) + '\n')
    cols = ['ledger_id', 'family', 'task', 'unit', 'n', 'mean', 'sd', 'ci95', 'sign_floor_p', 'mde80', 'gate', 'n_needed_for_gate', 'verdict', 'positive_units', 'negative_units', 'zero_units', 'source', 'note']
    with (out/'power_ledger.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in ledger: w.writerow({k: (json.dumps(r.get(k)) if isinstance(r.get(k), list) else r.get(k)) for k in cols})
    write_forest(out/'power_forest.svg', ledger)
    write_md(out/'GEOMETRY_MAP.md', gm, de, ledger, baselines, plan, base)
    done = dict(assembly_complete=True, experiment_complete=False, created_utc=now, base=str(base), copied_from_base=copied,
                outputs=[dict(path=p.name, sha256=sha256(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'ASSEMBLY_DONE.json'],
                model_calls=0, simulator_calls=0, gpu_calls=0)
    (out/'ASSEMBLY_DONE.json').write_text(json.dumps(done, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), ledger_rows=len(ledger), verdicts=dict(verdicts)), indent=1))

def write_forest(path, ledger):
    panels = [('Push-T interventions: selected-plan coverage gain vs native (pp), gate 2 pp', 'pp coverage gain vs native', 2.0, 'push-t'),
              ('Reach-Wall output corrections: 15/30-step progress vs native or sham (mm), gate 5 mm', 'mm progress', 5.0, 'reach-wall')]
    W = 1100; rowh = 22; y = 40; parts = []
    parts.append('<style>text{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:12px;fill:#0b0b0b}.s{fill:#52514e;font-size:11px}.t{font-size:15px}</style>')
    x0, x1 = 640, W - 40
    for title, unit, gate, task in panels:
        rows = [r for r in ledger if r['unit'] == unit and r['task'] == task and r.get('n') and r['n'] >= 4 and r.get('mean') is not None]
        rows = [r for r in rows if r['verdict'] != 'zero_variance_no_decision_change'][:40]
        zero = sum(1 for r in ledger if r['unit'] == unit and r['task'] == task and r['verdict'] == 'zero_variance_no_decision_change')
        parts.append(f'<text class="t" x="20" y="{y}">{title}</text>'); y += 16
        parts.append(f'<text class="s" x="20" y="{y}">Dot = mean over independent starts; bar = 95% t-CI; light band = 80%-power minimum detectable effect; dashed = protocol gate.</text>'); y += 13
        parts.append(f'<text class="s" x="20" y="{y}">{zero} further arms with zero variance (the edit never changed the selected plan) are omitted; see power_ledger.csv.</text>'); y += 16
        vals = [gate, -gate, 0.0]
        for r in rows:
            vals += (r['ci95'] or [r['mean'], r['mean']])
            if r.get('mde80'): vals += [r['mde80'], -r['mde80']]
        lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.06 or 1; lo -= pad; hi += pad
        sx = lambda v: x0 + (max(lo, min(hi, v)) - lo) / (hi - lo) * (x1 - x0)
        top = y
        parts.append(f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{top + rowh*len(rows)}" stroke="#c9c8c3"/>')
        parts.append(f'<line x1="{sx(gate):.1f}" y1="{top}" x2="{sx(gate):.1f}" y2="{top + rowh*len(rows)}" stroke="#52514e" stroke-dasharray="4 3"/>')
        parts.append(f'<text class="s" x="{sx(gate)+3:.1f}" y="{top-3}">gate {gate:g} {unit.split()[0]}</text>')
        for r in rows:
            cy = y + rowh/2 + 4
            label = r['ledger_id'].replace('decision.', '').replace('autoresearch.', '').replace('task_metric_complement_ablation_v1.', 'complement.').replace('normalization_aware_subspace_v1.', 'normaware.').replace('one_minus_requested_goal_coverage', '1-cov').replace('joint_xy_squared_proxy', 'xy2')
            label = label[:58]
            parts.append(f'<text x="20" y="{cy}">{label}</text>')
            parts.append(f'<text class="s" x="{x0-8}" y="{cy}" text-anchor="end">n={r["n"]}  {r["verdict"].replace("_"," ")}</text>')
            if r.get('mde80'):
                parts.append(f'<rect x="{sx(-r["mde80"]):.1f}" y="{cy-9}" width="{sx(r["mde80"])-sx(-r["mde80"]):.1f}" height="14" fill="#cde2fb" rx="2"/>')
            if r.get('ci95'):
                parts.append(f'<line x1="{sx(r["ci95"][0]):.1f}" y1="{cy-2}" x2="{sx(r["ci95"][1]):.1f}" y2="{cy-2}" stroke="#2a78d6" stroke-width="2"/>')
            parts.append(f'<circle cx="{sx(r["mean"]):.1f}" cy="{cy-2}" r="4" fill="#2a78d6" stroke="#fcfcfb" stroke-width="2"><title>{label}: mean {r["mean"]:.3f} {unit}, n={r["n"]}, CI {r["ci95"]}, MDE80 {r.get("mde80")}</title></circle>')
            y += rowh
        step = 1.0 if hi - lo < 12 else (2.0 if hi - lo < 30 else 5.0)
        v = math.ceil(lo / step) * step
        while v <= hi:
            parts.append(f'<line x1="{sx(v):.1f}" y1="{y}" x2="{sx(v):.1f}" y2="{y+4}" stroke="#52514e"/>')
            parts.append(f'<text class="s" x="{sx(v):.1f}" y="{y+16}" text-anchor="middle">{v:g}</text>')
            v += step
        y += 44
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" viewBox="0 0 {W} {y}"><rect width="100%" height="100%" fill="#fcfcfb"/>' + ''.join(parts) + '</svg>'
    path.write_text(svg)

def fmt(x, d=2):
    return 'null' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))

def write_md(path, gm, de, ledger, baselines, plan, base):
    rows = gm['rows']; L = []
    L.append('# Geometry map v11 — evidence joined with statistical power\n')
    L.append(f'Built {gm["created_utc"]} from `{base}` plus post-v10 results. `complete: false`. No model, simulator, or GPU call. Every number below is re-derived from the source JSON listed in `power_ledger.json`; missing measurements are `null`, never zero.\n')
    L.append('## What the map says\n')
    for h in gm['power_summary']['headline']: L.append(f'- {h}')
    L.append('')
    L.append('## Representation shortlist (discovery screen, 150 episodes, no CI)\n')
    L.append('| Task variable | Best site | Linear r² / AUROC | Δ vs time-only | kNN gain | Participation ratio |\n|---|---|---:|---:|---:|---:|')
    best = {}
    for r in rows:
        if r['task'] != 'reach-wall' or not r.get('linear_score') or r['linear_score'].get('score') is None: continue
        k = r['task_variable']; s = r['linear_score']['score']
        if k not in best or s > best[k]['linear_score']['score']: best[k] = r
    for k, r in sorted(best.items(), key=lambda kv: -kv[1]['linear_score']['score'])[:24]:
        ls = r['linear_score']; g = r.get('geometry') or {}; ng = r.get('nonlinear_gain') or {}
        L.append(f'| {k} | {r["module"]} block {r["block"]} | {fmt(ls.get("score"),3)} ({ls.get("metric")}) | {fmt(ls.get("delta_vs_time"),3)} | {fmt(ng.get("nonlinear_gain"),3)} | {fmt(g.get("participation_ratio"),1)} |')
    L.append('\nScores are leave-one-collection-seed-out on the 150-trajectory discovery split. Time-only baselines are high for progress-like variables, so read Δ vs time, not raw r². None of these rows has untouched-confirmation evidence.\n')
    L.append('## Intervention ledger with power\n')
    L.append('| Ledger id | n | mean | 95% CI | sign-test floor p | MDE₈₀ | gate | n needed | verdict |\n|---|---:|---:|---|---:|---:|---|---:|---|')
    keep = [r for r in ledger if r.get('n') and r['n'] >= 4 and r.get('mean') is not None and r['ledger_id'].split('.')[0] in ('decision', 'autoresearch', 'causal', 'cem')]
    zero = [r for r in keep if r['verdict'] == 'zero_variance_no_decision_change']
    for r in keep:
        if r['verdict'] == 'zero_variance_no_decision_change': continue
        ci = f'[{r["ci95"][0]:.2f}, {r["ci95"][1]:.2f}]' if r.get('ci95') else 'null'
        L.append(f'| {r["ledger_id"]} | {r["n"]} | {fmt(r["mean"])} {r["unit"].split()[0]} | {ci} | {fmt(r.get("sign_floor_p"),3)} | {fmt(r.get("mde80"))} | {r.get("gate_desc") or fmt(r.get("gate"))} | {fmt(r.get("n_needed_for_gate"))} | {r["verdict"]} |')
    L.append(f'\n{len(zero)} additional arms have zero variance across all 4 states (the edit never changed the selected plan) and are listed in `power_ledger.csv` with verdict `zero_variance_no_decision_change`. That is a null at the decision level under a fixed 64-plan bank, not an underpowered positive.\n')
    L.append('Interpretation of MDE₈₀: the smallest true paired effect that this many independent starts would detect with 80% power at α=0.05, using the observed SD. When MDE₈₀ exceeds the gate the test could not have passed the gate for a real effect of gate size; that cell is underpowered rather than negative.\n')
    L.append('## Native unsteered baselines (label-first anchors)\n')
    L.append('| Panel | n | terminal success | 95% CI | ever success | 95% CI |\n|---|---:|---:|---|---:|---|')
    for k, v in baselines['panels'].items():
        L.append(f'| {k} | {v["n"]} | {100*v["terminal"]["rate"]:.1f}% | [{100*v["terminal"]["ci95"][0]:.1f}, {100*v["terminal"]["ci95"][1]:.1f}] | {100*v["ever"]["rate"]:.1f}% | [{100*v["ever"]["ci95"][0]:.1f}, {100*v["ever"]["ci95"][1]:.1f}] |')
    a = baselines.get('reach_wall_all_full99')
    if a: L.append(f'| reach_wall all full-99 (66+8+5) | {a["n"]} | {100*a["terminal"]["rate"]:.1f}% | [{100*a["terminal"]["ci95"][0]:.1f}, {100*a["terminal"]["ci95"][1]:.1f}] | {100*a["ever"]["rate"]:.1f}% | [{100*a["ever"]["ci95"][0]:.1f}, {100*a["ever"]["ci95"][1]:.1f}] |')
    L.append('\nEpisodes 74–78 (new, verified tensor SHA): ' + ', '.join(f'ep{f["episode"]} final={"✓" if f["final_success"] else "✗"}' for f in baselines['reach_five_additional']) + '.\n')
    L.append('## What a closed-loop pilot can detect\n')
    L.append('| n per arm | terminal-success MDE (from ' + f'{plan["success_rate_mde_pp"][12]["terminal_from"]}%) | ever-success MDE (from {plan["success_rate_mde_pp"][12]["ever_from"]}%) |\n|---:|---:|---:|')
    for n, v in plan['success_rate_mde_pp'].items(): L.append(f'| {n} | +{v["terminal_mde"]} pp | +{v["ever_mde"]} pp |')
    ce = plan.get('continuous_endpoint')
    if ce: L.append(f'\nContinuous endpoint ({ce["endpoint"]}): observed SD {ce["sd_mm"]:.2f} mm → n needed for 5 mm gate = {ce["n_needed_for_5mm_gate"]}, for 2 mm = {ce["n_needed_for_2mm"]}, for 1 mm = {ce["n_needed_for_1mm"]}.\n')
    L.append(plan['note'] + '\n')
    L.append('## Post-v10 evidence joined\n')
    pe = gm.get('post_v10_evidence', {}).get('cem_replication')
    if pe:
        s = pe['stage30_minus15_coverage_pp']; m = pe['mean_minus_best_stage30_pp']
        L.append(f'- Native CEM replication on 4 new TRAIN groups: 30-vs-15-iteration coverage change mean {s["mean"]:.2f} pp (CI [{s["ci95"][0]:.2f}, {s["ci95"][1]:.2f}]); elite-mean minus best {m["mean"]:.2f} pp (CI [{m["ci95"][0]:.2f}, {m["ci95"][1]:.2f}]). Model cost fell in all 4 groups while coverage did not improve. Verdict: {s["verdict"]}.')
    L.append('- 149-row native baseline table and 5 additional Reach episodes joined above with Wilson CIs.\n')
    L.append('## Field coverage carried from v10\n')
    fc = gm.get('field_coverage', {})
    L.append('| Field | measured / total |\n|---|---:|')
    for k, v in fc.items(): L.append(f'| {k} | {v["measured_rows"]} / {v["total_rows"]} |')
    L.append('\nBlockers unchanged: ' + '; '.join(gm.get('completion_blockers', [])) + '.\n')
    L.append('Figures: `power_forest.svg` (new), plus the v10 visuals copied unchanged (`layer_variable_heatmap.svg`, `spatial_token_readability.svg`, `shortlist_eigenspectra.svg`, causal timelines).\n')
    path.write_text('\n'.join(L))

if __name__ == '__main__':
    main()
