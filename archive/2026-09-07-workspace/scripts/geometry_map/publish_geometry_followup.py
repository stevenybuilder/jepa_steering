#!/usr/bin/env python3
"""Publish measured follow-ups without replacing the earlier map or its claims."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_rows(report):
    if not report.get('complete'):
        raise ValueError('Incomplete source')
    rows = []
    for item in report['geometry_views']:
        for mode in ('native', 'actual_past_context'):
            measure = item['modes'][mode]['max_orthogonal_fraction_of_chord']
            rows.append(dict(task='Push-T', metric='normalized_bending', block=item['block'],
                horizon=item['horizon'], radius=item['radius'], condition=mode,
                value=measure['mean'], n_initial_states=measure['n_initial_states'],
                independent_values=measure['initial_state_values'],
                scope='development; actual past is privileged offline information'))
    for item in report['forecast_views']:
        native = item['native_actual_mse']['mean']
        actual = item['actual_past_actual_mse']['mean']
        rows.append(dict(task='Push-T', metric='prediction_error_reduction_percent', block=None,
            horizon=item['horizon'], radius=item['radius'], condition='actual_past_vs_native',
            value=100*(native-actual)/native if native else None,
            n_initial_states=report['initial_states'], independent_values=None,
            scope='ratio of state-balanced means; not task-success improvement'))
    return rows


def render(rows):
    panels = [(r, m) for r in (1, 4) for m in ('native', 'actual_past_context')]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="780" viewBox="0 0 920 780">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial;fill:#18232e;font-size:14px}</style>',
        '<text x="24" y="28" font-size="21">Push-T: curvature across predictor blocks and imagined time</text>',
        '<text x="24" y="52">Four development starts. Bending / endpoint chord; same 0–0.65 color scale.</text>',
        '<text x="24" y="74">Actual-past context is an offline diagnostic, not deployable steering.</text>']
    for panel, (radius, mode) in enumerate(panels):
        top = 118 + panel*157
        parts.append(f'<text x="24" y="{top}">Radius {radius}: {mode}</text>')
        subset = [r for r in rows if r['metric']=='normalized_bending' and r['radius']==radius and r['condition']==mode]
        for bi, block in enumerate((1, 3, 5)):
            parts.append(f'<text x="24" y="{top+37+bi*32}">Block {block} (0-based)</text>')
            for hi, horizon in enumerate((1, 3, 6)):
                matches = [r for r in subset if r['block']==block and r['horizon']==horizon]
                if len(matches)!=1:
                    raise ValueError('Expected exactly one measured block/horizon cell')
                value = matches[0]['value']; strength = min(max(value/.65, 0), 1)
                color = f'rgb({int(245-120*strength)},{int(247-60*strength)},250)'
                x=235+hi*200; y=top+15+bi*32
                parts.extend([f'<rect x="{x}" y="{y}" width="192" height="29" fill="{color}"/>',
                    f'<text x="{x+12}" y="{y+20}">H{horizon}: {value:.4f}</text>'])
    parts.append('</svg>')
    return '\n'.join(parts)


def insert_panel(original, panel):
    if '<body>' in original:
        return original.replace('<body>', '<body>'+panel, 1)
    if '<h1>' in original:
        return original.replace('<h1>', panel+'<h1>', 1)
    raise ValueError('Unrecognized base HTML')


def verified_report(root, directory, filename, sources):
    """Only publish compact results whose final verification receipt pins their SHA."""
    folder = root/directory
    receipt_path = folder/'VERIFIED.json'
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get('complete'):
        raise ValueError('Incomplete verification: '+directory)
    expected = receipt.get('aggregate_sha256') if filename == 'AGGREGATE.json' else None
    if expected is None:
        expected = next(r['sha256'] for r in receipt['files'] if r['path'] == filename)
    path = folder/filename
    if digest(path) != expected:
        raise ValueError('Verified summary SHA differs: '+str(path))
    report = json.loads(path.read_text())
    if not report.get('complete'):
        raise ValueError('Incomplete measured report')
    sources[directory] = dict(path=str(path.resolve()), sha256=expected,
        verification_path=str(receipt_path.resolve()), verification_sha256=digest(receipt_path))
    return report


def decision_evidence(root):
    sources, rows = {}, []
    action = verified_report(root, 'action_responsive_subspace_v1', 'AGGREGATE.json', sources)
    norm = verified_report(root, 'normalization_aware_subspace_v1', 'results-v1/summary.json', sources)
    costs = verified_report(root, 'task_metric_complement_ablation_v1', 'results-v1/AGGREGATE.json', sources)
    objective = verified_report(root, 'objective_alignment_v1', 'AGGREGATE.json', sources)
    def add(group, arm, values, choices, channel='predicted', control=None, source=None):
        if len(values) != 4 or any(not math.isfinite(v) for v in values):
            raise ValueError('Four finite equal-weight state measurements required')
        avg = sum(values)/4
        positive = sum(v > 0 for v in values)
        rows.append(dict(group=group, arm=arm, channel=channel, metric='selected_requested_goal_coverage_gain_pp',
            state_values=[100*v for v in values], mean=100*avg, positive_states=positive,
            selected_indices=choices, matched_control_mean_pp=100*control if control is not None else None,
            continuation_gate=('fail' if not (avg >= .01 and positive >= 3 and avg > control) else 'pass_exploratory_only') if control is not None else 'not_applicable_control_or_diagnostic',
            n_initial_states=4, plans_per_state=64, reused_development=True,
            heldout_performance='untested', source=source))
    for arm, item in action['arms'].items():
        control = None
        if arm.startswith('targeted_'):
            sign = arm.split('_')[-1]
            control = max(v['mean_coverage_gain'] for k,v in action['arms'].items() if k.startswith('random_') and k.endswith(sign))
        add('Predictor: action-responsive rank4', arm, item['coverage_gain_by_state'], item['choices'],
            control=control, source='action_responsive_subspace_v1')
    by_arm = {r['arm']:r for r in norm['aggregate']}
    for arm, item in by_arm.items():
        control = None
        if 'semantic' in arm:
            control = by_arm[arm.replace('semantic', 'random')]['mean_coverage_delta']
        add('Predictor: normalization-aware P5', arm, item['coverage_deltas'], item['choices'],
            control=control, source='normalization_aware_subspace_v1')
    for channel, items in costs['channels'].items():
        by_arm = {r['arm']:r for r in items}
        for arm, item in by_arm.items():
            control = by_arm[arm.replace('/learned/', '/rotated_control/')]['mean_coverage_gain'] if '/learned/' in arm else None
            add('Scoring: fixed complement ablation', arm, item['coverage_gain_by_state'], item['selected_indices'],
                channel=channel, control=control, source='task_metric_complement_ablation_v1')
    terminal = next(h for h in objective['horizons'] if h['horizon'] == 6)
    per_state = terminal['per_state']
    result = dict(complete=True, source_verified=True, rows=rows, sources=sources,
        sample_scope='4 repeatedly reused DEVELOPMENT initial states, 64 fixed plans per state; not n=256 independent episodes.',
        heldout_performance='Untested for these operators; baseline collection does not establish held-out intervention efficacy.',
        objective_alignment=dict(scope='Technical/decision-cost diagnostic using true future encodings, not a deployable operator.',
            native_coverage_regret_pp=[100*r['native_predicted']['coverage_regret'] for r in per_state],
            actual_encoding_cost_coverage_regret_pp=[100*r['actual_encoding_cost_oracle']['coverage_regret'] for r in per_state],
            interpretation='Even actual-future encoding cost need not choose the best physical outcome. Encoder-distance accuracy is not task-utility alignment.'),
        technical_findings=[
            'Action-responsive rank4 edits and their controls: all selected-plan coverage gains are zero.',
            'P5 output-dose calibration was feasible on all252 noncentral candidates; calibrated positive yields mean +0.460pp in1/4 states, gate fails.',
            'Complement removal: pointwise XY span-only mean +1.160pp in2/4 states, exactly the same coverage gains as its rotated control; gate fails.',
            'Complement energy/contrast and geometric readability are descriptive mechanisms, not held-out physical performance.'],
        proposed_decision_matrix=[
            dict(branch='Unsteered', status='Reference baseline', missing_test='Same-start full-task success comparator for each frozen intervention.'),
            dict(branch='Prediction correction', status='Bounded development pilots; no operator passes continuation gate', missing_test='A repeatable frozen correction plus paired full-task success.'),
            dict(branch='Scoring correction', status='Fixed-bank ablations completed; learned specificity not established', missing_test='Frozen task-aligned score tested in native planning and paired full-task success.'),
            dict(branch='Search correction', status='CEM audit separate / awaiting completed verified integration', missing_test='Freeze any search correction after audit; no completed correction method or full-task efficacy claim.')],
        matrix_scope='Proposed comparison structure, NOT four finished methods, not authorization for additional experiments. Future primary endpoint: native full-task success.',
        cem_audit=dict(status='pending_verified_integration', outcome=None),
        conclusion='No confirmed leaderboard or promoted operator; retain every measured arm and all nulls.')
    result['metric_details'] = dict(
        requested_goal_coverage='Primary selected-plan physical polygon overlap; changes above in percentage points, not success rates.',
        stepwise_physical_progress=dict(status='not_aggregated_for_these_interventions', value=None,
            note='Do not invent a progress curve from endpoint coverage or substitute model-predicted motion.'),
        prediction_error=dict(action_responsive={k:v['equal_state_forecast_error_reduction_percent_by_horizon'] for k,v in action['arms'].items()},
            normalization_visual_H6_MSE_by_arm={r['arm']:r['visual_H6_MSE'] for r in norm['aggregate']},
            scope='Full native feature prediction error against cached actual encodings; distinct from physical utility.'),
        candidate_ranking_and_native_cost=dict(source='objective_alignment_v1', horizons=objective['horizons'],
            scope='Cost/coverage and cost/XY rankings, cost error and margins retained separately for H1–6; candidate pairs are dependent.'))
    cem_root=root/'cem_search_audit_v1/summary-v1'
    if (cem_root/'DONE.json').is_file():
        receipt=json.loads((cem_root/'DONE.json').read_text())
        expected=next(r['sha256'] for r in receipt['outputs'] if r['path']=='AGGREGATE.json')
        path=cem_root/'AGGREGATE.json'
        if not receipt['complete'] or digest(path)!=expected:
            raise ValueError('CEM compact source checksum failed')
        cem=json.loads(path.read_text())
        if not cem['complete'] or cem['independent_seen_development_states']!=4:
            raise ValueError('CEM audit scope differs')
        sources['cem_search_audit_v1']=dict(path=str(path.resolve()),sha256=expected,
            verification_path=str((cem_root/'DONE.json').resolve()),verification_sha256=digest(cem_root/'DONE.json'))
        result['cem_audit']=dict(status='complete_compact_source_verified', report=cem,
            scope='Separate fresh native CEM audit: 4 seen DEV states,30 optimization iterations;30 raw-step fixed physical forks, not256-plan replication or closed-loop held efficacy.',
            geometry_warning='Elite action-space spread and proposal entropy are not model uncertainty or proof of a curved latent manifold; negative differential entropy is valid.',
            interpretation='Final mean minus best-sampled coverage is mixed, mean -0.291pp; no consistent averaging failure.')
        result['proposed_decision_matrix'][-1]['status']='Native CEM audit complete; averaging result mixed, no search correction validated'
    return result


def decision_label(row):
    arm = row['arm'].replace('joint_xy_squared_proxy', 'XY').replace('one_minus_requested_goal_coverage', 'coverage')
    return arm.replace('rotated_control', 'rotated').replace('with_complement', '+complement').replace('calibrated_', 'cal. ').replace('_', ' ')


def render_decisions(evidence):
    rows = [r for r in evidence['rows'] if r['channel'] == 'predicted']
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="'+str(155+len(rows)*29)+'">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b;font-size:12px}</style>',
        '<text x="20" y="28" style="font-size:20px">Push-T fixed-bank decisions: coverage change, not held-out success</text>',
        '<text x="20" y="51">4 reused DEV states ×64 plans. Percentage points vs native predicted choice; equal-state mean. No operator promoted.</text>',
        '<text x="20" y="72">Blue/red = gain/loss, same ±3pp scale. Full oracle-channel rows and all source hashes are in decision_evidence.json / CSV.</text>']
    for x, label in [(20,'Arm (all measured predicted-channel variants)'),(610,'DEV0'),(695,'DEV1'),(780,'DEV2'),(865,'DEV3'),(950,'Mean'),(1030,'Positive'),(1100,'Control mean'),(1210,'Gate')]:
        out.append(f'<text x="{x}" y="101">{label}</text>')
    last = None
    for i, row in enumerate(rows):
        y = 116+i*29
        if last != row['group']:
            out.append(f'<line x1="20" x2="1300" y1="{y-5}" y2="{y-5}" stroke="#64748b"/>')
        last = row['group']
        label = decision_label(row)
        prefix = {'Predictor: action-responsive rank4':'Action', 'Predictor: normalization-aware P5':'P5', 'Scoring: fixed complement ablation':'Cost'}[row['group']]
        out.append(f'<text x="20" y="{y+18}">{html.escape(prefix+": "+label)}</text>')
        for j, value in enumerate(row['state_values']+[row['mean']]):
            strength = min(abs(value)/3, 1)
            rgb = (40,110,175) if value >= 0 else (180,65,55)
            fill = '#'+''.join(f'{round(248+strength*(v-248)):02x}' for v in rgb)
            x=600+j*85
            out.append(f'<rect x="{x}" y="{y}" width="80" height="24" fill="{fill}"/><text x="{x+5}" y="{y+17}">{value:+.3f}</text>')
        control = '—' if row['matched_control_mean_pp'] is None else f"{row['matched_control_mean_pp']:+.3f}"
        gate = 'FAIL' if row['continuation_gate']=='fail' else '—'
        out.append(f'<text x="1040" y="{y+18}">{row["positive_states"]}/4</text><text x="1110" y="{y+18}">{control}</text><text x="1220" y="{y+18}">{gate}</text>')
    return '\n'.join(out+['</svg>'])


def decision_panel(evidence, output):
    panel='<section id="latest-decision-evidence" style="padding:20px;border:2px solid #426b8e"><h2>Latest measured decision evidence</h2>'
    ledger=evidence.get('scope_ledger')
    if ledger:
        panel+='<p><strong>The map is not all n=4.</strong> '+html.escape(ledger['descriptive_scope'])+' '+html.escape(ledger['missing_count_scope'])+'</p>'
    panel+='<p>'+html.escape(evidence['sample_scope'])+'</p><p>'+html.escape(evidence['heldout_performance'])+'</p>'
    panel+='<p><a href="decision_evidence.json">Verified source data / scope</a> · <a href="decision_evidence.csv">All numerical rows</a></p>'
    panel+='<img alt="Per-state requested-goal coverage gains for every predicted-channel action, normalization, and scoring ablation" src="decision_evidence_heatmap.svg">'
    panel+='<h3>Technical finding ≠ geometry proxy ≠ physical selection ≠ held-out performance</h3><ul>'
    panel+=''.join('<li>'+html.escape(v)+'</li>' for v in evidence['technical_findings'])+'</ul>'
    obj=evidence['objective_alignment']
    panel+='<p>Objective alignment at H6: actual-encoding cost still has physical coverage regret '+', '.join(f'{v:.2f}pp' for v in obj['actual_encoding_cost_coverage_regret_pp'])+' across DEV0–3. This oracle is not a usable runtime correction.</p>'
    panel+='<p>Metric separation: requested-goal coverage, combined agent/block XY distance, H1–6 feature prediction error, candidate ranking, and native model cost are distinct. Stepwise physical-progress changes for these interventions are not yet aggregated; the JSON marks them unmeasured rather than deriving them from endpoint coverage.</p>'
    panel+='<h3>Proposed comparison matrix — not four finished methods</h3><table><tr><th>Branch</th><th>Status</th><th>Missing test</th></tr>'
    for r in evidence['proposed_decision_matrix']:
        panel+='<tr>'+''.join('<td>'+html.escape(r[k])+'</td>' for k in ('branch','status','missing_test'))+'</tr>'
    panel+='</table><p>'+html.escape(evidence['matrix_scope'])+'</p><p>Source reports: '
    panel+=' · '.join('<a href="'+html.escape(os.path.relpath(v['path'], output))+'">'+html.escape(k)+'</a>' for k,v in evidence['sources'].items())
    panel+='</p>'
    cem=evidence['cem_audit']
    if cem['status']=='complete_compact_source_verified':
        panel+='<h3>Separate native CEM audit: mean versus best sampled plan</h3><p>'+html.escape(cem['scope'])+'</p>'
        panel+='<table><tr><th>Iteration</th><th>Coverage change DEV0 /1 /2 /3 (pp)</th><th>Mean (pp)</th><th>Mean model-cost difference</th></tr>'
        for stage,item in cem['report']['stages'].items():
            c=item['mean_minus_best_requested_coverage']
            panel+='<tr><td>'+html.escape(stage)+'</td><td>'+', '.join(f'{100*v:+.3f}' for v in c['by_state'])+'</td><td>'+f"{100*c['equal_state_mean']:+.3f}"+'</td><td>'+f"{item['mean_minus_best_model_cost']['equal_state_mean']:+.6f}"+'</td></tr>'
        panel+='</table><p>'+html.escape(cem['interpretation'])+'</p><p>'+html.escape(cem['geometry_warning'])+'</p>'
    return panel+'</section>'


def publish(root, base, output, decisions=False, refresh_draft=False):
    source = root/'action_path_curvature_v1/actual-context-v1/summary-v2.json'
    report = json.loads(source.read_text()); rows = measured_rows(report)
    evidence = {}
    patterns = ['action_ranking_pilot_v1/verified-v1/*.json',
        'action_ranking_jacobian_v1/verified-v1/*.json',
        'reach_native_coordinates_v1/results-v1/*.json',
        'reach_hmm_routing_v1/results-v2/*.json',
        'density_geometry_pilot_v1/**/*.json']
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.name.startswith('episode-') or path.name.startswith('near-dev-') or path.name in ('DONE.json','protocol.json'):
                evidence[str(path.relative_to(root))] = dict(sha256=digest(path), data=json.loads(path.read_text()))
    svg = render(rows)
    original=(base/'index.html').read_text()
    original=re.sub(r'<nav aria-label="Latest publication".*?</nav>', '', original, flags=re.DOTALL)
    insert_panel(original, '')  # Validate before creating any output.
    if output.exists() and not refresh_draft:
        raise FileExistsError('Published versions are immutable: '+str(output))
    if refresh_draft:
        prior=json.loads((output/'ASSEMBLY_DONE.json').read_text())
        if prior.get('base')!=str(base.resolve()):
            raise ValueError('Draft rebuild must retain its original base snapshot')
    shutil.copytree(base, output, dirs_exist_ok=refresh_draft, ignore=shutil.ignore_patterns('NAVIGATION_UPDATE.json'))
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
        source=dict(path=str(source.resolve()), sha256=digest(source)), evidence=evidence,
        conclusion='Geometry measurements are not proof of useful steering; no operator promoted.',
        scope='Development only; tasks and original independent-state counts remain separate.')
    if (base/'research_followup.json').is_file():
        previous=json.loads((base/'research_followup.json').read_text())
        payload['evidence']={**previous.get('evidence',{}), **evidence}
        payload['previous_publication']=dict(path=str((base/'research_followup.json').resolve()), sha256=digest(base/'research_followup.json'))
    decision = decision_evidence(root) if decisions else None
    if decision:
        original_data=json.loads((base/'geometry_map.json').read_text())
        map_rows=original_data['rows']
        counted=[r for r in map_rows if r.get('independent_episode_count')==150 and r.get('sample_count')==2850]
        missing=sum(r.get('independent_episode_count') is None for r in map_rows)
        decision['scope_ledger']=dict(historical_map_rows=len(map_rows), descriptive_counted_rows=len(counted),
            descriptive_trajectory_ids=150, dependent_step_samples=2850, distinct_initial_state_groups=None,
            rows_without_recorded_episode_count=missing,
            descriptive_scope=f'{len(counted)} historical descriptive rows use150 discovery trajectories and2850 dependent step samples, not2850 independent observations. The trajectory-count field does not establish150 distinct physics initial states.',
            missing_count_scope=f'{missing} historical rows do not record that count: UNKNOWN; their individual source scopes still apply.',
            causal_scope='Recent decision/intervention panels reuse4 DEV states; independent-state counts are not pooled across experiments.')
        payload['decision_evidence']=decision
        (output/'decision_evidence.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
        with (output/'decision_evidence.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(decision['rows'][0]));writer.writeheader();writer.writerows(decision['rows'])
        (output/'decision_evidence_heatmap.svg').write_text(render_decisions(decision))
    (output/'research_followup.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    with (output/'research_followup.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'curvature_history_heatmap.svg').write_text(svg)
    target=output/'geometry_map.json'; data=json.loads(target.read_text())
    data['followups']['research_iteration_2026_09_06']=payload
    if decision:
        data['followups']['decision_evidence_2026_09_06']=decision
    data['complete']=False
    target.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    page=output/'index.html'
    panel='<section style="padding:20px;border:2px solid #426b8e"><h2>Latest measured research follow-up</h2><p>Development evidence, not a validated steering interface. Earlier map measurements below retain their original scope.</p><p><a href="research_followup.json">Machine-readable results and source hashes</a> | <a href="research_followup.csv">Numerical heatmap table</a></p><img alt="Curvature by predictor block, horizon, and history condition" src="curvature_history_heatmap.svg" style="max-width:100%"></section>'
    page.write_text(insert_panel(original, decision_panel(decision, output) if decision else panel))
    outputs=[dict(path=str(p.relative_to(output)),sha256=digest(p)) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='ASSEMBLY_DONE.json']
    (output/'ASSEMBLY_DONE.json').write_text(json.dumps(dict(assembly_complete=True,experiment_complete=False,
        created_utc=datetime.now(timezone.utc).isoformat(), base=str(base.resolve()), base_index_sha256=digest(base/'index.html'),
        previous_rows_preserved=len(data['rows']), decision_rows=len(decision['rows']) if decision else 0,
        outputs=outputs),indent=2,allow_nan=False)+'\n')
    return dict(output=str(output), measured_rows=len(rows), evidence_files=len(evidence), complete=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','base','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--decisions',action='store_true')
    parser.add_argument('--refresh-draft',action='store_true',help='Explicitly rebuild the current unpublished draft from the same base; never delete previous snapshots.')
    args=parser.parse_args()
    print(json.dumps(publish(args.root, args.base, args.output, args.decisions,args.refresh_draft)))
