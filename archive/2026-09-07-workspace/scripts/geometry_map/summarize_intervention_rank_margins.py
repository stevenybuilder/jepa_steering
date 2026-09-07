"""Explain fixed-bank selections from measured costs; no model or dose fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def rank_margins(native, edited):
    if len(native) != len(edited) or len(native) < 2:
        raise ValueError('Aligned costs for at least two candidates required')
    if not all(math.isfinite(v) for v in (*native, *edited)):
        raise ValueError('Finite costs required')
    best = min(range(len(native)), key=native.__getitem__)
    selected = min(range(len(edited)), key=edited.__getitem__)
    changes = [e - b for b, e in zip(native, edited)]
    comparisons = []
    for j in range(len(native)):
        if j == best:
            continue
        gap = native[j] - native[best]
        relative = changes[j] - changes[best]
        remaining = edited[j] - edited[best]
        if not math.isclose(gap + relative, remaining, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Cost-gap decomposition failed')
        comparisons.append(dict(candidate=j, native_gap=gap,
            relative_cost_change=relative, remaining_gap=remaining,
            fraction_of_positive_gap_closed=max(0., -relative) / gap if gap > 0 else None,
            absolute_cost_change_bound=abs(changes[j]) + abs(changes[best])))
    rms = math.sqrt(statistics.mean(v*v for v in changes))
    center = statistics.mean(changes)
    centered_rms = math.sqrt(statistics.mean((v-center)**2 for v in changes))
    fractions = [r['fraction_of_positive_gap_closed'] for r in comparisons
                 if r['fraction_of_positive_gap_closed'] is not None]
    return dict(native_choice=best, edited_choice=selected, choice_changed=best != selected,
        native_minimum_competitor_gap=min(r['native_gap'] for r in comparisons),
        minimum_remaining_gap=min(r['remaining_gap'] for r in comparisons),
        maximum_positive_gap_fraction_closed=max(fractions) if fractions else None,
        mean_cost_shift=center, cost_shift_rms=rms,
        action_contrast_shift_rms=centered_rms,
        contrast_fraction_of_cost_shift=centered_rms/rms if rms else None,
        strict_choice_preservation_bound=all(r['native_gap'] > r['absolute_cost_change_bound']
                                            for r in comparisons),
        competitors=comparisons)


def summarize(paths, baseline_arm='native'):
    states = []
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        if not report.get('complete'):
            raise ValueError('Completed state report required')
        rows = report.get('score', report.get('rows'))
        if not isinstance(rows, list):
            raise ValueError('Per-arm cost rows required')
        native = next(row for row in rows if row['arm'] == baseline_arm)
        base = native['cost_by_horizon'][-1]
        arms = []
        for row in rows:
            result = rank_margins(base, row['cost_by_horizon'][-1])
            if result['edited_choice'] != row['choice']:
                raise ValueError('Recomputed argmin differs from saved choice')
            arms.append(dict(arm=row['arm'], **result))
        key = report.get('key')
        if key is None:
            key = f"episode-{report['episode']:03d}"
        states.append(dict(key=key, source=str(path.resolve()),
            source_sha256=hashlib.sha256(raw).hexdigest(), arms=arms))
    if len({s['key'] for s in states}) != len(states):
        raise ValueError('Repeated state report')
    return dict(complete=True, states=states, independent_states=len(states), baseline_arm=baseline_arm,
        method='Exact terminal-cost gap and relative cost-change decomposition; fixed bank only',
        limitations=['Descriptive after-run diagnosis, not an independent efficacy test',
                     'Bound uses observed absolute cost changes, not an ACPC latent-distance bound',
                     'No linear extrapolation to larger doses or untouched candidate plans',
                     'Dependent plans are not independent episodes'],
        model_calls=0, simulator_calls=0, fitting_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-arm', default='native')
    args = parser.parse_args()
    result = summarize(args.reports, args.baseline_arm)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True, states=len(result['states']), output=str(args.output))))
