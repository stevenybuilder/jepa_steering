#!/usr/bin/env python3
"""Compare partial residual imports to a native forecast of matching hybrid actions."""
import argparse
import json
from pathlib import Path
import time
import torch
from decompose_actual_context_paths import checked, sha
from summarize_action_path_curvature import average, episode_interval


def horizon_mse(prediction, reference):
    if prediction.shape != reference.shape or prediction.shape != (6, 1, 16, 16, 384):
        raise ValueError('Matching full-spatial six-future tensors required')
    return (prediction.float()-reference.float()).square().flatten(1).mean(1)


def receipt(root):
    result = json.loads((root/'DONE.json').read_text())
    if not result['complete']: raise ValueError('Incomplete source before load')
    return result, {r['path']: r for r in result['outputs']}


def main():
    p = argparse.ArgumentParser()
    for name in ('interior', 'native', 'hybrid', 'donor', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); torch.set_num_threads(1)
    receipts = {key: receipt(getattr(a, key)) for key in ('interior', 'native', 'hybrid', 'donor')}
    native_done, native_rows = receipts['native']
    if native_done.get('plans') != 128 or native_done.get('held_access') is not False:
        raise ValueError('Only the predeclared development hybrid reference')
    expected = {f'near-dev-{e:03}-pair{pair}-radius{radius}-t{sign}025-H{h}.pt'
                for e in range(4) for pair in range(4) for radius in (1, 4)
                for sign in ('minus', 'plus') for h in (1, 3)}
    if {key for key in native_rows if key.endswith('.pt')} != expected: raise ValueError('Native coverage changed')
    def load(key, filename):
        return checked(getattr(a, key)/filename, receipts[key][1][filename]['sha256'])
    a.output.mkdir(parents=True, exist_ok=False); rows = []
    for episode in range(4):
        for pair in range(4):
            for radius in (1, 4):
                for sign, t in (('minus', -.25), ('plus', .25)):
                    stem = f'near-dev-{episode:03}-pair{pair}-radius{radius}'
                    model = load('interior', stem+f'-{sign}025.pt')
                    donor = load('donor', stem+f'-t{sign}025.pt')
                    for j, h in enumerate((1, 3, 6)):
                        name = stem+f'-t{sign}025-H{h}.pt'
                        truth = donor if h == 6 else load('hybrid', name)
                        native = None if h == 6 else load('native', name)
                        forecast = model['natural_donor_visual'] if h == 6 else native['predicted_visual']
                        if h != 6:
                            if native['hybrid_truth_sha256'] != receipts['hybrid'][1][name]['sha256']:
                                raise ValueError('Native model/physical source mismatch')
                            if not torch.equal(native['raw_actions'], truth['raw_actions']): raise ValueError('Native/physics actions mismatch')
                        if not torch.equal(model['raw_target_actions'][:5*h], truth['raw_actions'][:5*h]):
                            raise ValueError('Imported donor prefix action mismatch')
                        physical = truth['actual_visual']
                        error = horizon_mse(forecast, physical)
                        conditions = {}
                        for i, condition in enumerate(model['conditions']):
                            pred = model['patched_visual'][j, :, i]
                            actual_error = horizon_mse(pred, physical)
                            native_difference = horizon_mse(pred, forecast)
                            conditions[condition] = dict(actual_hybrid_mse_by_horizon=actual_error.tolist(),
                                actual_hybrid_mse_after_patch=float(actual_error[h-1:].mean()),
                                native_hybrid_forecast_mse_by_horizon=native_difference.tolist(),
                                native_hybrid_forecast_mse_after_patch=float(native_difference[h-1:].mean()))
                        rows.append(dict(episode=episode, pair=pair, radius=radius, t=t, patch_horizon=h,
                            native_hybrid_actual_mse_by_horizon=error.tolist(),
                            native_hybrid_actual_mse_after_patch=float(error[h-1:].mean()), conditions=conditions,
                            source_native='existing full donor forecast' if h == 6 else receipts['native'][1][name]['sha256'],
                            source_truth=(receipts['donor'][1][stem+f'-t{sign}025.pt'] if h == 6
                                          else receipts['hybrid'][1][name])['sha256']))
    views = []
    for radius in (1, 4):
        for h in (1, 3, 6):
            panel = [r for r in rows if r['radius'] == radius and r['patch_horizon'] == h]
            def grouped(fn): return [average(fn(r) for r in panel if r['episode'] == e) for e in range(4)]
            native = episode_interval(grouped(lambda r: r['native_hybrid_actual_mse_after_patch']))
            conditions = {}
            for name in panel[0]['conditions']:
                conditions[name] = {metric: episode_interval(grouped(lambda r, metric=metric, name=name:
                    r['conditions'][name][metric])) for metric in
                    ('actual_hybrid_mse_after_patch', 'native_hybrid_forecast_mse_after_patch')}
                conditions[name]['actual_error_minus_native_hybrid'] = episode_interval(grouped(lambda r, name=name:
                    r['conditions'][name]['actual_hybrid_mse_after_patch']-r['native_hybrid_actual_mse_after_patch']))
            views.append(dict(radius=radius, horizon=h, native_hybrid_actual_mse_after_patch=native, conditions=conditions))
    result = dict(complete=True, targets=64, cases=192, initial_states=4, rows=rows, views=views,
        limitations=['The native reference executes the full donor-prefix/central-suffix action plan in imagination; no residual import',
            'A donor latest-token P3 import does not replace full visual/proprio history or remaining central AdaLN action conditions',
            'Native hybrid and imported-recipient errors share exactly matched physical hybrid truth, but differ in internal conditioning',
            'Full spatial encoded-image fidelity only, not observed policy success or a deployable privileged donor intervention',
            'H6 native reference reuses existing full donor forecast; all four development states and dependent radii/directions retained'])
    target = a.output/'JOIN.json'; target.write_text(json.dumps(result, indent=2)+'\n')
    (a.output/'DONE.json').write_text(json.dumps(dict(complete=True, seconds=time.monotonic()-started,
        outputs=[dict(path=target.name, sha256=sha(target), bytes=target.stat().st_size)],
        source_done_sha256={k: sha(getattr(a, k)/'DONE.json') for k in receipts}, gpu_calls=0, simulator_calls=0), indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=time.monotonic()-started, targets=64, cases=192)), flush=True)


if __name__ == '__main__': main()
