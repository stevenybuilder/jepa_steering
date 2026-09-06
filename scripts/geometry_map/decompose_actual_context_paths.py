#!/usr/bin/env python3
"""Separate physical, encoded-image, and forecast action-path geometry offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from action_path_geometry import sampled_path_geometry
from summarize_action_path_curvature import average, episode_interval


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def checked(path, checksum):
    if sha(path) != checksum: raise ValueError('Checksum failed before tensor loading')
    return torch.load(path, map_location='cpu', weights_only=False)


def physical_spaces(states):
    """Native state is agentXY, blockXY, theta, agent velocityXY."""
    states = torch.as_tensor(states).double()
    if states.ndim != 3 or states.shape[:2] != (5, 7) or states.shape[-1] != 7:
        raise ValueError('Five paths, seven observation times, seven native state fields required')
    if not torch.isfinite(states).all(): raise ValueError('Nonfinite physical state')
    return {'agent_xy_pixels': states[..., :2], 'block_xy_pixels': states[..., 2:4],
            'block_angle_unit_circle': torch.stack([states[..., 4].cos(), states[..., 4].sin()], -1)}


def compact_geometry(samples, unit):
    result = sampled_path_geometry(samples)
    result['coordinate_units'] = unit
    result['metric'] = 'Euclidean in this explicitly named space; no mixing pixels and angles'
    result['max_orthogonal_distance'] = (max(result['chord_orthogonal_distance'])
        if result['chord_orthogonal_distance'] is not None else None)
    return result


def summarize(rows):
    views = []
    for radius in (1, 4):
        for horizon in range(1, 7):
            panel = [r for r in rows if r['radius'] == radius and r['horizon'] == horizon]
            for space in panel[0]['spaces']:
                geometry = [r['spaces'][space] for r in panel]
                metrics = {}
                for field in ('path_length', 'chord_length', 'path_chord_ratio',
                              'max_orthogonal_fraction_of_chord', 'max_orthogonal_distance'):
                    grouped = []
                    counts = []
                    for episode in range(4):
                        values = [r['spaces'][space][field] for r in panel if r['episode'] == episode
                                  and r['spaces'][space][field] is not None]
                        counts.append(len(values))
                        grouped.append(average(values) if values else None)
                    valid = [v for v in grouped if v is not None]
                    stats = episode_interval(valid) if valid else {'mean': None, 'n_initial_states': 0}
                    stats['values_by_original_initial_state'] = grouped
                    stats['valid_directions_by_initial_state'] = counts
                    metrics[field] = stats
                views.append(dict(radius=radius, horizon=horizon, space=space, metrics=metrics,
                    degenerate_chords=sum(g['degenerate_endpoint_chord'] for g in geometry), paths=len(panel)))
    return views


def main():
    p = argparse.ArgumentParser()
    for name in ('input', 'truth_index', 'source_verified', 'output'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); torch.set_num_threads(1)
    done = json.loads((a.input/'DONE.json').read_text())
    protocol = json.loads((a.input/'protocol.json').read_text())
    verification = json.loads(a.source_verified.read_text())
    index = json.loads(a.truth_index.read_text())
    if not all(x['complete'] for x in (done, verification, index)) or done['paths'] != 32:
        raise ValueError('Complete SHA-verified source required')
    if protocol['truth_index_sha256'] != sha(a.truth_index): raise ValueError('Frozen truth index changed')
    entries = index['rows']
    if any(r['episode'] not in range(4) or r['split'] != 'development_external' for r in entries):
        raise ValueError('Only the original four development states before any tensor load')
    lookup = {(r['episode'], r['pair'], float(r['coefficient'])): r for r in entries}
    sources = {r['path']: r for r in done['outputs']}
    expected = {f'near-dev-{e:03}-pair{pair}-radius{radius}.pt'
                for e in range(4) for pair in range(4) for radius in (1, 4)}
    if {name for name in sources if name.endswith('.pt')} != expected: raise ValueError('Frozen path coverage changed')
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []; cache = {}
    for episode in range(4):
        cache.clear()
        for pair in range(4):
            for radius in (1, 4):
                name = f'near-dev-{episode:03}-pair{pair}-radius{radius}.pt'
                value = checked(a.input/name, sources[name]['sha256'])
                state_paths = []
                for position, t in enumerate((-1., -.5, 0., .5, 1.)):
                    entry = lookup[episode, pair, radius*t]; path = entry['path']
                    if path not in cache:
                        physical = checked(path, entry['sha256'])
                        states = (torch.as_tensor(physical['observed_states']) if 'observed_states' in physical
                                  else torch.as_tensor(physical['truth']['states'])[::5])
                        cache[path] = (states.clone(), torch.as_tensor(physical['raw_actions']).clone())
                    states, actions = cache[path]
                    if not torch.equal(actions, value['raw_actions'][position]): raise ValueError('Physical/model actions mismatch')
                    state_paths.append(states)
                spaces = physical_spaces(torch.stack(state_paths))
                for horizon in range(1, 7):
                    geometries = {key: compact_geometry(x[:, horizon], 'pixels' if key.endswith('pixels') else 'unit circle')
                                  for key, x in spaces.items()}
                    for key in ('native_predicted_visual', 'actual_past_predicted_visual', 'actual_visual'):
                        if value[key].shape != (6, 5, 1, 16, 16, 384): raise ValueError('Full spatial feature layout changed')
                        geometries[key] = compact_geometry(value[key][horizon-1], 'raw encoder-feature units')
                    rows.append(dict(episode=episode, pair=pair, radius=radius, horizon=horizon,
                                     real_raw_step=5*horizon, spaces=geometries, actions_exact=True))
                print(json.dumps(dict(event='path_decomposition_complete', episode=episode, pair=pair, radius=radius)), flush=True)
    report = dict(complete=True, rows=rows, views=summarize(rows), initial_states=4, paths=32,
        primary_history_views=[2, 5], source_done_sha256=sha(a.input/'DONE.json'), truth_index_sha256=sha(a.truth_index),
        limitations=['Same four development states; directions, radii and horizons are dependent, descriptive intervals only',
            'Physical pixels, angle-unit-circle and raw feature coordinates are scored separately, not a mixed metric',
            'Unit-circle angle embedding itself curves under rotation; it is not an intrinsic dynamics curvature estimate',
            'Actual encoded image bending can include rendering, encoder nonlinearities and contact dynamics',
            'Teacher-forced forecasts use privileged actual PAST observations only, not prospective information',
            'Finite sampled action-response paths do not establish a semantic manifold, density, safety or steering utility'])
    target = a.output/'DECOMPOSITION.json'; target.write_text(json.dumps(report, indent=2)+'\n')
    receipt = dict(complete=True, outputs=[dict(path=target.name, sha256=sha(target), bytes=target.stat().st_size)],
                   seconds=time.monotonic()-started, gpu_calls=0, simulator_calls=0, source_verified_sha256=sha(a.source_verified))
    (a.output/'DONE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=receipt['seconds'], paths=32, views=len(report['views']))), flush=True)


if __name__ == '__main__': main()
