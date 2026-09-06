"""Verified terminal forecast-error data; never load sealed sources."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verified_load(root, row):
    p = root / row['path']
    if sha(p) != row['sha256']:
        raise ValueError('Source checksum mismatch: ' + str(p))
    return torch.load(p, map_location='cpu', weights_only=False)


def visual(value):
    a = torch.as_tensor(value).float().numpy().reshape(-1)
    if a.size != 98304 or not np.isfinite(a).all():
        raise ValueError('Expected finite native 256x384 visual')
    return a


def feature(pred, context, actions):
    current = torch.as_tensor(context['visual']).float().numpy().reshape(-1, 384).mean(0)
    action = torch.as_tensor(actions).float().numpy().reshape(-1)
    if action.size != 60:
        raise ValueError('Native 6x10 action feature required')
    return np.concatenate([current, pred.reshape(256,384).mean(0), action]).astype(np.float32)


def finish(out, arrays, receipt, started):
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'data.npz'
    np.savez_compressed(path, **{k: np.asarray(v) for k,v in arrays.items()})
    receipt.update(complete=True, seconds=time.monotonic()-started, gpu_used=False,
                   outputs=[dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size)])
    (out / 'DONE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


def coarse(root, out):
    started=time.monotonic()
    done=json.loads((root/'DONE.json').read_text())
    if not done.get('complete'):
        raise ValueError('Incomplete source')
    selected=[r for r in done['outputs'] if r['path'].endswith('.pt') and r.get('split')=='fit']
    if not selected:
        raise ValueError('No FIT entries; do not open development or sealed tensors')
    a={k:[] for k in ['x','residual','predicted','goal','group','candidate','native_cost','physical_state','goal_state','raw_action_norm']}
    for row in selected:
        b=verified_load(root,row)
        if b['row']['split']!='fit' or not b['same_state_actions_and_two_physical_replays_exact']:
            raise ValueError('FIT/replay contract')
        g=visual(b['goal_encoded']['visual'])
        for i,c in enumerate(b['candidates']):
            p=visual(c['predicted_visual'][-1]);truth=visual(c['actual_visual'][-1])
            a['x'].append(feature(p,b['context'],b['normalized_actions'][i]))
            a['residual'].append(truth-p);a['predicted'].append(p);a['goal'].append(g)
            a['group'].append(row['source_id']);a['candidate'].append(i)
            a['native_cost'].append(float(torch.as_tensor(c['native_cost_by_horizon']).flatten()[-1]))
            a['physical_state'].append(c['states'][-1]);a['goal_state'].append(b['goal_state'])
            a['raw_action_norm'].append(float(b['raw_actions'][i].square().mean().sqrt()))
    finish(out,a,dict(kind='fit',groups=[r['source_id'] for r in selected],
        source_root=str(root),source_done_sha256=sha(root/'DONE.json'),sources=selected,
        split_checked_before_tensor_load=True,features='current visual mean384 + native H6 visual mean384 + normalized60 actions',
        target='actual encoded H6 full visual minus native forecast,98304 coordinates',rows=len(a['x'])),started)


def expanded(root,out):
    started=time.monotonic();done=json.loads((root/'DONE.json').read_text())
    if not done['complete'] or done['plans']!=64 or done['episode'] not in range(4) or done['held_access']:
        raise ValueError('Only four previously seen DEVELOPMENT banks')
    entries={r['path']:r for r in done['outputs']}
    frozen=verified_load(root,entries['FROZEN_INPUTS.pt'])
    summary=json.loads((root/'summary.json').read_text())
    if sha(root/'summary.json')!=entries['summary.json']['sha256'] or summary['native_objective_sum_all_diffs'] or summary['native_objective_alpha']!=.1:
        raise ValueError('Native terminal alpha.1 contract')
    a={k:[] for k in ['x','residual','predicted','goal','group','candidate','native_cost','physical_state','goal_state','raw_action_norm']}
    goal=visual(frozen['goal_encoded']['visual'])
    for i in range(64):
        b=verified_load(root,entries[f'candidate-{i:03d}.pt'])
        if b['episode']!=done['episode'] or b['split']!='development_external':
            raise ValueError('Development grouping')
        p=visual(b['predicted_visual'][-1]);truth=visual(b['actual_visual'][-1])
        a['x'].append(feature(p,frozen['original_context'],b['normalized_actions']))
        a['residual'].append(truth-p);a['predicted'].append(p);a['goal'].append(goal)
        a['group'].append(done['episode']);a['candidate'].append(i)
        a['native_cost'].append(summary['rows'][i]['native_full_objective_cost'])
        a['physical_state'].append(b['truth']['states'][-1]);a['goal_state'].append(frozen['goal_state'])
        a['raw_action_norm'].append(float(b['raw_actions'].square().mean().sqrt()))
    finish(out,a,dict(kind='development',episode=done['episode'],source_root=str(root),
        source_done_sha256=sha(root/'DONE.json'),rows=64,
        split_checked_before_tensor_load=True,actual_residual_for_scoring_only=True),started)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['coarse','expanded'])
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    globals()[args.mode](args.source,args.output)
