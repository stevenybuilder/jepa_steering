"""Locate nonlinear reconstruction gains within versus outside the fitted PCA chart.

Cached exploratory diagnostic only: no additional fitting choices or physical labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from learn_intrinsic_chart import rbf_reconstruct


def error_parts(error, basis):
    projected = error @ basis.T
    within = float(np.mean(np.sum(projected**2, axis=1)))
    total = float(np.mean(np.sum(error**2, axis=1)))
    return dict(total_squared_l2=total, within_chart_squared_l2=within,
                outside_chart_squared_l2=max(0., total-within))


def evaluate(knots):
    if knots.shape != (4, 4, 9, 64) or not np.isfinite(knots).all():
        raise ValueError('Unexpected or nonfinite source')
    rows = []
    for held in range(4):
        train = knots[:, [i for i in range(4) if i != held]].reshape(-1, 64)
        test = knots[:, held].reshape(-1, 64)
        mean = train.mean(0)
        _, _, vt = np.linalg.svd(train-mean, full_matrices=False)
        for rank in (2, 4, 8):
            basis = vt[:rank]
            tr = (train-mean) @ basis.T
            te = (test-mean) @ basis.T
            linear = mean + te @ basis
            nonlinear = rbf_reconstruct(tr, te, train)
            a = error_parts(linear-test, basis)
            b = error_parts(nonlinear-test, basis)
            rows.append(dict(held_direction=held, rank=rank, linear=a, nonlinear=b,
                outside_error_reduction_fraction=1-b['outside_chart_squared_l2']/a['outside_chart_squared_l2'],
                nonlinear_coordinate_distortion_squared_l2=b['within_chart_squared_l2']))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    states = []
    for state in range(4):
        path = args.inputs / f'density-compact-{state}.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads(path.with_suffix('.json').read_text())
        if not receipt.get('complete') or receipt['sha256'] != digest:
            raise ValueError('Source checksum mismatch')
        with np.load(path, allow_pickle=False) as source:
            rows = evaluate(source['knot_coordinates'].astype(np.float64))
        states.append(dict(state=state, source_sha256=digest, rows=rows))
    result = dict(complete=True, independent_states=4, states=states,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_code_sha256=hashlib.sha256(Path(__file__).with_name('learn_intrinsic_chart.py').read_bytes()).hexdigest(),
        scope='Same fixed PCA64 preprojection; direction holdout within each state; no unseen-episode or causal claim.',
        protocol='Ranks 2/4/8 unchanged; same TRAIN-only RBF median bandwidth and ridge .001; error decomposition only.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, states=4, output=str(args.output))))
