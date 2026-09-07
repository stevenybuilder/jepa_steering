"""Descriptive action-space geometry of CEM elites, not model uncertainty."""
import math
import argparse
import hashlib
import json
from pathlib import Path


def summarize_elites(elites, delivered_mean, previous_mean=None):
    """Accept flattened normalized action sequences; no physical outcome inputs.

    Participation rank is computed through the small elite Gram matrix. It is
    bounded by k-1, not an estimate of the world's intrinsic dimensionality.
    """
    k = len(elites)
    d = len(delivered_mean)
    if k < 2 or d < 1 or any(len(x) != d for x in elites):
        raise ValueError('At least two aligned nonempty elite vectors required')
    vectors = list(elites) + [delivered_mean]
    if previous_mean is not None:
        if len(previous_mean) != d:
            raise ValueError('Previous mean has incompatible shape')
        vectors.append(previous_mean)
    if any(not math.isfinite(v) for x in vectors for v in x):
        raise ValueError('Nonfinite action')
    mean = [math.fsum(x[j] for x in elites) / k for j in range(d)]
    centered = [[x[j] - mean[j] for j in range(d)] for x in elites]
    sqdist = [math.fsum(v*v for v in x) for x in centered]
    trace = math.fsum(sqdist)
    gram_squared = math.fsum(
        math.fsum(a*b for a, b in zip(x, y)) ** 2
        for x in centered for y in centered)
    variances = [math.fsum(x[j]*x[j] for x in centered)/(k-1)
                 for j in range(d)]
    radius = math.sqrt(trace/k)
    nearest = min(math.sqrt(math.fsum((x[j]-delivered_mean[j])**2
                                    for j in range(d))) for x in elites)
    return {
        'elite_count': k, 'action_sequence_dimensions': d,
        'elite_rms_radius': radius,
        'mean_to_nearest_elite_l2': nearest,
        'mean_to_nearest_elite_over_rms_radius': nearest/radius if radius else None,
        'mean_arithmetic_maxabs_error': max(abs(x-y) for x,y in zip(mean,delivered_mean)),
        'mean_update_l2': (math.sqrt(math.fsum((x-y)**2 for x,y in
                            zip(delivered_mean,previous_mean)))
                           if previous_mean is not None else None),
        'elite_covariance_participation_rank': trace*trace/gram_squared if gram_squared else 0.,
        'participation_rank_upper_bound': min(k-1,d),
        'zero_variance_action_coordinates': sum(v == 0 for v in variances),
        'diagonal_gaussian_entropy_nats': (
            .5*math.fsum(math.log(2*math.pi*math.e*v) for v in variances)
            if all(v > 0 for v in variances) else None),
        'entropy_scope': 'Fitted diagonal action proposal only; undefined if singular; not predictive uncertainty.',
        'interpretation_limit': 'Neither separation from elites nor shrinking spread proves multimodality, off-manifold actions, or bad control.',
    }


def analyze_capture(capture):
    if not capture.get('complete'):
        raise ValueError('Completed capture required')
    seen=set()
    rows=[]
    for row in capture['rows']:
        key=(row['episode'],row['iteration'])
        if key in seen:
            raise ValueError('Repeated state/iteration')
        seen.add(key)
        stats=summarize_elites(row['elites_flat'],row['delivered_mean_flat'],row['previous_mean_flat'])
        if stats['mean_arithmetic_maxabs_error']>2e-6:
            raise ValueError('Delivered mean not consistent with zero-momentum arithmetic mean')
        rows.append(dict(episode=key[0],iteration=key[1],**stats))
    states=sorted(set(row['episode'] for row in rows))
    if states!=sorted(capture['states']):
        raise ValueError('State inventory mismatch')
    for state in states:
        if sorted(r['iteration'] for r in rows if r['episode']==state)!=list(range(1,31)):
            raise ValueError('Incomplete native 30-iteration capture')
    return dict(complete=True,independent_seen_development_states=len(states),rows=rows,
        scope='Descriptive post-run analysis; no physical outcomes supplied; 30 iterations are not independent samples',
        GPU_calls=0,simulator_calls=0,model_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.capture.read_bytes()
    result=analyze_capture(json.loads(raw))
    result['source_sha256']=hashlib.sha256(raw).hexdigest()
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    for row in result['rows']:
        if row['iteration'] in (1,15,30):
            print(json.dumps({k:row[k] for k in ['episode','iteration','elite_rms_radius',
                'elite_covariance_participation_rank','mean_to_nearest_elite_over_rms_radius',
                'diagonal_gaussian_entropy_nats']}))
