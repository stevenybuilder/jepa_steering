#!/usr/bin/env python3
"""Test a mean/radius explanation of action-path bending using cached tensors.

This is an origin-dependent diagnostic, not proof that residuals live on spheres.
No network calls, target-fitted interpolation, or task-performance inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from action_path_geometry import central_estimates


def direction_scale_estimate(donors):
    """Four donors only; affine midpoint fits to mean, radius, and unit direction."""
    if donors.ndim != 3 or donors.shape[0] != 4 or not donors.is_floating_point():
        raise ValueError('Require four [token, channel] floating-point donors')
    if not torch.isfinite(donors).all():
        raise ValueError('Nonfinite donor')
    x = donors.double()
    mean = x.mean(-1, keepdim=True)
    centered = x-mean
    radius = centered.norm(dim=-1, keepdim=True)
    unit = centered/radius.clamp_min(1e-24)
    mean_unit = unit.mean(0)
    unit_length = mean_unit.norm(dim=-1, keepdim=True)
    result = mean.mean(0)+radius.mean(0)*mean_unit/unit_length.clamp_min(1e-24)
    # A zero/cancelling direction does not define a sphere interpolation.
    degenerate = (unit_length < 1e-12) | (radius.min(0).values < 1e-24)
    result = torch.where(degenerate, x.mean(0), result)
    return result.to(donors.dtype), int(degenerate.sum())


def diagnose_path(samples):
    if samples.ndim != 3 or samples.shape[0] != 5 or not torch.isfinite(samples).all():
        raise ValueError('Require five finite [token, channel] samples')
    x = samples.double()
    chord = x[-1]-x[0]
    displacement = x[2]-x[0]
    chord_sq = chord.square().sum()
    normal = displacement-chord*(displacement*chord).sum()/chord_sq.clamp_min(1e-24)
    # Independent orthogonal basis per token: constant channel direction, then
    # the centered true midpoint radius. The midpoint is used for scoring only.
    constant = normal.mean(-1, keepdim=True).expand_as(normal)
    radial = x[2]-x[2].mean(-1, keepdim=True)
    radial = radial/radial.norm(dim=-1, keepdim=True).clamp_min(1e-24)
    radial_component = ((normal-constant)*radial).sum(-1, keepdim=True)*radial
    remaining = normal-constant-radial_component
    energies = [float(v.square().sum()) for v in (constant, radial_component, remaining)]
    total = float(normal.square().sum())
    estimates = central_estimates(samples[[0,1,3,4]])
    spherical, fallback = direction_scale_estimate(samples[[0,1,3,4]])
    estimates['mean_radius_direction_equal_data'] = spherical
    mean = x.mean(-1)
    radius = (x-x.mean(-1,keepdim=True)).norm(dim=-1)
    return dict(
        chord_length=float(chord_sq.sqrt()), central_chord_normal_l2=total**.5,
        normal_energy_fractions=dict(zip(('token_mean','token_radius','remaining_direction'),
            [v/total if total > 1e-24 else None for v in energies])),
        orthogonal_decomposition_error=abs(sum(energies)-total),
        token_radius_relative_range_median=float(((radius.max(0).values-radius.min(0).values)/radius.mean(0).clamp_min(1e-24)).median()),
        token_mean_range_rms=float((mean.max(0).values-mean.min(0).values).square().mean().sqrt()),
        midpoint_mse={name:float((value.double()-x[2]).square().mean()) for name,value in estimates.items()},
        direction_estimate_degenerate_tokens=fallback,
        degenerate_endpoint_chord=bool(chord_sq < 1e-24))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    from complete_cached_geometry import sha256,write_json
    done=json.loads((a.source/'DONE.json').read_text())
    paths=[r for r in done['outputs'] if r['path'].endswith('.pt')]
    if not done['complete'] or len(paths)!=32:
        raise ValueError('Expected complete frozen32path study')
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',dict(source_done_sha256=sha256(a.source/'DONE.json'),
        script_sha256=sha256(__file__),paths=32,fit_samples=[-1,-.5,.5,1],target=0,
        metric='Native full-spatial Euclidean',model_calls=0,simulator_calls=0,
        limitations=['Token mean/radius decomposition depends on the native origin and channel coordinates',
            'Normalization geometry is a motivation, not evidence these prenorm residuals lie on spheres',
            'Midpoint used for scoring/decomposition only; interpolation sees four donors',
            'Four source states, not32independent episodes; descriptive, no task-success claim']))
    torch.set_num_threads(2);start=time.monotonic();rows=[]
    for entry in paths:
        path=a.source/entry['path']
        if sha256(path)!=entry['sha256']:raise ValueError('Source tensor checksum mismatch')
        value=torch.load(path,map_location='cpu',weights_only=False)
        source=value['source']['key']
        if source not in [f'near-dev-{i:03}' for i in range(4)]:raise ValueError('Unexpected source split')
        sites=[(b,h) for b in value['block_indices'] for h in value['horizons']]
        for (block,horizon),samples in zip(sites,value['residual_samples'],strict=True):
            rows.append(dict(source=source,pair=value['path']['pair'],radius=value['path']['radius'],
                block=block,horizon=horizon,**diagnose_path(samples)))
        print(json.dumps(dict(event='path_decomposed',path=path.name)),flush=True)
    groups=defaultdict(list)
    for r in rows:groups[r['block'],r['horizon'],r['radius']].append(r)
    summary=[]
    for (block,horizon,radius),group in sorted(groups.items()):
        per_state=[]
        for state in sorted(set(r['source'] for r in group)):
            rs=[r for r in group if r['source']==state]
            mse={m:sum(r['midpoint_mse'][m] for r in rs)/len(rs) for m in rs[0]['midpoint_mse']}
            fractions={k:sum(r['normal_energy_fractions'][k] or 0 for r in rs)/len(rs) for k in rs[0]['normal_energy_fractions']}
            per_state.append(dict(source=state,midpoint_mse=mse,normal_energy_fractions=fractions))
        summary.append(dict(block=block,horizon=horizon,radius=radius,independent_source_states=4,per_state=per_state,
            midpoint_mse={m:sum(r['midpoint_mse'][m] for r in per_state)/4 for m in per_state[0]['midpoint_mse']},
            normal_energy_fractions={k:sum(r['normal_energy_fractions'][k] for r in per_state)/4 for k in per_state[0]['normal_energy_fractions']}))
    write_json(a.output/'summary.json',dict(complete=True,rows=rows,groups=summary,seconds=time.monotonic()-start))
    write_json(a.output/'DONE.json',dict(complete=True,seconds=time.monotonic()-start,outputs=[dict(path='summary.json',sha256=sha256(a.output/'summary.json'))],model_calls=0))


if __name__=='__main__':main()
