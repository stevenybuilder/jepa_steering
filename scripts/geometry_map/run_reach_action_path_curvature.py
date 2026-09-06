#!/usr/bin/env python3
"""Seen-development Reach replication; NEW model-only paths, cached central truth.

The Push geometry/hook implementations are imported unchanged. Radii are raw
Reach XYZ units, not quantitatively matched to Push perturbations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from action_path_geometry import central_estimates, sampled_path_geometry
from run_action_path_curvature import (BLOCKS, HORIZONS, METHODS, BATCH, TIMES,
    ResidualCapture, ResidualReplacement, tensor_mse_by_horizon, interpolation_diagnostics)
from capture_horizon_coordinates import exact, tensor_hash
from complete_cached_geometry import sha256, write_json

EPISODES = (0, 1, 4, 7)
CHECKPOINT = 'c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'


def frozen_directions():
    """Four shared, reproducible orthogonal 18D directions; no outcomes used."""
    q, r = np.linalg.qr(np.random.default_rng(90506).normal(size=(18, 4)), mode='reduced')
    q *= np.where(np.diag(r) < 0, -1., 1.)[None]
    return torch.from_numpy((q.T.reshape(4, 6, 3) * (.05*np.sqrt(18))).copy()).float()


def fixed_reach_path(center, directions, pair, radius):
    if center.shape != (30, 4) or directions.shape != (4, 6, 3) or pair not in range(4) or radius not in (1, 4):
        raise ValueError('Only frozen H6 four XYZ directions and radii1/4')
    delta = torch.zeros_like(center)
    delta[:, :3] = directions[pair].repeat_interleave(5, dim=0).to(center)
    raw = torch.stack([center + (radius*t)*delta for t in TIMES])
    exact(raw[2], center, 'unchanged raw native midpoint')
    exact(raw[:, :, 3], center[None, :, 3].expand(5, -1), 'unchanged gripper')
    return raw, dict(pair=pair, radius=radius, xyz_rms_target=.05*radius,
        xyz_direction_rms=float(delta[:, :3].square().mean().sqrt()),
        raw_direction_l2=float(delta.norm()), gripper_unchanged_exact=True,
        raw_xyz_outside_unit_box_counts=(raw[:, :, :3].abs()>1).sum((1,2)).tolist(),
        raw_xyz_max_absolute=raw[:, :, :3].abs().flatten(1).max(1).values.tolist(),
        model_action_clipping=False, simulator_calls=0,
        nominal_radius_not_quantitatively_matched_to_push=True)


def checked_rows(manifest):
    rows = manifest.get('sources', [])
    if not manifest.get('complete') or [r.get('episode') for r in rows] != list(EPISODES):
        raise ValueError('Exactly predeclared seen development0/1/4/7 required before tensor access')
    if any(r.get('split') != 'development' or r.get('sealed', True) for r in rows):
        raise ValueError('Held or unknown sources refused before tensor access')
    return rows


def check_original_goal(encoded, raw, bank):
    """Respect both immutable reference schemas, without inventing old FP32 data."""
    checks=bank['initial_checks']
    hashes={k:tensor_hash(encoded[k]) for k in ('visual','proprio')}
    if 'fresh_goal_sha256' in checks:
        if hashes != checks['fresh_goal_sha256']:
            raise ValueError('Original fullprecision goal encoding SHA changed')
        for k in raw:
            if tensor_hash(raw[k]) != checks['original_raw_goal_sha256'][k]:
                raise ValueError('Original raw goal SHA changed')
        mode='original fullprecision and raw goal SHA exact'
    else:
        if checks.get('goal_pixels') != 0 or checks.get('goal_proprio') != 0:
            raise ValueError('Legacy reference lacks exact raw goal identity')
        for k in raw:
            saved=bank['goal']['encoded_'+k]
            exact(encoded[k].cpu().to(saved.dtype),saved,'legacy original cache-dtype goal '+k)
        mode='legacy original cache-dtype equality exact; original FP32 hash unavailable, fresh FP32 frozen across new conditions'
    return hashes,mode


@torch.no_grad()
def run_source(args, row, wm, prep, cfg, directions, global_start):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from intervene_head_spatial_mean import batch_context
    source = args.inputs/row['path']
    if sha256(source) != row['sha256']:
        raise ValueError('Reference SHA mismatch before torch.load')
    bank = torch.load(source, map_location='cpu', weights_only=False)
    if bank['episode'] != row['episode'] or bank['split'] != 'development' or not bank['fresh30action_physics_and_pixels_repeat_exact']:
        raise ValueError('Reference identity/split/physical exactness failed')
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=prep)
    raw_goal = {k: bank['goal'][k].clone() for k in ('visual', 'proprio')}
    agent.set_goal(TensorDict(raw_goal, batch_size=[]))
    goal_hashes,goal_contract = check_original_goal(agent.goal_state_enc,raw_goal,bank)
    z = batch_context(bank['context'], BATCH).to(agent.device)
    central_plan = bank['normalized_actions'][:, None].repeat(1, BATCH, 1).contiguous().to(agent.device)
    central_saved = central_plan.clone()
    context_hashes = {k: tensor_hash(z[k]) for k in z.keys()}
    blocks = wm.model.predictor.predictor_blocks
    baseline = wm.unroll(z.clone(), act_suffix=central_plan)
    with ResidualCapture(blocks) as central_capture:
        central = wm.unroll(z.clone(), act_suffix=central_plan)
    for k in ('visual', 'proprio'):
        exact(baseline[k], central[k], 'read-only native capture '+k)
    base_visual = baseline['visual'][1:].float().cpu()
    actual_visual = torch.cat(bank['truth']['encoded_visual'], 0).reshape_as(base_visual[:, 0])
    actual = actual_visual[:, None].expand_as(base_visual)
    baseline_error = tensor_mse_by_horizon(base_visual, actual)
    source_meta = dict(episode=row['episode'], key=row['key'], split='development', already_seen=True,
        source_path=str(source), source_sha256=row['sha256'], context_sha256=context_hashes,
        effective_fresh_goal_sha256=goal_hashes, original_goal_contract=goal_contract,original_context_and_goal_unchanged=True,
        native_repeat_and_capture_exact=True, batch_shape=BATCH,
        historical_single_vs_batch_visual_maxabs=float((base_visual[:,0]-bank['predictions']['visual']).abs().max()),
        central_physical_future_existing=True, new_path_physical_futures_observed=False,
        context_raw_step=int(bank['arrays']['context_real_raw_step'][0]), raw_offsets=[5,10,15,20,25,30])
    outputs = []
    for pair in range(4):
        for radius in (1, 4):
            if time.monotonic()-global_start > 1800:
                raise RuntimeError('Authorized process budget reached before next path')
            start = time.monotonic()
            raw, path_meta = fixed_reach_path(bank['raw_actions'], directions, pair, radius)
            normalized = prep.normalize_actions(raw.reshape(5, 6, 5, 4)).reshape(5, 6, 20)
            # Raw->normalized round-trip may differ by one ULP; all conditions
            # instead retain the ORIGINAL normalized central native commands.
            central_rounding = float((normalized[2]-bank['normalized_actions']).abs().max())
            normalized_delta = (prep.normalize_actions((bank['raw_actions']+torch.cat([
                directions[pair].repeat_interleave(5,0), torch.zeros(30,1)],dim=1)).reshape(6,5,4)).reshape(6,20)
                - prep.normalize_actions(bank['raw_actions'].reshape(6,5,4)).reshape(6,20))
            normalized = torch.stack([bank['normalized_actions']+(radius*t)*normalized_delta for t in TIMES])
            exact(normalized[2], bank['normalized_actions'], 'exact original normalized midpoint')
            effective = prep.denormalize_actions(normalized.reshape(5,30,4))
            tolerance = 16*torch.finfo(normalized.dtype).eps*max(1.,float(raw.abs().max()),float(normalized.abs().max()))
            torch.testing.assert_close(effective, raw, rtol=0, atol=tolerance)
            path_meta.update(normalization_raw_roundtrip_maxabs=float((effective-raw).abs().max()),
                original_center_normalize_roundtrip_maxabs=central_rounding, roundoff_tolerance=tolerance,
                normalized_path_policy='original native normalized center plus affine normalized direction; raw roundtrip checked')
            padded = torch.cat([normalized, normalized[2:3].repeat(BATCH-5,1,1)])
            path_plan = padded.transpose(0,1).contiguous().to(agent.device)
            path_saved = path_plan.clone()
            with ResidualCapture(blocks) as sampled:
                natural = wm.unroll(z.clone(), act_suffix=path_plan)
            for k in ('visual', 'proprio'):
                exact(natural[k][:,2], baseline[k][:,2], 'same-batch native midpoint '+k)
            site_rows, patched_visual, patched_proprio, replacements = [], [], [], []
            for block in BLOCKS:
                for horizon in HORIZONS:
                    samples = sampled.values[block,horizon][:5]
                    exact(samples[2], central_capture.values[block,horizon][2], 'held midpoint residual')
                    noncentral = samples[[0,1,3,4]]
                    estimates = central_estimates(noncentral)
                    replacement = torch.stack([central_capture.values[block,horizon][0]]+[estimates[m] for m in METHODS])
                    with ResidualReplacement(blocks[block], horizon, replacement.to(agent.device)):
                        changed = wm.unroll(z.clone(), act_suffix=central_plan)
                    for k in ('visual', 'proprio'):
                        exact(changed[k][:,0], baseline[k][:,0], 'exact self replacement '+k)
                        exact(changed[k][:horizon], baseline[k][:horizon], 'unchanged prepatch predictions '+k)
                    visual = changed['visual'][1:].float().cpu()
                    native_mse = tensor_mse_by_horizon(visual, base_visual)
                    actual_mse = tensor_mse_by_horizon(visual, actual)
                    cost = agent.objective(changed, central_plan, keepdims=True).float().cpu()
                    methods = []
                    for i, name in enumerate(METHODS, 1):
                        methods.append(dict(method=name, **interpolation_diagnostics(estimates[name],noncentral,samples[2]),
                            downstream_native_fullspatial_mse_by_horizon=native_mse[:,i].tolist(),
                            downstream_native_fullspatial_mse_after_patch=float(native_mse[horizon-1:,i].mean()),
                            actual_future_fullspatial_mse_by_horizon=actual_mse[:,i].tolist(),
                            actual_future_mse_delta_vs_native_by_horizon=(actual_mse[:,i]-baseline_error[:,i]).tolist(),
                            actual_future_mse_delta_after_patch=float((actual_mse[horizon-1:,i]-baseline_error[horizon-1:,i]).mean()),
                            native_goal_cost_by_horizon=cost[:,i].reshape(-1).tolist()))
                    site_rows.append(dict(block=block,intervention_horizon=horizon,self_patch_exact=True,
                        geometry=sampled_path_geometry(samples),methods=methods))
                    patched_visual.append(visual)
                    patched_proprio.append(changed['proprio'][1:].float().cpu())
                    replacements.append(replacement)
            exact(central_plan, central_saved, 'unchanged native commands')
            exact(path_plan, path_saved, 'unchanged sampled commands')
            if context_hashes != {k:tensor_hash(z[k]) for k in z.keys()} or goal_hashes != {k:tensor_hash(agent.goal_state_enc[k]) for k in raw_goal}:
                raise ValueError('Context/goal mutated')
            pt = args.output/f"reach-dev-{row['episode']:03d}-pair{pair}-radius{radius}.pt"
            torch.save(dict(complete=True,source=source_meta,path=path_meta,raw_path_actions=raw,
                normalized_path_actions=normalized,sample_t=TIMES,block_indices=BLOCKS,horizons=HORIZONS,
                residual_samples=torch.stack([sampled.values[b,h][:5] for b in BLOCKS for h in HORIZONS]),
                central_estimates_and_self=torch.stack(replacements),
                natural_predicted_visual=natural['visual'][1:].float().cpu(),
                patched_predicted_visual=torch.stack(patched_visual),patched_predicted_proprio=torch.stack(patched_proprio),
                native_central_predicted_visual=base_visual,actual_central_visual=actual_visual,
                condition_order=['exact_self_patch']+list(METHODS)),pt)
            torch.cuda.synchronize(); seconds=time.monotonic()-start
            report = pt.with_suffix('.json')
            write_json(report,dict(complete=True,source=source_meta,path=path_meta,rows=site_rows,
                baseline_actual_fullspatial_mse_by_horizon=baseline_error[:,0].tolist(),seconds=seconds,
                full_tensor_sha256=sha256(pt),full_tensor_bytes=pt.stat().st_size,
                limitations=['Four already-seen source states; directions/horizons dependent, not32 robot episodes',
                    'Four noncentral samples only for every fit; true midpoint scoring/self-control only',
                    'Raw full-spatial geometry, no pooling/PCA/whitening or manifold-density inference',
                    'New paths model-only; central exact30-action physical truth reused from earlier capture',
                    'Native prediction restoration and actual encoded future accuracy are distinct',
                    'Cubic central interpolation equals four-point quadratic LS; no cubic-specific claim',
                    'Raw XYZ path unprojected/no clipping; differs from effective simulator controls and Push radius units']))
            outputs.extend(dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,key=row['key']) for p in (pt,report))
            print(json.dumps(dict(event='reach_path_complete',episode=row['episode'],pair=pair,radius=radius,seconds=seconds)),flush=True)
            if row['episode']==0 and pair==0 and radius==1:
                estimate=seconds*32
                write_json(args.output/'CANARY.json',dict(complete=True,guards_passed=True,seconds_per_path=seconds,
                    estimated_32path_seconds=estimate,budget_seconds=1800,within_budget=estimate<=1800,
                    native_midpoint_self_prefix_context_goal_exact=True,outcomes_not_used_for_gate=True,
                    fullspatial_sample_shape=list(samples.shape),native_visual_shape=list(base_visual.shape)))
                if estimate>1800: raise RuntimeError('Timing-only canary exceeds authorized budget')
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','manifest','repo','config','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();rows=checked_rows(json.loads(args.manifest.read_text()))
    args.output.mkdir(parents=True,exist_ok=False)
    directions=frozen_directions();np.save(args.output/'frozen_xyz_directions.npy',directions.numpy(),allow_pickle=False)
    protocol=dict(episodes=EPISODES,sources=rows,scope='qualitative cross-task response-geometry replication on seen development',
        seed=90506,orthogonal_directions_shape=[4,6,3],shared_directions_across_states=True,
        directions_sha256=sha256(args.output/'frozen_xyz_directions.npy'),raw_xyz_rms_at_radius1=.05,
        radii=[1,4],fixed_t=TIMES,gripper_unchanged=True,repeat_raw_steps_per_chunk=5,model_action_clipping=False,
        blocks=BLOCKS,horizons=HORIZONS,methods=METHODS,fit_times=[-1,-.5,.5,1],batch_shape=BATCH,
        primary_exploratory_view='P3/H3, both radii; other sites secondary',statistical_unit='four source states',
        simulator_calls=0,cem_calls=0,confirmation_access=False,max_process_gpu_seconds=1800,
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('run_reach_action_path_curvature.py','run_action_path_curvature.py','action_path_geometry.py')})
    write_json(args.output/'protocol.json',protocol)
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from causal_planner_forks import setup_cfg
    torch.set_num_threads(2);torch.manual_seed(90506);start=time.monotonic()
    try:
        wm,prep,provenance=load_headless_metaworld(args.repo)
        if sha256(provenance['checkpoint'])!=CHECKPOINT:raise ValueError('Pinned Reach checkpoint mismatch')
        wm.eval().requires_grad_(False);before=parameters_sha(wm)
        cfg=setup_cfg(args.config,args.output,wm);outputs=[]
        for row in rows:outputs.extend(run_source(args,row,wm,prep,cfg,directions,start))
        if parameters_sha(wm)!=before:raise ValueError('Frozen model parameters changed')
        for name in ('protocol.json','CANARY.json','frozen_xyz_directions.npy'):
            path=args.output/name;outputs.append(dict(path=name,sha256=sha256(path),bytes=path.stat().st_size))
        write_json(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,
            paths=32,states=4,sites_per_path=9,methods=METHODS,checkpoint_sha256=CHECKPOINT,
            parameter_sha256=before,weights_unchanged=True,provenance=provenance,cem_calls=0,simulator_calls=0))
    except Exception as exc:
        write_json(args.output/'FAILED.json',dict(complete=False,error=repr(exc),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':main()
