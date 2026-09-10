#!/usr/bin/env python3
"""Fixed natural action paths and held-midpoint causal interpolation diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from action_path_geometry import central_estimates, sampled_path_geometry

BLOCKS = (1, 3, 5)
HORIZONS = (1, 3, 6)
METHODS = ('endpoint_chord', 'near_chord', 'linear_equal_data', 'cubic_equal_data', 'reparameterized_chord', 'reflected_curvature')
BATCH = 1+len(METHODS)
TIMES = (-1., -.5, 0., .5, 1.)


def fixed_action_path(raw_actions, pair, radius):
    if raw_actions.shape != (9, 30, 2) or pair not in range(4) or radius not in (1, 4):
        raise ValueError('Only four frozen near-plan antithetic pairs and radii1/4')
    center = raw_actions[0]
    delta = raw_actions[1+2*pair]-center
    negative_delta = raw_actions[2+2*pair]-center
    tolerance = 4*torch.finfo(raw_actions.dtype).eps*max(1., float(raw_actions.abs().max()))
    if float((delta+negative_delta).abs().max()) > tolerance:
        raise ValueError('Source pair is not antithetic within source float rounding')
    path = torch.stack([center+(radius*t)*delta for t in TIMES])
    if not torch.equal(path[2], center):
        raise ValueError('Central raw command changed')
    return path, dict(pair=pair, radius=radius, raw_delta_l2=float(delta.norm()),
                     raw_delta_rms=float(delta.square().mean().sqrt()),
                     antithetic_rounding_maxabs=float((delta+negative_delta).abs().max()),
                     antithetic_tolerance=tolerance, clipping_applied=False)


def replace_newest(output, replacement):
    if not isinstance(output, torch.Tensor):
        raise TypeError('Pinned predictor block must return a tensor')
    if output.ndim != 3 or output.shape[1] < 256 or replacement.shape != (output.shape[0], 256, output.shape[2]):
        raise ValueError('Wrong native newest256 spatial replacement axes')
    result = output.clone()
    result[:, -256:] = replacement.to(output)
    return result


class ResidualCapture:
    def __init__(self, blocks):
        self.blocks = blocks; self.handles = []; self.calls = {i: 0 for i in BLOCKS}; self.values = {}
    def __enter__(self):
        for i in BLOCKS:
            self.handles.append(self.blocks[i].register_forward_hook(lambda m, a, o, i=i: self.capture(i, o)))
        return self
    def capture(self, block, out):
        self.calls[block] += 1
        h = self.calls[block]
        if not isinstance(out, torch.Tensor) or out.shape[-1] != 400:
            raise ValueError('Pinned predictor block residual layout changed')
        if h in HORIZONS:
            self.values[block, h] = out[:, -256:].detach().float().cpu().clone()
    def __exit__(self, *exc):
        for handle in self.handles: handle.remove()
        if exc[0] is None and list(self.calls.values()) != [6, 6, 6]:
            raise ValueError('Native H6 capture call mismatch')


class ResidualReplacement:
    def __init__(self, block, horizon, replacement):
        self.block = block; self.horizon = horizon; self.replacement = replacement; self.calls = 0; self.fired = 0
    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.patch); return self
    def patch(self, module, args, output):
        self.calls += 1
        if self.calls == self.horizon:
            self.fired += 1
            return replace_newest(output, self.replacement)
        return None
    def __exit__(self, *exc):
        self.handle.remove()
        if exc[0] is None and (self.calls != 6 or self.fired != 1):
            raise ValueError('Replacement must affect exactly one of six imagined steps')


def tensor_mse_by_horizon(x, y):
    # Native future tensors [H,batch,view,16,16,384]; retain H and condition.
    return (x.float()-y.float()).square().flatten(2).mean(-1)


def interpolation_diagnostics(estimate, noncentral, actual_midpoint):
    x = noncentral.double().flatten(1); e = estimate.double().flatten(); m = actual_midpoint.double().flatten()
    low, high = x.min(0).values, x.max(0).values
    excess = torch.maximum(low-e, e-high).clamp_min(0)
    center = x.mean(0); support = (x-center).norm(dim=1).max()
    return dict(midpoint_l2_error=float((e-m).norm()), midpoint_relative_l2_error=float((e-m).norm()/m.norm().clamp_min(1e-20)),
        estimate_l2_norm=float(e.norm()), actual_midpoint_l2_norm=float(m.norm()),
        estimate_offset_from_noncentral_mean=float((e-center).norm()),
        maximum_sample_offset_from_noncentral_mean=float(support),
        estimate_offset_over_sample_radius=float((e-center).norm()/support.clamp_min(1e-20)),
        coordinate_box_overshoot_l2=float(excess.norm()), coordinate_box_overshoot_fraction=float((excess>0).double().mean()))


@torch.no_grad()
def run_source(args, row, wm, prep, cfg, first_canary):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from capture_horizon_coordinates import exact
    from causal_response_transfer import restore_fullprecision_goal
    from complete_cached_geometry import sha256, write_json
    from intervene_head_spatial_mean import batch_context
    source = args.bank/row['path']
    if sha256(source) != row['sha256']: raise ValueError('Source hash failed before tensor loading')
    bank = torch.load(source, map_location='cpu', weights_only=False)
    if bank['row']['split'] != 'development_external': raise ValueError('Only original development sources')
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=prep)
    agent.set_goal(TensorDict(bank['raw_goal'], batch_size=[]))
    goal_diff = restore_fullprecision_goal(agent, bank['goal_encoded'])
    z = batch_context(bank['context'], BATCH).to(agent.device)
    central_plan = bank['normalized_actions'][0, :, None].repeat(1, BATCH, 1).to(agent.device)
    central_saved = central_plan.clone()
    blocks = wm.model.predictor.predictor_blocks
    baseline = wm.unroll(z.clone(), act_suffix=central_plan)
    with ResidualCapture(blocks) as central_capture:
        central = wm.unroll(z.clone(), act_suffix=central_plan)
    for k in ('visual', 'proprio'): exact(baseline[k], central[k], 'read-only capture repeat '+k)
    base_visual = baseline['visual'][1:].float().cpu()
    actual = bank['candidates'][0]['actual_visual'][:, None].expand_as(base_visual)
    base_actual_error = tensor_mse_by_horizon(base_visual, actual)
    historical = bank['candidates'][0]['predicted_visual']
    source_metadata = dict(key=row['key'], split=row['split'], source_path=str(source), source_sha256=row['sha256'],
        fullprecision_goal_restored_exact=True, fresh_goal_encoding_difference=goal_diff,
        historical_single_vs_batch_visual_maxabs=float((base_visual[:, 0]-historical).abs().max()),
        native_repeat_and_capture_exact=True, batch_shape=BATCH, actual_central_future_observed=True,
        other_new_path_futures_observed=False, context_raw_step=0, raw_offsets=[5,10,15,20,25,30])
    entries = []
    for pair in range(4):
        for radius in (1, 4):
            start = time.monotonic()
            raw, path_meta = fixed_action_path(bank['raw_actions'], pair, radius)
            normalized = prep.normalize_actions(raw.reshape(5, 6, 5, 2)).reshape(5, 6, 10)
            exact(normalized[2], bank['normalized_actions'][0], 'native central action normalization')
            effective = prep.denormalize_actions(normalized.reshape(5,30,2))
            path_meta['native_denormalized_raw_maxabs'] = float((effective-raw).abs().max())
            affine = torch.stack([normalized[2]+t*(normalized[4]-normalized[2]) for t in TIMES])
            tolerance=8*torch.finfo(normalized.dtype).eps*max(1.,float(normalized.abs().max()))
            path_meta['normalized_affine_path_maxabs']=float((affine-normalized).abs().max())
            path_meta['normalized_affine_roundoff_tolerance']=tolerance
            if path_meta['normalized_affine_path_maxabs']>tolerance: raise ValueError('Action normalization bends the proposed straight action path')
            torch.testing.assert_close(effective,raw,rtol=0,atol=tolerance)
            padded_normalized=torch.cat([normalized,normalized[2:3].repeat(BATCH-5,1,1)])
            path_plan = padded_normalized.transpose(0, 1).contiguous().to(agent.device); path_saved = path_plan.clone()
            with ResidualCapture(blocks) as sampled:
                natural = wm.unroll(z.clone(), act_suffix=path_plan)
            for k in ('visual', 'proprio'):
                exact(natural[k][:, 2], baseline[k][:, 2], 'natural midpoint versus central-action batch '+k)
            site_rows=[]; patched_visual=[]; patched_proprio=[]; patch_values=[]
            for block in BLOCKS:
                for horizon in HORIZONS:
                    samples = sampled.values[block, horizon][:5]
                    exact(samples[2], central_capture.values[block, horizon][2], 'held native midpoint residual')
                    # The only fitting inputs are four nonzero path samples; midpoint used for scoring/control only.
                    noncentral = samples[[0, 1, 3, 4]]
                    estimated = central_estimates(noncentral)
                    replacement = torch.stack([central_capture.values[block, horizon][0]]+[estimated[m] for m in METHODS])
                    with ResidualReplacement(blocks[block], horizon, replacement.to(agent.device)):
                        changed = wm.unroll(z.clone(), act_suffix=central_plan)
                    for k in ('visual', 'proprio'):
                        exact(changed[k][:, 0], baseline[k][:, 0], 'self-patch exact '+k)
                        exact(changed[k][:horizon], baseline[k][:horizon], 'predictions before intervention unchanged '+k)
                    visual = changed['visual'][1:].float().cpu()
                    mse_native = tensor_mse_by_horizon(visual, base_visual)
                    mse_actual = tensor_mse_by_horizon(visual, actual)
                    cost = agent.objective(changed, central_plan, keepdims=True).float().cpu()
                    methods=[]
                    for i, name in enumerate(METHODS, 1):
                        methods.append(dict(method=name, **interpolation_diagnostics(estimated[name], noncentral, samples[2]),
                            downstream_native_fullspatial_mse_by_horizon=mse_native[:, i].tolist(),
                            downstream_native_fullspatial_mse_after_patch=float(mse_native[horizon-1:,i].mean()),
                            actual_future_fullspatial_mse_by_horizon=mse_actual[:, i].tolist(),
                            actual_future_mse_delta_vs_native_by_horizon=(mse_actual[:, i]-base_actual_error[:, i]).tolist(),
                            actual_future_mse_delta_after_patch=float((mse_actual[horizon-1:,i]-base_actual_error[horizon-1:,i]).mean()),
                            native_goal_cost_by_horizon=cost[:, i].reshape(-1).tolist()))
                    site_rows.append(dict(block=block, intervention_horizon=horizon, geometry=sampled_path_geometry(samples),
                        self_patch_exact=True, methods=methods))
                    patched_visual.append(visual); patched_proprio.append(changed['proprio'][1:].float().cpu()); patch_values.append(replacement)
            exact(central_plan, central_saved, 'central actions fixed across conditions')
            exact(path_plan, path_saved, 'natural path actions fixed')
            name=f"{row['key']}-pair{pair}-radius{radius}"
            full=args.output/(name+'.pt')
            torch.save(dict(complete=True, source=source_metadata, path=path_meta, raw_path_actions=raw,
                normalized_path_actions=normalized, sample_t=TIMES, block_indices=BLOCKS, horizons=HORIZONS,
                residual_samples=torch.stack([sampled.values[b,h][:5] for b in BLOCKS for h in HORIZONS]),
                central_estimates_and_self=torch.stack(patch_values),
                natural_predicted_visual=natural['visual'][1:].float().cpu(),
                patched_predicted_visual=torch.stack(patched_visual), patched_predicted_proprio=torch.stack(patched_proprio),
                native_central_predicted_visual=base_visual, actual_central_visual=actual[:, 0],
                condition_order=['exact_self_patch']+list(METHODS)), full)
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            report=full.with_suffix('.json')
            write_json(report, dict(complete=True, source=source_metadata, path=path_meta, rows=site_rows,
                baseline_actual_fullspatial_mse_by_horizon=base_actual_error[:,0].tolist(), seconds=seconds,
                full_tensor_sha256=sha256(full), full_tensor_bytes=full.stat().st_size,
                limitations=['Only four initial states; action directions/horizons/patches are dependent, exploratory units',
                    'Natural midpoint used only for scoring and exact control, never interpolation fitting',
                    'Fullspatial representation fidelity is not physical steering success or output probability',
                    'Only central action physical truth is observed; sampled path futures are model predictions',
                    'Five sampled points do not identify a manifold or a density model',
                    'Cubic central interpolation equals four-point quadratic LS at zero; no specifically cubic mechanism']))
            for p in (full, report): entries.append(dict(path=p.name, sha256=sha256(p), bytes=p.stat().st_size, key=row['key']))
            print(json.dumps(dict(event='path_complete', key=row['key'], pair=pair, radius=radius, seconds=seconds)), flush=True)
            if first_canary:
                write_json(args.output/'CANARY.json',dict(complete=True, guards_passed=True, seconds_per_path=seconds,
                    estimated_32path_seconds=seconds*32, max_cuda_memory_allocated=torch.cuda.max_memory_allocated(),
                    native_and_self_patch_exact=True, outcomes_not_used_for_gate=True, within_900seconds=seconds*32<=900))
                print(json.dumps(dict(event='curvature_canary', seconds=seconds, projected_seconds=seconds*32)),flush=True)
                if seconds*32>900: raise RuntimeError('Timing-only canary exceeds authorized15GPU-minute estimate; no next path launched')
                first_canary=False
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','repo','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    from complete_cached_geometry import sha256, write_json
    from capture_pusht_calibration import parameter_sha
    from collect_pusht_bank import config
    from model_loader import load_headless
    done=json.loads((args.bank/'DONE.json').read_text())
    rows=sorted([r for r in done['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not done['complete'] or [r['key'] for r in rows]!=[f'near-dev-{i:03}' for i in range(4)] or any(r['split']!='development_external' for r in rows):
        raise ValueError('Only four fixed development receipts may be opened')
    checkpoint_sha=sha256(args.checkpoint)
    if checkpoint_sha!='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb': raise ValueError('Checkpoint mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',dict(rows=rows, checkpoint_sha256=checkpoint_sha,
        script_sha256=sha256(__file__), geometry_helper_sha256=sha256(Path(__file__).with_name('action_path_geometry.py')),
        fixed_t=TIMES, radii=[1,4], pair_indices=list(range(4)), blocks=BLOCKS, horizons=HORIZONS, methods=METHODS,
        central_fit_inputs=[-1,-.5,.5,1], same_batch_all_conditions=BATCH, central_actual_future_only=True,
        timing_gate='First fixed path projected32total≤900seconds; no outcome gate or path selection',
        primary_exploratory_view='P3/H3; radii both fixed, remaining sites secondary',
        statistic_unit='four initial states; average directions within state before descriptive interval',
        exploratory=True, cem_calls=0, simulator_calls=0, native_weights_frozen=True))
    torch.set_num_threads(2); torch.manual_seed(90505)
    cfg=config(args.repo,args.output)
    wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False); before=parameter_sha(wm); start=time.monotonic(); entries=[]
    for i,row in enumerate(rows): entries.extend(run_source(args,row,wm,prep,cfg,i==0))
    if parameter_sha(wm)!=before: raise ValueError('Native parameters changed')
    write_json(args.output/'DONE.json',dict(complete=True, outputs=entries, seconds=time.monotonic()-start,
        parameter_sha256=before, provenance=provenance, paths=32, sites_per_path=9, methods=len(METHODS),
        simulator_calls=0, cem_calls=0, original_heldout_access=False))


if __name__=='__main__': main()
