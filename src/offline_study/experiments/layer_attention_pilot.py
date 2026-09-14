"""Read-only JEPA-WM predictor instrumentation; fitting-stimulus engineering only.

The SDPA observer receives the actual post-RoPE Q/K and original mask. It calls
the original kernel unchanged. No attention matrix is saved; query chunks bound
scratch memory. Run in a dedicated single-threaded process (SDPA is patched only
inside the context). This is not a probe-training or causal-ablation experiment.
"""
from __future__ import annotations
from offline_study._paths import source_path

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def attention_summary(q, k, *, T, H, W, action_tokens=0, attn_mask=None,
                      is_causal=False, scale=None, chunk=64):
    """Per-example/head distances conditional on visual-query -> visual-key mass.

Conditioning-token mass is reported separately, never assigned a spatial position.
All-masked visual queries fail closed rather than producing misleading zeros.
"""
    if q.ndim != 4 or k.shape != q.shape or chunk < 1:
        raise ValueError('Require equal, four-dimensional self-attention Q/K')
    b, heads, n, dim = q.shape
    if n != T * (H * W + action_tokens):
        raise ValueError('Token layout does not match declared grid')
    ids = torch.arange(n, device=q.device)
    frame = ids // (H * W + action_tokens)
    local = ids % (H * W + action_tokens) - action_tokens
    visual = local >= 0
    x, y = local.clamp_min(0) % W, local.clamp_min(0) // W
    sums = torch.zeros(b, heads, 4, device=q.device, dtype=torch.float64)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        logits = (q[:, :, start:end].float() @ k.float().transpose(-2, -1))
        logits *= dim ** -0.5 if scale is None else scale
        if attn_mask is not None:
            mask = attn_mask[..., start:end, :] if attn_mask.shape[-2] != 1 else attn_mask
            if mask.dtype == torch.bool:
                logits.masked_fill_(~mask, float('-inf'))
            else:
                logits += mask
        if is_causal:
            logits.masked_fill_(ids[None, :] > ids[start:end, None], float('-inf'))
        weights = logits.softmax(-1)[:, :, visual[start:end], :]
        if not torch.isfinite(weights).all():
            raise ValueError('Nonfinite attention on a visual query')
        dx = x[start:end, None] - x[None, :]
        dy = y[start:end, None] - y[None, :]
        ds = (dx.square() + dy.square()).float().sqrt()[visual[start:end]]
        dt = (frame[start:end, None] - frame[None, :]).abs().float()[visual[start:end]]
        vv = weights * visual
        sums[..., 0] += (vv * ds).sum((-2, -1)).double()
        sums[..., 1] += (vv * dt).sum((-2, -1)).double()
        sums[..., 2] += vv.sum((-2, -1)).double()
        sums[..., 3] += (weights * ~visual).sum((-2, -1)).double()
    denom = sums[..., 2]
    if not (denom > 0).all():
        raise ValueError('No visual attention mass')
    return {'spatial_distance_patches': (sums[..., 0] / denom).cpu().tolist(),
            'temporal_distance_frames': (sums[..., 1] / denom).cpu().tolist(),
            'conditioning_mass': (sums[..., 3] / int(visual.sum())).cpu().tolist(),
            'visual_mass': (denom / int(visual.sum())).cpu().tolist(),
            'T': T, 'H': H, 'W': W, 'action_tokens': action_tokens,
            'query_chunk': chunk}


class PredictorObserver:
    def __init__(self, predictor):
        self.predictor = predictor
        self.rows, self.handles, self.stack = [], [], []

    def __enter__(self):
        self.original = F.scaled_dot_product_attention

        def observe(q, k, v, *args, **kwargs):
            if self.stack:
                layer, params = self.stack[-1]
                mask = kwargs.get('attn_mask', args[0] if args else None)
                dropout = kwargs.get('dropout_p', args[1] if len(args) > 1 else 0.)
                causal = kwargs.get('is_causal', args[2] if len(args) > 2 else False)
                if dropout or kwargs.get('enable_gqa', False):
                    raise ValueError('Pilot only supports dropout-free, non-GQA inference')
                self.rows.append({'layer': layer, **attention_summary(q, k,
                    T=params['T'], H=params['H'], W=params['W'],
                    action_tokens=params.get('action_tokens', 0), attn_mask=mask,
                    is_causal=causal, scale=kwargs.get('scale'))})
            return self.original(q, k, v, *args, **kwargs)

        try:
            for layer, block in enumerate(self.predictor.predictor_blocks):
                if block.training:
                    raise ValueError('Predictor must be in eval mode')
                def pre(module, args, kwargs, layer=layer):
                    bound = inspect.signature(module.forward).bind(*args, **kwargs)
                    bound.apply_defaults()
                    self.stack.append((layer, bound.arguments))
                def post(module, args, kwargs, output):
                    self.stack.pop()
                self.handles.append(block.attn.register_forward_pre_hook(pre, with_kwargs=True))
                self.handles.append(block.attn.register_forward_hook(post, with_kwargs=True,
                                                                    always_call=True))
            F.scaled_dot_product_attention = observe
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()
        self.stack.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('vendor', 'checkpoint', 'stimulus', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--stimulus-receipt-sha256', required=True)
    parser.add_argument('--trace-cem', action='store_true',
                        help='Also validate native 15-iteration CEM trace on this fitting clip; no simulator')
    args = parser.parse_args()
    # Explicit hash supplied by the pre-launch contract, not inferred after loading.
    receipt_path = args.stimulus / 'STIMULUS.json'
    if file_hash(receipt_path) != args.stimulus_receipt_sha256:
        raise ValueError('Stimulus receipt changed')
    receipt = json.loads(receipt_path.read_text())
    if (receipt['task'] != 'mw-reach' or receipt['selected_fit_row']['index'] != 10587
            or receipt['selected_fit_row']['split'] != 'fit'
            or receipt['development_or_protected_outcomes_accessed'] is not False):
        raise ValueError('Require the existing original Reach fitting stimulus')
    for name, digest in receipt['files'].items():
        if Path(name).name != name or file_hash(args.stimulus / name) != digest:
            raise ValueError('Stimulus bytes changed')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one authorized GPU; no CPU fallback')
    args.output.mkdir(parents=True, exist_ok=False)
    from offline_study.models.backends import JepaBackend
    from offline_study.validation.fixed_combined_check import load_stimulus, assert_bytes, rng_signature
    from tensordict import TensorDict
    backend = JepaBackend(args.vendor, args.checkpoint, receipt['checkpoint_sha256'],
                          'metaworld', 'cuda:0', 'float32')
    value = load_stimulus(args.stimulus)
    with torch.no_grad():
        obs = {k: v.to('cuda:0') for k, v in value['observations'].items()}
        actions = value['actions'].to('cuda:0')
        visual, proprio, _ = backend.model.model.encode(obs, actions)
        context = TensorDict({'visual': visual[:, :1], 'proprio': proprio[:, :1]}, batch_size=[])
        suffix = actions[:, :6].transpose(0, 1).contiguous()
        backend.predict(context, suffix)  # warm-up on the same fitting row only
        torch.cuda.synchronize()
        start = time.monotonic()
        native = backend.predict(context, suffix)
        torch.cuda.synchronize()
        native_seconds = time.monotonic() - start
        before_rng = rng_signature('cuda:0')
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with PredictorObserver(backend.predictor) as observer:
            measured = backend.predict(context, suffix)
        torch.cuda.synchronize()
        observed_seconds = time.monotonic() - start
        for key in native.keys():
            assert_bytes(measured[key], native[key], 'instrumented ' + key)
        if rng_signature('cuda:0') != before_rng:
            raise ValueError('Observer changed RNG state')
    counts = {layer: sum(r['layer'] == layer for r in observer.rows) for layer in range(6)}
    if counts != dict.fromkeys(range(6), 6):
        raise ValueError('Expected all six blocks at every H6 predictor step: ' + str(counts))
    cem_report = None
    if args.trace_cem:
        from offline_study.experiments.candidate_score_trace import CandidateScoreTrace
        from offline_study.planning.planning_contract import prepare
        from evals.simu_env_planning.planning.planning.planner import CEMPlanner
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
        contract = prepare(args.vendor, 'reach')
        cfg = contract['config']
        # The goal is an existing recorded fitting future, not a new simulator goal
        # or a ground-truth outcome for arbitrary candidate actions.
        target = TensorDict({'visual': visual[:, 6:7], 'proprio': proprio[:, 6:7]}, batch_size=[])
        def make_planner():
            generator = torch.Generator(device='cuda:0').manual_seed(20260913)
            planner = CEMPlanner(unroll=backend.model.unroll, action_dim=backend.model.action_dim,
                                 local_generator=generator, **cfg['planner'])
            planner.set_objective(ReprTargetDistMPCObjective(cfg, target,
                                      **cfg['planner']['planning_objective']))
            return planner
        native_planner, traced_planner = make_planner(), make_planner()
        with torch.no_grad():
            reference_plan = native_planner.plan(context.clone(), steps_left=6)
            before_trace_rng = rng_signature('cuda:0')
            with CandidateScoreTrace(traced_planner) as trace:
                traced_plan = trace.run(context.clone(), steps_left=6)
            assert_bytes(traced_plan.actions, reference_plan.actions, 'traced CEM actions')
            assert_bytes(traced_planner.local_generator.get_state(),
                         native_planner.local_generator.get_state(), 'traced CEM generator')
            if before_trace_rng != rng_signature('cuda:0'):
                raise ValueError('Trace changed global RNG')
        with (args.output / 'native-candidate-trace.pt').open('xb') as f:
            torch.save(trace.payload(), f)
        cem_report = {'native_only': True, 'iterations': len(trace.iterations),
                      'candidates_per_iteration': 300, 'seed': 20260913,
                      'goal': 'encoded fitting clip frame 6', 'output_byte_parity': True,
                      'trace_sha256': file_hash(args.output / 'native-candidate-trace.pt'),
                      'observer_source_sha256': file_hash(source_path('candidate_score_trace.py')),
                      'planner_source_sha256': file_hash(args.vendor / 'evals/simu_env_planning/planning/planning/planner.py')}
    report = {'status': 'fitting_only_engineering_complete', 'scientific_efficacy_measurement': False,
              'fresh_confirmation': False, 'unique_trajectories': 1,
              'source_sha256': file_hash(__file__), 'stimulus_receipt_sha256': args.stimulus_receipt_sha256,
              'checkpoint_sha256': receipt['checkpoint_sha256'], 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'output_byte_parity': True, 'rng_unchanged': True,
              'native_seconds': native_seconds, 'observed_seconds': observed_seconds,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'attention': observer.rows,
              'cem_trace': cem_report}
    with (args.output / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'attention'}))


if __name__ == '__main__':
    main()
