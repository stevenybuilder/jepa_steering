"""Bounded processing audit on the first CEM population of an excluded scenario.

Never a scientific episode, precision amendment, confirmation gate or efficacy
measurement. No live source/model is changed; modes run only in this process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import statistics
import time

import numpy as np
import torch


MODES = ('strict_fp32', 'tf32_diagnostic', 'bf16_diagnostic')


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode()
                          + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def use_mode(backend, mode):
    if mode not in MODES:
        raise ValueError('Unregistered diagnostic mode')
    torch.set_float32_matmul_precision('high' if mode == 'tf32_diagnostic' else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32_diagnostic'
    torch.backends.cudnn.allow_tf32 = mode == 'tf32_diagnostic'
    backend.autocast_dtype = torch.bfloat16 if mode == 'bf16_diagnostic' else None


def compare(reference, actual):
    if set(reference.keys()) != set(actual.keys()):
        raise ValueError('Forecast output keys changed')
    output = {}
    for key in reference.keys():
        a, b = reference[key], actual[key]
        if a.shape != b.shape or not torch.isfinite(b).all():
            raise ValueError('Invalid diagnostic forecast')
        difference = b.float() - a.float()
        output[key] = {'shape': list(a.shape), 'dtype': str(b.dtype),
            'bitwise_equal': torch.equal(a, b),
            'max_absolute_error': float(difference.abs().max()),
            'relative_l2_error': float(difference.norm() / a.float().norm().clamp_min(1e-30))}
    return output


class CapturedPopulation(Exception):
    pass


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('vendor', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from offline_study.author_fit import source_hash
    from offline_study.backends import JepaBackend
    from offline_study.operator_fit import _model_versions
    from offline_study.planning_contract import prepare
    from offline_study.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
    from offline_study.protocol import sha256, write_json
    from offline_study.vendor import use_vendor
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError('Ten-minute processing-audit limit; no science launched')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    contract = {'role': 'excluded_first_native_cem_population_processing_audit',
        'task': 'reach', 'scenario_seed': SMOKE_SEED, 'modes': list(MODES),
        'timed_repetitions': 5, 'warmups_per_mode': 2, 'gpu_seconds_limit': 600,
        'checkpoint_sha256': CHECKPOINTS['metaworld'], 'source_sha256': source_hash(),
        'audit_source_sha256': sha256(Path(__file__)), 'complete_episode': False,
        'scientific_efficacy_measurement': False, 'fresh_confirmation': False,
        'precision_change_authorized_in_scientific_runs': False,
        'mode_order': 'forward_then_reverse_timing_blocks',
        'profiled_timing_excluded_from_speed_comparison': True}
    write_json(args.output / 'protocol.json', contract)
    backend = None
    try:
        use_vendor(args.vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        random.seed(SMOKE_SEED); np.random.seed(SMOKE_SEED); torch.manual_seed(SMOKE_SEED)
        backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINTS['metaworld'],
                              'metaworld', 'cuda:0', 'float32')
        versions = _model_versions(backend.model)
        cfg = OmegaConf.create(prepare(args.vendor, 'reach')['config'])
        cfg.meta.seed = cfg.local_seed = SMOKE_SEED
        agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
        captured = {}
        def capture(context, act_suffix=None, **kwargs):
            if act_suffix is None or tuple(act_suffix.shape[:2]) != (6, 300) or kwargs:
                raise ValueError('Unexpected first native full-CEM call')
            captured.update(context=context.clone(), actions=act_suffix.clone())
            raise CapturedPopulation()
        agent.planner.unroll = capture
        env = make_env(cfg)
        try:
            run_episode(cfg, backend, agent, env, SMOKE_SEED)
        except CapturedPopulation:
            pass
        finally:
            env.close()
        if not captured:
            raise ValueError('No real excluded engineering population captured')
        context, actions = captured['context'], captured['actions']
        original_hashes = {key: tensor_hash(context[key]) for key in context.keys()}
        original_hashes['actions'] = tensor_hash(actions)
        write_json(args.output / 'INPUTS.json', {'hashes': original_hashes,
            'context_shapes': {key: list(context[key].shape) for key in context.keys()},
            'actions_shape': list(actions.shape), 'source': 'actual first full native CEM population',
            'simulator_episode_deliberately_not_completed': True})
        reference = backend.predict(context, actions)
        timings = {mode: [] for mode in MODES}
        comparisons = {}
        for mode in (*MODES, *reversed(MODES)):
            use_mode(backend, mode)
            for _ in range(2):
                actual = backend.predict(context, actions)
            torch.cuda.synchronize()
            comparisons[mode] = compare(reference, actual)
            if mode == 'strict_fp32' and not all(v['bitwise_equal'] for v in comparisons[mode].values()):
                raise ValueError('Strict native repetition changed')
            for _ in range(5):
                torch.cuda.synchronize(); before = time.monotonic()
                actual = backend.predict(context, actions)
                torch.cuda.synchronize(); timings[mode].append(time.monotonic() - before)
            write_json(args.output / 'progress.json', {'finished_timing_block': mode,
                'timings': timings, 'scientific_efficacy_measurement': False})
        use_mode(backend, 'strict_fp32')
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_flops=True) as prof:
            actual = backend.predict(context, actions)
            torch.cuda.synchronize()
        if not all(v['bitwise_equal'] for v in compare(reference, actual).values()):
            raise ValueError('Profile instrumentation changed native forecast')
        events = []
        for item in prof.key_averages():
            events.append({'operator': item.key, 'calls': item.count,
                'self_cpu_microseconds': item.self_cpu_time_total,
                'self_device_microseconds': getattr(item, 'self_device_time_total', 0.),
                'flops_reported_by_profiler': item.flops})
        events.sort(key=lambda row: row['self_device_microseconds'], reverse=True)
        prof.export_chrome_trace(str(args.output / 'native_trace.json'))
        after = {key: tensor_hash(context[key]) for key in context.keys()}
        after['actions'] = tensor_hash(actions)
        if after != original_hashes or versions != _model_versions(backend.model) or source_hash() != contract['source_sha256']:
            raise ValueError('Input/model/source changed during diagnostic')
        report = {'status': 'bounded_native_processing_audit_complete',
            'protocol_sha256': sha256(args.output / 'protocol.json'),
            'input_sha256': sha256(args.output / 'INPUTS.json'),
            'trace_sha256': sha256(args.output / 'native_trace.json'),
            'seconds': time.monotonic() - started, 'device': torch.cuda.get_device_name(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'timings': {mode: {'seconds': values, 'median_seconds': statistics.median(values)} for mode, values in timings.items()},
            'forecast_differences': comparisons, 'operators': events,
            'gpu_profiler_events_present': any(row['self_device_microseconds'] > 0 for row in events),
            'profiler_flops_are_incomplete_not_model_flop_count': True,
            'parameters_and_inputs_unchanged': True, 'fresh_confirmation': False,
            'complete_episode': False, 'scientific_efficacy_measurement': False,
            'speedup_does_not_establish_full_episode_speed_or_task_success': True,
            'diagnostic_precision_not_activated_in_science': True}
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'report.json')})
        print(json.dumps({key: report[key] for key in ('status', 'seconds', 'timings', 'forecast_differences', 'gpu_profiler_events_present')}), flush=True)
    except BaseException as exc:
        write_json(args.output / 'FAILED.json', {'error': str(exc), 'scientific_efficacy_measurement': False})
        raise
    finally:
        signal.alarm(0)
        if backend is not None:
            use_mode(backend, 'strict_fp32')


if __name__ == '__main__':
    main()
