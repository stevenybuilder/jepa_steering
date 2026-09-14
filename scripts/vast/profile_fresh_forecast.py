"""Strict-FP32 first-forecast diagnostic; never a completed episode or gate.

Capture only excluded Reach input 0 through frozen execute(). Compare the
unchanged scientific observer path with its direct native adapter as an upper
bound on removable observer overhead, not an approved production optimization.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import statistics
import time

FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
UUID = 'ecbbfa2e-25d1-2a99-96fb-7f7435371af2'


class Captured(BaseException):
    pass


def validate_output(project, output):
    project, output = project.resolve(), output.resolve()
    if output.parent != project / 'engineering-diagnostics' or output.exists():
        raise ValueError('Require a new immediate engineering-diagnostics child')
    return output


def summarize(timings):
    if set(timings) != {'observer', 'direct_adapter'} or any(len(v) != 2 for v in timings.values()):
        raise ValueError('Require two paired timings per path')
    medians = {k: statistics.median(v) for k, v in timings.items()}
    if any(v <= 0 for v in medians.values()):
        raise ValueError('Invalid elapsed timing')
    return {'seconds': timings, 'median_seconds': medians,
            'observer_over_direct_ratio': medians['observer'] / medians['direct_adapter'],
            'upper_bound_removable_fraction': 1 - medians['direct_adapter'] / medians['observer'],
            'tiny_sample_not_significance_test': True}


def run(project, output):
    import random
    import numpy as np
    import torch
    from offline_study.experiments import fresh_confirmation as fc
    from offline_study.runtime.intervention_runner import _model_versions
    from offline_study.core.protocol import write_json
    from torch.profiler import profile, ProfilerActivity
    output = validate_output(project, output)
    if hashlib.sha256((project / 'fresh-freeze-v2/protocol.json').read_bytes()).hexdigest() != FREEZE:
        raise ValueError('Wrong scientific freeze')
    if torch.cuda.device_count() != 1 or str(torch.cuda.get_device_properties(0).uuid) != UUID:
        raise ValueError('Only the explicitly reserved A100 GPU0 is authorized')
    lock = Path('/tmp', 'fresh-confirmation-' + UUID + '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'PROFILE_INTENT.json', {'role': 'excluded_first_forecast_only',
        'scientific_efficacy_measurement': False, 'episode': 0, 'task': 'reach',
        'device_uuid': UUID, 'freeze_sha256': FREEZE, 'strict_fp32': True,
        'outer_limit_seconds': 180, 'profile_limit_seconds': 120})
    capture = {}
    original = fc.CheckedObserver.__call__
    def intercept(observer, context, act_suffix=None, **kwargs):
        if capture or act_suffix is None or tuple(act_suffix.shape[:2]) != (6, 300) or kwargs:
            raise ValueError('Unexpected first forecast shape')
        capture.update(observer=observer, context=context.clone(), actions=act_suffix.clone())
        raise Captured()
    args = argparse.Namespace(project=project, freeze=project / 'fresh-freeze-v2',
        task='reach', episode=0, mode='engineering', output=output / 'excluded-partial')
    try:
        fc.CheckedObserver.__call__ = intercept
        try:
            fc.execute(args)
        except Captured:
            pass
    finally:
        fc.CheckedObserver.__call__ = original
    if not capture or list(output.rglob('DONE.json')):
        raise ValueError('Capture missing or full-episode DONE unexpectedly published')
    c = capture['observer']
    if c.arm != 'native' or not c.engineering or c.backend.autocast_dtype is not None:
        raise ValueError('Expected excluded strict native forecast')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('Strict FP32 settings changed')
    context, actions = capture['context'], capture['actions']
    def digest(tensor):
        value = tensor.detach().cpu().contiguous()
        return hashlib.sha256(str((value.dtype, tuple(value.shape))).encode() + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def inputs():
        return {**{k: digest(context[k]) for k in context.keys()}, 'actions': digest(actions)}
    def weights():
        h = hashlib.sha256()
        for name, tensor in c.backend.model.state_dict().items():
            h.update(name.encode()); h.update(digest(tensor).encode())
        return h.hexdigest()
    before_inputs, before_weights = inputs(), weights()
    versions = _model_versions(c.backend.model)
    source = fc.source_hash()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
    observer_path = output / 'observer'
    observer_path.mkdir()
    observer = fc.CheckedObserver(c.adapter, c.backend, observer_path, c.arm,
                                  c.fitted, c.cp, c.cb, engineering=False, all_arms=False)
    paths = {'observer': observer, 'direct_adapter': c.adapter}
    profile_start = time.monotonic()
    longest_call = 0.
    def call(name):
        nonlocal longest_call
        if time.monotonic() - profile_start + max(10., 2 * longest_call) >= 120:
            raise TimeoutError('Two-minute forecast profiling bound')
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state(rng[3])
        torch.cuda.synchronize()
        started = time.monotonic()
        result = paths[name](context, actions)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        longest_call = max(longest_call, seconds)
        return result, seconds
    with torch.no_grad():
        reference, _ = call('observer')  # Warm-up/reference, excluded from timings.
        timings = {'observer': [], 'direct_adapter': []}
        for name in ('observer', 'direct_adapter', 'direct_adapter', 'observer'):
            actual, seconds = call(name)
            if set(actual.keys()) != set(reference.keys()) or any(not torch.equal(actual[k], reference[k]) for k in reference.keys()):
                raise ValueError('Bitwise forecast parity failed')
            timings[name].append(seconds)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            actual, _ = call('observer')
        if any(not torch.equal(actual[k], reference[k]) for k in reference.keys()):
            raise ValueError('Profiler instrumentation changed forecasts')
    prof.export_chrome_trace(str(output / 'observer_trace.json'))
    events = [{'operator': e.key, 'calls': e.count, 'self_cpu_us': e.self_cpu_time_total,
               'self_device_us': getattr(e, 'self_device_time_total', 0.)} for e in prof.key_averages()]
    events.sort(key=lambda e: e['self_device_us'], reverse=True)
    sync = [e for e in events if any(s in e['operator'].lower() for s in ('synchronize', 'memcpy', '_local_scalar_dense', 'aten::item', 'aten::to', 'aten::_to_copy'))]
    if before_inputs != inputs() or before_weights != weights() or versions != _model_versions(c.backend.model) or source != fc.source_hash():
        raise ValueError('Input/weight/source mutation detected')
    if list(output.rglob('DONE.json')):
        raise ValueError('A diagnostic must never publish episode/scientific DONE')
    report = {'role': 'excluded_first_forecast_only', 'scientific_efficacy_measurement': False,
        'complete_episode': False, 'receiving_gate_pass_claimed': False, 'device_uuid': UUID,
        'freeze_sha256': FREEZE, 'precision': 'strict_fp32_no_tf32', 'arm': 'native',
        'shape': [6, 300], 'input_hashes': before_inputs, 'weight_sha256': before_weights,
        'bitwise_outputs_equal': True, 'inputs_weights_source_unchanged': True,
        'timing': summarize(timings), 'operators': events, 'sync_copy_operators': sync,
        'profile_seconds': time.monotonic() - profile_start,
        'gpu_profiler_events_present': any(e['self_device_us'] > 0 for e in events),
        'scope_limit': 'One native Reach forecast; engineering-only all-arm parity work is excluded from observer timing. Direct adapter is an upper bound, not approved production code.'}
    write_json(output / 'PROFILE_REPORT.json', report)
    print(json.dumps({'report': str(output / 'PROFILE_REPORT.json'), 'timing': report['timing'],
                      'bitwise_outputs_equal': True, 'complete_episode': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def timeout(*_):
        raise TimeoutError('180-second outer diagnostic bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    try:
        run(args.project.resolve(), args.output)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
