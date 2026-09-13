"""Bounded, excluded-input receiving checks; never scientific outcome evidence."""
import contextlib
import random

import numpy as np
import torch

from .fixed_response import FixedResponseIntervention
from .fixed_response_check import reference_fields
from .intervention_runner import _model_versions
from .interventions import PredictorIntervention
from .metaworld_component_behavior import verify_energy
from .planning_panel_engineering import H6StaticPlanningIntervention
from .planning_support_check import source_coupling_fields

ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only',
        'matched_random_coupling', 'joint', 'visual_only', 'action_condition_only', 'zero_dose')
COUPLING = {'coupling_only': 'joint_equal_standardized_energy',
            'matched_random_coupling': 'matched_random_equal_standardized_energy',
            **{name: name for name in ('joint', 'visual_only', 'action_condition_only')}}


def context_shape_key(context, actions):
    if actions.ndim != 3 or not 1 <= actions.shape[0] <= 6 or actions.shape[1] not in (1, 300):
        raise ValueError('Receiving checks require actual H1-H6, population300/mean1 shapes')
    return (int(actions.shape[0]), int(actions.shape[1]),
            tuple(context['visual'].shape), tuple(context['proprio'].shape))


def _same_output(actual, expected):
    for key in ('visual', 'proprio'):
        if not torch.isfinite(actual[key]).all() or not torch.equal(actual[key], expected[key]):
            raise ValueError('Independent source-hook/native reference parity failed')


def _hooks(model):
    return tuple((id(module), tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                 for module in model.modules())


@contextlib.contextmanager
def _capture_b3(predictor):
    """Capture precisely the independently referenced H3/B3 field, not all6 blocks."""
    if len(predictor.predictor_blocks) != 6:
        raise ValueError('Unexpected predictor architecture')
    state = {'horizon': 0, 'field': None}
    def advance(module, args):
        state['horizon'] += 1
    def capture(module, args, output):
        if state['horizon'] == 3:
            if state['field'] is not None or output.ndim != 3 or output.shape[1] < 256:
                raise ValueError('Unexpected H3/B3 capture')
            state['field'] = output[:, -256:].detach().clone()
    handles = [predictor.register_forward_pre_hook(advance),
               predictor.predictor_blocks[3].register_forward_hook(capture)]
    try:
        yield state
        if state['horizon'] != 6 or state['field'] is None:
            raise ValueError('Incomplete H6/H3 reference capture')
    finally:
        for handle in handles:
            handle.remove()


def _one_call(adapter, backend, context, actions):
    original, count = backend.model.unroll, 0
    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)
    backend.model.unroll = counted
    try:
        output = adapter(context, actions)
    finally:
        backend.model.unroll = original
    if count != 1:
        raise ValueError('Scientific adapter must use exactly one model unroll')
    return output


def _refined_energy(energy, arm, horizon, count, dose):
    if energy['response_probe_rollouts'] or energy['native_shadow_rollouts']:
        raise ValueError('Online response probes or shadows appeared')
    if horizon == 6 and arm in ARMS[1:3]:
        requested, realized = (np.asarray(energy[k]) for k in ('requested_l2', 'realized_l2'))
        active = np.asarray(energy['active'], dtype=bool)
        coefficients = np.asarray(energy['coefficients'])
        if (requested.shape != (count,) or realized.shape != requested.shape
                or active.shape != requested.shape or coefficients.shape != (count, 4)
                or not np.isfinite(coefficients).all() or not np.isfinite(realized).all()
                or not np.allclose(requested, active * dose, rtol=1e-5, atol=1e-8)
                or not np.allclose(realized, requested, rtol=5e-4, atol=1e-8)):
            raise ValueError('Refined edit dose/coefficient contract changed')
    elif any(k in energy for k in ('requested_l2', 'realized_l2', 'coefficients')):
        raise ValueError('Native/short forecast received a refined edit')


@torch.no_grad()
def validate_all_arms(context, actions, backend, fitted, cp, cb):
    key = context_shape_key(context, actions)
    horizon, count = key[:2]
    json_key = [horizon, count, list(key[2]), list(key[3])]
    parameters, hooks = _model_versions(backend.model), _hooks(backend.predictor)
    originals = {name: context[name].clone() for name in ('visual', 'proprio')}
    original_actions = actions.clone()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(backend.device) if str(backend.device).startswith('cuda') else None
    if horizon == 6:
        with _capture_b3(backend.predictor) as capture:
            native = backend.predict(context, actions)
        field = capture['field']
    else:
        native, field = backend.predict(context, actions), None
    checks = []
    for arm in ARMS:
        adapter = (H6StaticPlanningIntervention(backend, cp, cb, COUPLING[arm]) if arm in COUPLING
                   else FixedResponseIntervention(backend, fitted, arm))
        actual = _one_call(adapter, backend, context, actions)
        if arm in ('native', 'zero_dose') or horizon < 6:
            expected = native
        else:
            fields = (reference_fields(field, adapter.bank, arm) if arm in ARMS[1:3]
                      else source_coupling_fields((cp, cb, COUPLING[arm]), count, backend.device))
            with PredictorIntervention(backend.predictor, fields):
                expected = backend.predict(context, actions)
            del fields
        _same_output(actual, expected)
        if arm in COUPLING:
            verify_energy([{'backend_calls': 1, 'horizon': horizon, 'candidates': count,
                            'energy': adapter.energy[-1]}], cp, cb, COUPLING[arm])
        else:
            energy = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                      for k, v in adapter.last_record.items()}
            _refined_energy(energy, arm, horizon, count, fitted['dose'])
        if (_model_versions(backend.model) != parameters or _hooks(backend.predictor) != hooks
                or not torch.equal(actions, original_actions)
                or any(not torch.equal(context[name], originals[name]) for name in originals)):
            raise ValueError('Receiving check mutated parameters, hooks, or inputs')
        checks.append({'arm': arm, 'context_key': json_key, 'source_parity_bitwise': True,
                       'backend_calls': 1, 'energy_valid': True, 'inputs_unchanged': True,
                       'parameters_unchanged': True, 'hooks_restored': True})
        del actual, expected, adapter
    current_numpy = np.random.get_state()
    if (python_rng != random.getstate() or numpy_rng[0] != current_numpy[0]
            or not np.array_equal(numpy_rng[1], current_numpy[1]) or numpy_rng[2:] != current_numpy[2:]
            or not torch.equal(torch_rng, torch.get_rng_state())
            or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(backend.device)))):
        raise ValueError('Receiving reference checks consumed RNG state')
    return checks
