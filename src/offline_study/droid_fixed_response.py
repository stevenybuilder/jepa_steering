"""DROID's twelve-block fixed-response adapter; no online probes or shadow model."""
import json
import math
from pathlib import Path

import torch

from .fixed_response import ARMS, ZERO_THRESHOLD, compose_map
from .protocol import sha256

METHOD = 'droid_fixed_response_rank4_v1'
BLOCK, HORIZON, PATCHES, FEATURES = 6, 3, 256, 1024


def validate_bank(bank):
    if bank.get('method') != METHOD or bank.get('schema_version') != 1:
        raise ValueError('Require the DROID-specific fitted operator')
    if not math.isfinite(bank['dose']) or bank['dose'] <= ZERO_THRESHOLD:
        raise ValueError('Invalid fit-only dose')
    width = PATCHES * FEATURES
    if (bank['mean'].shape != (width,) or bank['projection'].shape != (3, width)
            or bank['scale'].shape != (3,) or not (bank['scale'] > 0).all()
            or set(bank['operators']) != set(ARMS[2:])):
        raise ValueError('Wrong DROID feature/readout/condition contract')
    values = [bank[k] for k in ('mean', 'projection', 'scale')]
    for operator in bank['operators'].values():
        if operator['basis'].shape != (4, PATCHES, FEATURES) or operator['map'].shape != (4, 4):
            raise ValueError('Wrong DROID direction/map dimensions')
        basis = operator['basis'].double().flatten(1)
        if not torch.allclose(basis @ basis.T, torch.eye(4, device=basis.device).double(), atol=2e-5, rtol=2e-5):
            raise ValueError('Random/learned basis must be independently orthonormal')
        values += list(operator.values())
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError('Nonfinite fitting tensor')


def make_bank(target, bases, responses, dose, binding):
    bank = {'schema_version': 1, 'method': METHOD, 'binding': binding, 'dose': float(dose),
        'mean': target['mean'].float().cpu(), 'projection': target['basis'].float().cpu(),
        'scale': target['scale'].float().cpu(), 'operators': {}}
    for arm, key in zip(ARMS[2:], ('rank', 'rank_random')):
        bank['operators'][arm] = {'basis': bases[key][:4].float().cpu(),
            'map': compose_map(responses[key], target['weight'].to(responses[key].device)).cpu()}
    validate_bank(bank)
    return bank


def load_bank(directory, checkpoint_sha256):
    done = json.loads((directory / 'DONE.json').read_text())
    if done.get('status') != 'droid_fixed_fit_complete_not_behavioral_clearance' or (directory / 'FAILED.json').exists():
        raise ValueError('Incomplete DROID fixed-response fit')
    expected = ('contract.json', 'operator_bank.pt', 'mean_responses.pt', 'audit_metrics.json',
                'report.json', 'prefixes.json', 'native_captures.pt', 'PARITY.json')
    if set(done['files_sha256']) != set(expected):
        raise ValueError('Missing fitted artifact')
    for name, digest in done['files_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError('Fitted artifact checksum differs: ' + name)
    report = json.loads((directory / 'report.json').read_text())
    contract = json.loads((directory / 'contract.json').read_text())
    bank = torch.load(directory / 'operator_bank.pt', map_location='cpu', weights_only=True)
    if (bank['binding'] != contract['binding'] or bank['binding'] != report['binding']
            or bank['binding']['checkpoint_sha256'] != checkpoint_sha256
            or bank['binding']['task'] != 'droid' or bank['binding']['method'] != METHOD
            or report['evaluation_outcomes_accessed'] is not False):
        raise ValueError('Wrong task/checkpoint/fit provenance')
    validate_bank(bank)
    return bank


def transform_field(field, bank, arm):
    """FP32 thin-map evaluation; returns original activation dtype and audit tensors."""
    if field.ndim != 3 or field.shape[1:] != (PATCHES, FEATURES) or arm not in ARMS[2:]:
        raise ValueError('Not the full newest DROID B6 field or active arm')
    with torch.autocast(device_type=field.device.type, enabled=False):
        operator = bank['operators'][arm]
        score = ((field.float().flatten(1) - bank['mean']) @ bank['projection'].T) / bank['scale']
        phi = torch.cat([torch.ones_like(score[:, :1]), score], 1)
        coefficients = phi @ operator['map'].T
        delta = (coefficients @ operator['basis'].flatten(1)).reshape_as(field)
        norm = delta.flatten(1).norm(dim=1)
        if not torch.isfinite(norm).all():
            raise ValueError('Nonfinite correction; never drop a candidate')
        active = norm > ZERO_THRESHOLD
        factor = torch.where(active, bank['dose'] / norm.clamp_min(ZERO_THRESHOLD), 0.)
        delta = delta * factor[:, None, None]
        changed = torch.where(active[:, None, None], field + delta.to(field.dtype), field)
        record = {'coefficients': (coefficients * factor[:, None]).detach(),
            'requested_l2': delta.flatten(1).norm(dim=1).detach(),
            'realized_l2': (changed.float() - field.float()).flatten(1).norm(dim=1).detach(),
            'active': active.detach()}
        return changed, record


class FieldHook:
    def __init__(self, predictor, transform=None):
        self.predictor, self.transform = predictor, transform
        self.horizon, self.applications, self.handles, self.field = 0, 0, [], None

    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 12 or any(
                module._forward_hooks or module._forward_pre_hooks for module in self.predictor.modules()):
            raise ValueError('Require standalone twelve-block DROID predictor')
        self.handles = [self.predictor.register_forward_pre_hook(self._input),
            self.predictor.predictor_blocks[BLOCK].register_forward_hook(self._output)]
        return self

    def _input(self, module, args):
        self.horizon += 1
        if len(args) != 3 or args[2] is not None:
            raise ValueError('Native DROID must remain no-proprio')

    def _output(self, module, args, output):
        if self.horizon != HORIZON:
            return output
        if self.applications or output.ndim != 3 or output.shape[1] < PATCHES or output.shape[2] != FEATURES:
            raise ValueError('Changed or duplicated B6/H3 field')
        self.applications += 1
        self.field = output[:, -PATCHES:].detach()
        if self.transform is not None:
            changed = self.transform(self.field)
            if changed.shape != self.field.shape or changed.dtype != output.dtype:
                raise ValueError('Transformation changed the native tensor contract')
            result = output.clone()
            result[:, -PATCHES:] = changed
            return result
        return output

    def __exit__(self, kind, *_):
        for handle in self.handles:
            handle.remove()
        if kind is None and (self.horizon != HORIZON or self.applications != 1):
            raise ValueError('Require one B6/H3 application in the full three-step forecast')


class Intervention:
    def __init__(self, model, bank, arm):
        validate_bank(bank)
        if arm not in ARMS:
            raise ValueError('Unregistered DROID fixed-response condition')
        def stage(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=model.device, dtype=torch.float32)
            return {k: stage(v) for k, v in value.items()} if isinstance(value, dict) else value
        self.model, self.bank, self.arm = model, stage(bank), arm
        self.last_record = {}

    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if kwargs or act_suffix is None or act_suffix.ndim != 3 or not 1 <= len(act_suffix) <= 3 or act_suffix.shape[-1] != 7:
            raise ValueError('DROID requires the unchanged one-to-three-step seven-action suffix')
        self.last_record = {'backend_calls': 1, 'response_probe_rollouts': 0, 'native_shadow_rollouts': 0,
            'horizon': len(act_suffix)}
        if len(act_suffix) < HORIZON or self.arm in ARMS[:2]:
            return self.model.unroll(context, act_suffix=act_suffix)
        def transform(field):
            changed, record = transform_field(field, self.bank, self.arm)
            self.last_record.update(record)
            return changed
        with FieldHook(self.model.model.predictor, transform):
            result = self.model.unroll(context, act_suffix=act_suffix)
        if not isinstance(result, torch.Tensor) or not torch.isfinite(result).all():
            raise ValueError('DROID forecast must remain a finite native tensor')
        return result
