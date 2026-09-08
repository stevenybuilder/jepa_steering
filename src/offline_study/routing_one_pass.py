"""Candidate exact execution optimization for the frozen H1/H2-routed H3 edit.

No edit occurs before H3, so the same-pass H1/H2 activations equal the unedited
shadow prefix. Full receiving-model equivalence is required before replacing the
existing two-pass runner. Same fitted gates, dose, HMM, candidate and RNG semantics.
"""
import torch

from .fixed_response import FixedResponseHook
from .routing_hmm import route_prefix
from .routing_intervention import GATE_KEYS, RoutedFixedResponse


class CausalHook(FixedResponseHook):
    def __init__(self, predictor, bank, routing, arm):
        operator = 'matched_random_fixed_rank4' if arm.startswith('matched_random_') else 'fixed_rank4'
        super().__init__(predictor, bank, operator)
        self.routing, self.gate_arm = routing, arm
        self.prefix, self.gate_record = {}, None

    def _output(self, module, args, output):
        if self.horizon in (1, 2):
            if self.horizon in self.prefix or output.ndim != 3 or output.shape[1] < 256 or output.shape[2] != 400:
                raise ValueError('Changed native causal-prefix field')
            self.prefix[self.horizon] = output[:, -256:].detach().float().mean(1)
            return output
        if self.horizon == 3:
            if set(self.prefix) != {1, 2} or self.gate_record is not None:
                raise ValueError('Routing must precede exactly one H3 edit')
            gates = route_prefix(torch.stack([self.prefix[1], self.prefix[2]], 1), self.routing['model'])
            key = GATE_KEYS[self.gate_arm.removeprefix('matched_random_')]
            normalizer = self.routing['normalizers'][key]
            gate = (gates[key] / normalizer).float()
            if not torch.isfinite(gate).all() or not (gate > 0).all():
                raise ValueError('Nonfinite gate; no candidate filtering')
            self.bank = {**self.bank, 'dose': self.bank['dose'] * gate}
            self.gate_record = {'normalized_gate': gate.detach(), 'raw_gate': gates[key].detach(),
                'posterior': gates['posterior'].detach(), 'gate_normalizer': normalizer.detach()}
        return super()._output(module, args, output)

    def __exit__(self, kind, exc, traceback):
        super().__exit__(kind, exc, traceback)
        if kind is None and (set(self.prefix) != {1, 2} or self.gate_record is None):
            raise ValueError('Incomplete same-pass causal gate')


class RoutedFixedResponseOnePass(RoutedFixedResponse):
    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if kwargs or act_suffix is None or act_suffix.ndim != 3 or not 1 <= len(act_suffix) <= 6 or act_suffix.shape[1] < 1:
            raise ValueError('Require unchanged H1-H6 explicit actions')
        self.calls += 1
        self.last_record = {'backend_calls': 1, 'native_shadow_rollouts': 0, 'response_probe_rollouts': 0,
            'horizon': len(act_suffix), 'routing_horizons': [], 'future_routing_features_read': False,
            'runtime_implementation': 'causal_same_pass_v1'}
        if len(act_suffix) < 6 or self.arm in ('native', 'zero_dose'):
            return self.backend.predict(context, act_suffix)
        with CausalHook(self.backend.predictor, self.bank, self.routing, self.arm) as hook:
            result = self.backend.predict(context, act_suffix)
        if any(not torch.isfinite(result[key]).all() for key in ('visual', 'proprio')):
            raise ValueError('Nonfinite prediction; no candidate filtering')
        self.last_record.update(hook.record)
        self.last_record.update(hook.gate_record)
        self.last_record['routing_horizons'] = [1, 2]
        return result
