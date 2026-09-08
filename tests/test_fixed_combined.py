"""Synthetic CPU checks of the native-prefix combined adapter, not efficacy."""
import contextlib
import copy
import unittest
from unittest.mock import patch

import torch

from offline_study.fixed_combined import CombinedFixedResponseIntervention
from offline_study.fixed_response import FixedResponseIntervention
from offline_study.fixed_response_check import reference_fields
from offline_study.interventions import PredictorIntervention
from offline_study.planning_intervention import static_edits
from offline_study.support_operator import NativeFieldCapture
from test_fixed_response import fixture_bank


COMPONENTS = {
    'combined_fixed_rank4': ('joint_equal_standardized_energy', 'fixed_rank4'),
    'matched_random_combined_fixed_rank4': (
        'matched_random_equal_standardized_energy', 'matched_random_fixed_rank4'),
}


class Context(dict):
    def clone(self):
        return Context({k: v.clone() for k, v in self.items()})


class NonlinearBlock(torch.nn.Module):
    def __init__(self, index, trace):
        super().__init__()
        self.index, self.trace, self.visits = index, trace, 0
        self.gain = torch.nn.Parameter(torch.full((400,), .17 + index * .013), requires_grad=False)
        self.fail_on_visit, self.failure = None, None

    def forward(self, value, condition):
        self.visits += 1
        self.trace.append(self.index)
        if self.visits == self.fail_on_visit:
            raise self.failure
        batch, frames, channels = condition.shape
        expanded = condition[:, :, None, :].expand(batch, frames, 256, channels).reshape_as(value)
        # Nonlinearity is necessary: a coupled-field coefficient bug must not
        # accidentally look equivalent to native-feature coefficients.
        hidden = torch.tanh(value * self.gain.to(value.dtype) + expanded * .31)
        return value + hidden * (.035 * (self.index + 1)) + expanded * .011


class NonlinearPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trace, self.inputs = [], []
        self.embedding = torch.nn.Identity()
        self.predictor_blocks = torch.nn.ModuleList([NonlinearBlock(i, self.trace) for i in range(6)])
        self.use_activation_checkpointing = False
        self.proj_drop_prob = 0.
        self.eval()

    def forward(self, visual, actions, proprio):
        self.inputs.append({'values': [x.detach().clone() for x in (visual, actions, proprio)],
            'strides': [x.stride() for x in (visual, actions, proprio)],
            'pointers': [x.data_ptr() for x in (visual, actions, proprio)]})
        batch, frames = visual.shape[:2]
        value = self.embedding(visual).reshape(batch, frames * 256, 400)
        condition = actions + proprio * .23
        for block in self.predictor_blocks:
            value = block(value, condition)
        return value


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = NonlinearPredictor()
        self.unroll_calls = 0
        self.eval()

    def unroll(self, context, act_suffix):
        self.unroll_calls += 1
        visual, proprio = context['visual'], context['proprio']
        batch = act_suffix.shape[1]
        if visual.shape[0] == 1 and batch != 1:
            visual = visual.expand(batch, *visual.shape[1:])
            proprio = proprio.expand(batch, *proprio.shape[1:])
        if visual.shape[0] != batch or proprio.shape[:2] != visual.shape[:2]:
            raise ValueError('Synthetic fixture context batch mismatch')
        vs, ps = [visual[:, -1]], [proprio[:, -1]]
        for action in act_suffix:
            actions = action[:, None].expand(batch, visual.shape[1], 400)
            # The actual prefix caller must retain this noncontiguous layout;
            # the established visual coupling hook may clone its main input.
            output = self.predictor(visual.transpose(3, 4), actions, proprio)
            newest = output[:, -256:].reshape(batch, 1, 16, 16, 400)
            next_proprio = proprio[:, -1] + action * .1 + newest.flatten(1).mean(1, keepdim=True) * .01
            visual = torch.cat([visual[:, -1:], newest[:, None]], dim=1)
            proprio = torch.cat([proprio[:, -1:], next_proprio[:, None]], dim=1)
            vs.append(newest)
            ps.append(next_proprio)
        return {'visual': torch.stack(vs), 'proprio': torch.stack(ps)}


class Backend:
    def __init__(self):
        self.model, self.device = Model(), torch.device('cpu')
        self.allow_tf32, self.calls = False, 0

    @property
    def predictor(self):
        return self.model.predictor

    def autocast(self):
        return contextlib.nullcontext()

    def predict(self, context, actions):
        self.calls += 1
        return self.model.unroll(context.clone(), act_suffix=actions)


def coupling_fixture():
    protocol = {'category': 'vision_action_coupling', 'arms': []}
    visual = (torch.arange(256 * 400).remainder(41).float() / 127 + .1).reshape(1, 16, 16, 400)
    condition = torch.linspace(.03, .13, 400)
    bank = {'global_tensors': {'v': visual, 'a': condition, 'rv': -visual * .8, 'ra': -condition * .7}}
    for arm, v, a in [('joint_equal_standardized_energy', 'v', 'a'),
            ('matched_random_equal_standardized_energy', 'rv', 'ra')]:
        protocol['arms'].append({'name': arm, 'edits': [
            {'site': 'predictor_visual', 'horizon': 3, 'tensor': v, 'scale': .4},
            {'site': 'block_condition', 'block': 3, 'horizon': 3, 'tensor': a, 'scale': .5}]})
    return protocol, bank


def inputs(context_batch=2, candidates=2, horizon=6, dtype=torch.float32, zero_first=False):
    # Strided storage tests that neither original input nor its layout changes.
    storage = (torch.arange(context_batch * 2 * 256 * 800).remainder(47).float() / 163).to(dtype)
    visual = storage.reshape(context_batch, 2, 1, 16, 16, 800)[..., ::2]
    proprio = (torch.arange(context_batch * 2 * 800).remainder(13).float() / 97).to(dtype).reshape(context_batch, 2, 800)[..., ::2]
    actions = (torch.arange(horizon * candidates * 800).remainder(17).float() / 103).to(dtype).reshape(horizon, candidates, 800)[..., ::2]
    if zero_first:
        visual[0].zero_()
        proprio[0].zero_()
        actions[:, 0].zero_()
    return Context(visual=visual, proprio=proprio), actions


class BytePreservingCompiledReference(PredictorIntervention):
    """Original static compiler, with the original fixed map's inactive rule.

Generic static addition of +0 may change native -0 bytes. Rank-inactive rows
therefore retain their original field, independently of combined-adapter code.
"""
    def __init__(self, predictor, edits, active):
        super().__init__(predictor, edits)
        self.active = active

    def _replace(self, target, delta, label):
        result = super()._replace(target, delta, label)
        if label.startswith('block_output/'):
            mask = self.active.reshape((-1,) + (1,) * (target.ndim - 1))
            return torch.where(mask, result, target)
        return result


def reference(backend, fixed, protocol, coupling, arm, context, actions, *, coupled_features=False):
    coupling_arm, fixed_arm = COMPONENTS[arm]
    fields = static_edits(protocol, coupling, coupling_arm, actions.shape[1], 6, backend.device)
    with contextlib.ExitStack() as stack:
        capture = stack.enter_context(NativeFieldCapture(backend.predictor))
        if coupled_features:
            stack.enter_context(PredictorIntervention(backend.predictor, fields))
        native = backend.predict(context, actions)
    delta = reference_fields(capture.values[3], fixed, fixed_arm)
    active = delta[0].delta.flatten(1).norm(dim=1) > fixed['zero_threshold']
    # Compile anew, so prior engineering applications/counters are not reused.
    fields = static_edits(protocol, coupling, coupling_arm, actions.shape[1], 6, backend.device)
    with BytePreservingCompiledReference(backend.predictor, fields + delta, active):
        expected = backend.predict(context, actions)
    return expected, native, capture.values[3]


def same_bytes(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


class CombinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, arm='combined_fixed_rank4'):
        backend, fixed = Backend(), fixture_bank()
        fixed['operators']['matched_random_fixed_rank4']['map'] *= torch.tensor([-1., 1., -1., 1.])[:, None]
        protocol, coupling = coupling_fixture()
        adapter = CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, arm)
        return backend, fixed, protocol, coupling, adapter

    def assert_clean(self, predictor):
        for module in predictor.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_fp32_and_bf16_match_independent_native_capture_static_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for arm in COMPONENTS:
                with self.subTest(dtype=dtype, arm=arm):
                    backend, fixed, protocol, coupling, adapter = self.fixture(arm)
                    context, actions = inputs(dtype=dtype)
                    oracle_backend = Backend()
                    expected, native, _ = reference(oracle_backend, fixed, protocol, coupling, arm, context, actions)
                    self.assertEqual(oracle_backend.calls, 2)
                    self.assertEqual(len(oracle_backend.predictor.trace), 72)
                    actual = adapter(context, actions)
                    for key in ('visual', 'proprio'):
                        self.assertTrue(same_bytes(actual[key], expected[key]), key)
                        self.assertTrue(same_bytes(actual[key][:3], native[key][:3]))
                    self.assertFalse(same_bytes(actual['visual'][3:], native['visual'][3:]))
                    self.assertEqual(backend.calls, 1)
                    self.assertEqual(backend.model.unroll_calls, 1)
                    self.assertEqual(backend.predictor.trace, list(range(6)) * 2 + list(range(4)) + list(range(6)) * 4)
                    self.assertEqual([b.visits for b in backend.predictor.predictor_blocks], [7, 7, 7, 7, 6, 6])
                    self.assertEqual(adapter.last_record['native_prefix_replays'], 1)
                    self.assertEqual(adapter.last_record['extra_native_predictor_blocks'], 4)
                    self.assertEqual(adapter.last_record['main_predictor_blocks'], 36)
                    self.assertEqual(adapter.last_record['rank_applications'], 1)
                    self.assertEqual(adapter.last_record['response_probe_rollouts'], 0)
                    self.assertEqual(adapter.last_record['full_native_shadow_rollouts'], 0)
                    self.assertFalse(adapter.last_record['behavioral_launch_ready'])
                    self.assert_clean(backend.predictor)

    def test_coefficients_come_from_native_not_already_coupled_b3(self):
        backend, fixed, protocol, coupling, adapter = self.fixture()
        context, actions = inputs()
        correct, _, native_field = reference(Backend(), fixed, protocol, coupling, adapter.arm, context, actions)
        wrong, _, coupled_field = reference(Backend(), fixed, protocol, coupling, adapter.arm, context, actions, coupled_features=True)
        self.assertFalse(same_bytes(native_field, coupled_field))
        self.assertFalse(same_bytes(correct['visual'], wrong['visual']))
        actual = adapter(context, actions)
        self.assertTrue(same_bytes(actual['visual'], correct['visual']))
        self.assertFalse(same_bytes(actual['visual'], wrong['visual']))

    def test_all_short_horizons_are_true_native_with_no_coupling_or_replay(self):
        # Actual H6StaticPlanningIntervention/planning_support policy, not the
        # broader StaticPlanningIntervention's H3 behavior on short rollouts.
        for arm in COMPONENTS:
            for horizon in range(1, 6):
                with self.subTest(arm=arm, horizon=horizon):
                    backend, _, _, _, adapter = self.fixture(arm)
                    context, actions = inputs(horizon=horizon, dtype=torch.bfloat16)
                    expected = Backend().predict(context, actions)
                    actual = adapter(context, actions)
                    for key in expected:
                        self.assertTrue(same_bytes(actual[key], expected[key]))
                    self.assertEqual(backend.calls, 1)
                    self.assertEqual(len(backend.predictor.trace), horizon * 6)
                    self.assertEqual(adapter.last_record['native_prefix_replays'], 0)
                    self.assertEqual(adapter.last_record['rank_applications'], 0)
                    self.assert_clean(backend.predictor)

    def test_singleton_and_batched_context_inputs_layouts_and_parameters_unchanged(self):
        for context_batch, candidates in ((1, 3), (3, 3)):
            backend, fixed, protocol, coupling, adapter = self.fixture()
            context, actions = inputs(context_batch, candidates)
            self.assertFalse(context['visual'].is_contiguous())
            self.assertFalse(actions.is_contiguous())
            versions = {k: (v._version, v.stride(), v.clone()) for k, v in context.items()}
            before_action = (actions._version, actions.stride(), actions.clone())
            parameters = {k: v.clone() for k, v in backend.model.state_dict().items()}
            expected_backend = Backend()
            expected, _, _ = reference(expected_backend, fixed, protocol, coupling, adapter.arm, context, actions)
            actual = adapter(context, actions)
            self.assertTrue(same_bytes(actual['visual'], expected['visual']))
            for key, (version, stride, value) in versions.items():
                self.assertEqual(context[key]._version, version)
                self.assertEqual(context[key].stride(), stride)
                self.assertTrue(same_bytes(context[key], value))
            self.assertEqual(actions._version, before_action[0])
            self.assertEqual(actions.stride(), before_action[1])
            self.assertTrue(same_bytes(actions, before_action[2]))
            for key, before in parameters.items():
                self.assertTrue(same_bytes(backend.model.state_dict()[key], before))
            native_h3 = expected_backend.predictor.inputs[2]
            prefix, main_h3 = backend.predictor.inputs[2:4]
            self.assertEqual(prefix['strides'], native_h3['strides'])
            for before, after in zip(native_h3['values'], prefix['values']):
                self.assertTrue(same_bytes(before, after))
            # Source visual coupling can clone visual; action/proprio objects
            # must be the identical caller inputs for replay and main forward.
            self.assertEqual(prefix['pointers'][1:], main_h3['pointers'][1:])
            self.assert_clean(backend.predictor)

    def test_zero_rank_treatment_is_per_candidate_and_does_not_remove_coupling(self):
        for dtype in (torch.float32, torch.bfloat16):
            backend, fixed, protocol, coupling, _ = self.fixture()
            fixed['operators']['fixed_rank4']['map'][:, 0] = 0
            adapter = CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, 'combined_fixed_rank4')
            context, actions = inputs(dtype=dtype, zero_first=True)
            static_backend = Backend()
            edits = static_edits(protocol, coupling, 'joint_equal_standardized_energy', 2, 6, 'cpu')
            with PredictorIntervention(static_backend.predictor, edits):
                static = static_backend.predict(context, actions)
            actual = adapter(context, actions)
            self.assertEqual(adapter.last_record['active'].tolist(), [False, True])
            for key in ('visual', 'proprio'):
                self.assertTrue(same_bytes(actual[key][:, 0], static[key][:, 0]))
            self.assertFalse(same_bytes(actual['visual'][:, 1], static['visual'][:, 1]))
            self.assertEqual(float(adapter.last_record['requested_l2'][0]), 0.)
            self.assertAlmostEqual(float(adapter.last_record['requested_l2'][1]), fixed['dose'], places=5)

    def test_training_checkpointing_and_wrong_architecture_are_rejected(self):
        for changed in ('train', 'child_train', 'checkpoint', 'blocks', 'tf32', 'dropout'):
            backend, fixed = Backend(), fixture_bank()
            protocol, coupling = coupling_fixture()
            if changed == 'train': backend.predictor.train()
            if changed == 'child_train': backend.predictor.predictor_blocks[2].train()
            if changed == 'checkpoint': backend.predictor.use_activation_checkpointing = True
            if changed == 'blocks': backend.predictor.predictor_blocks = backend.predictor.predictor_blocks[:5]
            if changed == 'tf32': backend.allow_tf32 = True
            if changed == 'dropout': backend.predictor.predictor_blocks[1].proj_drop_prob = .1
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, 'combined_fixed_rank4')
            self.assertEqual(backend.calls, 0)
            self.assert_clean(backend.predictor)

    def test_changed_runtime_eval_policy_is_rechecked_on_call(self):
        backend, _, _, _, adapter = self.fixture()
        backend.predictor.train()
        with self.assertRaises(ValueError):
            adapter(*inputs())
        self.assertEqual(backend.calls, 0)

    def test_wrong_coupling_site_time_gate_scale_or_arm_are_rejected(self):
        changes = [lambda p, b: p['arms'][0]['edits'][0].update(horizon=2),
            lambda p, b: p['arms'][0]['edits'][1].update(block=2),
            lambda p, b: p['arms'][0]['edits'][0].update(site='block_output', block=3),
            lambda p, b: p['arms'][0]['edits'][0].update(gate='invented'),
            lambda p, b: p['arms'][0]['edits'][0].update(scale=float('nan')),
            lambda p, b: p['arms'].pop(),
            lambda p, b: p['arms'].append(copy.deepcopy(p['arms'][0])),
            lambda p, b: b['global_tensors']['v'].fill_(float('inf'))]
        for change in changes:
            backend, fixed = Backend(), fixture_bank()
            protocol, coupling = coupling_fixture()
            change(protocol, coupling)
            with self.assertRaises(ValueError):
                CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, 'combined_fixed_rank4')
            self.assertEqual(backend.calls, 0)

    def test_unknown_arm_or_extra_kwargs_are_not_silently_accepted(self):
        backend, fixed, protocol, coupling, adapter = self.fixture()
        with self.assertRaises(ValueError):
            CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, 'legacy_combined')
        with self.assertRaises(ValueError):
            adapter(*inputs(), extra_steps=1)
        self.assertEqual(backend.calls, 0)
        self.assert_clean(backend.predictor)

    def test_nonfinite_runtime_failure_cleans_every_hook(self):
        backend, _, _, _, adapter = self.fixture()
        context, actions = inputs()
        context['visual'][0, -1, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            adapter(context, actions)
        self.assert_clean(backend.predictor)

    def test_foreign_exceptions_propagate_from_prefix_and_main_without_masking(self):
        class ForeignFailure(BaseException):
            pass
        for block, visit in ((2, 3), (4, 3)):
            backend, _, _, _, adapter = self.fixture()
            error = ForeignFailure('fixture foreign failure')
            target = backend.predictor.predictor_blocks[block]
            target.fail_on_visit, target.failure = visit, error
            with self.assertRaises(ForeignFailure) as caught:
                adapter(*inputs())
            self.assertIs(caught.exception, error)
            self.assert_clean(backend.predictor)

    def test_preexisting_predictor_or_descendant_hooks_are_preserved_and_rejected(self):
        for site, pre in (('predictor', True), ('block', False), ('embedding', True)):
            backend, fixed = Backend(), fixture_bank()
            protocol, coupling = coupling_fixture()
            module = {'predictor': backend.predictor, 'block': backend.predictor.predictor_blocks[3],
                      'embedding': backend.predictor.embedding}[site]
            handle = (module.register_forward_pre_hook(lambda *args: None) if pre else
                      module.register_forward_hook(lambda *args: None))
            try:
                with self.assertRaises(ValueError):
                    CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, 'combined_fixed_rank4')
                self.assertTrue(module._forward_pre_hooks if pre else module._forward_hooks)
            finally:
                handle.remove()
            self.assert_clean(backend.predictor)

    def test_global_hooks_are_not_implicitly_replayed(self):
        from torch.nn.modules import module as module_state
        for pre in (True, False):
            backend, fixed = Backend(), fixture_bank()
            protocol, coupling = coupling_fixture()
            handle = (module_state.register_module_forward_pre_hook(lambda *args: None) if pre else
                      module_state.register_module_forward_hook(lambda *args: None))
            try:
                with self.assertRaises(ValueError):
                    CombinedFixedResponseIntervention(backend, fixed, protocol, coupling, 'combined_fixed_rank4')
            finally:
                handle.remove()
            self.assertEqual(backend.calls, 0)

    def test_failed_partial_hook_registration_cleans_up_without_masking_exception(self):
        backend, _, _, _, adapter = self.fixture()
        error = RuntimeError('synthetic registration failure')
        with patch.object(backend.predictor.predictor_blocks[2], 'register_forward_pre_hook', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                adapter(*inputs())
        self.assertIs(caught.exception, error)
        self.assertEqual(backend.calls, 0)
        self.assert_clean(backend.predictor)

    def test_reentrant_call_is_rejected_and_lock_is_released_after_failure(self):
        backend, _, _, _, adapter = self.fixture()
        context, actions = inputs()
        with patch.object(backend, 'predict', side_effect=lambda c, a: adapter(c, a)):
            with self.assertRaisesRegex(ValueError, 'reentrant'):
                adapter(context, actions)
        self.assert_clean(backend.predictor)
        actual = adapter(context, actions[:2])
        expected = Backend().predict(context, actions[:2])
        self.assertTrue(same_bytes(actual['visual'], expected['visual']))

    def test_private_prefix_exception_from_another_owner_is_not_swallowed(self):
        from offline_study.fixed_combined import _PrefixComplete
        backend, _, _, _, adapter = self.fixture()
        error = _PrefixComplete(object())
        target = backend.predictor.predictor_blocks[1]
        target.fail_on_visit, target.failure = 3, error
        with self.assertRaises(_PrefixComplete) as caught:
            adapter(*inputs())
        self.assertIs(caught.exception, error)
        self.assert_clean(backend.predictor)

    def test_original_standalone_adapter_still_rejects_composition(self):
        backend, fixed, protocol, coupling, _ = self.fixture()
        adapter = FixedResponseIntervention(backend, fixed, 'fixed_rank4')
        edits = static_edits(protocol, coupling, 'joint_equal_standardized_energy', 2, 6, 'cpu')
        context, actions = inputs()
        with self.assertRaisesRegex(ValueError, 'coexist'):
            with PredictorIntervention(backend.predictor, edits):
                adapter(context, actions)
        self.assert_clean(backend.predictor)


if __name__ == '__main__':
    unittest.main()
