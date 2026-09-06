import json
import sys
from pathlib import Path

import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'geometry_map'))
from task_metric_complement_ablation_v1 import assert_fold, decompose, gate, score, component_diagnostics


def fixture():
    x = np.array([[1., 2., 3.], [2., 1., 1.], [3., 0., 4.]])
    b = np.eye(3)[:2]
    q = np.array([[0., -1.], [1., 0.]])
    return x, b, q


def test_uncentered_energy_decomposition_and_ones_identity():
    x, b, q = fixture()
    c, z = decompose(x, b, q, {'ones': np.ones(2)})
    assert np.array_equal(z, x[:, :2])
    assert np.array_equal(c['complement_only'], x[:, 2] ** 2)
    assert np.array_equal(c['native_full'], c['ones/learned/with_complement'])
    assert np.array_equal(c['native_full'], c['ones/rotated_control/with_complement'])


def test_exact_old_formula_and_rotation_preserves_complement():
    x, b, q = fixture(); w = np.array([.2, 3.])
    c, z = decompose(x, b, q, {'frozen': w})
    for name, coords in [('learned', z), ('rotated_control', z @ q)]:
        old = np.sum(x*x, axis=1) + (coords**2) @ (w-1)
        assert np.allclose(c[f'frozen/{name}/with_complement'], old, atol=1e-12)
        assert np.allclose(c[f'frozen/{name}/with_complement'] - c[f'frozen/{name}/span_only'], x[:, 2]**2)


def test_complement_can_change_rank_without_any_refit():
    x = np.array([[1., 9.], [2., 0.]])
    c, _ = decompose(x, np.array([[1., 0.]]), np.eye(1), {'f': np.array([2.])})
    assert c['f/learned/span_only'].argmin() == 0
    assert c['f/learned/with_complement'].argmin() == 1


def test_nonorthogonal_and_negative_weights_fail():
    x, b, q = fixture()
    with unittest.TestCase().assertRaises(ValueError): decompose(x, b*2, q, {'w': np.ones(2)})
    with unittest.TestCase().assertRaises(ValueError): decompose(x, b, q*2, {'w': np.ones(2)})
    with unittest.TestCase().assertRaises(ValueError): decompose(x, b, q, {'w': np.array([-1., 1.])})


def test_state_exclusion_and_gate_not_candidate_count():
    assert_fold({'held_state': 2, 'train_states': [0, 1, 3]}, 2)
    with unittest.TestCase().assertRaises(ValueError): assert_fold({'held_state': 2, 'train_states': [0, 1, 2]}, 2)
    assert not gate([0, 0, 0, .1], [0, 0, 0, 0])['pass_exploratory_gate']
    assert gate([.02, .02, .02, 0], [0, 0, 0, 0])['pass_exploratory_gate']
    assert not gate([.02]*4, [.02]*4)['pass_exploratory_gate']


def test_scoring_coverage_and_xy_are_distinct():
    s = np.zeros((3, 7)); s[:, 0] = [0, 3, 4]
    r = score([2, 0, 1], s, np.zeros(7), [.1, .9, .2], [0, 1, 2])
    assert r['selected_index'] == 1
    assert r['goal_coverage_delta_vs_baseline'] == .8
    assert r['xy_delta_vs_baseline_px'] == 3


def test_mean_energy_does_not_imply_rank_dominance():
    x = np.array([[1., 100.], [2., 100.], [3., 100.]])
    c, _ = decompose(x, np.array([[1., 0.]]), np.eye(1), {})
    d = component_diagnostics(c)
    assert d['complement_energy_fraction_mean'] > .99
    assert d['components']['complement_only']['pair_difference_rms'] == 0
    assert d['components']['unweighted_span_only']['pair_difference_rms'] > 0


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(f) for n, f in sorted(globals().items()) if n.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
