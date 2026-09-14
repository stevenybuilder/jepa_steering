"""Check that media capture preserves execution and rejects partial episodes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('capture_episode', Path(__file__).parents[1]/'scripts/capture_jepa_episode.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class Env:
    dt = .0125
    _last_rand_vec = np.zeros(6)
    _target_pos = np.zeros(3)
    def __init__(self):
        self.data = SimpleNamespace(time=0., **{key:np.zeros(2) for key in capture.PHYSICS})
        self._prev_obs = np.zeros(18)
    def _get_obs(self):
        obs=np.concatenate([np.full(18,self.data.time),self._prev_obs,np.zeros(3)])
        self._prev_obs=np.full(18,self.data.time)
        return obs
    def step(self, action):
        self.data.time += self.dt
        return self._get_obs(), 1., False, False, {'success':0.}


def test_capture_preserves_actions_results_and_restores_hooks():
    raw=Env(); env=SimpleNamespace(proprio_env=SimpleNamespace(unwrapped=raw))
    action=np.array([.1,.2,.3,.4]); record={}
    class Evaluator:
        def unroll_agent(self, env):
            for _ in range(100): env.proprio_env.unwrapped.step(action)
            return 'original result'
    original=Evaluator.unroll_agent
    with capture.capture_unroll(Evaluator,record):
        assert Evaluator().unroll_agent(env)=='original result'
    assert Evaluator.unroll_agent is original
    np.testing.assert_array_equal(action,[.1,.2,.3,.4])
    assert len(record['actions'])==100
    assert record['frames'][2]['state'][18] == pytest.approx(.0125)
    capture.validate_capture(record)
    record['frames'].pop()
    with pytest.raises(ValueError,match='Incomplete'): capture.validate_capture(record)


def test_capture_restores_hooks_after_failure():
    raw=Env(); env=SimpleNamespace(proprio_env=SimpleNamespace(unwrapped=raw))
    class Evaluator:
        def unroll_agent(self, env): raise RuntimeError('failed run')
    original=Evaluator.unroll_agent; step=raw.step
    with pytest.raises(RuntimeError):
        with capture.capture_unroll(Evaluator,{}): Evaluator().unroll_agent(env)
    assert Evaluator.unroll_agent is original and raw.step==step


def test_initial_snapshot_does_not_advance_observation_history():
    env=Env(); env.data.time=1.
    capture.snapshot(env)
    np.testing.assert_array_equal(env._prev_obs,np.zeros(18))
