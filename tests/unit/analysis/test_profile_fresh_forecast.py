import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[3] / 'scripts/vast/profile_fresh_forecast.py'
spec = importlib.util.spec_from_file_location('forecast_profile', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_only_new_engineering_output_allowed(tmp_path):
    output = tmp_path / 'engineering-diagnostics/forecast-v1'
    assert m.validate_output(tmp_path, output) == output
    for wrong in (tmp_path / 'results-v2/profile', tmp_path / 'fresh-freeze-v2/profile'):
        with pytest.raises(ValueError): m.validate_output(tmp_path, wrong)
    output.mkdir(parents=True)
    with pytest.raises(ValueError): m.validate_output(tmp_path, output)


def test_paired_timing_upper_bound_not_guaranteed_speedup():
    result = m.summarize({'observer': [10., 12.], 'direct_adapter': [8., 10.]})
    assert result['observer_over_direct_ratio'] == 11 / 9
    assert result['tiny_sample_not_significance_test']
    slower = m.summarize({'observer': [1., 1.], 'direct_adapter': [2., 2.]})
    assert slower['upper_bound_removable_fraction'] == -1


def test_incomplete_timings_rejected():
    with pytest.raises(ValueError): m.summarize({'observer': [1.], 'direct_adapter': [1.]})


def test_capture_escapes_episode_completion():
    assert issubclass(m.Captured, BaseException) and not issubclass(m.Captured, Exception)
