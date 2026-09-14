import numpy as np
import pytest

from analysis.mechanism.lcfm_replication_summary import aggregate, estimate, selection_metrics, wilson


def costs(values, elites=None):
    return {'costs': list(values), 'elite_indices': list(range(10)) if elites is None else elites}


def test_wilson_handles_zero_and_all_and_wider_family_interval():
    assert wilson(0, 100, .95) == pytest.approx([0., .03699349820698568])
    assert wilson(100, 100, .95) == pytest.approx([.9630065017930143, 1.])
    low, high = wilson(50, 100, .95)
    assert [low, high] == pytest.approx([.4038315303659956, .5961684696340044])
    family = wilson(50, 100, .975)
    assert family[0] < low and family[1] > high


def test_winner_tie_changes_id_without_excess_cost():
    reference = np.arange(300, dtype=float)+1
    reference[1] = reference[0]
    current = reference.copy(); current[1] = .5
    result = selection_metrics(costs(reference), costs(current))
    assert result['winner_changed'] == 1
    assert result['selected_in_reference_minimum_set'] == 1
    assert result['excess_reference_cost'] == 0
    assert result['reference_minimum_ties'] == 2


def test_excess_cost_uses_coherent_reference_and_retains_zero_denominator():
    reference = np.arange(300, dtype=float)
    current = reference.copy(); current[2] = -1
    result = selection_metrics(costs(reference), costs(current))
    assert result['excess_reference_cost'] == 2
    assert result['excess_reference_cost_percent'] is None
    reference += 2
    result = selection_metrics(costs(reference), costs(current))
    assert result['excess_reference_cost_percent'] == 100


def test_minimum_and_elite_cutoff_ties_reported_separately():
    values = np.arange(300, dtype=float)
    values[8:12] = 8
    result = selection_metrics(costs(values), costs(values))
    assert result['reference_minimum_ties'] == 1
    assert result['reference_top10_cutoff_ties'] == 4
    assert result['reference_top10_overlap'] == 10


def test_undefined_context_does_not_shrink_denominator():
    weights = np.full((3, 100), .01)
    values = np.ones(100); values[-1] = np.nan
    result = estimate(values, weights)
    assert result['n'] == 100 and result['n_defined'] == 99
    assert result['mean'] is None and result['marginal_95_low'] is None


def test_partial_or_duplicated_population_rejected_before_reading_metrics(tmp_path):
    rows = [{'task': t, 'episode': i} for t in ('reach', 'reach-wall') for i in range(100)]
    with pytest.raises(ValueError, match='Complete 200'):
        aggregate(rows[:-1], tmp_path/'partial')
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match='Complete 200'):
        aggregate(rows, tmp_path/'duplicated')
    assert not (tmp_path/'partial').exists()
