from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_fit_apply_audit import aggregate, audit_episode  # noqa: E402


def _operator() -> dict[str, np.ndarray]:
    covariance = np.eye(2)[None]
    return {
        "regime_weights": np.ones(1), "regime_means": np.zeros((1, 2)), "regime_var": np.ones((1, 2)),
        "trust_metrics": np.eye(2)[None],
        "success_weights": np.ones(1), "success_means": np.array([[1.0, 0.0]]),
        "success_covariances": covariance,
        "failure_weights": np.ones(1), "failure_means": np.array([[-1.0, 0.0]]),
        "failure_covariances": covariance,
        "local_contrastive_conceptors": np.array([[[0.8, 0.0], [0.0, 0.5]]]),
        "action_gram": np.eye(2),
    }


def test_nearby_candidate_population_passes_field_checks() -> None:
    chosen = np.array([0.5, 0.5])
    candidates = chosen + np.array([[0.01, 0.0], [-0.01, 0.01], [0.0, -0.01]])
    row = audit_episode(chosen, candidates, _operator(), step_size=0.1, radius=0.25)
    report = aggregate([row], {
        "min_regime_agreement": 0.75, "min_edit_cosine": 0.8,
        "max_density_gap": 0.25, "max_abs_log_ratio": np.log(2.0),
    })
    assert report["fit_apply_equivalent"]
    assert row["energy_edit_cosine_median"] > 0.99


def test_replan_rows_are_collapsed_within_episode_before_thresholding() -> None:
    chosen = np.array([0.5, 0.5])
    nearby = chosen + np.array([[0.01, 0.0], [-0.01, 0.01]])
    rows = []
    for episode in range(2):
        for replan in range(3):
            row = audit_episode(chosen, nearby, _operator(), step_size=0.1, radius=0.25)
            row.update({"ep": episode, "replan": replan})
            rows.append(row)
    report = aggregate(rows, {
        "min_regime_agreement": 0.75, "min_edit_cosine": 0.8,
        "max_density_gap": 0.25, "max_abs_log_ratio": np.log(2.0),
    })
    assert report["fit_apply_equivalent"]
    assert [row["ep"] for row in report["episode_metrics"]] == [0, 1]
