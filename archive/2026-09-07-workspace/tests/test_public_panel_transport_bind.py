from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_transport_bind import bind_transport  # noqa: E402


def test_bind_transport_conjugates_source_operator_into_target_coordinates(tmp_path: Path) -> None:
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    source_regime = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    target_regime = source_regime @ rotation
    local = np.stack([np.diag([0.8, 0.2]), np.diag([0.6, 0.1])])
    source = tmp_path / "source.npz"
    target = tmp_path / "target.npz"
    modules = tmp_path / "modules.npz"
    common = {
        "coordinate_rank": np.asarray(2),
        "regime_weights": np.asarray([0.5, 0.5]),
        "success_weights": np.asarray([0.5, 0.5]),
        "failure_weights": np.asarray([0.5, 0.5]),
        "covariance_floor": np.asarray(1e-3),
    }
    np.savez_compressed(
        source,
        **common,
        regime_means=source_regime,
        success_means=source_regime + [0.2, 0.0],
        failure_means=source_regime - [0.2, 0.0],
        success_covariances=np.stack([np.eye(2), 2 * np.eye(2)]),
        failure_covariances=np.stack([2 * np.eye(2), np.eye(2)]),
        local_contrastive_conceptors=local,
        global_contrastive_conceptor=np.diag([0.7, 0.3]),
    )
    np.savez_compressed(
        target,
        **common,
        regime_means=target_regime,
        success_means=np.zeros((2, 2)),
        failure_means=np.zeros((2, 2)),
        success_covariances=np.stack([np.eye(2), np.eye(2)]),
        failure_covariances=np.stack([np.eye(2), np.eye(2)]),
        local_contrastive_conceptors=np.zeros((2, 2, 2)),
        hidden_mean=np.zeros(2),
        encoder=np.eye(2),
        decoder=np.eye(2),
    )
    np.savez_compressed(
        modules,
        transport_rotation=rotation,
        transport_mean_source=np.zeros(2),
        transport_mean_target=np.zeros(2),
        transport_eligible=np.asarray(True),
    )
    out = tmp_path / "transported.npz"
    report = bind_transport(source, target, modules, out)
    assert report["source_for_target_regime"] == [0, 1]
    with np.load(out) as fitted:
        np.testing.assert_allclose(fitted["success_means"], (source_regime + [0.2, 0.0]) @ rotation)
        np.testing.assert_allclose(
            fitted["local_contrastive_conceptors"][0], rotation.T @ local[0] @ rotation
        )
        assert fitted["relational_transport_eligible"].item()

