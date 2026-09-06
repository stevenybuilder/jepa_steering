from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_V03,
    NULL_CONTROL_HAZARD,
    FactorialCell,
    paired_did,
    relational_did,
    replay_gate,
    sha256_array,
)


def test_factorial_cell_id_is_stable() -> None:
    cell = FactorialCell("pair7", 7, "gentle_contact", 1, 0)
    assert cell.cell_id == "pair7__h1a0"
    assert cell.to_dict()["cell_id"] == cell.cell_id


def test_factorial_cell_rejects_invalid_levels() -> None:
    with pytest.raises(ValueError):
        FactorialCell("bad", 0, "contact", 3, 0)
    with pytest.raises(ValueError):
        FactorialCell("bad", 0, "contact", 0, 2)
    assert FactorialCell("ok", 0, "contact", NULL_CONTROL_HAZARD, 1).cell_id == "ok__h2a1"


def test_relational_did_subtracts_null_factor() -> None:
    values = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 0.0, (1, 1): 4.0, (2, 0): 0.0, (2, 1): 2.0}
    assert paired_did(values) == 3.0
    assert paired_did(values, NULL_CONTROL_HAZARD) == 1.0
    assert relational_did(values) == 2.0
    with pytest.raises(ValueError):
        relational_did({(0, 0): 0.0, (0, 1): 1.0, (1, 0): 0.0, (1, 1): 4.0})


def test_paired_did_uses_preregistered_sign() -> None:
    values = {(0, 0): 3.0, (0, 1): 5.0, (1, 0): 4.0, (1, 1): 11.0}
    assert paired_did(values) == 5.0


def test_paired_did_supports_activation_tensors() -> None:
    values = {
        (0, 0): np.asarray([0.0, 1.0]),
        (0, 1): np.asarray([1.0, 3.0]),
        (1, 0): np.asarray([2.0, 5.0]),
        (1, 1): np.asarray([6.0, 12.0]),
    }
    np.testing.assert_array_equal(paired_did(values), np.asarray([3.0, 5.0]))


def test_array_hash_includes_shape_and_dtype() -> None:
    base = np.arange(4, dtype=np.float32)
    assert sha256_array(base) == sha256_array(base.copy())
    assert sha256_array(base) != sha256_array(base.reshape(2, 2))
    assert sha256_array(base) != sha256_array(base.astype(np.float64))



def _row(**over):
    base = {
        "replay_max_state_error": 0.0,
        "replay_initial_state_error": 0.0,
        "replay_force_error_n": 0.0,
        "replay_contact_count_equal": True,
        "replay_frames_bit_exact": True,
        "replay_max_pixel_error": 0,
    }
    base.update(over)
    return base


def test_replay_gate_v02_requires_bit_exact_frames() -> None:
    assert replay_gate(_row(), PROTOCOL_VERSION)["passed"]
    r = replay_gate(_row(replay_frames_bit_exact=False, replay_max_pixel_error=1), PROTOCOL_VERSION)
    assert not r["passed"] and r["reason"] == "frames_not_bit_exact"


def test_replay_gate_v03_allows_one_lsb_one_pixel_only_when_verified() -> None:
    jitter = _row(replay_frames_bit_exact=False, replay_max_pixel_error=1)
    assert not replay_gate(jitter, PROTOCOL_VERSION_V03)["passed"]  # unverified count fails closed
    assert replay_gate(jitter, PROTOCOL_VERSION_V03, differing_pixels_per_frame=1)["passed"]
    assert not replay_gate(jitter, PROTOCOL_VERSION_V03, differing_pixels_per_frame=2)["passed"]
    two_lsb = _row(replay_frames_bit_exact=False, replay_max_pixel_error=2)
    assert replay_gate(two_lsb, PROTOCOL_VERSION_V03, differing_pixels_per_frame=1)["reason"] == "pixel_diff_exceeds_1_lsb"


def test_replay_gate_never_forgives_physics_mismatch() -> None:
    bad = _row(replay_max_state_error=1e-6)
    for proto in (PROTOCOL_VERSION, PROTOCOL_VERSION_V03):
        assert replay_gate(bad, proto)["reason"] == "physics_replay_mismatch"


# --------------------------------------------------------------------------- #
# protocol v0.8 (driving): frame gate, hazard level 3, identity DiD
# --------------------------------------------------------------------------- #

from protocol import (  # noqa: E402
    HAZARD_LEVELS,
    OBJECT_HAZARD,
    PROTOCOL_VERSION_V08,
    V08_CELL_ORDER,
    did_by_identity,
    hazard_levels_for,
)


def test_v08_constants_and_levels() -> None:
    assert PROTOCOL_VERSION_V08 == "cgs-metadrive-pilot-v0.8"
    assert OBJECT_HAZARD == 3 and HAZARD_LEVELS == (0, 1, 2, 3)
    assert hazard_levels_for(PROTOCOL_VERSION_V03) == (0, 1, 2) and hazard_levels_for(PROTOCOL_VERSION_V08) == (0, 1, 2, 3)
    assert len(V08_CELL_ORDER) == 8 and set(V08_CELL_ORDER) == {(h, a) for h in range(4) for a in (0, 1)}
    # egg cells still reject level 3; v0.8 cells accept it
    with pytest.raises(ValueError):
        FactorialCell("bad", 0, "drive", 3, 0)
    cell = FactorialCell("s7", 7, "drive", 3, 1, PROTOCOL_VERSION_V08)
    assert cell.cell_id == "s7__h3a1" and cell.to_dict()["protocol_version"] == PROTOCOL_VERSION_V08
    assert FactorialCell("e", 1, "egg", 1, 0).to_dict()["protocol_version"] == PROTOCOL_VERSION


def test_did_by_identity_contrast() -> None:
    v = {(0, 0): 10.0, (0, 1): 9.0, (1, 0): 10.0, (1, 1): 4.0, (2, 0): 10.0, (2, 1): 8.5, (3, 0): 10.0, (3, 1): 9.0}
    d = did_by_identity(v)
    assert d["pedestrian"] == paired_did(v, 1) == -5.0
    assert d["object"] == paired_did(v, 3) == 0.0
    assert d["identity_x_action"] == -5.0
    assert d["null_factor"] == -0.5 and d["pedestrian_relational"] == -4.5 and d["object_relational"] == 0.5
    no_null = {k: x for k, x in v.items() if k[0] != 2}
    assert set(did_by_identity(no_null)) == {"pedestrian", "object", "identity_x_action"}
    with pytest.raises(ValueError):
        did_by_identity({k: x for k, x in v.items() if k[0] != 3})


def _drow(**over):
    base = {"replay_max_state_error": 0.0, "replay_initial_state_error": 0.0, "replay_contact_count_equal": True,
            "replay_frames_bit_exact": True, "replay_max_pixel_error": 0}
    base.update(over)
    return base


def test_replay_gate_v08_frame_gate() -> None:
    assert replay_gate(_drow(), PROTOCOL_VERSION_V08)["reason"] == "ok"
    # no force channel in MetaDrive: replay_force_error_n optional
    assert "replay_force_error_n" not in _drow() and replay_gate(_drow(), PROTOCOL_VERSION_V08)["passed"]
    jit = _drow(replay_frames_bit_exact=False, replay_max_pixel_error=8)
    r = replay_gate(jit, PROTOCOL_VERSION_V08)
    assert not r["passed"] and r["reason"] == "differing_pixel_count_unverified"
    assert replay_gate(jit, PROTOCOL_VERSION_V08, differing_pixels_per_frame=16)["reason"] == "ok_render_jitter_le16px"
    assert replay_gate(jit, PROTOCOL_VERSION_V08, differing_pixels_per_frame=17)["reason"] == "more_than_16_differing_pixels"
    # count recorded by the generator in the row
    assert replay_gate(_drow(replay_frames_bit_exact=False, replay_max_pixel_error=3, replay_max_differing_pixels_per_frame=5), PROTOCOL_VERSION_V08)["passed"]
    assert replay_gate(_drow(replay_frames_bit_exact=False, replay_max_pixel_error=17), PROTOCOL_VERSION_V08, 1)["reason"] == "pixel_diff_exceeds_16_lsb"
    # physics must be bit-exact: 1e-9 state error passes v0.3 but fails v0.8
    tiny = _drow(replay_max_state_error=1e-9, replay_force_error_n=0.0)
    assert replay_gate(tiny, PROTOCOL_VERSION_V03)["passed"]
    assert replay_gate(tiny, PROTOCOL_VERSION_V08)["reason"] == "physics_replay_mismatch"
    assert replay_gate(_drow(replay_contact_count_equal=False), PROTOCOL_VERSION_V08)["reason"] == "physics_replay_mismatch"
