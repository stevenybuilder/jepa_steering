"""Frozen bookkeeping primitives for the CGS factorial pilot.

This module deliberately has no simulator or model dependencies.  It defines the
cell identities, canonical hashes, and the one allowed difference-in-differences
contrast used by the pilot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROTOCOL_VERSION = "cgs-robocasa-pilot-v0.2"
# v0.3 (2026-09-01): identical to v0.2 except the frame-replay rule. v0.2 required
# bit-identical replay frames. Seed 201 showed that the EGL renderer can flip a
# single pixel by 1/255 on repeated renders of the SAME unstepped state while
# every per-step MuJoCo state is bit-identical (see verify_render_jitter.py and
# artifacts/cgs_pilot/render_jitter_seed_201.json). v0.3 therefore allows at most
# one differing pixel per frame with |diff| <= 1/255, still with exact state,
# force, and contact-count replay. Decided by the user after seeing that data and
# before any model analysis; both versions are reported.
PROTOCOL_VERSION_V03 = "cgs-robocasa-pilot-v0.3"
REPLAY_STATE_TOL = 1e-8
REPLAY_FORCE_TOL_N = 1e-6
V03_MAX_PIXEL_ABS_DIFF = 1
V03_MAX_DIFFERING_PIXELS_PER_FRAME = 1
CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
# v0.3 null factor: hazard level 2 = H0' (second off-path placement at a pixel
# displacement matched to H0->H1). The (H0'-H0) x A interaction is a same-size
# visual-change x action interaction with no route relation; the relational
# effect is (H1-H0) x A minus (H0'-H0) x A. Added after the arXiv review noted
# that AdaLN action modulation produces a nonzero elementwise interaction at
# every nonlinear site for ANY hazard-token change.
# Driving pilot (MetaDrive, protocol v0.7 amendment; metadrive_hazard_pilot.py). Replay rule preregistered 2026-09-02
# before any driving data: exact physics replay (fresh process, same in-process history => bit-exact ego states) and
# at most 16 differing pixels per frame with |diff| <= 16/255 (EGL renderer jitter measured in the 20-seed pilot).
PROTOCOL_VERSION_DRIVE = "cgs-metadrive-pilot-v0.7"
DRIVE_MAX_PIXEL_ABS_DIFF = 16  # amended 2026-09-02 (pre-outcome): renderer jitter up to 14/255 measured in the 20-seed pilot
DRIVE_MAX_DIFFERING_PIXELS_PER_FRAME = 16
NULL_CONTROL_HAZARD = 2
# v0.8 (2026-09-02, MetaDrive driving pilot; experiment_design.md "v0.7 stimulus
# instantiation"): hazard level 3 = a second identity ON the path (envelope-matched
# cone in the ego lane). The identity x action contrast is DiD(level 1) - DiD(level 3)
# (``did_by_identity``); which identity carries the physical consequence is an arm
# property (arm A: pedestrian solid / cone ghost; arm B: reversed).
OBJECT_HAZARD = 3
EGG_HAZARD_LEVELS = (0, 1, NULL_CONTROL_HAZARD)
HAZARD_LEVELS = (0, 1, NULL_CONTROL_HAZARD, OBJECT_HAZARD)
NULL_CELL_ORDER = ((0, 0), (0, 1), (NULL_CONTROL_HAZARD, 0), (NULL_CONTROL_HAZARD, 1))
IDENTITY_CELL_ORDER = ((0, 0), (0, 1), (OBJECT_HAZARD, 0), (OBJECT_HAZARD, 1))
# Full v0.8 scene: 8 cells = levels {0, 1, 2, 3} x actions {0 = brake, 1 = throttle}.
V08_CELL_ORDER = tuple((h, a) for h in HAZARD_LEVELS for a in (0, 1))
PROTOCOL_VERSION_V08 = "cgs-metadrive-pilot-v0.8"
# v0.9 (2026-09-03): same 8-cell scene and the SAME replay/frame gate as v0.8; only the stimulus factors vary
# (metadrive_hazard_pilot.py --prefix-throttle/--lateral-offset/--randomize/--fov/--settle-steps). Rows and provenance
# of runs that used any of those flags carry this string; unchanged runs keep the v0.7/v0.8 strings.
PROTOCOL_VERSION_V09 = "cgs-metadrive-pilot-v0.9"
# v0.8 frame gate (preregistered from the MetaDrive spike's jitter measurement,
# before any factorial data): physics / ego state bit-exact on replay (no
# tolerance); renders may differ on at most 16 pixels per frame with
# |diff| <= 16/255; anything worse fails closed. No contact-force channel exists
# in MetaDrive, so ``replay_force_error_n`` is optional under v0.8.
V08_STATE_TOL = 0.0
V08_MAX_PIXEL_ABS_DIFF = 16  # amended 2026-09-02 (pre-outcome), matches experiment_design.md tolerances
V08_MAX_DIFFERING_PIXELS_PER_FRAME = 16
HAZARD_LEVEL_NAMES = {
    "egg": {0: "off_path_control", 1: "on_path_hazard", 2: "matched_displacement_control"},
    "driving": {0: "sidewalk_pedestrian", 1: "pedestrian_in_lane", 2: "matched_displacement_sidewalk_pose", 3: "cone_in_lane"},
}
ACTION_LEVEL_NAMES = {"egg": {0: "gentle", 1: "aggressive"}, "driving": {0: "brake", 1: "throttle"}}


def hazard_levels_for(protocol: str) -> tuple[int, ...]:
    """Admissible hazard levels under a protocol version (v0.2/v0.3: 0/1/2; v0.8 adds 3)."""

    return HAZARD_LEVELS if protocol == PROTOCOL_VERSION_V08 else EGG_HAZARD_LEVELS


@dataclass(frozen=True)
class FactorialCell:
    """One member of a paired hazard x candidate-action quartet."""

    pair_id: str
    seed: int
    risk_family: str
    hazard: int
    candidate_action: int
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.hazard not in hazard_levels_for(self.protocol_version) or self.candidate_action not in (0, 1):
            raise ValueError(
                "hazard must be 0, 1, or 2 (null control)"
                + (", or 3 (object on path)" if self.protocol_version == PROTOCOL_VERSION_V08 else "")
                + "; candidate_action must be binary"
            )

    @property
    def cell_id(self) -> str:
        return f"{self.pair_id}__h{self.hazard}a{self.candidate_action}"

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version": PROTOCOL_VERSION, **asdict(self), "cell_id": self.cell_id}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)}).encode()
    return sha256_bytes(header + b"\0" + array.tobytes())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_did(values: Mapping[tuple[int, int], Any], hazard_level: int = 1) -> Any:
    """Return (Y_h1 - Y_h0) - (Y01 - Y00) for a complete paired quartet.

    ``hazard_level`` selects which hazard contrast is taken against H0: 1 for the
    real hazard, ``NULL_CONTROL_HAZARD`` for the H0' null-factor contrast.
    """

    needed = {(0, 0), (0, 1), (hazard_level, 0), (hazard_level, 1)}
    missing = needed.difference(values)
    if missing:
        raise ValueError(f"incomplete factorial quartet; missing {sorted(missing)}")
    return (values[(hazard_level, 1)] - values[(hazard_level, 0)]) - (values[(0, 1)] - values[(0, 0)])


def relational_did(values: Mapping[tuple[int, int], Any]) -> Any:
    """(H1-H0)xA interaction minus the (H0'-H0)xA null-factor interaction."""

    return paired_did(values, 1) - paired_did(values, NULL_CONTROL_HAZARD)


def did_by_identity(values: Mapping[tuple[int, int], Any]) -> dict[str, Any]:
    """v0.8 identity x action contrast from a complete 8-cell (or 6-cell 0/1/3) scene.

    Returns the hazard x action DiD for the pedestrian (level 1 vs 0), for the
    object (level 3 vs 0) and their difference ``identity_x_action`` =
    DiD(pedestrian) - DiD(object). Under arm A (pedestrian solid) the difference
    is predicted positive in the consequence currency; under arm B negative. The
    optional relational versions subtract the H0' null-factor DiD when the
    ``hazard == 2`` cells are present.
    """

    ped = paired_did(values, 1)
    obj = paired_did(values, OBJECT_HAZARD)
    out: dict[str, Any] = {"pedestrian": ped, "object": obj, "identity_x_action": ped - obj}
    if (NULL_CONTROL_HAZARD, 0) in values and (NULL_CONTROL_HAZARD, 1) in values:
        null = paired_did(values, NULL_CONTROL_HAZARD)
        out["null_factor"] = null
        out["pedestrian_relational"] = ped - null
        out["object_relational"] = obj - null
    return out


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(row)) + "\n")


def replay_gate(row: Mapping[str, Any], protocol: str = PROTOCOL_VERSION_V03,
                differing_pixels_per_frame: int | None = None) -> dict[str, Any]:
    """Evaluate the replay-determinism gate for one manifest row.

    ``differing_pixels_per_frame`` is the maximum over frames of the number of
    differing pixels between the two replays. The v0.2 generator does not record
    it (only ``replay_max_pixel_error``), so for v0.3 it must be supplied from
    ``verify_render_jitter.py`` for every cell that is not bit-exact; if it is
    missing for such a cell the v0.3 gate fails closed.

    v0.8 (driving): physics/ego state must be bit-exact (error 0.0), the
    contact flag equal when recorded (``replay_contact_count_equal`` or
    ``replay_contact_equal``), force error optional; renders may differ on at
    most ``V08_MAX_DIFFERING_PIXELS_PER_FRAME`` pixels per frame with
    ``|diff| <= V08_MAX_PIXEL_ABS_DIFF``. The per-frame count comes from
    ``differing_pixels_per_frame`` or, when the generator records it, from the
    row field ``replay_max_differing_pixels_per_frame``; a non-bit-exact cell
    without a count fails closed.
    """

    if protocol in (PROTOCOL_VERSION_V08, PROTOCOL_VERSION_V09):  # v0.9 = v0.8 replay gate
        return _replay_gate_v08(row, differing_pixels_per_frame)
    physics_ok = (
        float(row["replay_max_state_error"]) <= REPLAY_STATE_TOL
        and float(row.get("replay_initial_state_error", 0.0)) <= REPLAY_STATE_TOL
        and float(row["replay_force_error_n"]) <= REPLAY_FORCE_TOL_N
        and bool(row["replay_contact_count_equal"])
    )
    bit_exact = bool(row["replay_frames_bit_exact"])
    if protocol == PROTOCOL_VERSION:
        passed = physics_ok and bit_exact
        reason = "ok" if passed else ("frames_not_bit_exact" if physics_ok else "physics_replay_mismatch")
    elif protocol == PROTOCOL_VERSION_V03:
        if not physics_ok:
            passed, reason = False, "physics_replay_mismatch"
        elif bit_exact:
            passed, reason = True, "ok"
        elif int(row["replay_max_pixel_error"]) > V03_MAX_PIXEL_ABS_DIFF:
            passed, reason = False, "pixel_diff_exceeds_1_lsb"
        elif differing_pixels_per_frame is None:
            passed, reason = False, "differing_pixel_count_unverified"
        elif differing_pixels_per_frame > V03_MAX_DIFFERING_PIXELS_PER_FRAME:
            passed, reason = False, "more_than_one_differing_pixel"
        else:
            passed, reason = True, "ok_render_jitter_1px"
    elif protocol == PROTOCOL_VERSION_DRIVE:
        n = row.get("replay_max_differing_pixels_per_frame", differing_pixels_per_frame)
        if not physics_ok:
            passed, reason = False, "physics_replay_mismatch"
        elif bit_exact:
            passed, reason = True, "ok"
        elif int(row["replay_max_pixel_error"]) > DRIVE_MAX_PIXEL_ABS_DIFF:
            passed, reason = False, "pixel_diff_exceeds_8_lsb"
        elif n is None:
            passed, reason = False, "differing_pixel_count_unverified"
        elif int(n) > DRIVE_MAX_DIFFERING_PIXELS_PER_FRAME:
            passed, reason = False, "more_than_16_differing_pixels"
        else:
            passed, reason = True, "ok_render_jitter_le16px"
    else:
        raise ValueError(f"unknown protocol {protocol}")
    return {"protocol": protocol, "passed": bool(passed), "reason": reason,
            "physics_ok": bool(physics_ok), "frames_bit_exact": bit_exact}


def _replay_gate_v08(row: Mapping[str, Any], differing_pixels_per_frame: int | None) -> dict[str, Any]:
    contact_equal = row.get("replay_contact_count_equal", row.get("replay_contact_equal", True))
    physics_ok = (
        float(row["replay_max_state_error"]) <= V08_STATE_TOL
        and float(row.get("replay_initial_state_error", 0.0)) <= V08_STATE_TOL
        and float(row.get("replay_force_error_n", 0.0)) <= REPLAY_FORCE_TOL_N
        and bool(contact_equal)
    )
    bit_exact = bool(row["replay_frames_bit_exact"])
    if differing_pixels_per_frame is None and row.get("replay_max_differing_pixels_per_frame") is not None:
        differing_pixels_per_frame = int(row["replay_max_differing_pixels_per_frame"])
    if not physics_ok:
        passed, reason = False, "physics_replay_mismatch"
    elif bit_exact:
        passed, reason = True, "ok"
    elif int(row["replay_max_pixel_error"]) > V08_MAX_PIXEL_ABS_DIFF:
        passed, reason = False, f"pixel_diff_exceeds_{V08_MAX_PIXEL_ABS_DIFF}_lsb"
    elif differing_pixels_per_frame is None:
        passed, reason = False, "differing_pixel_count_unverified"
    elif differing_pixels_per_frame > V08_MAX_DIFFERING_PIXELS_PER_FRAME:
        passed, reason = False, f"more_than_{V08_MAX_DIFFERING_PIXELS_PER_FRAME}_differing_pixels"
    else:
        passed, reason = True, f"ok_render_jitter_le{V08_MAX_DIFFERING_PIXELS_PER_FRAME}px"
    return {"protocol": PROTOCOL_VERSION_V08, "passed": bool(passed), "reason": reason,
            "physics_ok": bool(physics_ok), "frames_bit_exact": bit_exact}
