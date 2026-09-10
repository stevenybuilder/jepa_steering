#!/usr/bin/env python3
"""Spatial token groups on the 16x16 patch grid of the JEPA-WM predictor.

Per cell, ``<stimulus_dir>/masks/<cell_id>.npz`` (written by
``validate_hazard_labels.py``) carries, for frames ``[context, future1, future2,
future3]``: ``egg_mask`` bool [4,256,256], ``robot_mask`` bool [4,256,256],
``eef_px`` float [4,2] (x,y), ``egg_px`` float [4,2], plus ``cam_pos``, ``cam_mat``,
``fovy``.  From these we derive, per frame, the token groups

- ``egg``        patches overlapping ``egg_mask``, dilated by one patch (8-neighbourhood)
- ``gripper``    patches overlapping ``robot_mask`` within a 3-patch Chebyshev radius of ``eef_px``
- ``corridor``   patches on the straight segment ``eef_px[0] -> egg_px[0]`` (context frame),
                 excluding ``egg`` and ``gripper``
- ``background`` everything else
- ``gripper_corridor`` = gripper U corridor (the primary localization target)
- ``all``        all 256 tokens (whole-residual reference)

Frame-to-step mapping: the tokens the predictor sees at imagined step ``s`` are
the last-frame tokens, i.e. frame ``s`` (0 = context, s>=1 = the predicted frame
``s``).  ``frame_for_step(s, offset)`` defaults to ``offset=0``; the coordinator's
``s+1`` convention is available with ``offset=1`` (it indexes the object positions
of the frame being predicted rather than the frame being read).  The endpoint
region for the final predicted latent uses frame ``n_steps`` (= future3).

Fallback: when the mask file is missing, ``egg`` is derived from the manifest's
``target_bbox_xyxy`` (pixels in the 256x256 render), ``gripper``/``corridor`` are
empty, and ``source == "bbox_fallback"``.

Driving domain (protocol v0.8, MetaDrive). ``masks/<cell_id>.npz`` carries
``hazard_mask`` bool [T,256,256] (pedestrian or cone) and ``corridor_mask`` bool
[T,256,256] (ego-lane polygon ahead). Native groups:

- ``hazard``          patches overlapping ``hazard_mask``, dilated by one patch
- ``corridor``        patches overlapping ``corridor_mask``, excluding ``hazard``
- ``background``      everything else
- ``hazard_corridor`` = hazard U corridor (the primary localization / gate region)
- ``all``

Egg-name aliases are exposed so every consumer runs unchanged: ``egg`` ->
``hazard``, ``gripper_corridor`` -> ``corridor``, ``gripper`` -> empty (no
manipulator in the scene). ``alias_note(domain)`` returns the mapping that the
scripts write into their JSON under ``token_group_source``. The domain is set
per process with ``set_domain`` (from ``--domain``) and, independently, detected
from the mask file's keys, so a driving mask can never be read as an egg mask.
Bbox fallback in driving uses ``hazard_bbox_xyxy`` (or ``target_bbox_xyxy``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


GRID = 16
PATCH = 16
IMG = GRID * PATCH
GROUP_NAMES = ("egg", "gripper", "corridor", "background", "gripper_corridor", "all")
CANDIDATE_GROUPS = ("gripper_corridor", "egg", "gripper", "corridor")
DOMAINS = ("egg", "driving")
DRIVING_GROUP_NAMES = ("hazard", "corridor", "background", "hazard_corridor", "all")
DRIVING_CANDIDATE_GROUPS = ("hazard_corridor", "hazard", "corridor")
# egg-name -> driving-name (consumers keep using the egg names); "" = empty group
DRIVING_ALIASES = {"egg": "hazard", "gripper": "", "gripper_corridor": "corridor"}
# reverse aliases so driving names also resolve on egg groups (never used by the egg pipeline)
EGG_ALIASES = {"hazard": "egg", "hazard_corridor": "gripper_corridor"}
# region = the endpoint token region scored by the gate / patching (egg+gripper vs hazard+corridor)
REGION_GROUPS = {"egg": ("egg", "gripper"), "driving": ("hazard", "corridor")}
PRIMARY_GROUP = {"egg": "gripper_corridor", "driving": "hazard_corridor"}
_DOMAIN = ["egg"]


def set_domain(domain: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    _DOMAIN[0] = domain
    return domain


def current_domain() -> str:
    return _DOMAIN[0]


def group_names(domain: str | None = None, include_aliases: bool = False) -> tuple[str, ...]:
    """Group names for a domain; with ``include_aliases`` the egg names are accepted too (driving)."""

    domain = domain or current_domain()
    if domain == "egg":
        return GROUP_NAMES
    return DRIVING_GROUP_NAMES + (tuple(DRIVING_ALIASES) if include_aliases else ())


def candidate_groups(domain: str | None = None) -> tuple[str, ...]:
    domain = domain or current_domain()
    return CANDIDATE_GROUPS if domain == "egg" else DRIVING_CANDIDATE_GROUPS + tuple(k for k, v in DRIVING_ALIASES.items() if v)


def alias_note(domain: str | None = None) -> dict[str, Any] | None:
    """What the JSON outputs record under ``token_group_source`` for a non-egg domain (None for egg)."""

    domain = domain or current_domain()
    if domain == "egg":
        return None
    return {
        "domain": domain,
        "native_groups": list(DRIVING_GROUP_NAMES),
        "aliases": {k: (v or "(empty: no manipulator in the driving scene)") for k, v in DRIVING_ALIASES.items()},
        "masks": "hazard_mask (pedestrian or cone) and corridor_mask (ego-lane polygon ahead), [T,256,256] bool",
        "primary_group": PRIMARY_GROUP[domain],
        "region_groups": list(REGION_GROUPS[domain]),
    }


@dataclass
class CellGroups:
    source: str  # "masks" | "bbox_fallback"
    frames: list[dict[str, np.ndarray]] = field(default_factory=list)  # per frame: name -> sorted int token indices
    domain: str = "egg"

    def group(self, frame: int, name: str) -> np.ndarray:
        frame = min(max(frame, 0), len(self.frames) - 1)
        table = self.frames[frame]
        if name in table:
            return table[name]
        alias = (DRIVING_ALIASES if self.domain == "driving" else EGG_ALIASES).get(name)
        if alias is None:
            raise KeyError(f"unknown token group {name!r} for domain {self.domain!r}")
        if alias == "":
            return np.zeros(0, dtype=np.int64)
        return table[alias]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def frame_for_step(step: int, offset: int = 0) -> int:
    return step + offset


def _patch_of_px(xy: np.ndarray) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    return int(np.clip(y // PATCH, 0, GRID - 1)), int(np.clip(x // PATCH, 0, GRID - 1))  # (row, col)


def _mask_to_patches(mask: np.ndarray) -> np.ndarray:
    """bool [256,256] -> bool [16,16], true where any pixel of the patch is set."""

    m = np.asarray(mask, dtype=bool)
    if m.shape != (IMG, IMG):
        raise ValueError(f"mask shape {m.shape} != {(IMG, IMG)}")
    return m.reshape(GRID, PATCH, GRID, PATCH).any(axis=(1, 3))


def _dilate(patches: np.ndarray, radius: int = 1) -> np.ndarray:
    out = patches.copy()
    rows, cols = np.nonzero(patches)
    for r, c in zip(rows, cols):
        out[max(0, r - radius) : r + radius + 1, max(0, c - radius) : c + radius + 1] = True
    return out


def _segment_patches(a_px: np.ndarray, b_px: np.ndarray) -> np.ndarray:
    grid = np.zeros((GRID, GRID), dtype=bool)
    a = np.asarray(a_px, dtype=np.float64)
    b = np.asarray(b_px, dtype=np.float64)
    n = int(max(2, np.ceil(np.abs(b - a).max() / (PATCH / 2)) + 1))
    for t in np.linspace(0.0, 1.0, n):
        r, c = _patch_of_px(a + t * (b - a))
        grid[r, c] = True
    return grid


def _to_index(grid: np.ndarray) -> np.ndarray:
    return np.flatnonzero(grid.reshape(-1)).astype(np.int64)


def _finalize(egg: np.ndarray, gripper: np.ndarray, corridor: np.ndarray) -> dict[str, np.ndarray]:
    egg = egg.copy()
    gripper = gripper & ~egg
    corridor = corridor & ~egg & ~gripper
    background = ~(egg | gripper | corridor)
    return {
        "egg": _to_index(egg),
        "gripper": _to_index(gripper),
        "corridor": _to_index(corridor),
        "background": _to_index(background),
        "gripper_corridor": _to_index(gripper | corridor),
        "all": np.arange(GRID * GRID, dtype=np.int64),
    }


def groups_from_masks(masks: dict[str, np.ndarray], gripper_radius: int = 3, egg_dilate: int = 1) -> CellGroups:
    egg_mask = np.asarray(masks["egg_mask"], dtype=bool)
    robot_mask = np.asarray(masks["robot_mask"], dtype=bool)
    eef_px = np.asarray(masks["eef_px"], dtype=np.float64)
    egg_px = np.asarray(masks["egg_px"], dtype=np.float64)
    n_frames = egg_mask.shape[0]
    corridor_seg = _segment_patches(eef_px[0], egg_px[0])
    frames = []
    for f in range(n_frames):
        egg = _dilate(_mask_to_patches(egg_mask[f]), egg_dilate)
        robot = _mask_to_patches(robot_mask[f])
        r0, c0 = _patch_of_px(eef_px[f])
        near = np.zeros((GRID, GRID), dtype=bool)
        near[max(0, r0 - gripper_radius) : r0 + gripper_radius + 1, max(0, c0 - gripper_radius) : c0 + gripper_radius + 1] = True
        gripper = robot & near
        frames.append(_finalize(egg, gripper, corridor_seg))
    return CellGroups(source="masks", frames=frames)


def groups_from_bbox(bbox_xyxy: Any, n_frames: int = 4, egg_dilate: int = 1) -> CellGroups:
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    grid = np.zeros((GRID, GRID), dtype=bool)
    r0, c0 = _patch_of_px(np.asarray([x0, y0]))
    r1, c1 = _patch_of_px(np.asarray([x1, y1]))
    grid[min(r0, r1) : max(r0, r1) + 1, min(c0, c1) : max(c0, c1) + 1] = True
    egg = _dilate(grid, egg_dilate)
    empty = np.zeros((GRID, GRID), dtype=bool)
    frames = [_finalize(egg, empty, empty) for _ in range(n_frames)]
    return CellGroups(source="bbox_fallback", frames=frames)


def _finalize_driving(hazard: np.ndarray, corridor: np.ndarray) -> dict[str, np.ndarray]:
    hazard = hazard.copy()
    corridor = corridor & ~hazard
    background = ~(hazard | corridor)
    return {
        "hazard": _to_index(hazard),
        "corridor": _to_index(corridor),
        "background": _to_index(background),
        "hazard_corridor": _to_index(hazard | corridor),
        "all": np.arange(GRID * GRID, dtype=np.int64),
    }


def groups_from_driving_masks(masks: dict[str, np.ndarray], hazard_dilate: int = 1) -> CellGroups:
    hazard_mask = np.asarray(masks["hazard_mask"], dtype=bool)
    corridor_mask = np.asarray(masks["corridor_mask"], dtype=bool)
    if hazard_mask.ndim == 2:
        hazard_mask = hazard_mask[None]
    if corridor_mask.ndim == 2:
        corridor_mask = np.repeat(corridor_mask[None], hazard_mask.shape[0], axis=0)
    if corridor_mask.shape[0] != hazard_mask.shape[0]:
        raise ValueError(f"corridor_mask has {corridor_mask.shape[0]} frames, hazard_mask {hazard_mask.shape[0]}")
    frames = []
    for f in range(hazard_mask.shape[0]):
        hazard = _dilate(_mask_to_patches(hazard_mask[f]), hazard_dilate)
        frames.append(_finalize_driving(hazard, _mask_to_patches(corridor_mask[f])))
    return CellGroups(source="masks", frames=frames, domain="driving")


def groups_from_driving_bbox(bbox_xyxy: Any, n_frames: int = 4, hazard_dilate: int = 1) -> CellGroups:
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    grid = np.zeros((GRID, GRID), dtype=bool)
    r0, c0 = _patch_of_px(np.asarray([x0, y0]))
    r1, c1 = _patch_of_px(np.asarray([x1, y1]))
    grid[min(r0, r1) : max(r0, r1) + 1, min(c0, c1) : max(c0, c1) + 1] = True
    hazard = _dilate(grid, hazard_dilate)
    empty = np.zeros((GRID, GRID), dtype=bool)
    return CellGroups(source="bbox_fallback", frames=[_finalize_driving(hazard, empty) for _ in range(n_frames)], domain="driving")


def load_cell_groups(stimulus_dir: Path, row: dict[str, Any], n_frames: int = 4, domain: str | None = None) -> CellGroups:
    """Domain-aware loader. ``domain`` defaults to ``current_domain()``; a mask file whose
    keys identify the other domain wins over the requested domain (never mis-read a mask)."""

    domain = domain or current_domain()
    path = Path(stimulus_dir) / "masks" / f"{row['cell_id']}.npz"
    if path.exists():
        with np.load(path) as handle:
            keys = set(handle.files)
            if "hazard_mask" in keys and "egg_mask" not in keys:
                domain = "driving"
            elif "egg_mask" in keys and "hazard_mask" not in keys:
                domain = "egg"
            if domain == "driving":
                masks = {k: handle[k] for k in ("hazard_mask", "corridor_mask")}
                return groups_from_driving_masks(masks)
            masks = {k: handle[k] for k in ("egg_mask", "robot_mask", "eef_px", "egg_px")}
        return groups_from_masks(masks)
    if domain == "driving":
        bbox = row.get("hazard_bbox_xyxy", row.get("target_bbox_xyxy"))
        if bbox is None:
            raise KeyError(f"no masks/{row['cell_id']}.npz and no hazard_bbox_xyxy/target_bbox_xyxy in the manifest row")
        return groups_from_driving_bbox(bbox, n_frames=n_frames)
    return groups_from_bbox(row["target_bbox_xyxy"], n_frames=n_frames)


def union(*indices: np.ndarray) -> np.ndarray:
    if not indices:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate([np.asarray(i, dtype=np.int64) for i in indices]))


def summarize(groups: CellGroups) -> dict[str, Any]:
    if groups.domain == "egg":
        return {
            "source": groups.source,
            "n_frames": groups.n_frames,
            "sizes": [{name: int(len(frame[name])) for name in GROUP_NAMES} for frame in groups.frames],
        }
    return {
        "source": groups.source,
        "domain": groups.domain,
        "n_frames": groups.n_frames,
        "sizes": [{name: int(len(groups.group(f, name))) for name in group_names(groups.domain, include_aliases=True)} for f in range(groups.n_frames)],
        "token_group_source": alias_note(groups.domain),
    }


def cache_token_mask_driving(stimulus_dir: Path, cell_id: str, n_frames: int) -> tuple[np.ndarray, bool]:
    """Drop-in replacement for ``latent_cache.load_token_mask`` in the driving domain: the cache's
    per-frame token mask is hazard U corridor (the v0.8 gate region) instead of egg U robot.
    ``run_wave_drive.sh`` installs it with ``latent_cache.load_token_mask = token_groups.cache_token_mask_driving``
    (latent_cache.py itself is frozen)."""

    path = Path(stimulus_dir) / "masks" / f"{cell_id}.npz"
    if not path.exists():
        return np.ones((n_frames, GRID, GRID), dtype=bool), False
    with np.load(path) as z:
        hazard = np.asarray(z["hazard_mask"], dtype=bool)
        corridor = np.asarray(z["corridor_mask"], dtype=bool)
    if corridor.ndim == 2:
        corridor = np.repeat(corridor[None], hazard.shape[0], axis=0)
    tok = (hazard | corridor).reshape(hazard.shape[0], GRID, PATCH, GRID, PATCH).any(axis=(2, 4))
    if tok.shape[0] != n_frames:
        raise ValueError(f"mask {path} has {tok.shape[0]} frames, expected {n_frames}")
    return tok, True
