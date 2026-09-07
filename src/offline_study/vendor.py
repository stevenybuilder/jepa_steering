"""Explicit, pinned local upstream checkout; no silent source upgrades."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import VENDOR_COMMIT

VENDOR_SNAPSHOT_SHA256 = "797c4a2765b57cc2e6d17be23d7031a8b0427092706034210897998e6f27de90"


def use_vendor(path: Path) -> Path:
    path = path.resolve()
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        commit_marker = path / ".source-commit"
        archive_marker = path / ".source-archive-sha256"
        if not commit_marker.is_file() or not archive_marker.is_file():
            raise RuntimeError("Upstream snapshot is missing provenance markers") from None
        actual = commit_marker.read_text().strip()
        if archive_marker.read_text().strip() != VENDOR_SNAPSHOT_SHA256:
            raise RuntimeError("Upstream snapshot archive SHA256 marker is invalid") from None
    if actual != VENDOR_COMMIT:
        raise RuntimeError(f"Upstream must be {VENDOR_COMMIT}; found {actual}")
    if (path / ".git").exists() and subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise RuntimeError("Upstream checkout has modified tracked files")
    sys.path.insert(0, str(path))
    return path


def open_dataset(kind: str, root: Path, pool: str):
    """Raw official observations/actions; normalization happens once in the model adapter."""
    if kind == "metaworld":
        from app.plan_common.datasets.metaworld_hf_dset import MetaworldHFDataset
        return MetaworldHFDataset(data_path=str(root), n_rollout=None, transform=None,
                                  normalize_action=False, filter_tasks=None, with_reward=False)
    if kind == "pusht":
        from app.plan_common.datasets.pusht_dset import PushTDataset
        return PushTDataset(data_path=str(root / pool), n_rollout=None, transform=None,
                            normalize_action=False, with_velocity=True)
    raise ValueError(kind)
