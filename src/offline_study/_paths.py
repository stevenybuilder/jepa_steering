"""Source locations and identities for the reorganized package.

Current code never borrows archived bytes to satisfy an execution guard. Historical
analysis can explicitly select an original snapshot; its recorded SHA remains required.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def source_files(root=None):
    """All Python sources, including subpackage initializers, in stable path order."""
    root = PACKAGE_ROOT if root is None else Path(root).resolve()
    paths = sorted(root.rglob('*.py'), key=lambda p: p.relative_to(root).as_posix())
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError('Source tree contains a symlink or escaped path')
    return paths


def source_path(name, root=None):
    """Resolve a relative source path or a unique historical flat basename."""
    root = PACKAGE_ROOT if root is None else Path(root).resolve()
    name = Path(name)
    if name.is_absolute() or '..' in name.parts or name.suffix != '.py':
        raise ValueError('Expected a relative Python source name')
    if name.parts[:2] == ('src', 'offline_study'):
        name = Path(*name.parts[2:])
    path = root / name
    if path.is_file():
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Source path escapes its root')
        return path
    if len(name.parts) != 1:
        raise FileNotFoundError(path)
    matches = [p for p in source_files(root) if p.name == name.name]
    if len(matches) != 1:
        raise ValueError(f'Source {name} has {len(matches)} matches under {root}; require an explicit path')
    return matches[0]


def package_source_hash(root=None):
    """Hash every current package source by relative path and exact bytes."""
    root = PACKAGE_ROOT if root is None else Path(root).resolve()
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def frozen_source_path(name, expected_sha256, root=None):
    """Locate analysis evidence only when it matches the original recorded bytes."""
    root = root if root is not None else os.environ.get('JEPA_FROZEN_SOURCE_ROOT')
    try:
        path = source_path(name, root)
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError('Original source snapshot required: set JEPA_FROZEN_SOURCE_ROOT '
                         'to its src/offline_study directory') from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f'Frozen source bytes differ for {name}. Set JEPA_FROZEN_SOURCE_ROOT '
                         'to the original src/offline_study snapshot; do not rebind its SHA.')
    return path


def frozen_analysis_path(name, expected_sha256):
    """Resolve a hash-bound analysis helper from the same explicit snapshot."""
    selected = os.environ.get('JEPA_FROZEN_SOURCE_ROOT')
    root = Path(selected).resolve().parents[1] if selected else PACKAGE_ROOT.parents[1]
    return frozen_source_path(Path('analysis/mechanism') / name, expected_sha256, root)


def snapshot_source_files(root):
    """Preserve legacy flat snapshot scope; structured snapshots bind every file."""
    root = Path(root).resolve()
    return source_files(root) if (root / '_paths.py').is_file() else sorted(root.glob('*.py'))
