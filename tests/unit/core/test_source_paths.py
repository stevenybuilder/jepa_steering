import hashlib
from pathlib import Path

import pytest

from offline_study._paths import (
    PACKAGE_ROOT, frozen_analysis_path, frozen_source_path, package_source_hash,
    snapshot_source_files, source_files, source_path,
)


def put(root, name, body='pass\n'):
    p = root/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_legacy_name_and_explicit_path_resolve_current_module():
    p = source_path('lcfm_replication.py')
    assert p == PACKAGE_ROOT/'experiments/lcfm_replication.py'
    assert source_path('src/offline_study/lcfm_replication.py') == p
    assert source_path('experiments/lcfm_replication.py') == p


def test_reject_traversal_ambiguous_missing_and_symlink(tmp_path):
    for name in ('../outside.py', '/outside.py', 'data.json'):
        with pytest.raises(ValueError): source_path(name, tmp_path)
    put(tmp_path, 'a/duplicate.py'); put(tmp_path, 'b/duplicate.py')
    with pytest.raises(ValueError, match='2 matches'): source_path('duplicate.py', tmp_path)
    with pytest.raises(ValueError, match='0 matches'): source_path('missing.py', tmp_path)
    (tmp_path/'linked.py').symlink_to(tmp_path/'a/duplicate.py')
    with pytest.raises(ValueError): source_path('linked.py', tmp_path)


def test_package_hash_binds_nested_names_and_bytes(tmp_path):
    put(tmp_path, '__init__.py'); nested=put(tmp_path, 'deep/model.py')
    expected=hashlib.sha256()
    for p in source_files(tmp_path):
        expected.update(p.relative_to(tmp_path).as_posix().encode()+b'\0'+p.read_bytes())
    baseline=package_source_hash(tmp_path)
    assert baseline == expected.hexdigest()
    nested.write_text('changed\n')
    assert package_source_hash(tmp_path) != baseline
    baseline=package_source_hash(tmp_path);nested.rename(nested.with_name('renamed.py'))
    assert package_source_hash(tmp_path) != baseline


def test_legacy_flat_scope_retained_and_structured_scope_complete(tmp_path):
    flat=put(tmp_path, 'model.py');put(tmp_path, 'refined_panel/helper.py')
    assert snapshot_source_files(tmp_path) == [flat]
    put(tmp_path, '_paths.py')
    assert len(snapshot_source_files(tmp_path)) == 3


def test_frozen_evidence_requires_original_hash_and_explicit_snapshot(tmp_path, monkeypatch):
    source=tmp_path/'src/offline_study'; original=put(source, 'original.py')
    helper=put(tmp_path, 'analysis/mechanism/helper.py', 'x=1\n')
    monkeypatch.setenv('JEPA_FROZEN_SOURCE_ROOT',str(source))
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    assert frozen_source_path('original.py',digest) == original
    assert frozen_analysis_path('helper.py', hashlib.sha256(helper.read_bytes()).hexdigest()) == helper
    original.write_text('modified\n')
    with pytest.raises(ValueError,match='do not rebind'): frozen_source_path('original.py',digest)


def test_live_source_resolution_never_borrows_frozen_snapshot(tmp_path, monkeypatch):
    put(tmp_path, 'lcfm_replication.py', '# archived bytes\n')
    monkeypatch.setenv('JEPA_FROZEN_SOURCE_ROOT',str(tmp_path))
    assert source_path('lcfm_replication.py') == PACKAGE_ROOT/'experiments/lcfm_replication.py'
