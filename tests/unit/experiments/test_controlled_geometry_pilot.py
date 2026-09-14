import json
from pathlib import Path

import pytest
import torch

from offline_study.experiments.controlled_geometry_pilot import ANCHORS, OFFSETS, CaptureH3, action_batch, predictor_precision, summarize_fields, tensor_hash, write_json, validate_manifest, REQUIRED_SOURCE, file_hash, tree_hash


def polynomial(coefficients):
    x = torch.tensor(OFFSETS, dtype=torch.float64)[:, None]
    return sum(torch.tensor(c, dtype=torch.float64)[None] * x**i
               for i, c in enumerate(coefficients))


def test_affine_center_and_derivative():
    row = summarize_fields(polynomial([[2., -3.], [4., 7.]]))
    assert row['linear_center_mse'] < 1e-28
    assert row['cubic_center_mse'] < 1e-28
    for diff in row['finite_differences']:
        assert diff['central_first_derivative_rms'] == pytest.approx((32.5)**.5)
        assert diff['central_second_derivative_rms'] < 1e-10


def test_quadratic_compatible_center_not_third_order_identification():
    fields = polynomial([[2., -3.], [4., 7.], [8., -2.], [12., 20.]])
    row = summarize_fields(fields)
    assert row['linear_center_mse'] > 0
    assert row['cubic_center_mse'] < 1e-28
    for diff in row['finite_differences']:
        assert diff['central_second_derivative_rms'] == pytest.approx((136.)**.5)


def test_gram_independent_reconstruction():
    gen = torch.Generator().manual_seed(42)
    fields = torch.randn(5, 3, 7, generator=gen, dtype=torch.float64)
    row = summarize_fields(fields)
    gram = torch.tensor(row['centered_anchor_gram_mean_inner_product'], dtype=torch.float64)
    deltas = fields[list(ANCHORS)].reshape(4, -1) - fields[2].reshape(1, -1)
    assert torch.allclose(gram, deltas @ deltas.T / 21)
    for name in ('linear', 'cubic'):
        assert row[name + '_center_mse'] == pytest.approx(row[name + '_gram_mse'], abs=1e-14)


def test_roundtrip_has_own_center_and_exact_unchanged_fraction():
    fields = polynomial([[1., 2.], [0.001, 0.001]]).float()
    original = summarize_fields(fields)
    rounded = summarize_fields(fields.bfloat16().float())
    assert original['exact_unchanged_fraction'] == [0.] * 4
    assert rounded['exact_unchanged_fraction'] == [1.] * 4
    assert rounded['linear_center_mse'] == 0
    assert rounded['donor_response_rms'] == [0.] * 4
    assert rounded['center_sha256'] == tensor_hash(fields.bfloat16().float()[2])


@pytest.mark.parametrize('bad', [torch.ones(4, 2), torch.full((5, 2), float('nan'))])
def test_invalid_fields_rejected(bad):
    with pytest.raises(ValueError):
        summarize_fields(bad)


def test_actions_only_literal_h3_coordinate_changes():
    bank = torch.zeros(6, 300, 20)
    bank[:, 1:] = 17
    before = bank.clone()
    actual = action_batch(bank)
    expected = torch.zeros(6, 5, 20)
    expected[2, :, 0] = torch.tensor(OFFSETS)
    assert torch.equal(actual, expected)
    assert torch.equal(bank, before)
    bank[0, 0, 0] = 1
    with pytest.raises(ValueError, match='zero-action'):
        action_batch(bank)


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    def forward(self, x):
        for block in self.predictor_blocks:
            x = block(x)
        return x


def test_hook_parity_horizon_and_cleanup():
    predictor = Predictor()
    x = torch.arange(5 * 257 * 2).reshape(5, 257, 2).float()
    with CaptureH3(predictor) as capture:
        for _ in range(6):
            assert torch.equal(predictor(x), x)
    capture.validate()
    assert len(capture.fields) == 6
    for field in capture.fields.values():
        assert torch.equal(field, x[:, -256:])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in predictor.modules())


def test_hook_cleanup_on_failure_and_incomplete_capture():
    predictor = Predictor()
    with pytest.raises(RuntimeError):
        with CaptureH3(predictor):
            raise RuntimeError('fixture')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in predictor.modules())
    with CaptureH3(predictor) as capture:
        predictor(torch.zeros(5, 256, 2))
    with pytest.raises(ValueError, match='Incomplete'):
        capture.validate()


def test_precision_restored_even_on_error():
    backend = type('Backend', (), {'precision': 'float32', 'autocast_dtype': None})()
    with pytest.raises(RuntimeError):
        with predictor_precision(backend, 'bfloat16'):
            assert backend.autocast_dtype == torch.bfloat16
            raise RuntimeError('fixture')
    assert backend.precision == 'float32' and backend.autocast_dtype is None


def test_exclusive_receipts_and_protocol():
    protocol = json.loads((Path(__file__).parents[3] / 'paper/data/controlled_geometry_protocol.json').read_text())
    assert protocol['offsets'] == list(OFFSETS)
    assert protocol['off_center_evaluations'] == []
    assert protocol['context_precision'] == 'float32_shared_encoded_once'
    assert sum(map(len, protocol['scenarios'].values())) == 64


def test_no_overwrite(tmp_path):
    path = tmp_path / 'DONE.json'
    write_json(path, {'ok': True})
    with pytest.raises(FileExistsError):
        write_json(path, {'ok': False})


def test_manifest_exact_sources_and_cohort_fail_closed(tmp_path):
    protocol = json.loads((Path(__file__).parents[3] / 'paper/data/controlled_geometry_protocol.json').read_text())
    source, vendor = tmp_path / 'source', tmp_path / 'vendor'
    source.mkdir()
    vendor.mkdir()
    for name in REQUIRED_SOURCE:
        (source / name).write_text('# synthetic fixture only\n')
    (vendor / 'model.py').write_text('# fixture\n')
    (vendor / 'config.yaml').write_text('fixture: true\n')
    manifest = {key: protocol[key] for key in ('scenarios', 'input_manifest_sha256', 'checkpoint_sha256')}
    manifest['source_sha256'] = {name: file_hash(source / name) for name in REQUIRED_SOURCE}
    manifest['vendor_source_sha256'] = tree_hash(vendor)
    validate_manifest(manifest, protocol, source, vendor)
    (vendor / 'config.yaml').write_text('fixture: changed\n')
    with pytest.raises(ValueError, match='Vendor'):
        validate_manifest(manifest, protocol, source, vendor)
    manifest['vendor_source_sha256'] = tree_hash(vendor)
    manifest['source_sha256'].pop('backends.py')
    with pytest.raises(ValueError, match='Missing'):
        validate_manifest(manifest, protocol, source, vendor)
