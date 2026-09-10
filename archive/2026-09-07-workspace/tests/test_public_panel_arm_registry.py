from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_arm_registry import freeze_registry  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    stimulus = tmp_path / "stimulus"
    stimulus.mkdir()
    _write_json(stimulus / "manifest_meta.json", {"schema_version": "panel-p-paired-stimulus-v1"})
    (stimulus / "fit.jsonl").write_text("{}\n")
    (stimulus / "evaluation.jsonl").write_text("{}\n")
    operator = tmp_path / "operator.npz"
    operator.write_bytes(b"operator")
    evidence = {}
    for name, field in (
        ("fit_apply", "fit_apply_equivalent"),
        ("identity_action_hash", "passed"),
        ("action_dose", "passed"),
        ("development_utility", "passed"),
    ):
        path = tmp_path / f"{name}.json"
        _write_json(path, {field: True})
        evidence[name] = {"name": name, "path": str(path), "required_true_field": field}
    spec = {
        "primary_endpoint": "paired native simulator success",
        "primary_comparison": "selected_utility_arm_vs_unsteered",
        "selected_on_split": "development",
        "method_hyperparameters": {
            "coordinate_rank": 8, "shrinkage": 0.25, "covariance_floor": 1e-3,
            "support_quantile": 0.99, "conceptor_aperture": 1.0,
            "conceptor_rcond": 1e-12, "bootstrap_draws": 200,
        },
        "artifacts": {"sonar": str(operator)},
        "evidence": list(evidence.values()),
        "arms": [
            {"name": "unsteered"},
            {"name": "identity"},
            {
                "name": "sonar_coast_global", "operator_artifact": "sonar",
                "site": "L00.resid_post", "beta": 0.2,
                "hook_schedule": "all_physical_replans", "displacement_mode": "joint",
            },
        ],
    }
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, spec)
    return spec_path, stimulus


def test_registry_hashes_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    spec, stimulus = _fixture(tmp_path)
    out = tmp_path / "registry.json"
    registry = freeze_registry(spec, stimulus, out)
    assert registry["primary_comparison"] == "selected_utility_arm_vs_unsteered"
    assert set(registry["evidence"]) == {
        "fit_apply", "identity_action_hash", "action_dose", "development_utility"
    }
    with pytest.raises(FileExistsError):
        freeze_registry(spec, stimulus, out)


def test_hmm_arm_requires_all_hmm_specific_evidence(tmp_path: Path) -> None:
    spec_path, stimulus = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["arms"].append({
        "name": "sonar_coast_hmm", "operator_artifact": "sonar",
        "site": "L00.resid_post", "beta": 0.2,
        "hook_schedule": "all_physical_replans", "displacement_mode": "joint",
    })
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="hmm_deployment"):
        freeze_registry(spec_path, stimulus, tmp_path / "registry.json")


def test_admitted_frankenstein_arm_requires_module_gate_evidence(tmp_path: Path) -> None:
    spec_path, stimulus = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    treated = next(arm for arm in spec["arms"] if arm["name"] not in {"unsteered", "identity"})
    treated["frankenstein_mode"] = "admitted"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="frankenstein_modules"):
        freeze_registry(spec_path, stimulus, tmp_path / "registry.json")


def test_registry_rejects_unknown_frankenstein_ablation(tmp_path: Path) -> None:
    spec_path, stimulus = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    treated = next(arm for arm in spec["arms"] if arm["name"] not in {"unsteered", "identity"})
    treated["frankenstein_mode"] = "admitted"
    treated["frankenstein_ablate"] = ["made_up_module"]
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="invalid Frankenstein ablations"):
        freeze_registry(spec_path, stimulus, tmp_path / "registry.json")


def test_registry_requires_admitted_gate_for_attention_restoration(tmp_path: Path) -> None:
    spec_path, stimulus = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    treated = next(arm for arm in spec["arms"] if arm["name"] not in {"unsteered", "identity"})
    treated["attention_restore_site"] = "L00.attn_out"
    treated["attention_restore_strength"] = 1.0
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="cannot restore attention"):
        freeze_registry(spec_path, stimulus, tmp_path / "registry.json")


def test_registry_allows_same_operator_family_with_unique_ablation_ids(tmp_path: Path) -> None:
    spec_path, stimulus = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    original = next(arm for arm in spec["arms"] if arm["name"] == "sonar_coast_global")
    original["id"] = "sonar_coast_global_full"
    ablation = dict(original)
    ablation["id"] = "sonar_coast_global_second_condition"
    spec["arms"].append(ablation)
    _write_json(spec_path, spec)
    registry = freeze_registry(spec_path, stimulus, tmp_path / "registry.json")
    assert registry["arm_condition_ids"][-2:] == [
        "sonar_coast_global_full", "sonar_coast_global_second_condition"
    ]
