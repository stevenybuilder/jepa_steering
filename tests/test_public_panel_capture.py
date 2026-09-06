from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_capture import finalize, flush_episode, infer_n_spatial, site_specs  # noqa: E402


class _Capture:
    n_spatial = 2

    def __init__(self) -> None:
        self.plan_buf = {
            ("L00.resid_post", 0): [np.ones((3, 2), dtype=np.float32)],
            ("L00.resid_post", 1): [2.0 * np.ones((4, 2), dtype=np.float32)],
        }

    def pop(self) -> dict[str, np.ndarray]:
        return {
            "L00.resid_post__mean": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            "L00.resid_post__mean_all": np.array([[1.5, 1.5], [2.5, 2.5]], dtype=np.float32),
        }


def test_all_plan_capture_keeps_tagged_populations_outside_replan_axis(tmp_path: Path) -> None:
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    capture = _Capture()
    state = {
        "cap": capture,
        "chosen_actions": [np.ones((1, 1)), np.ones((1, 1))],
        "steps_left": [100, 85],
        "planner_subsample": 256,
        "planner_capture_scope": "all_plans",
        "_act_dir": activation_dir,
        "episodes": [],
        "ep": 0,
        "errors": [],
        "extra_forward_s": 0.0,
    }
    flush_episode(state, {"ep": 0, "success": 1, "n_plan_calls": 2})
    with np.load(activation_dir / "ep000.npz") as arrays:
        assert arrays["L00.resid_post__mean"].shape == (2, 2)
        assert arrays["L00.resid_post__planner_call000"].shape == (3, 2)
        assert arrays["L00.resid_post__planner_call001"].shape == (4, 2)
        assert np.array_equal(
            arrays["L00.resid_post__planner"], arrays["L00.resid_post__planner_call000"]
        )
    index = finalize(state, tmp_path, {})
    assert index["planner_capture_scope"] == "all_plan_calls_tagged_by_physical_replan"
    saved = json.loads((activation_dir / "index.json").read_text())
    assert saved["episodes"][0]["n_planner_rows_by_call"] == {"0": 3, "1": 4}


def test_site_specs_supports_released_dino_wm_transformer_layout() -> None:
    attention = object()
    feed_forward = object()
    predictor = SimpleNamespace(
        transformer=SimpleNamespace(layers=[[attention, feed_forward]])
    )
    specs = list(site_specs(predictor))
    assert specs == [
        ("L00.resid_post", feed_forward, "ff_resid"),
        ("L00.attn_out", attention, "tensor"),
        ("L00.mlp_out", feed_forward, "tensor"),
    ]


def test_n_spatial_falls_back_without_an_attention_mask() -> None:
    assert infer_n_spatial(SimpleNamespace(), fallback=49) == 49
