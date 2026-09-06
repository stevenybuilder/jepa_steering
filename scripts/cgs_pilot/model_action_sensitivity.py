#!/usr/bin/env python3
"""Run the Phase-0 action-conditioning preflight on frozen CGS quartets.

This is intentionally a model-level gate, not a mechanistic conclusion. It
checks that candidate actions change predicted latent futures and that matched
action/trajectory pairs are closer than swapped pairs before layer localization.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from protocol import sha256_file


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0)).item())


def load_model(
    repo: Path,
    config_path: Path,
    checkpoint: Path,
    model_name: str,
    device: str,
):
    """Use the official loader while disabling the optional image decoder."""

    sys.path.insert(0, str(repo))
    import hubconf  # type: ignore

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["folder"] = "/"
    config["model_kwargs"]["checkpoint"] = str(checkpoint.resolve())
    heads = config["model_kwargs"]["pretrain_kwargs"].setdefault("heads_cfg", {})
    heads["pretrain_dec_path"] = None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
        yaml.safe_dump(config, handle)
        handle.flush()
        model, preprocessor = hubconf._load_model_with_config(
            handle.name,
            model_name=model_name,
            device=device,
            pretrained=False,
        )
    model.eval()
    return model, preprocessor


def encode_frames(model, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    visual = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model.encode(visual)


def predict_cell(model, artifact: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(artifact) as cell:
        context = np.asarray(cell["context_frames"])
        future = np.asarray(cell["true_future_frames"])
        actions = np.asarray(cell["model_actions"])
    # The released planner seeds autoregressive unrolls from one current frame;
    # ``ctxt_window`` controls the rolling predicted history, not extra observed
    # frames without their preceding actions.
    z_context = encode_frames(model, context[-1:], device)
    z_future = encode_frames(model, future, device)
    act = torch.from_numpy(actions).to(device=device, dtype=torch.float32).unsqueeze(1)
    with torch.inference_mode():
        z_rollout = model.unroll(z_context, act_suffix=act)
    context_steps = int(z_context.shape[1])
    return {
        "context": z_context.detach().cpu(),
        "target": z_future.detach().cpu(),
        "prediction": z_rollout[context_steps:].detach().cpu(),
    }


def analyze_pair(cells: dict[tuple[int, int], dict[str, torch.Tensor]]) -> dict[str, Any]:
    pred = {key: value["prediction"][-1] for key, value in cells.items()}
    target = {key: value["target"][:, -1] for key, value in cells.items()}
    context = {key: value["context"][:, -1] for key, value in cells.items()}

    action_separation_rms = {
        str(hazard): rms(pred[(hazard, 0)], pred[(hazard, 1)]) for hazard in (0, 1)
    }
    empirical_separation_rms = {
        str(hazard): rms(target[(hazard, 0)], target[(hazard, 1)]) for hazard in (0, 1)
    }
    matched = np.mean([rms(pred[key], target[key]) for key in sorted(cells)])
    swapped = np.mean(
        [rms(pred[(hazard, action)], target[(hazard, 1 - action)]) for hazard in (0, 1) for action in (0, 1)]
    )
    displacement = {
        key: rms(pred[key], context[key]) for key in sorted(cells)
    }
    scalar_did = (
        displacement[(1, 1)]
        - displacement[(1, 0)]
        - displacement[(0, 1)]
        + displacement[(0, 0)]
    )
    interaction = (
        pred[(1, 1)] - pred[(1, 0)] - pred[(0, 1)] + pred[(0, 0)]
    )
    return {
        "predicted_action_separation_rms": action_separation_rms,
        "predicted_action_separation_cosine": {
            str(hazard): cosine_distance(pred[(hazard, 0)], pred[(hazard, 1)]) for hazard in (0, 1)
        },
        "empirical_action_separation_rms": empirical_separation_rms,
        "matched_prediction_rms": float(matched),
        "swapped_prediction_rms": float(swapped),
        "matched_vs_swapped_margin_rms": float(swapped - matched),
        "predicted_displacement_rms": {
            f"h{hazard}a{action}": value for (hazard, action), value in displacement.items()
        },
        "scalar_displacement_did": float(scalar_did),
        "vector_interaction_rms": float(torch.sqrt(torch.mean(interaction.float() ** 2)).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="dino_wm_droid")
    parser.add_argument("--model-label", default="DINO-WM DROID method preflight")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)
    model, _ = load_model(
        args.repo, args.config, args.checkpoint, args.model_name, str(device)
    )
    rows = [json.loads(line) for line in (args.artifacts / "manifest.jsonl").read_text().splitlines()]
    grouped: dict[str, dict[tuple[int, int], dict[str, torch.Tensor]]] = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], {})[(row["hazard"], row["candidate_action"])] = predict_cell(
            model, args.artifacts / row["artifact"], device
        )
    incomplete = [pair_id for pair_id, cells in grouped.items() if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}]
    if incomplete:
        raise RuntimeError(f"incomplete factorial quartets: {incomplete}")

    pairs = {pair_id: analyze_pair(cells) for pair_id, cells in grouped.items()}
    separations = [
        value
        for result in pairs.values()
        for value in result["predicted_action_separation_rms"].values()
    ]
    margins = [result["matched_vs_swapped_margin_rms"] for result in pairs.values()]
    result = {
        "model_label": args.model_label,
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.artifacts / "manifest.jsonl"),
        "torch_version": torch.__version__,
        "device": str(device),
        "stimulus_summary": json.loads((args.artifacts / "summary.json").read_text()),
        "pair_results": pairs,
        "phase0_model_gate": {
            "pair_count": len(pairs),
            "finite_metrics": bool(all(np.isfinite(separations + margins))),
            "nonzero_action_sensitivity": bool(min(separations) > 1e-7),
            "mean_predicted_action_separation_rms": float(np.mean(separations)),
            "positive_pairing_margin_fraction": float(np.mean(np.asarray(margins) > 0)),
            "mean_matched_vs_swapped_margin_rms": float(np.mean(margins)),
        },
        "interpretation_scope": (
            "Pipeline/action-conditioning preflight only; do not attribute these results "
            "to JEPA-WM when model_label identifies a baseline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["phase0_model_gate"], indent=2))


if __name__ == "__main__":
    main()
