#!/usr/bin/env python3
"""Fail closed on corrupt, non-finite, or shape-incompatible capture shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from protocol import file_sha256


POOLED_SHAPES = {
    "encoder": (20, 12, 384),
    "encoder_cls": (20, 12, 384),
    "predictor": (19, 6, 400),
    "encoded_final": (20, 384),
    "predicted_final": (19, 384),
    "predicted_proprio": (19, 16),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    done = json.loads((args.capture_dir / "DONE.json").read_text())
    errors = []
    checked = 0
    for item in done["outputs"]:
        path = args.capture_dir / item["path"]
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        if file_sha256(path) != item["sha256"]:
            errors.append(f"hash mismatch {path.name}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["mode"] == "pooled":
            for key, shape in POOLED_SHAPES.items():
                observed = tuple(payload["features"][key].shape)
                if observed != shape:
                    errors.append(f"{path.name} {key} shape {observed} != {shape}")
        for family in ("features", "labels"):
            for key, value in payload[family].items():
                if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                    errors.append(f"{path.name} {family}.{key} is non-finite")
        if payload["meta"]["task"] != "mw-reach-wall":
            errors.append(f"{path.name} wrong task")
        checked += 1
    report = {"complete": not errors and checked == done["episode_count"], "checked": checked, "errors": errors}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
