"""Freeze the registered equal-budget controls using an unchanged, verified fit."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from offline_study.interventions.interventions import validate_operator_bank
from offline_study.fitting.operator_fit import make_protocol
from offline_study.core.protocol import sha256, write_json


def amend(source: Path, output: Path) -> dict:
    done = json.loads((source / "DONE.json").read_text())
    for name in ("fit_receipt.json", "fit_selection.json", "protocol.json", "operator_bank.pt"):
        key = name.rsplit(".", 1)[0] + "_sha256"
        if done.get(key) != sha256(source / name):
            raise ValueError(f"Source fit checksum mismatch: {name}")
    original = json.loads((source / "protocol.json").read_text())
    receipt = json.loads((source / "fit_receipt.json").read_text())
    if (original["category"] != "vision_action_coupling" or
            receipt["development_outcomes_accessed"] or receipt["holdout_outcomes_accessed"]):
        raise ValueError("Amendment requires a verified fit-only coupling receipt")
    bank = torch.load(source / "operator_bank.pt", map_location="cpu", weights_only=True)
    validate_operator_bank(bank, sha256(source / "protocol.json"), list(bank["rows"].values()))
    frozen_at = datetime.now(timezone.utc).isoformat()
    receipt = {**receipt, "amendment": {
        "reason": "Add the two equal-standardized-energy arms required by the active plan",
        "source_fit_done_sha256": sha256(source / "DONE.json"),
        "source_protocol_sha256": sha256(source / "protocol.json"),
        "source_bank_sha256": sha256(source / "operator_bank.pt"),
        "directions_refitted": False, "existing_doses_changed": False,
        "prior_eight_arm_development_exists": True, "created_at": frozen_at,
    }}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "fit_receipt.json", receipt)
    shutil.copyfile(source / "fit_selection.json", output / "fit_selection.json")
    protocol = make_protocol(
        receipt["task"], receipt["manifest_sha256"], receipt["checkpoint_sha256"],
        sha256(output / "fit_receipt.json"), receipt["visual_delivered_l2"],
        receipt["action_condition_delivered_l2"], frozen_at,
    )
    new_arms = {arm["name"]: arm for arm in protocol["arms"]}
    if any(new_arms.get(arm["name"]) != arm for arm in original["arms"]):
        raise ValueError("Amendment would change an existing registered arm")
    write_json(output / "protocol.json", protocol)
    bank["protocol_sha256"] = sha256(output / "protocol.json")
    torch.save(bank, output / "operator_bank.pt")
    result = {**done, "status": "vision_action_coupling_protocol_amended_without_refitting",
              "arms": len(protocol["arms"])}
    for name in ("fit_receipt.json", "fit_selection.json", "protocol.json", "operator_bank.pt"):
        result[name.rsplit(".", 1)[0] + "_sha256"] = sha256(output / name)
    write_json(output / "DONE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(amend(args.source, args.output)), flush=True)


if __name__ == "__main__":
    main()
