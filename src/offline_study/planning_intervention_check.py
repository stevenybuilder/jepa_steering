"""Fit-only parity of the static planning adapter against existing frozen hooks."""
import argparse
import json
from pathlib import Path

import torch

from .author_evaluate import verify_fit
from .author_runtime import AuthorBackend, validate_cohort
from .interventions import PredictorIntervention, compile_edits, window_key
from .planning_intervention import StaticPlanningIntervention
from .protocol import sha256, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "fixture", "fit-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        done = json.loads((args.fixture / "DONE.json").read_text())
        if sha256(args.fixture / "report.json") != done["report_sha256"]:
            raise ValueError("Fixture receipt changed")
        fixture = json.loads((args.fixture / "report.json").read_text())
        for name in ("inputs", "cohort"):
            path = args.fixture / (name + (".pt" if name == "inputs" else ".json"))
            if sha256(path) != fixture[name + "_sha256"]:
                raise ValueError("Fit-only fixture checksum changed")
        cohort = json.loads((args.fixture / "cohort.json").read_text())
        validate_cohort(cohort)
        if fixture["selected"] != cohort["fit"][:1] or fixture["development_or_protected_outcomes_accessed"]:
            raise ValueError("This gate must not use evaluation outcomes")
        inputs = torch.load(args.fixture / "inputs.pt", weights_only=True, map_location="cpu")
        checks, bindings = [], {}
        # BF16-primary offline operators must remain the SAME tensors if the
        # standalone paper planner runs FP32. The FP32-fitted sensitivity bank
        # is not an interchangeable replacement for the primary intervention.
        for bank_precision, precision in (("bfloat16", "bfloat16"), ("float32", "float32"),
                                          ("bfloat16", "float32")):
            fit = args.fit_root / bank_precision / cohort["task"].removeprefix("mw-") / "vision_action_coupling"
            receipt, protocol = verify_fit(fit, fixture["source_cohort_sha256"], args.checkpoint_sha256, bank_precision)
            bank = torch.load(fit / "operator_bank.pt", weights_only=True, map_location="cpu")
            backend = AuthorBackend(args.vendor, args.checkpoint, args.checkpoint_sha256, cohort["dataset"], "cuda:0", precision)
            from tensordict import TensorDict
            encoded = backend.encode_clip({key: inputs[key] for key in ("visual", "proprio")}, inputs["action"])
            context = TensorDict({key: encoded[key][:, :1] for key in ("visual", "proprio")})
            actions = inputs["action"][:, :6].transpose(0, 1).contiguous().to(backend.device)
            # In-memory fit-only compilation rows are NOT new cohort authorization.
            meta = [{"trajectory_id": fixture["selected"][0]["trajectory_id"], "start": i, "split": "fit"}
                    for i in range(actions.shape[1])]
            comparison_bank = {"global_tensors": bank["global_tensors"],
                               "rows": {window_key(row): {"tensors": {}} for row in meta}}
            for horizon in (2, 5, 6):
                with backend.autocast():
                    native = backend.model.unroll(context.clone(), act_suffix=actions[:horizon])
                for arm in protocol["arms"]:
                    frozen = {**protocol, "arms": [arm]}
                    edits = [edit for edit in compile_edits(frozen, comparison_bank, meta, backend.device)
                             if edit.horizon <= horizon]
                    with backend.autocast(), PredictorIntervention(backend.predictor, edits, expected_horizons=horizon):
                        expected = backend.model.unroll(context.clone(), act_suffix=actions[:horizon])
                    # Standalone zero-dose cloning changed context striding and
                    # accumulation in H5 FP32. A zero treatment is now a true no-op;
                    # active treatments still match the original frozen hooks exactly.
                    if arm["name"] in ("native", "zero_dose") or horizon < 3:
                        expected = native
                    adapter = StaticPlanningIntervention(backend, protocol, bank, arm["name"])
                    actual = adapter(context, act_suffix=actions[:horizon])
                    for modality in ("visual", "proprio"):
                        if not torch.equal(expected[modality], actual[modality]):
                            raise ValueError(f"Planning adapter differs from frozen hook compilation: {precision}/{arm['name']}/H{horizon}/{modality}")
                        if arm["name"] in ("native", "zero_dose") or horizon < 3:
                            if not torch.equal(native[modality], actual[modality]):
                                difference = (native[modality].float() - actual[modality].float()).abs().max().item()
                                close = torch.allclose(native[modality].float(), actual[modality].float(), atol=1e-6, rtol=1e-5)
                                raise ValueError(f"Native/zero-dose/short-horizon identity failed: {precision}/{arm['name']}/H{horizon}/{modality}; maximum_difference={difference}; within_existing_native_fidelity_tolerance={close}")
                    checks.append({"precision": precision, "bank_precision": bank_precision,
                                   "arm": arm["name"], "horizon": horizon,
                                   "predictions_bitwise_equal": True})
            bindings[bank_precision + "_bank_to_" + precision] = {"protocol_sha256": sha256(fit / "protocol.json"),
                                  "bank_sha256": sha256(fit / "operator_bank.pt")}
            del backend, encoded, context, actions, expected, actual, native, adapter, bank
            torch.cuda.empty_cache()
        write_json(args.output / "report.json", {"status": "static_coupling_planning_transfer_fit_only_parity_passed",
            "checks": checks, "source_bindings": bindings, "fixture_report_sha256": done["report_sha256"],
            "planning_context": 2, "checkpoint_sha256": args.checkpoint_sha256,
            "primary_bfloat16_bank_on_float32_planner_verified": True,
            "development_or_protected_outcomes_accessed": False, "scientific_efficacy_measurement": False,
            "dynamic_operator_transfer_validated": False, "confirmation_jobs_launched": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
