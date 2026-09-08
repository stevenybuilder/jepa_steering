"""Complete source-aligned Wall training histories with native validation monitoring.

Three explicit seeds,50 epochs,16x8 logical training batches; no outcome-based
checkpoint/seed selection. Physical all-reduce order and unpublished author RNG
histories are not claimed. Epoch checkpoints retain exact local resume state.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate

from .behavioral_development import verified_report
from .droid_native import assert_same
from .navigation_input_check import CONFIGS, SelectedFrameSlicer, rng_state, restore_rng
from .navigation_training_inputs import verify_inputs
from .protocol import sha256, write_json
from .training_pilot import (VirtualRankBatchSampler, build_model, native_step,
                             accumulated_update, verify_accumulation)
from .vendor import use_vendor

SEEDS = (234, 235, 236)


class ValidationFrames:
    """Eight-frame counterpart of the already-verified training frame reader."""
    def __init__(self, original):
        from app.plan_common.datasets.traj_dset import TrajSlicerDataset, TrajSubset
        from app.plan_common.datasets.wall_dset import WallDataset
        if (type(original) is not TrajSlicerDataset or original.num_frames != 8 or
                original.frameskip != 5 or original.action_skip != 1 or original.process_actions != "concat"):
            raise ValueError("Wrong native validation slicer")
        base = original.dataset
        while type(base) is TrajSubset:
            base = base.dataset
        if type(base) is not WallDataset:
            raise ValueError("Wall-only validated training runner")
        self.original, self.base = original, base

    def __getitem__(self, index):
        from app.plan_common.datasets.traj_dset import TrajSubset
        row, start, end = self.original.slices[index]
        subset = self.original.dataset
        while type(subset) is TrajSubset:
            row = int(subset.indices[row])
            subset = subset.dataset
        if subset is not self.base:
            raise ValueError("Validation row mapping changed")
        frames = list(range(start, end, 5))
        image = torch.load(self.base.data_path / f"obses/episode_{row:03d}.pth",
                           map_location="cpu", weights_only=True, mmap=True)[frames] / 255.
        obs = {"visual": self.base.transform(image), "proprio": self.base.proprios[row, frames]}
        return obs, self.base.actions[row, start:end].reshape(8, -1), self.base.states[row, frames], torch.zeros(8)


def verify_validation_reader(original, adapted):
    initial = rng_state()
    try:
        for index in (0, 1, len(original) - 1):
            before = rng_state()
            expected = original[index]
            expected_rng = rng_state()
            restore_rng(before)
            actual = adapted[index]
            assert_same(expected, actual)
            # numpy RNG is not used by this deterministic transform.
            actual_rng = rng_state()
            if (expected_rng[0] != actual_rng[0] or not torch.equal(expected_rng[2], actual_rng[2]) or
                    not np.array_equal(expected_rng[1][1], actual_rng[1][1])):
                raise ValueError("Validation frame optimization changed RNG consumption")
    finally:
        restore_rng(initial)
    return {"indices": [0, 1, len(original)-1], "pixels_actions_states_rewards_rng_exact": True}


def checkpoint_payload(model, cfg, epoch, cpu_rngs, cuda_rngs, scheduler, wd, loader_rng, binding, validation_events):
    data = {"predictor": model.predictor.state_dict(), "opt": model.optimizer.state_dict(),
            "scaler": model.scaler.state_dict(), "epoch": epoch}
    for name in ("action_encoder", "proprio_encoder"):
        module = getattr(model, name)
        if module is not None and not cfg["model"][name].get(name + "_inpred", False):
            data[name] = module.state_dict()
    data["study_resume"] = {"binding": binding, "epoch": epoch,
        "cpu_rngs": [state.clone() for state in cpu_rngs], "cuda_rngs": [state.clone() for state in cuda_rngs],
        "scheduler_step": scheduler._step, "wd_step": wd._step, "loader_rng": loader_rng.get_state(),
        "validation_events": validation_events, "epoch_boundary_only": True}
    return data


def restore_checkpoint(data, model, cfg, scheduler, wd, loader_rng, binding):
    state = data["study_resume"]
    if (state["binding"] != binding or not state["epoch_boundary_only"] or state["epoch"] != data["epoch"] or
            state["scheduler_step"] != 418 * data["epoch"] or state["wd_step"] != 418 * data["epoch"] or
            state["validation_events"] != 2 * data["epoch"] or len(state["cpu_rngs"]) != 16 or len(state["cuda_rngs"]) != 16):
        raise ValueError("Changed training protocol or non-epoch resume")
    model.predictor.load_state_dict(data["predictor"], strict=True)
    for name in ("action_encoder", "proprio_encoder"):
        module = getattr(model, name)
        external = module is not None and not cfg["model"][name].get(name + "_inpred", False)
        if external != (name in data):
            raise ValueError("Native checkpoint component coverage changed")
        if external:
            module.load_state_dict(data[name], strict=True)
    model.optimizer.load_state_dict(data["opt"])
    model.scaler.load_state_dict(data["scaler"])
    scheduler._step, wd._step = state["scheduler_step"], state["wd_step"]
    loader_rng.set_state(state["loader_rng"])
    return [r.clone() for r in state["cpu_rngs"]], [r.clone() for r in state["cuda_rngs"]], state["validation_events"]


@torch.no_grad()
def monitor(run, model, validation, batches, event, cpu_rngs, cuda_rngs):
    batch_ids = batches[event % len(batches)]
    values = []
    for rank in range(16):
        indices = batch_ids[rank*4:(rank+1)*4]
        obs, action, state, reward = default_collate([validation[index] for index in indices])
        torch.set_rng_state(cpu_rngs[rank])
        torch.cuda.set_rng_state(cuda_rngs[rank], 0)
        obs = {key: value.to("cuda:0", dtype=torch.bfloat16) for key, value in obs.items()}
        result = run(obs, action.to("cuda:0", dtype=torch.bfloat16),
                     state.to("cuda:0", dtype=torch.bfloat16), reward.to("cuda:0", dtype=torch.bfloat16), train=False)
        metrics = {key: float(value) for key, value in result[3].items() if key.startswith("data_traj/")}
        if not metrics or not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Native validation monitoring failed")
        cpu_rngs[rank], cuda_rngs[rank] = torch.get_rng_state(), torch.cuda.get_rng_state(0)
        values.append({"logical_rank": rank, "validation_clip_indices": indices, "metrics": metrics})
    return {"event": event, "logical_batches": values, "global_clips": len(batch_ids),
            "source_step_train_false": True, "noisy_and_recorded_action_rollouts_retained": True}


def training_config(vendor):
    cfg = yaml.safe_load((vendor / CONFIGS["wall"]).read_text())
    opt = cfg["optimization"]["transition_model"]
    if (cfg["data"]["seed"] != 234 or cfg["data"]["custom"]["split_ratio"] != .9 or
            cfg["data"]["loader"]["batch_size"] != 8 or cfg["nodes"] * cfg["tasks_per_node"] != 16 or
            opt["num_epochs"] != 50 or opt["betas"] != [.9, .999] or cfg["meta"]["light_eval_freq"] != 200 or
            cfg["data"]["validation"]["val_dataset_batch_size"] != 4 or cfg["meta"]["save_every_freq"] != 1):
        raise ValueError("Source Wall optimization/evaluation contract changed")
    return cfg


def validation_step(vendor, model, cfg, scheduler, wd):
    step = native_step(vendor, model, scheduler, wd)
    step.__globals__.update(do_data_traj_rollout_eval=True, do_energy_landscape_eval=False,
        cfgs_data_traj_rollout_eval=cfg["meta"]["data_traj_rollout_eval"], data_traj_decode_gt=True)
    return step


def verify_saved_resume(path, model, cfg, scheduler, wd, loader_rng, binding, next_batch):
    """Actual source checkpoint loader, followed by a matched post-resume update."""
    from app.vjepa_wm.utils import load_checkpoint_state_dict
    data = torch.load(path, map_location="cpu", weights_only=True)
    reference, ref_schedule, ref_wd = copy.deepcopy((model, scheduler, wd))
    resumed, resumed_schedule, resumed_wd = copy.deepcopy((model, scheduler, wd))
    with torch.no_grad():
        for parameter in resumed.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_checkpoint_state_dict(data, predictor=resumed.predictor, action_encoder=resumed.action_encoder,
        proprio_encoder=resumed.proprio_encoder, opt=resumed.optimizer, scaler=resumed.scaler)
    assert_same(reference.state_dict(), resumed.state_dict())
    assert_same(reference.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_same(reference.scaler.state_dict(), resumed.scaler.state_dict())
    resumed_loader_rng = torch.Generator()
    resumed_cpu, resumed_cuda, events = restore_checkpoint(data, resumed, cfg,
        resumed_schedule, resumed_wd, resumed_loader_rng, binding)
    expected_cpu = [state.clone() for state in data["study_resume"]["cpu_rngs"]]
    cuda_before = torch.cuda.get_rng_state(0)
    expected = accumulated_update(reference, ref_schedule, ref_wd, next_batch, expected_cpu)
    torch.cuda.set_rng_state(cuda_before, 0)
    actual = accumulated_update(resumed, resumed_schedule, resumed_wd, next_batch, resumed_cpu)
    assert_same(reference.state_dict(), resumed.state_dict())
    assert_same(reference.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_same(reference.scaler.state_dict(), resumed.scaler.state_dict())
    assert_same(expected_cpu, resumed_cpu)
    assert_same(expected, actual)
    assert_same(data["study_resume"]["cuda_rngs"], resumed_cuda)
    if events != 2 or not torch.equal(resumed_loader_rng.get_state(), data["study_resume"]["loader_rng"]):
        raise ValueError("Resume lost validation cursor or loader generator")
    torch.cuda.set_rng_state(cuda_before, 0)
    return {"status":"actual_native_checkpoint_loader_and_resumed_update_exact",
        "checkpoint_sha256":sha256(path), "actual_upstream_loader_used":True,
        "next_update_parameters_optimizer_scaler_cpu_rng_exact":True,
        "validation_cursor_cuda_rng_loader_state_restored":True,
        "extra_updates_were_discarded_engineering_clones_only":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "data-root", "input-receipt", "input-check", "pilot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--engineering-proof", type=Path)
    parser.add_argument("--engineering-only", action="store_true", help="One complete epoch; never a50-epoch history claim")
    args = parser.parse_args()
    if args.engineering_only and args.resume_from:
        raise ValueError("Engineering epoch must start from initialization, not a selected checkpoint")
    if not args.engineering_only and (not args.resume_from or not args.engineering_proof):
        raise ValueError("Full training continues only from a verified first epoch and explicit engineering proof")
    use_vendor(args.vendor)
    cfg = training_config(args.vendor)
    inputs, inputs_hash = verify_inputs(args.data_root, args.input_receipt)
    pilot, pilot_hash = verified_report(args.pilot)
    pp = json.loads((args.pilot / "protocol.json").read_text())
    if (pilot["status"] != "native_training_accumulation_pilot_passed" or pilot["task"] != "wall" or
            pilot["updates_per_epoch"] != 418 or pilot["protocol_sha256"] != sha256(args.pilot / "protocol.json") or
            pilot["parity_sha256"] != sha256(args.pilot / "PARITY.json") or
            pp["source_sha256"] != sha256(Path(__file__).with_name("training_pilot.py")) or
            pp["native_training_source_sha256"] != sha256(args.vendor / "app/vjepa_wm/train.py") or
            inputs["native_config_sha256"] != sha256(args.vendor / CONFIGS["wall"])):
        raise ValueError("Original source training equivalence proof changed")
    args.output.mkdir(parents=True, exist_ok=False)
    binding = {"seed": args.seed, "input_report_sha256": inputs_hash,
        "native_config_sha256": sha256(args.vendor / CONFIGS["wall"]),
        "training_source_sha256": sha256(Path(__file__)), "pilot_report_sha256": pilot_hash}
    if not args.engineering_only:
        proof, _ = verified_report(args.engineering_proof)
        proof_protocol = json.loads((args.engineering_proof / "protocol.json").read_text())
        if (proof["status"] != "one_epoch_training_engineering_complete" or proof["seed"] != args.seed or
                proof["protocol_sha256"] != sha256(args.engineering_proof / "protocol.json") or
                proof["resume_parity_sha256"] != sha256(args.engineering_proof / "RESUME_PARITY.json") or
                any(proof_protocol[key] != value for key,value in binding.items())):
            raise ValueError("Training initialization or saved/resumed step has no matching completed proof")
    write_json(args.output / "protocol.json", {**binding, "role": "training_engineering_full_epoch" if args.engineering_only else "complete_wall_training_history",
        "epochs": 50, "optimizer_updates_per_epoch": 418, "global_training_batch": 128,
        "logical_training_batches": "16x8 with original DistributedSampler/drop_last",
        "training_rows": 1728, "validation_rows": 192, "training_clips": 53568,
        "native_validation_every_within_epoch_updates": 200, "global_validation_batch": 64,
        "source_validation_noisy_and_recorded_action_H6_both": True,
        "checkpoint_policy": "all50 epochs saved immutably; primary late window41..50; no outcome selection",
        "planning_episodes_per_checkpoint_condition": 96, "planning_evaluation_is_separate_required_job": True,
        "training_seed_triplet": list(SEEDS), "authors_exact_seed_triplet_or_collective_order_claimed": False,
        "resume_checkpoint": str(args.resume_from) if args.resume_from else None,
        "fresh_confirmation": False})
    started = time.monotonic()
    try:
        from app.plan_common.datasets.transforms import make_transforms
        from app.plan_common.datasets.wall_dset import WallDataset
        from app.plan_common.datasets.traj_dset import get_train_val_sliced
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataset = WallDataset(data_path=str(args.data_root), n_rollout=None, normalize_action=True,
                              transform=make_transforms(img_size=224, **cfg["data_aug"]))
        train, val, train_clips, val_clips = get_train_val_sliced(dataset, train_fraction=.9, random_seed=234,
            num_frames=4, num_frames_val=8, frameskip=5, action_skip=1)
        audited = json.loads((args.input_check / "wall.json").read_text())
        if ((len(dataset), len(train), len(val), len(train_clips), len(val_clips)) != (1920,1728,192,53568,2112) or
                list(train.indices) != audited["native_train_indices"] or list(val.indices) != audited["native_validation_indices"]):
            raise ValueError("Complete native split or trajectory counts changed")
        sampler = VirtualRankBatchSampler(len(train_clips))
        validation_batches = list(VirtualRankBatchSampler(len(val_clips), microbatch=4))
        if len(sampler) != 418 or len(validation_batches) != 33:
            raise ValueError("Source sampler length changed")
        validation = ValidationFrames(val_clips)
        reader_proof = verify_validation_reader(val_clips, validation)
        loader_rng = torch.Generator().manual_seed(2026090725)
        loader = torch.utils.data.DataLoader(SelectedFrameSlicer(train_clips), batch_sampler=sampler,
            num_workers=4, pin_memory=True, persistent_workers=True, generator=loader_rng)
        model, scheduler, wd = build_model(cfg, dataset, len(sampler))
        encoder_versions = [(name, parameter._version) for name, parameter in model.encoder.named_parameters()]
        cpu_rngs = [torch.get_rng_state().clone() for _ in range(16)]
        cuda_rngs = [torch.cuda.get_rng_state(0).clone() for _ in range(16)]
        native_monitor = validation_step(args.vendor, model, cfg, scheduler, wd)
        start_epoch, events = 0, 0
        inherited = []
        if args.resume_from:
            data = torch.load(args.resume_from, map_location="cpu", weights_only=True)
            cpu_rngs, cuda_rngs, events = restore_checkpoint(data, model, cfg, scheduler, wd, loader_rng, binding)
            start_epoch = data["epoch"]
            inherited = data["study_resume"].get("checkpoint_history", [])
            inherited.append({"epoch": start_epoch, "path": str(args.resume_from), "sha256": sha256(args.resume_from)})
            if not 1 <= start_epoch < 50:
                raise ValueError("No incomplete full training history to resume")
            del data
        history = list(inherited)
        for epoch in range(start_epoch, 1 if args.engineering_only else 50):
            sampler.epoch = epoch
            epoch_start, losses = time.monotonic(), []
            for step, batch in enumerate(loader):
                before = time.monotonic()
                if epoch == 0 and step == 0:
                    parity = verify_accumulation(args.vendor, model, scheduler, wd, batch, cpu_rngs)
                    write_json(args.output / "PARITY.json", {"training": parity, "validation_reader": reader_proof})
                    values = {"loss": parity["loss"], "lr": model.optimizer.param_groups[0]["lr"]}
                else:
                    values = accumulated_update(model, scheduler, wd, batch, cpu_rngs)
                losses.append(values["loss"])
                if (step + 1) % 200 == 0:
                    monitored = monitor(native_monitor, model, validation, validation_batches, events, cpu_rngs, cuda_rngs)
                    write_json(args.output / f"validation-{events:03d}.json", {"epoch": epoch+1, "update": step+1, **monitored})
                    events += 1
                if (step + 1) % 20 == 0:
                    progress = {"epoch": epoch+1, "epoch_updates": step+1, "total_epochs": 50,
                        "global_updates": epoch*418+step+1, "last_update_seconds": time.monotonic()-before,
                        "loss": values["loss"], "seconds": time.monotonic()-started}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            if len(losses) != 418 or encoder_versions != [(name,p._version) for name,p in model.encoder.named_parameters()]:
                raise ValueError("Incomplete epoch or frozen encoder changed")
            checkpoint = args.output / f"jepa-e{epoch}.pth.tar"
            payload = checkpoint_payload(model, cfg, epoch+1, cpu_rngs, cuda_rngs, scheduler, wd, loader_rng, binding, events)
            payload["study_resume"]["checkpoint_history"] = list(history)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
            history.append({"epoch": epoch+1, "path": str(checkpoint), "sha256": sha256(checkpoint)})
            write_json(args.output / f"epoch-{epoch+1:03d}.json", {"epoch": epoch+1, "updates":418,
                "mean_training_loss": float(np.mean(losses)), "seconds": time.monotonic()-epoch_start,
                "checkpoint": history[-1], "validation_events_so_far": events})
            write_json(args.output / "CHECKPOINTS.json", {"history": history, "required_epochs": list(range(1,51))})
            print(json.dumps({"epoch_complete":epoch+1,"checkpoint_sha256":history[-1]["sha256"],"seconds":time.monotonic()-started}), flush=True)
        resume_parity_hash = None
        if args.engineering_only:
            sampler.epoch = 1
            next_batch = next(iter(loader))
            proof = verify_saved_resume(checkpoint, model, cfg, scheduler, wd, loader_rng, binding, next_batch)
            write_json(args.output / "RESUME_PARITY.json", proof)
            resume_parity_hash = sha256(args.output / "RESUME_PARITY.json")
        write_json(args.output / "report.json", {"status": "one_epoch_training_engineering_complete" if args.engineering_only else "all50_wall_training_epochs_complete",
            "protocol_sha256":sha256(args.output/"protocol.json"), "seed":args.seed,
            "checkpoint_manifest_sha256":sha256(args.output/"CHECKPOINTS.json"), "checkpoints":len(history),
            "seconds":time.monotonic()-started, "validation_events":events,
            "resume_parity_sha256":resume_parity_hash,
            "training_history_complete":not args.engineering_only, "behavioral_evaluations_complete":False,
            "fresh_confirmation":False, "all_study_tasks_complete":False})
        write_json(args.output/"DONE.json", {"report_sha256":sha256(args.output/"report.json")})
    except Exception as exc:
        write_json(args.output/"FAILED.json", {"error":str(exc),"training_history_complete":False})
        raise


if __name__ == "__main__":
    main()
