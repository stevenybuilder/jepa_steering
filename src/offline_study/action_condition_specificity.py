"""Finite H3 action-condition specificity diagnostic; no training or simulator.

New experiment, separate from completed attention/replay/CEM pilots. Native
candidate ranking is a reference, not a physical action-quality label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


TASKS = ("reach", "reach-wall")
BANKS = ("original", "fresh")
LAYERS = tuple(range(6))
MODALITIES = ("visual", "proprio", "official")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value):
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def random_seed(task, episode, bank, layer):
    text = f"action-condition-v1|{task}|{episode}|{bank}|{layer}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & ((1 << 63)-1)


def initial_population(cfg, seed, device):
    """Port only pinned CEM's initial population; never call its optimizer."""
    p = cfg["planner"]
    if p["horizon"] != 6 or p["num_samples"] != 300 or torch.get_default_dtype() != torch.float32:
        raise ValueError("Unexpected native candidate contract")
    mean = torch.zeros(6, 20, device=device)
    std = p["var_scale"] * torch.ones(6, 20, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.empty(6, 300, 20, device=device)
    actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
        6, 300, 20, device=std.device, generator=generator)
    actions[:, 0, :] = mean
    if p.get("max_norms") is not None:
        for h in range(6):
            for dims, maxnorm in zip(p.get("max_norm_dims", [[0, 1, 2], [6]]), p["max_norms"]):
                actions[h, :, dims] = torch.clip(actions[h, :, dims], min=-maxnorm, max=maxnorm)
    return actions, tensor_hash(generator.get_state())


def verify_native_sampler(cfg, seed, actions, final_rng_sha, planner_class):
    """Stop actual native CEM at its first cost call, before any forecast."""

    class PopulationCaptured(Exception):
        pass

    generator = torch.Generator(device=actions.device).manual_seed(seed)
    planner = planner_class(unroll=None,action_dim=20,local_generator=generator,**cfg["planner"])
    captured = []
    def observe(candidate_actions, context):
        captured.append(candidate_actions.detach().clone())
        raise PopulationCaptured()
    planner.cost_function = observe
    try:
        planner.plan(None,steps_left=6)
    except PopulationCaptured:
        pass
    if len(captured) != 1 or not torch.equal(captured[0],actions) or tensor_hash(generator.get_state()) != final_rng_sha:
        raise ValueError("Fresh bank or private RNG differs from actual native initial sampler")


def edited_condition(z, mode, seed=None):
    if z.ndim != 3 or z.shape[0] != 300 or z.dtype != torch.float32 or not torch.isfinite(z).all():
        raise ValueError("Require finite FP32 candidate,time,condition layout")
    if mode not in ("permute", "random"):
        raise ValueError("Unknown condition intervention")
    newest = z[:, -1]
    donors = torch.roll(newest, shifts=-1, dims=0)
    target = (donors.double()-newest.double()).norm(dim=1)
    if mode == "permute":
        replacement = donors
    else:
        generator = torch.Generator(device=z.device).manual_seed(seed)
        noise = torch.randn(newest.shape, dtype=z.dtype, device=z.device, generator=generator).double()
        length = noise.norm(dim=1)
        if torch.any(length == 0):
            raise ValueError("Zero isotropic draw")
        delta = (noise*(target/length)[:, None]).to(z.dtype)
        replacement = newest + delta
    result = z.clone()
    result[:, -1] = replacement
    delivered = (result[:, -1].double()-newest.double()).norm(dim=1)
    positive = target > 0
    relative = torch.zeros_like(target)
    relative[positive] = (delivered[positive]-target[positive]).abs()/target[positive]
    if torch.any(relative > 1e-5) or torch.any(delivered[~positive] != 0):
        raise ValueError("Actually delivered per-candidate norm does not match")
    if not torch.equal(result[:, :-1], z[:, :-1]):
        raise ValueError("Earlier condition positions changed")
    return result, dict(requested_delta_l2=target.cpu().tolist(), delivered_delta_l2=delivered.cpu().tolist(),
                       relative_norm_error=relative.cpu().tolist(), zero_target_count=int((~positive).sum()),
                       max_relative_norm_error=float(relative.max()), native_condition_sha256=tensor_hash(z),
                       edited_condition_sha256=tensor_hash(result), condition_shape=list(z.shape),
                       random_seed=seed if mode == "random" else None)


class ConditionHook:
    """Capture/zero-clone every H3 z, or edit one H3 block's newest z only."""
    def __init__(self, predictor, *, cache=None, layer=None, mode="capture", seed=None):
        self.predictor, self.cache, self.layer, self.mode, self.seed = predictor, cache, layer, mode, seed
        self.counts = [0]*6
        self.captured, self.audit, self.handles = {}, None, []

    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 6 or self.mode not in ("capture", "permute", "random"):
            raise ValueError("Unexpected predictor or hook mode")
        try:
            for layer, block in enumerate(self.predictor.predictor_blocks):
                if block.training:
                    raise ValueError("Only frozen eval-mode predictor is supported")
                def pre(module, args, kwargs, layer=layer):
                    self.counts[layer] += 1
                    if self.counts[layer] != 3:
                        return None
                    if len(args) < 2:
                        raise ValueError("Expected actual AdaLN positional x,z inputs")
                    z = args[1]
                    if self.mode == "capture":
                        self.captured[layer] = z.detach().clone()
                        replacement = z.clone()  # Explicit native zero-edit parity.
                    elif layer == self.layer:
                        cached = self.cache[layer]
                        if z.shape != cached.shape or z.dtype != cached.dtype or not torch.equal(
                                z.detach().contiguous().view(torch.uint8),cached.contiguous().view(torch.uint8)):
                            raise ValueError("Intervention entry differs from cached native condition")
                        replacement, self.audit = edited_condition(z, self.mode, self.seed)
                    else:
                        return None
                    return (args[0], replacement, *args[2:]), kwargs
                self.handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, typ, value, traceback):
        for handle in self.handles:
            handle.remove()
        if typ is None and (self.counts != [6]*6 or (self.mode != "capture" and self.audit is None)):
            raise ValueError("Hook did not traverse exactly all six rollout horizons")


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid ranking vector")
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start+end-1)/2
        start = end
    return ranked


def agreement(native, changed, native_elites, changed_elites):
    native, changed = np.asarray(native, float), np.asarray(changed, float)
    if native.shape != (300,) or changed.shape != (300,):
        raise ValueError("Exactly the same 300 candidates required")
    for ids, costs in ((native_elites, native), (changed_elites, changed)):
        ids = np.asarray(ids)
        if ids.shape != (10,) or ids.dtype.kind not in "iu" or len(set(ids)) != 10 or ids.min() < 0 or ids.max() >= 300:
            raise ValueError("Invalid actual elite IDs")
        if costs[ids].max() > np.delete(costs, ids).min():
            raise ValueError("Elite IDs do not select native minimum costs")
    a, b = average_ranks(native), average_ranks(changed)
    a -= a.mean(); b -= b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    delta = changed-native
    return dict(spearman=None if denominator == 0 else float(a@b/denominator),
                top10_overlap_count=len(set(native_elites)&set(changed_elites)),
                centered_cost_change_rms=float(np.sqrt(np.mean((delta-delta.mean())**2))),
                cost_change_mean=float(delta.mean()), cost_change_rms=float(np.sqrt(np.mean(delta**2))))


def layer_intervals(values):
    """Paired n8 x six layers; max-stat family is fixed before outcomes."""
    values = np.asarray(values,dtype=float)
    if values.shape != (8,6):
        raise ValueError("Require all eight scenarios and six layers")
    defined = np.isfinite(values)
    if not defined.all():
        return [dict(layer=l,n=8,n_defined=int(defined[:,l].sum()),mean=None,
                     marginal_95_low=None,marginal_95_high=None,simultaneous_95_low=None,
                     simultaneous_95_high=None,undefined_reason="Complete six-layer family contains undefined rank agreement")
                for l in LAYERS]
    weights = np.random.default_rng(20260913).multinomial(8,np.full(8,1/8),size=20000)/8
    means = values.mean(0)
    draws = weights@values
    radius = float(np.quantile(np.max(np.abs(draws-means),axis=1),.95))
    return [dict(layer=l,n=8,n_defined=8,mean=float(means[l]),
                 marginal_95_low=float(np.quantile(draws[:,l],.025)),marginal_95_high=float(np.quantile(draws[:,l],.975)),
                 simultaneous_95_low=float(means[l]-radius),simultaneous_95_high=float(means[l]+radius))
            for l in LAYERS]


def summarize_complete(cases):
    """CPU-only all16 aggregation; callers must first verify source receipts."""
    registry = {(c["input_binding"]["task"],c["input_binding"]["episode"]):c for c in cases}
    if len(cases) != 16 or set(registry) != {(t,e) for t in TASKS for e in range(8)}:
        raise ValueError("No partial or duplicate action-condition cohort summary")
    rows = []
    for task in TASKS:
        for bank in BANKS:
            for modality in MODALITIES:
                values = {"excess_spearman_loss":np.empty((8,6)),"excess_top10_overlap_loss":np.empty((8,6))}
                for episode in range(8):
                    arms = registry[task,episode]["banks"][bank]
                    native = arms["native"]["scores"][modality]
                    for layer in LAYERS:
                        compare = {}
                        for mode in ("permute","random"):
                            current = arms[f"{mode}_B{layer}"]["scores"][modality]
                            compare[mode] = agreement(native["costs"],current["costs"],native["elite_indices"],current["elite_indices"])
                        r,p = compare["random"],compare["permute"]
                        values["excess_spearman_loss"][episode,layer] = np.nan if r["spearman"] is None or p["spearman"] is None else r["spearman"]-p["spearman"]
                        values["excess_top10_overlap_loss"][episode,layer] = (r["top10_overlap_count"]-p["top10_overlap_count"])/10
                for metric,matrix in values.items():
                    rows.extend(dict(task=task,bank=bank,modality=modality,metric=metric,**r) for r in layer_intervals(matrix))
    return rows


def score_vectors(forecast, target, objective, actions):
    vectors = {k:(target[k]-forecast[k]).pow(2).mean(dim=tuple(range(2,forecast[k].ndim)))[-1]
               for k in ("visual", "proprio")}
    official = objective(forecast, actions)
    if not torch.equal(official, vectors["visual"]+.1*vectors["proprio"]):
        raise ValueError("Modality costs do not reproduce exact native weighted objective")
    vectors["official"] = official
    if any(v.shape != (300,) or not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Invalid candidate goal costs")
    return {k:dict(costs=v.detach().cpu().tolist(), elite_indices=torch.topk(-v,10,dim=0).indices.cpu().tolist())
            for k,v in vectors.items()}


def run_case(backend, cfg, data, binding, output, manifest_sha256, protocol_sha256, runtime_provenance):
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    # The pinned planner imports nevergrad, whose initialization consumes NumPy
    # RNG. Complete dependency initialization before the experimental baseline;
    # never restore or mask RNG changes from sampling/hooks/forecasts themselves.
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from .fixed_combined_check import assert_bytes, rng_signature

    if set(data) != {"initial", "goal", "actions"} or data["actions"].shape != (6,300,20):
        raise ValueError("Unexpected frozen scenario input")
    objective_cfg = cfg["planner"]["planning_objective"]
    if objective_cfg["objective_type"] != "L2" or objective_cfg["alpha"] != .1 or objective_cfg["sum_all_diffs"]:
        raise ValueError("Unexpected official goal objective")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial = TensorDict(data["initial"],batch_size=[])
    goal = TensorDict(data["goal"],batch_size=[])
    obs_type = cfg["task_specification"]["obs"]
    context = backend.model.encode(prepare_obs(obs_type,initial).to(backend.device).unsqueeze(0),act=True)
    target = backend.model.encode(prepare_obs(obs_type,goal).to(backend.device).unsqueeze(0),act=False).detach()
    objective = ReprTargetDistMPCObjective(cfg,target,**objective_cfg)
    context_before = {k:tensor_hash(context[k]) for k in context.keys()}
    target_before = {k:tensor_hash(target[k]) for k in target.keys()}
    versions = {(kind,k):v._version for kind,iterator in (("parameter",backend.model.named_parameters()),("buffer",backend.model.named_buffers())) for k,v in iterator}
    rng_before = rng_signature(backend.device)
    fresh_seed = binding["candidate_seed"] ^ 0x40000000
    fresh, private_rng_sha = initial_population(cfg,fresh_seed,backend.device)
    verify_native_sampler(cfg,fresh_seed,fresh,private_rng_sha,CEMPlanner)
    banks = {"original":data["actions"].to(backend.device),"fresh":fresh}
    if tensor_hash(banks["original"]) == tensor_hash(fresh):
        raise ValueError("Fresh bank duplicates original action bank")
    records, bank_receipts = {}, {}
    for bank_name, actions in banks.items():
        bank_start = time.monotonic()
        action_sha = tensor_hash(actions)
        if actions.dtype != torch.float32 or torch.count_nonzero(actions[:,0]):
            raise ValueError("Unexpected native action dtype/mean candidate")
        native = backend.predict(context,actions)
        native_scores = score_vectors(native,target,objective,actions)
        with ConditionHook(backend.predictor) as capture:
            zero = backend.predict(context,actions)
        for modality in ("visual","proprio"):
            assert_bytes(zero[modality],native[modality],"H1-H6 native zero-condition parity "+modality)
        if score_vectors(zero,target,objective,actions) != native_scores:
            raise ValueError("Zero-condition goal-cost parity failed")
        del zero,native
        records[bank_name] = {"native":dict(scores=native_scores)}
        for layer in LAYERS:
            for mode in ("permute","random"):
                seed = random_seed(binding["task"],binding["episode"],bank_name,layer)
                with ConditionHook(backend.predictor,cache=capture.captured,layer=layer,mode=mode,seed=seed) as hook:
                    forecast = backend.predict(context,actions)
                scores = score_vectors(forecast,target,objective,actions)
                records[bank_name][f"{mode}_B{layer}"] = dict(scores=scores, condition_audit=hook.audit,
                    agreement={m:agreement(native_scores[m]["costs"],scores[m]["costs"],native_scores[m]["elite_indices"],scores[m]["elite_indices"])
                               for m in MODALITIES})
                del forecast
        if tensor_hash(actions) != action_sha:
            raise ValueError("Recipient action input changed")
        bank_receipts[bank_name] = dict(actions_sha256=action_sha,actions_shape=list(actions.shape),
            dtype=str(actions.dtype),candidate0_zero=True,scientific_arms=13,forecast_calls=14,
            native_zero_full_forecast_and_score_byte_parity=True,
            native_condition_sha256={str(l):tensor_hash(z) for l,z in capture.captured.items()},
            seconds=time.monotonic()-bank_start)
        # No scores printed: the first complete case is an operational timing gate.
        print(json.dumps(dict(task=binding["task"],episode=binding["episode"],bank=bank_name,
                              seconds=bank_receipts[bank_name]["seconds"],forecast_calls=14)),flush=True)
    if context_before != {k:tensor_hash(context[k]) for k in context.keys()} or target_before != {k:tensor_hash(target[k]) for k in target.keys()}:
        raise ValueError("Encoded inputs mutated")
    after_versions = {(kind,k):v._version for kind,iterator in (("parameter",backend.model.named_parameters()),("buffer",backend.model.named_buffers())) for k,v in iterator}
    if versions != after_versions:
        raise ValueError("Model parameter or buffer versions changed")
    if rng_before != rng_signature(backend.device):
        raise ValueError("Experimental global RNG changed after setup")
    payload = dict(input_binding=binding,execution_manifest_sha256=manifest_sha256,protocol_sha256=protocol_sha256,
                   banks=records,bank_receipts=bank_receipts)
    dump(output/"scores.json",payload)
    with (output/"actions.pt").open("xb") as stream:
        torch.save({k:v.detach().cpu() for k,v in banks.items()},stream)
    files = {name:file_hash(output/name) for name in ("scores.json","actions.pt")}
    report = dict(**runtime_provenance,status="complete_action_condition_case",input_binding=binding,execution_manifest_sha256=manifest_sha256,
                  protocol_sha256=protocol_sha256,files=files,bank_receipts=bank_receipts,
                  fresh_bank_seed=fresh_seed,fresh_bank_generator_final_sha256=private_rng_sha,
                  fresh_bank_native_sampler_byte_and_rng_parity=True,
                  vendor_sampler_import_scope="Dependency initialization before experimental RNG snapshot; no RNG restoration",
                  input_context_sha256=context_before,target_sha256=target_before,
                  global_rng_unchanged=True,parameters_unchanged=True,inputs_unchanged=True,
                  physical_outcomes_measured=False,fresh_confirmation=False,training_performed=False,
                  precision="strict FP32 TF32 disabled",scientific_forward_count=26,total_forward_count=28,
                  full_h6_retained=False,total_seconds=time.monotonic()-started)
    dump(output/"report.json",report)
    dump(output/"DONE.json",dict(report_sha256=file_hash(output/"report.json"),files=files))
    print(json.dumps(dict(status=report["status"],task=binding["task"],episode=binding["episode"],
                          total_seconds=report["total_seconds"],total_forward_count=28)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor","checkpoint","inputs","manifest","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--task",choices=TASKS,required=True)
    parser.add_argument("--episode",type=int,choices=range(8),required=True)
    args = parser.parse_args()
    if file_hash(args.manifest) != args.manifest_sha256 or file_hash(args.protocol) != args.protocol_sha256:
        raise ValueError("Prospective execution/protocol binding changed")
    manifest, protocol = json.loads(args.manifest.read_text()),json.loads(args.protocol.read_text())
    expected = {task:list(range(8)) for task in TASKS}
    if manifest["scenarios"] != expected or protocol["scenarios"] != expected or manifest["protocol_sha256"] != args.protocol_sha256:
        raise ValueError("Changed complete16 scientific registry")
    if file_hash(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes changed")
    if file_hash(args.inputs/"INPUT_MANIFEST.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Input manifest bytes changed")
    source_root = Path(__file__).resolve().parent
    if "action_condition_specificity.py" not in manifest["source_sha256"]:
        raise ValueError("Execution source is not bound")
    for name,digest in manifest["source_sha256"].items():
        if Path(name).name != name or file_hash(source_root/name) != digest:
            raise ValueError("Frozen diagnostic source changed")
    for name,digest in manifest["vendor_source_sha256"].items():
        path = (args.vendor/name).resolve()
        if not path.is_relative_to(args.vendor.resolve()) or file_hash(path) != digest:
            raise ValueError("Frozen native vendor source changed")
    registry = json.loads((args.inputs/"INPUT_MANIFEST.json").read_text())["records"]
    selected = [r for r in registry if r["task"] == args.task and r["episode"] == args.episode]
    if len(selected) != 1:
        raise ValueError("Nonunique frozen scenario")
    binding = selected[0]
    source = args.inputs/args.task/f"episode-{args.episode:03d}"/"inputs.pt"
    if file_hash(source) != binding["inputs_sha256"]:
        raise ValueError("Scenario input bytes changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one assigned GPU")
    uuid = str(getattr(torch.cuda.get_device_properties(0),"uuid","unavailable"))
    if uuid.removeprefix("GPU-").lower() != args.gpu_uuid.removeprefix("GPU-").lower():
        raise ValueError("Physical GPU differs from assigned UUID")
    if os.environ.get("JEPA_VERIFIED_LOCAL_DINO") != "1":
        raise ValueError("Require verified local DINO source/weight cache, no moving network resolution")
    from .backends import JepaBackend
    from .planning_contract import prepare
    torch.set_num_threads(1)
    backend = JepaBackend(args.vendor,args.checkpoint,manifest["checkpoint_sha256"],"metaworld","cuda:0","float32",allow_tf32=False)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or backend.provenance.get("dino_loader_source") != "verified_local_cache_no_network_branch_resolution":
        raise ValueError("Unverified precision or external encoder source")
    cfg = prepare(args.vendor,args.task)["config"]
    data = torch.load(source,map_location="cpu",weights_only=True)
    runtime_provenance = dict(gpu_uuid=uuid,gpu=str(torch.cuda.get_device_properties(0)),
                              backend_provenance=backend.provenance,torch_version=torch.__version__,
                              tf32_matmul=False,tf32_cudnn=False)
    with torch.no_grad():
        run_case(backend,cfg,data,binding,args.output,args.manifest_sha256,args.protocol_sha256,runtime_provenance)


if __name__ == "__main__":
    main()
