"""Checkpoint-native branch interventions with post-projection/gate dose control.

Head edits act on concatenated attention heads before the output projection;
their dose is measured after that projection and AdaLN attention gate. MLP
edits act on the MLP output and use its own AdaLN gate. Only newest-frame
tokens are edited. All stochastic controls use a private CPU generator.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def spatial_mask(name, device):
    i = torch.arange(256, device=device)
    row, col = i // 16, i % 16
    choices = {"all": torch.ones(256, dtype=torch.bool, device=device), "top": row < 8,
               "bottom": row >= 8, "left": col < 8, "right": col >= 8,
               "center": (row >= 4) & (row < 12) & (col >= 4) & (col < 12),
               "checker": (row + col) % 2 == 0}
    if name not in choices:
        raise ValueError("Unknown spatial mask")
    return choices[name]


class ComponentPatch:
    def __init__(self, block, candidate, delta_fn, cap_radius, sham=False, identity=False):
        self.block, self.candidate, self.delta_fn = block, candidate, delta_fn
        self.cap_radius, self.sham, self.identity = float(cap_radius), sham, identity
        if not self.cap_radius > 0:
            raise ValueError("Positive shared residual-space budget required")
        self.site = candidate["site"]
        if self.site not in ("residual", "attention_preproj", "mlp_output"):
            raise ValueError("Unknown intervention site")
        if candidate["pulse"] not in range(7) or candidate["sign"] not in (-1, 1) or candidate["dose"] < 0:
            raise ValueError("Invalid pulse, sign, or dose")
        if self.site != "attention_preproj" and candidate.get("head") is not None:
            raise ValueError("Head selection only applies before attention projection")
        self.records, self.handles = [], []
        self.norm_targets = None
        self.reset()

    def reset(self):
        self.step = 0
        self.gate_msa = self.gate_mlp = self.block_input = self.mlp_residual_input = None
        self.pending = None

    def active(self):
        return not self.identity and self.candidate["dose"] != 0 and self.candidate["pulse"] in (0, self.step)

    def _block_pre(self, module, args):
        self.step += 1
        if self.step > 6:
            raise RuntimeError("Reset ComponentPatch before each native six-step unroll")
        self.block_input = args[0]
        if self.block_input.ndim != 3 or self.block_input.shape[1] % 256:
            raise RuntimeError("Expected native [batch,frames*256,channels] residual")
        self.gate_msa = self.gate_mlp = self.pending = None

    def _modulation(self, module, args, output):
        if not self.active() or self.site == "residual":
            return
        b, n, d = self.block_input.shape
        if output.ndim != 3 or output.shape[0] != b or output.shape[-1] != 6*d or n % output.shape[1]:
            raise RuntimeError("Unexpected AdaLN modulation/token axes")
        chunks = output.chunk(6, dim=-1)
        self.gate_msa = chunks[2].repeat_interleave(n//output.shape[1], dim=1)
        self.gate_mlp = chunks[5].repeat_interleave(n//output.shape[1], dim=1)

    def _head_support(self, d):
        head = self.candidate.get("head")
        if head is None:
            return 0, d
        heads = int(self.block.attn.num_heads)
        if d % heads or not 0 <= head < heads:
            raise RuntimeError("Head specification does not match checkpoint")
        width = d // heads
        return head*width, (head+1)*width

    def _map(self, delta, gate):
        if self.site == "attention_preproj":
            delta = F.linear(delta, self.block.attn.proj.weight.to(delta), None)
        return delta if gate is None else delta * gate[:, -256:].to(delta)

    def _delta(self, current, gate):
        raw = self.delta_fn(current.float(), self.step)
        if raw.shape != current.shape or not torch.isfinite(raw).all():
            raise RuntimeError("delta_fn must return finite newest-token raw directions")
        raw = raw.to(torch.float32)
        lo, hi = self._head_support(raw.shape[-1]) if self.site == "attention_preproj" else (0, raw.shape[-1])
        channel_mask = torch.zeros(raw.shape[-1], device=raw.device); channel_mask[lo:hi] = 1
        raw = raw * channel_mask * spatial_mask(self.candidate["mask"], raw.device)[None, :, None]
        mapped = self._map(raw, gate)
        norm = mapped.flatten(1).norm(dim=-1)
        factor = (self.cap_radius/norm.clamp_min(1e-30)).clamp(max=1)
        # Dose AFTER the base cap: saturation must not make .125 and1 identical.
        semantic = raw*factor[:, None, None]*(self.candidate["sign"]*self.candidate["dose"])
        desired = self._map(semantic, gate).flatten(1).norm(dim=-1)
        local_desired = desired.clone()
        delta = semantic
        if self.sham:
            if self.norm_targets is not None:
                if self.step not in self.norm_targets:
                    raise RuntimeError("Missing pre-recorded semantic norm at active horizon")
                desired = torch.as_tensor(self.norm_targets[self.step], device=raw.device, dtype=raw.dtype)
                if desired.shape != local_desired.shape or not torch.isfinite(desired).all() or (desired < 0).any():
                    raise RuntimeError("Invalid semantic-run norm targets")
                if (desired > self.cap_radius*self.candidate["dose"] + 1e-5).any():
                    raise RuntimeError("Semantic norm targets exceed frozen delivered budget")
            # Shuffle ONLY within the selected head's support, then match dose
            # after the actual output projection and action-conditioned gate.
            gen = torch.Generator(device="cpu").manual_seed(2026090607)
            perm = torch.randperm(hi-lo, generator=gen).to(raw.device)
            signs = (2*torch.randint(0, 2, (hi-lo,), generator=gen)-1).to(raw)
            delta = torch.zeros_like(semantic)
            delta[..., lo:hi] = semantic[..., lo:hi][..., perm]*signs
            available = self._map(delta, gate).flatten(1).norm(dim=-1)
            if ((desired > 0) & (available == 0)).any():
                raise RuntimeError("Nonzero semantic residual dose has zero mapped sham")
            delta = delta*(desired/available.clamp_min(1e-30))[:, None, None]
        requested = self._map(delta, gate).flatten(1).norm(dim=-1)
        record = {"step": self.step, "site": self.site, "head": self.candidate.get("head"),
                  "semantic_requested_residual_norm": desired.detach().cpu(),
                  "same_input_semantic_residual_norm": local_desired.detach().cpu(),
                  "norm_target_source": "recorded_semantic_trajectory" if self.sham and self.norm_targets is not None else "same_input_semantic_direction",
                  "requested_residual_norm": requested.detach().cpu(),
                  "normmatch_max_abs": float((requested-desired).abs().max()),
                  "uncapped_residual_norm": norm.detach().cpu(),
                  "cap_saturated": (norm > self.cap_radius).detach().cpu()}
        return delta, record

    def _attention_pre(self, module, args):
        if not self.active():
            return
        if self.gate_msa is None:
            raise RuntimeError("Attention gate not captured before output projection")
        original = args[0]
        delta, record = self._delta(original[:, -256:], self.gate_msa)
        edited = original.clone(); edited[:, -256:] += delta.to(original)
        # Exact native linear dtype/operation for the unedited branch reference.
        unedited_branch = F.linear(original, module.weight, module.bias)
        self.pending = (record, unedited_branch)
        return (edited,) + tuple(args[1:])

    def _attention_post(self, module, args, output):
        if self.pending is None:
            return
        record, unedited = self.pending
        # eval-mode dropout/drop-path are identity; compare the actual rounded
        # attention residual addition, not just a float32 intended delta.
        a = self.block_input + output*self.gate_msa
        b = self.block_input + unedited*self.gate_msa
        record["actual_rounded_residual_norm"] = (a[:, -256:].float()-b[:, -256:].float()).flatten(1).norm(dim=-1).detach().cpu()
        self.records.append(record); self.pending = None

    def _norm2_pre(self, module, args):
        self.mlp_residual_input = args[0]

    def _mlp_post(self, module, args, output):
        if not self.active():
            return
        if self.gate_mlp is None or self.mlp_residual_input is None:
            raise RuntimeError("MLP gate/residual input not captured")
        delta, record = self._delta(output[:, -256:], self.gate_mlp)
        edited = output.clone(); edited[:, -256:] += delta.to(output)
        a = self.mlp_residual_input + edited*self.gate_mlp
        b = self.mlp_residual_input + output*self.gate_mlp
        record["actual_rounded_residual_norm"] = (a[:, -256:].float()-b[:, -256:].float()).flatten(1).norm(dim=-1).detach().cpu()
        self.records.append(record)
        return edited

    def _residual_post(self, module, args, output):
        if not self.active():
            return
        delta, record = self._delta(output[:, -256:], None)
        edited = output.clone(); edited[:, -256:] += delta.to(output)
        record["actual_rounded_residual_norm"] = (edited[:, -256:].float()-output[:, -256:].float()).flatten(1).norm(dim=-1).detach().cpu()
        self.records.append(record)
        return edited

    def __enter__(self):
        if self.block.training:
            raise RuntimeError("Only frozen eval-mode branch experiments supported")
        self.handles.append(self.block.register_forward_pre_hook(self._block_pre))
        if self.site == "residual":
            self.handles.append(self.block.register_forward_hook(self._residual_post))
        else:
            self.handles.append(self.block.adaLN_modulation.register_forward_hook(self._modulation))
            if self.site == "attention_preproj":
                if not isinstance(self.block.attn.proj, torch.nn.Linear):
                    self.__exit__(); raise RuntimeError("Expected native linear attention projection")
                self.handles.append(self.block.attn.proj.register_forward_pre_hook(self._attention_pre))
                self.handles.append(self.block.attn.proj.register_forward_hook(self._attention_post))
            else:
                self.handles.append(self.block.norm2.register_forward_pre_hook(self._norm2_pre))
                self.handles.append(self.block.mlp.register_forward_hook(self._mlp_post))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []
