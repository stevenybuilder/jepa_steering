"""Engineering adapter for unchanged coupling + the refined native-feature map.

One backend forecast, with an explicitly counted native H3 predictor-prefix replay
through B3. This is NOT zero extra predictor work, a coupled-feature refit, or the
legacy candidate-response combined operator. No behavioral launch is granted here.
"""
from __future__ import annotations

import torch
import threading

from offline_study.interventions.fixed_response import FixedResponseIntervention, ZERO_THRESHOLD
from offline_study.interventions.interventions import PredictorIntervention
from offline_study.planning.planning_intervention import static_edits


METHOD = "native_prefix_fixed_combined_v1"
ARM_COMPONENTS = {
    "combined_fixed_rank4": ("joint_equal_standardized_energy", "fixed_rank4"),
    "matched_random_combined_fixed_rank4": (
        "matched_random_equal_standardized_energy", "matched_random_fixed_rank4"),
}


def validate_coupling(protocol, bank):
    """Only the existing upstream-H3 coupling scopes can share a native H1/H2."""
    if protocol.get("category") != "vision_action_coupling":
        raise ValueError("Require the existing coupling protocol")
    for name, _ in ARM_COMPONENTS.values():
        arms = [arm for arm in protocol.get("arms", []) if arm.get("name") == name]
        if len(arms) != 1 or len(arms[0].get("edits", [])) != 2:
            raise ValueError("Require both frozen equal-budget coupling components")
        sites = set()
        for edit in arms[0]["edits"]:
            site = (edit.get("site"), edit.get("block"))
            if (site not in {("predictor_visual", None), ("block_condition", 3)} or
                    edit.get("horizon") != 3 or "gate" in edit or
                    edit.get("token_start") is not None or edit.get("token_end") is not None):
                raise ValueError("Only predictor-visual/H3 and condition/B3/H3 coupling is allowed")
            if site in sites:
                raise ValueError("Duplicate coupling component")
            sites.add(site)
        # Reuse the source compiler's tensor/finite/scale checks without a model.
        edits = static_edits(protocol, bank, name, 1, 6, "cpu")
        if len(edits) != 2 or any(not torch.isfinite(edit.delta).all() for edit in edits):
            raise ValueError("Require two finite nonzero-scale source components")


def validate_predictor(predictor):
    if len(predictor.predictor_blocks) != 6:
        raise ValueError("Native-prefix contract requires six predictor blocks")
    if any(module.training for module in predictor.modules()):
        raise ValueError("Native-prefix replay requires an entirely eval-mode predictor")
    if getattr(predictor, "use_activation_checkpointing", False):
        raise ValueError("Activation checkpointing is not supported by prefix replay")
    if any(getattr(module, "proj_drop_prob", 0.) != 0. for module in predictor.modules()):
        raise ValueError("Native-prefix replay requires zero SDPA dropout, even in eval mode")
    from torch.nn.modules import module as module_state
    if (module_state._global_forward_hooks or module_state._global_forward_pre_hooks):
        raise ValueError("Global forward hooks are incompatible with native-prefix replay")
    if any(module._forward_hooks or module._forward_pre_hooks for module in predictor.modules()):
        raise ValueError("Combined adapter cannot coexist with preexisting predictor hooks")


class _PrefixComplete(Exception):
    def __init__(self, owner):
        self.owner = owner


class NativePrefixCombinedHook(PredictorIntervention):
    """Replay the actual native caller, not a hand-reimplemented transformer.

At H3 the outer predictor pre-hook has not changed any input. The nested call
uses exactly those args (no candidate expansion, clone or layout conversion),
skips our edits, and exits after native B3. Then the original call proceeds with
the source coupling hooks. No full unroll is called by the nested prefix.
"""
    def __init__(self, predictor, edits, bank, arm):
        super().__init__(predictor, edits, expected_horizons=6)
        self.bank, self.arm = bank, arm
        self.replaying = False
        self.native_output = None
        self.prefix_blocks, self.main_blocks = [], []
        self.prefix_calls, self.rank_applications = 0, 0
        self.record = None

    def __enter__(self):
        validate_predictor(self.predictor)
        try:
            return super().__enter__()
        except BaseException:
            for handle in self.handles:
                handle.remove()
            raise

    def _predictor_input(self, module, args):
        if len(args) != 3:
            raise ValueError("Require the pinned positional visual/action/proprio caller")
        if self.replaying:
            return None
        if self.horizon + 1 == 3:
            if self.prefix_calls:
                raise ValueError("Native H3 prefix may run only once")
            self.prefix_calls += 1
            self.replaying = True
            completed = False
            try:
                module(*args)
            except _PrefixComplete as exc:
                if exc.owner is not self:
                    raise
                completed = True
            finally:
                self.replaying = False
            if not completed or self.native_output is None or self.prefix_blocks != [0, 1, 2, 3]:
                raise ValueError("Native prefix did not terminate exactly after B3")
        return super()._predictor_input(module, args)

    def _block_input(self, block, args, kwargs):
        if self.replaying:
            return None
        return super()._block_input(block, args, kwargs)

    def _block_output(self, block, output):
        if self.replaying:
            self.prefix_blocks.append(block)
            if self.prefix_blocks != list(range(block + 1)) or block > 3:
                raise ValueError("Unexpected native prefix block order")
            if block == 3:
                self.native_output = output.detach()
                raise _PrefixComplete(self)
            return output
        self.main_blocks.append(block)
        output = super()._block_output(block, output)
        if self.horizon != 3 or block != 3:
            return output
        if self.rank_applications or self.native_output is None:
            raise ValueError("Missing native feature or repeated refined edit")
        self.rank_applications += 1
        native = self.native_output
        if (not isinstance(output, torch.Tensor) or output.ndim != 3 or
                output.shape[1] < 256 or output.shape[2] != 400 or
                native.shape != output.shape or native.dtype != output.dtype or
                native.device != output.device):
            raise ValueError("Native/coupled B3 fields differ in dimensions, dtype or device")
        # Same operation order as FixedResponseHook; crucially the projection
        # reads native, not coupled, H3. No inverse or future target is online.
        with torch.autocast(device_type=output.device.type, enabled=False):
            field = native[:, -256:]
            spec = self.bank["operators"][self.arm]
            score = ((field.float().flatten(1) - self.bank["mean"]) @
                     self.bank["projection"].T) / self.bank["scale"]
            phi = torch.cat([torch.ones_like(score[:, :1]), score], 1)
            raw = phi @ spec["map"].T
            delta = (raw @ spec["basis"].flatten(1)).reshape_as(field)
            norm = delta.flatten(1).norm(dim=1)
            if not torch.isfinite(norm).all():
                raise ValueError("Nonfinite correction; no candidate may be discarded")
            active = norm > self.bank["zero_threshold"]
            factor = torch.where(active, self.bank["dose"] / norm.clamp_min(ZERO_THRESHOLD), 0.)
            delta = delta * factor[:, None, None]
            coupled = output[:, -256:]
            changed = coupled + delta.to(output.dtype)
            changed = torch.where(active[:, None, None], changed, coupled)
            result = output.clone()
            result[:, -256:] = changed
            self.record = {"coefficients": (raw * factor[:, None]).detach(),
                "requested_l2": delta.flatten(1).norm(dim=1).detach(),
                "realized_l2": (changed.float() - coupled.float()).flatten(1).norm(dim=1).detach(),
                "active": active.detach()}
        self.native_output = None
        return result

    def __exit__(self, kind, exc, traceback):
        try:
            super().__exit__(kind, exc, traceback)
            if kind is None and (self.prefix_calls != 1 or self.prefix_blocks != [0, 1, 2, 3] or
                    self.main_blocks != list(range(6)) * 6 or self.rank_applications != 1):
                raise ValueError("Require six main predictor calls and exactly one four-block native prefix")
        finally:
            self.native_output = None


class CombinedFixedResponseIntervention:
    """Engineering-only composition; input/exposure/source gates belong to launcher."""
    def __init__(self, backend, fixed_bank, coupling_protocol, coupling_bank, arm):
        if arm not in ARM_COMPONENTS:
            raise ValueError("Unregistered refined combined arm")
        validate_predictor(backend.predictor)
        validate_coupling(coupling_protocol, coupling_bank)
        coupling_arm, fixed_arm = ARM_COMPONENTS[arm]
        # Reuse the fixed map's unchanged validation and one-time FP32 staging.
        staged = FixedResponseIntervention(backend, fixed_bank, fixed_arm)
        self.backend, self.bank, self.fixed_arm = backend, staged.bank, fixed_arm
        self.coupling_protocol, self.coupling_bank = coupling_protocol, coupling_bank
        self.coupling_arm, self.arm = coupling_arm, arm
        self.calls, self.last_record = 0, None
        self._lock = threading.Lock()

    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise ValueError("Concurrent/reentrant combined adapter use is not supported")
        try:
            return self._call(context, act_suffix, **kwargs)
        finally:
            self._lock.release()

    def _call(self, context, act_suffix=None, **kwargs):
        if (kwargs or act_suffix is None or act_suffix.ndim != 3 or
                not 1 <= len(act_suffix) <= 6 or act_suffix.shape[1] < 1):
            raise ValueError("Require explicit H1-H6 actions and unchanged rollout defaults")
        validate_predictor(self.backend.predictor)
        self.calls += 1
        horizon, batch = act_suffix.shape[:2]
        self.last_record = {"method": METHOD, "arm": self.arm, "horizon": horizon,
            "candidates": batch, "backend_calls": 1, "response_probe_rollouts": 0,
            "full_native_shadow_rollouts": 0, "native_prefix_replays": 0,
            "extra_native_predictor_blocks": 0, "main_predictor_blocks": horizon * 6,
            "rank_applications": 0, "behavioral_launch_ready": False}
        if horizon < 6:
            # Preserve PlanningSupportIntervention and the existing behavioral
            # H6StaticPlanningIntervention: ALL components stay native on H1-H5.
            edits = []
            result = self.backend.predict(context, act_suffix)
        else:
            edits = static_edits(self.coupling_protocol, self.coupling_bank, self.coupling_arm,
                                 batch, horizon, self.backend.device)
            with NativePrefixCombinedHook(self.backend.predictor, edits, self.bank, self.fixed_arm) as hook:
                result = self.backend.predict(context, act_suffix)
            self.last_record.update(hook.record)
            self.last_record.update(native_prefix_replays=hook.prefix_calls,
                extra_native_predictor_blocks=len(hook.prefix_blocks),
                main_predictor_blocks=len(hook.main_blocks), rank_applications=hook.rank_applications)
        if any(not torch.isfinite(result[key]).all() for key in ("visual", "proprio")):
            raise ValueError("Nonfinite prediction; no candidate may be discarded")
        if any(edit.realized_l2 is None or edit.applications != 1 for edit in edits):
            raise ValueError("Missing exactly-once coupling delivery")
        self.last_record["coupling"] = [{"site": edit.site, "block": edit.block,
            "horizon": edit.horizon, "requested_l2": edit.delivered_l2,
            "realized_l2": edit.realized_l2.detach(), "applications": edit.applications} for edit in edits]
        return result
