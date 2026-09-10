"""Causal HMM/constant/memoryless gates around the fixed-response successor.

First experiment retains a separate, unedited native shadow. Its H3-H6 values
are never read by routing. No state is shared across candidates or calls.
"""
import torch

from .fixed_response import FixedResponseHook, validate_bank
from .routing_hmm import route_prefix, validate_model

ARMS=('native','constant_gate','memoryless_gate','hmm_filtered_gate',
    'matched_random_constant_gate','matched_random_memoryless_gate','matched_random_hmm_filtered_gate')
GATE_KEYS={'constant_gate':'static','memoryless_gate':'memoryless','hmm_filtered_gate':'hmm'}


class NativePrefix:
    def __init__(self,predictor):
        self.predictor=predictor; self.horizon=0; self.values={}; self.handles=[]

    def __enter__(self):
        if (len(self.predictor.predictor_blocks)!=6 or
                any(m._forward_hooks or m._forward_pre_hooks for m in self.predictor.modules())):
            raise ValueError('Native shadow requires an untouched six-block predictor')
        self.handles=[self.predictor.register_forward_pre_hook(self._input),
            self.predictor.predictor_blocks[3].register_forward_hook(self._output)]
        return self

    def _input(self,module,args):
        self.horizon+=1

    def _output(self,module,args,output):
        if self.horizon in (1,2):
            if self.horizon in self.values or output.ndim!=3 or output.shape[1]<256 or output.shape[2]!=400:
                raise ValueError('Changed native prefix field')
            self.values[self.horizon]=output[:,-256:].detach().float().mean(1)
        # No tensor return: never edit the shadow. H3-H6 values are not stored.

    def __exit__(self,kind,exc,traceback):
        for handle in self.handles: handle.remove()
        if kind is None and (self.horizon!=6 or set(self.values)!={1,2}):
            raise ValueError('Expected full native H6 shadow with exactly H1/H2 features')


class RoutedFixedResponse:
    def __init__(self,backend,bank,routing,arm):
        validate_bank(bank); validate_model(routing['model'])
        if arm not in ARMS+('zero_dose',) or getattr(backend,'allow_tf32',False):
            raise ValueError('Unregistered routed arm or precision')
        def stage(value,dtype):
            if isinstance(value,torch.Tensor): return value.to(device=backend.device,dtype=dtype)
            if isinstance(value,dict): return {k:stage(v,dtype) for k,v in value.items()}
            return value
        self.backend,self.arm=backend,arm
        self.bank=stage(bank,torch.float32);self.routing=stage(routing,torch.float64)
        normalizers=self.routing['normalizers']
        if set(normalizers)!={'static','memoryless','hmm'} or any(
                v.numel()!=1 or not torch.isfinite(v).all() or not bool(v>0) for v in normalizers.values()):
            raise ValueError('Invalid frozen fit-only gate normalizers')
        self.calls=0;self.last_record=None

    @torch.no_grad()
    def __call__(self,context,act_suffix=None,**kwargs):
        if kwargs or act_suffix is None or act_suffix.ndim!=3 or not 1<=len(act_suffix)<=6 or act_suffix.shape[1]<1:
            raise ValueError('Require unchanged H1-H6 explicit actions')
        self.calls+=1
        self.last_record={'backend_calls':1,'native_shadow_rollouts':0,'response_probe_rollouts':0,
            'horizon':len(act_suffix),'routing_horizons':[],'future_routing_features_read':False}
        if len(act_suffix)<6 or self.arm in ('native','zero_dose'):
            return self.backend.predict(context,act_suffix)
        with NativePrefix(self.backend.predictor) as capture:
            native=self.backend.predict(context,act_suffix)
        prefix=torch.stack([capture.values[1],capture.values[2]],1)
        gates=route_prefix(prefix,self.routing['model'])
        name=self.arm.removeprefix('matched_random_'); key=GATE_KEYS[name]
        gate=(gates[key]/self.routing['normalizers'][key]).float()
        if not torch.isfinite(gate).all() or not (gate>0).all():
            raise ValueError('Nonfinite gate; no candidate filtering')
        arm='matched_random_fixed_rank4' if self.arm.startswith('matched_random_') else 'fixed_rank4'
        bank={**self.bank,'dose':self.bank['dose']*gate}
        with FixedResponseHook(self.backend.predictor,bank,arm) as hook:
            result=self.backend.predict(context,act_suffix)
        for key in ('visual','proprio'):
            if not torch.isfinite(result[key]).all() or not torch.equal(result[key][:3],native[key][:3]):
                raise ValueError('Future-only edit changed its native H1/H2 prefix or is nonfinite')
        self.last_record.update(hook.record)
        self.last_record.update(backend_calls=2,native_shadow_rollouts=1,routing_horizons=[1,2],
            normalized_gate=gate.detach(),raw_gate=gates[GATE_KEYS[name]].detach(),
            posterior=gates['posterior'].detach(),gate_normalizer=self.routing['normalizers'][GATE_KEYS[name]].detach())
        return result
