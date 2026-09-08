"""DROID-specific native H3 visual/B6-condition coupling, one model pass.

Twelve blocks,1024features,no proprioception. This is not the MetaWorld HMM or
fixed-response operator. Fits and full receiving-planner checks are separate.
"""
import math

import torch

from .operator_fit import fit_coupled_directions,orthogonal_random_control,permute_visual_direction

METHOD='droid_native_h3_b6_coupling_v1'
ARMS=('native','visual_only','action_condition_only','joint','joint_equal_standardized_energy',
      'permuted_visual','permuted_joint','matched_random','matched_random_equal_standardized_energy')


def fit_bank(visual,condition):
    if visual.shape!=(512,1,16,16,1024) or condition.shape!=(512,1024):
        raise ValueError('Require full128recordings x4native-prefix DROID captures')
    v,a,diagnostics=fit_coupled_directions(visual,condition,iterations=12)
    return {'method':METHOD,'visual':v,'action':a,'permuted_visual':permute_visual_direction(v,2026090702),
        'random_visual':orthogonal_random_control(v,2026090703),
        'random_action':orthogonal_random_control(a,2026090704),
        'visual_dose':.1*diagnostics['visual_score_robust_sigma'],
        'action_dose':.1*diagnostics['action_score_robust_sigma'],'fit_diagnostics':diagnostics}


def arm_fields(bank,arm,device):
    if bank['method']!=METHOD or arm not in ARMS+('zero_dose',):raise ValueError('Wrong DROID coupling method/arm')
    for name,shape in (('visual',(1,16,16,1024)),('action',(1024,)),('permuted_visual',(1,16,16,1024)),
            ('random_visual',(1,16,16,1024)),('random_action',(1024,))):
        if bank[name].shape!=shape or not torch.isfinite(bank[name]).all():raise ValueError('Wrong DROID field shape/finite values')
        if not torch.isclose(bank[name].double().norm(),torch.tensor(1.,dtype=torch.float64),atol=2e-5,rtol=0):
            raise ValueError('Unnormalized DROID direction')
    for name in ('visual_dose','action_dose'):
        if not math.isfinite(bank[name]) or bank[name]<=0:raise ValueError('Invalid fit-only dose')
    if arm in ('native','zero_dose'):return None,None
    visual='permuted_visual' if arm.startswith('permuted') else 'random_visual' if arm.startswith('matched_random') else 'visual'
    action='random_action' if arm.startswith('matched_random') else 'action'
    scale=2**-.5 if 'equal_standardized_energy' in arm else 1.
    v=None if arm=='action_condition_only' else bank[visual].to(device=device,dtype=torch.float32)*bank['visual_dose']*scale
    a=None if arm in ('visual_only','permuted_visual') else bank[action].to(device=device,dtype=torch.float32)*bank['action_dose']*scale
    return v,a


class Capture:
    def __init__(self,predictor):
        if len(predictor.predictor_blocks)!=12:raise ValueError('DROID requires12predictor blocks')
        self.predictor=predictor;self.horizon=0;self.visual=None;self.condition=None;self.handles=[]
    def __enter__(self):
        self.handles=[self.predictor.register_forward_pre_hook(self._input),
            self.predictor.predictor_blocks[6].register_forward_pre_hook(self._condition,with_kwargs=True)]
        return self
    def _input(self,module,args):
        self.horizon+=1
        if len(args)!=3 or args[2] is not None:raise ValueError('DROID must retain no-proprio native input')
        if self.horizon==3:
            if self.visual is not None or args[0].shape[2:]!=(1,16,16,1024):raise ValueError('Changed DROID capture site')
            self.visual=args[0][:,-1].detach().float().cpu().clone()
    def _condition(self,module,args,kwargs):
        if self.horizon==3:
            if self.condition is not None or args[1].shape[2:]!=(1024,):raise ValueError('Changed DROID condition')
            self.condition=args[1][:,-1].detach().float().cpu().clone()
    def __exit__(self,kind,*args):
        for handle in self.handles:handle.remove()
        if kind is None and (self.horizon!=3 or self.visual is None or self.condition is None):
            raise ValueError('Incomplete native H3 capture')


class Hook:
    def __init__(self,predictor,visual,action):
        if len(predictor.predictor_blocks)!=12:raise ValueError('Not the12-block DROID predictor')
        if visual is not None and visual.shape!=(1,16,16,1024):raise ValueError('Wrong visual field shape')
        if action is not None and action.shape!=(1024,):raise ValueError('Wrong condition field shape')
        self.predictor=predictor;self.visual=visual;self.action=action;self.horizon=0;self.handles=[];self.energy={}
    def __enter__(self):
        if self.predictor._forward_pre_hooks or any(b._forward_pre_hooks or b._forward_hooks for b in self.predictor.predictor_blocks):
            raise ValueError('Do not compose unrelated predictor hooks')
        self.handles=[self.predictor.register_forward_pre_hook(self._input),
            self.predictor.predictor_blocks[6].register_forward_pre_hook(self._condition,with_kwargs=True)]
        return self
    def _edit(self,value,field,key):
        original=value[:,-1];updated=value.clone();updated[:,-1]=original+field.to(original.dtype)
        actual=(updated[:,-1].float()-original.float()).flatten(1).norm(dim=1)
        self.energy[key]={'requested_l2':float(field.double().norm()),'realized_l2':actual.detach()}
        return updated
    def _input(self,module,args):
        self.horizon+=1
        if len(args)!=3 or args[2] is not None:raise ValueError('Unexpected DROID proprioceptive input')
        if self.horizon==3 and self.visual is not None:
            return (self._edit(args[0],self.visual,'visual'),*args[1:])
    def _condition(self,module,args,kwargs):
        if self.horizon==3 and self.action is not None:
            return (args[0],self._edit(args[1],self.action,'action'),*args[2:]),kwargs
    def __exit__(self,kind,*args):
        for handle in self.handles:handle.remove()
        expected={'visual' for _ in [0] if self.visual is not None}|{'action' for _ in [0] if self.action is not None}
        if kind is None and (self.horizon!=3 or set(self.energy)!=expected):raise ValueError('DROID hook coverage changed')


class Intervention:
    def __init__(self,model,bank,arm):
        self.model=model;self.arm=arm;self.visual,self.action=arm_fields(bank,arm,model.device)
        self.backend_calls=0;self.last_energy={}
    def __call__(self,context,act_suffix=None,**kwargs):
        if kwargs or act_suffix is None or act_suffix.ndim!=3 or act_suffix.shape[2]!=7:
            raise ValueError('DROID requires native three-dimensional seven-action suffix')
        self.backend_calls=1;self.last_energy={}
        if self.arm in ('native','zero_dose') or act_suffix.shape[0]<3:
            return self.model.unroll(context,act_suffix=act_suffix)
        if act_suffix.shape[0]!=3:raise ValueError('Do not apply theDROID H3contract to another horizon')
        with Hook(self.model.model.predictor,self.visual,self.action) as hook:
            result=self.model.unroll(context,act_suffix=act_suffix)
        self.last_energy=hook.energy
        return result
