"""Fixed shared-action, cached component replay on one registered fitting clip.

All components use the full arm's cached, pre-addition FP32 correction from the
same native input. Mean/centered components are not renormalized or recomputed
from their own changed rollouts. This is not a physical outcome experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .fixed_response import FixedResponseHook, FixedResponseIntervention, load_fitted_bank
from .fixed_combined_check import assert_bytes, load_stimulus, rng_signature
from .layer_attention_pilot import file_hash


def components(delta):
    if delta.ndim != 3 or len(delta) != 300 or not torch.isfinite(delta).all():
        raise ValueError('Require the complete finite 300-candidate field bank')
    common = delta.mean(dim=0, keepdim=True)
    return common, delta - common


class CaptureFull(FixedResponseHook):
    def _output(self, module, args, output):
        result = super()._output(module, args, output)
        if self.horizon == 3:
            field = output[:, -256:]
            spec = self.bank['operators'][self.arm]
            score = ((field.float().flatten(1) - self.bank['mean']) @
                     self.bank['projection'].T) / self.bank['scale']
            phi = torch.cat([torch.ones_like(score[:, :1]), score], 1)
            raw = phi @ spec['map'].T
            delta = (raw @ spec['basis'].flatten(1)).reshape_as(field)
            norm = delta.flatten(1).norm(dim=1)
            active = norm > self.bank['zero_threshold']
            factor = torch.where(active, self.bank['dose'] / norm.clamp_min(1e-10), 0.)
            delta = delta * factor[:, None, None]
            changed = torch.where(active[:, None, None], field + delta, field)
            assert_bytes(changed, result[:, -256:], 'independent cached full edit')
            self.native_field = field.detach().clone()
            self.full_delta = delta.detach().clone()
        return result


class ReplayField:
    def __init__(self, predictor, native_field, delta):
        self.predictor, self.native_field, self.delta = predictor, native_field, delta
        self.horizon = self.applications = 0

    def __enter__(self):
        if any(m._forward_hooks or m._forward_pre_hooks for m in self.predictor.modules()):
            raise ValueError('Replay requires an otherwise uninstrumented predictor')
        self.handles = [self.predictor.register_forward_pre_hook(self._input),
                        self.predictor.predictor_blocks[3].register_forward_hook(self._output)]
        return self

    def _input(self, module, args):
        self.horizon += 1

    def _output(self, module, args, output):
        if self.horizon != 3:
            return output
        self.applications += 1
        field = output[:, -256:]
        assert_bytes(field, self.native_field, 'same-input cached native field')
        if self.delta is None:
            return output
        result = output.clone()
        result[:, -256:] = field + self.delta
        return result

    def __exit__(self, kind, exc, tb):
        for handle in self.handles:
            handle.remove()
        if kind is None and (self.horizon != 6 or self.applications != 1):
            raise ValueError('Incomplete H6 replay')


def score_summary(reference, cost, native_elites, elites):
    delta = cost.double() - reference.double()
    centered = delta - delta.mean()
    ordered = cost.sort().values
    return {'score_change_mean': float(delta.mean()),
            'score_change_rms': float(delta.square().mean().sqrt()),
            'centered_score_change_rms': float(centered.square().mean().sqrt()),
            'native_elite_overlap_count': len(set(elites.tolist()) & set(native_elites.tolist())),
            'best_runnerup_margin': float(ordered[1] - ordered[0]),
            'elite_boundary_margin': float(ordered[10] - ordered[9])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('vendor', 'checkpoint', 'stimulus', 'fit', 'trace', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if file_hash(args.manifest) != args.manifest_sha256:
        raise ValueError('Unbound comparative manifest')
    manifest = json.loads(args.manifest.read_text())
    for label, path in [('source',Path(__file__)),('checkpoint',args.checkpoint),
                        ('stimulus_receipt',args.stimulus/'STIMULUS.json'),
                        ('fit_bank',args.fit/'operator_bank.pt'),('trace',args.trace)]:
        if file_hash(path) != manifest['sha256'][label]:
            raise ValueError('Comparative binding changed: '+label)
    if (manifest['native_populations'] != [0,14] or manifest['component_populations'] != [0]
            or manifest['frozen_arms'] != ['fixed_rank4','matched_random_fixed_rank4']
            or manifest['components'] != ['full','cached_full','zero','mean_only','centered_only']):
        raise ValueError('Unexpected finite arm/population registry')
    receipt = json.loads((args.stimulus/'STIMULUS.json').read_text())
    if (receipt['selected_fit_row']['index'] != 10587 or receipt['task'] != 'mw-reach'
            or receipt['development_or_protected_outcomes_accessed'] is not False):
        raise ValueError('Only original Reach fitting row10587 is registered')
    for name, digest in receipt['files'].items():
        if Path(name).name != name or file_hash(args.stimulus/name) != digest:
            raise ValueError('Stimulus bytes changed')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one CUDA device required')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.json').write_bytes(args.manifest.read_bytes())
    from .backends import JepaBackend
    from .planning_contract import prepare
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    backend = JepaBackend(args.vendor,args.checkpoint,manifest['sha256']['checkpoint'],
                          'metaworld','cuda:0','float32')
    bank = load_fitted_bank(args.fit,task='mw-reach',checkpoint_sha256=manifest['sha256']['checkpoint'])
    fixed = FixedResponseIntervention(backend,bank,'fixed_rank4')
    trace = torch.load(args.trace,map_location='cpu',weights_only=True)
    if len(trace['iterations']) != 15:
        raise ValueError('Incomplete native15-iteration trace')
    value = load_stimulus(args.stimulus)
    with torch.no_grad():
        obs = {k:v.to('cuda:0') for k,v in value['observations'].items()}
        actions = value['actions'].to('cuda:0')
        visual,proprio,_ = backend.model.model.encode(obs,actions)
        context = TensorDict({'visual':visual[:,:1],'proprio':proprio[:,:1]},batch_size=[])
        goal = TensorDict({'visual':visual[:,6:7],'proprio':proprio[:,6:7]},batch_size=[])
        cfg = prepare(args.vendor,'reach')['config']
        objective = ReprTargetDistMPCObjective(cfg,goal,**cfg['planner']['planning_objective'])
        rng = rng_signature('cuda:0')
        summaries, files = [], {}
        for population in manifest['native_populations']:
            row = trace['iterations'][population]
            action = row['candidate_actions'].to('cuda:0')
            if action.shape != (6,300,20):
                raise ValueError('Changed full native action bank')
            native = backend.predict(context,action)
            native_cost = objective(native,action).cpu()
            assert_bytes(native_cost,row['objective_costs'],'native score replay')
            native_elites = torch.topk(-native_cost.to('cuda:0'),10,dim=0).indices.cpu()
            assert_bytes(native_elites,row['elite_indices'],'native elite replay')

            def save(name, forecast, cost, extra=None):
                cost = cost.detach().cpu().clone()
                elites = torch.topk(-cost.to('cuda:0'),10,dim=0).indices.cpu()
                payload = {'population':population,'arm':name,'candidate_actions':action.cpu(),
                    'objective_costs':cost,'elite_indices':elites,
                    'forecast_h6':{k:forecast[k][-1].detach().cpu().clone() for k in ('visual','proprio')},
                    'forecast_scope':'actual final H6 embeddings for all300 candidates',
                    **(extra or {})}
                path = args.output/f'population{population:02d}-{name}.pt'
                with path.open('xb') as stream:
                    torch.save(payload,stream)
                files[path.name] = file_hash(path)
                summary = {'population':population,'arm':name,**score_summary(native_cost,cost,native_elites,elites)}
                summaries.append(summary)
                print(json.dumps(summary),flush=True)
                return cost

            save('native',native,native_cost)
            del native
            for arm in manifest['frozen_arms']:
                start = time.monotonic()
                with CaptureFull(backend.predictor,fixed.bank,arm) as capture:
                    full = backend.predict(context,action)
                full_cost = objective(full,action)
                save(arm+'-full',full,full_cost,{'coefficients':capture.record['coefficients'].cpu()})
                if population in manifest['component_populations']:
                    mean,centered = components(capture.full_delta)
                    component_costs = {}
                    for name,delta in [('cached_full',capture.full_delta),('zero',None),
                                       ('mean_only',mean),('centered_only',centered)]:
                        with ReplayField(backend.predictor,capture.native_field,delta):
                            forecast = backend.predict(context,action)
                        cost = objective(forecast,action)
                        if name == 'cached_full':
                            for key in ('visual','proprio'):
                                assert_bytes(forecast[key],full[key],'cached-full forecast '+key)
                            assert_bytes(cost,full_cost,'cached-full objective')
                        elif name == 'zero':
                            assert_bytes(cost.cpu(),native_cost,'zero objective')
                        component_costs[name] = save(arm+'-'+name,forecast,cost)
                        del forecast
                    total_energy = capture.full_delta.double().square().sum()
                    centered_energy = centered.double().square().sum()
                    residual = (full_cost.cpu().double()-native_cost.double())
                    residual -= residual.mean()
                    common_change = component_costs['mean_only'].double()-native_cost.double()
                    common_change -= common_change.mean()
                    denom = residual.square().sum()
                    summaries.append({'population':population,'arm':arm,'component_audit':True,
                        'cached_full_exact_forecast_and_score_parity':True,
                        'zero_exact_score_parity':True,
                        'centered_field_energy_fraction':float(centered_energy/total_energy),
                        'common_centered_cost_reconstruction':float(1-(residual-common_change).square().sum()/denom) if denom>0 else None,
                        'component_renormalization':False,
                        'seconds':time.monotonic()-start})
                del full,full_cost,capture
            if rng_signature('cuda:0') != rng:
                raise ValueError('Replay changed global RNG')
        report = {'status':'fixed_single_fit_row_candidate_replay_complete',
            'fresh_confirmation':False,'physical_outcomes_measured':False,
            'unique_trajectories':1,'native_populations':[0,14],
            'manifest_sha256':args.manifest_sha256,'files':files,'summaries':summaries,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'rng_unchanged':True}
        with (args.output/'report.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        print(json.dumps({k:v for k,v in report.items() if k not in ('files','summaries')}),flush=True)


if __name__ == '__main__':
    main()
