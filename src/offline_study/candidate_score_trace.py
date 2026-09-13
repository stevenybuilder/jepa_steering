"""Observe an unchanged, single-device CEM plan in a dedicated process.

Captures actual topk indices (including the kernel's tie decisions), candidate
actions and objective scores. Does not resample, rescore or modify the planner.
CPU transfers add overhead: use this for diagnostics, not throughput claims.
"""
import torch


class CandidateScoreTrace:
    def __init__(self, planner):
        self.planner = planner
        self.iterations = []
        self.selected_plan = None
        self.pending = None

    def __enter__(self):
        if getattr(self.planner, 'distribute_planner', False):
            raise ValueError('Trace pilot supports single-device CEM only')
        self.original_cost = self.planner.cost_function
        self.original_topk = torch.topk
        self.had_cost_override = 'cost_function' in self.planner.__dict__

        def cost(actions, z_init):
            if self.pending is not None:
                raise ValueError('Previous population did not reach native elite selection')
            result = self.original_cost(actions, z_init)
            if (actions.ndim != 3 or result.shape != (actions.shape[1],)
                    or not torch.isfinite(result).all() or not torch.isfinite(actions).all()):
                raise ValueError('Unexpected/nonfinite candidate population')
            self.pending = result.detach()
            self.iterations.append({'iteration': len(self.iterations),
                'candidate_actions': actions.detach().cpu().clone(),
                'objective_costs': result.detach().cpu().clone()})
            return result

        def topk(value, k, *args, **kwargs):
            result = self.original_topk(value, k, *args, **kwargs)
            if self.pending is not None and value.shape == self.pending.shape:
                dim = kwargs.get('dim', args[0] if args else -1)
                largest = kwargs.get('largest', args[1] if len(args) > 1 else True)
                if (k != self.planner.num_elites or dim not in (0, -1) or not largest
                        or not torch.equal(value, -self.pending)):
                    raise ValueError('Unexpected CEM elite selection; refusing inferred indices')
                row = self.iterations[-1]
                row['elite_indices'] = result.indices.detach().cpu().clone()
                ordered = row['objective_costs'].sort().values
                row['best_runnerup_margin'] = float(ordered[1] - ordered[0]) if len(ordered) > 1 else None
                row['elite_boundary_margin'] = float(ordered[k] - ordered[k-1]) if k < len(ordered) else None
                self.pending = None
            return result

        self.planner.cost_function = cost
        torch.topk = topk
        return self

    def run(self, *args, **kwargs):
        result = self.planner.plan(*args, **kwargs)
        if self.pending is not None or len(self.iterations) != self.planner.iterations:
            raise ValueError('Incomplete CEM trace')
        self.selected_plan = result.actions.detach().cpu().clone()
        self.final_mean = self.planner._prev_mean.detach().cpu().clone()
        return result

    def payload(self):
        if self.selected_plan is None:
            raise ValueError('No completed plan')
        return {'schema_version': 1, 'scope': 'diagnostic_not_original_confirmation_trace',
                'elite_indices_source': 'actual_native_torch_topk_return',
                'score_semantics': 'planner_objective_lower_is_better_not_physical_outcome',
                'iterations': self.iterations, 'selected_plan': self.selected_plan,
                'final_mean': self.final_mean}

    def __exit__(self, *exc):
        torch.topk = self.original_topk
        if self.had_cost_override:
            self.planner.cost_function = self.original_cost
        else:
            del self.planner.cost_function
