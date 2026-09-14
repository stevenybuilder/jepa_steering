import ast
import importlib.util
from pathlib import Path
import types
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("action_condition_specificity",ROOT/"src/offline_study/experiments/action_condition_specificity.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Block(torch.nn.Module):
    def forward(self,x,z,**kwargs):
        return x+z[:,-1,None]


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = torch.nn.ModuleList([Block() for _ in range(6)])

    def forward(self,z):
        rows = []
        for h in range(6):
            x = torch.zeros(300,2,z.shape[-1])
            for block in self.predictor_blocks:
                x = block(x,z,T=2)
            rows.append(x)
        return torch.stack(rows)


class ActionConditionTests(unittest.TestCase):
    def condition(self):
        return torch.randn(300,2,8,generator=torch.Generator().manual_seed(17))

    def test_newest_cyclic_permutation_and_prior_time_identity(self):
        z = self.condition();original = z.clone()
        changed,audit = m.edited_condition(z,"permute")
        self.assertTrue(torch.equal(changed[:,-1],torch.roll(z[:,-1],-1,0)))
        self.assertTrue(torch.equal(changed[:,:-1],z[:,:-1]))
        self.assertTrue(torch.equal(z,original))
        self.assertEqual(audit["max_relative_norm_error"],0.)

    def test_random_per_candidate_match_private_rng_and_reproducibility(self):
        z = self.condition();before = torch.get_rng_state().clone()
        _,permutation = m.edited_condition(z,"permute")
        edited,audit = m.edited_condition(z,"random",37)
        repeat,_ = m.edited_condition(z,"random",37)
        self.assertTrue(torch.equal(edited,repeat))
        self.assertTrue(torch.equal(torch.get_rng_state(),before))
        np.testing.assert_allclose(audit["delivered_delta_l2"],permutation["requested_delta_l2"],rtol=1e-6)
        self.assertFalse(torch.equal(edited[:,-1],torch.roll(z[:,-1],-1,0)))

    def test_zero_condition_differences_remain_exact_zero(self):
        z = torch.ones(300,2,8)
        edited,audit = m.edited_condition(z,"random",37)
        self.assertTrue(torch.equal(edited,z))
        self.assertEqual(audit["zero_target_count"],300)
        self.assertEqual(max(audit["delivered_delta_l2"]),0.)

    def test_hook_exact_native_parity_only_h3_and_single_block(self):
        predictor = Predictor().eval();z = self.condition()
        native = predictor(z)
        with m.ConditionHook(predictor) as capture:
            zero = predictor(z)
        self.assertTrue(torch.equal(zero,native));self.assertEqual(set(capture.captured),set(range(6)))
        for layer in range(6):
            with m.ConditionHook(predictor,cache=capture.captured,layer=layer,mode="permute") as hook:
                edited = predictor(z)
            self.assertEqual(hook.counts,[6]*6)
            self.assertFalse(torch.equal(edited[2],native[2]))
            self.assertTrue(torch.equal(edited[[0,1,3,4,5]],native[[0,1,3,4,5]]))
            self.assertTrue(all(not block._forward_pre_hooks for block in predictor.predictor_blocks))

    def test_changed_cache_and_incomplete_horizon_fail(self):
        predictor = Predictor().eval();z = self.condition()
        with m.ConditionHook(predictor) as capture: predictor(z)
        capture.captured[2][0,0,0] += 1
        with self.assertRaisesRegex(ValueError,"cached native"):
            with m.ConditionHook(predictor,cache=capture.captured,layer=2,mode="permute"):predictor(z)
        self.assertTrue(all(not block._forward_pre_hooks for block in predictor.predictor_blocks))
        with self.assertRaisesRegex(ValueError,"six rollout"):
            with m.ConditionHook(predictor):predictor.predictor_blocks[0](torch.zeros(300,2,8),z)

    def test_seed_registry_is_deterministic_and_distinct(self):
        seeds = [m.random_seed(t,e,b,l) for t in m.TASKS for e in range(8) for b in m.BANKS for l in m.LAYERS]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertEqual(m.random_seed("reach",0,"fresh",2),m.random_seed("reach",0,"fresh",2))

    def test_native_sampler_dependency_injected_before_rng_boundary(self):
        cfg={"planner":dict(horizon=6,num_samples=300,var_scale=1.,max_norms=None)}
        actions,rng=m.initial_population(cfg,123,"cpu")
        class NativeSamplerFixture:
            def __init__(self,unroll,action_dim,local_generator,**kwargs):
                self.generator=local_generator
            def plan(self,context,steps_left):
                sampled=torch.randn(6,300,20,generator=self.generator)
                sampled[:,0]=0
                self.cost_function(sampled,context)
        torch_before=torch.get_rng_state().clone()
        numpy_before=np.random.get_state()
        m.verify_native_sampler(cfg,123,actions,rng,NativeSamplerFixture)
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        self.assertEqual(numpy_before[0],np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_before[1],np.random.get_state()[1])
        self.assertEqual(numpy_before[2:],np.random.get_state()[2:])
        tree=ast.parse((ROOT/"src/offline_study/experiments/action_condition_specificity.py").read_text())
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="verify_native_sampler")
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(method)))

    @unittest.skipUnless(
        (ROOT/"vendor/jepa-wms/evals/simu_env_planning/planning/planning/planner.py").exists(),
        "Pinned upstream planner required; exercised in the local integration environment",
    )
    def test_sampler_matches_actual_native_plan_prefix_ast(self):
        # Execute the upstream plan body only through its first cost evaluation;
        # no optimizer, model, CUDA, or rewritten sampling reference is involved.
        source = ROOT/"vendor/jepa-wms/evals/simu_env_planning/planning/planning/planner.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="CEMPlanner")
        plan = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="plan")
        index = next(i for i,n in enumerate(plan.body) if isinstance(n,ast.For))
        loop = plan.body[index]
        cost_index = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="cost" for x in n.targets))
        plan.body = plan.body[:index]+loop.body[:cost_index]+[ast.Return(value=ast.Name(id="actions",ctx=ast.Load()))]
        plan.decorator_list=[]
        namespace={"torch":torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[plan],type_ignores=[])),"native_sampler_prefix","exec"),namespace)
        cfg={"planner":dict(horizon=6,num_samples=300,var_scale=.7,max_norms=[.15,.3],max_norm_dims=[[0,1,2],[6]])}
        before=torch.get_rng_state().clone()
        actual,after=m.initial_population(cfg,123,"cpu")
        planner=types.SimpleNamespace(**cfg["planner"],action_dim=20,device=torch.device("cpu"),decode_each_iteration=False,
                                      local_generator=torch.Generator().manual_seed(123))
        expected=namespace["plan"](planner,None,6)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(after,m.tensor_hash(planner.local_generator.get_state()))
        self.assertTrue(torch.equal(torch.get_rng_state(),before));self.assertEqual(int(torch.count_nonzero(actual[:,0])),0)
        self.assertLessEqual(float(actual[:,:,[0,1,2]].abs().max()),.15000001)

    def test_tied_ranks_and_actual_elites(self):
        np.testing.assert_array_equal(m.average_ranks([2,1,2,0]),[2.5,1,2.5,0])
        costs=np.arange(300,dtype=float)
        self.assertAlmostEqual(m.agreement(costs,costs+1,list(range(10)),list(range(10)))["spearman"],1.)
        reversed_cost=-costs
        self.assertAlmostEqual(m.agreement(costs,reversed_cost,list(range(10)),list(range(290,300)))["spearman"],-1.)
        self.assertIsNone(m.agreement(np.ones(300),np.ones(300),list(range(10)),list(range(10)))["spearman"])
        with self.assertRaises(ValueError):m.agreement(costs,costs,list(range(10)),list(range(20,30)))

    def test_scenario_bootstrap_family_and_undefined_not_dropped(self):
        values=np.arange(8)[:,None]*np.ones((1,6))/100
        rows=m.layer_intervals(values)
        self.assertEqual(len(rows),6)
        for row in rows:
            self.assertEqual(row["n"],8)
            self.assertAlmostEqual(row["mean"],.035)
            self.assertLess(row["simultaneous_95_low"],row["mean"])
            self.assertGreater(row["simultaneous_95_high"],row["mean"])
        values[0,2]=np.nan
        self.assertTrue(all(r["mean"] is None for r in m.layer_intervals(values)))
        with self.assertRaises(ValueError):m.layer_intervals(np.zeros((7,6)))
        with self.assertRaises(ValueError):m.summarize_complete([])


if __name__ == "__main__":
    unittest.main()
