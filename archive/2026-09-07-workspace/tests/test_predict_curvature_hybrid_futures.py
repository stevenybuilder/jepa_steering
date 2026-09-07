from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from predict_curvature_hybrid_futures import fixed_batch_plan,select_hybrids


class HybridForecastTests(unittest.TestCase):
    def test_identical_contiguous_batch8_without_source_mutation(self):
        x=torch.arange(60,dtype=torch.float32).reshape(6,10);saved=x.clone()
        plan=fixed_batch_plan(x)
        self.assertEqual(plan.shape,(6,8,10));self.assertTrue(plan.is_contiguous())
        for b in range(8):self.assertTrue(torch.equal(plan[:,b],x))
        plan[0,0,0]=999
        self.assertTrue(torch.equal(x,saved));self.assertTrue(torch.equal(plan[:,1],saved))
    def test_receipt_exact_development_set_before_load(self):
        names=[f'near-dev-{e:03d}-pair{p}-radius{r}-t{s}025-H{h}.pt' for e in range(4) for p in range(4) for r in (1,4) for s in ('minus','plus') for h in (1,3)]
        receipt=dict(complete=True,hybrid=True,held_access=False,outputs=[dict(path=n) for n in names])
        self.assertEqual(len(select_hybrids(receipt)),128)
        receipt['outputs'][0]['path']='near-dev-004-forbidden.pt'
        with self.assertRaises(ValueError):select_hybrids(receipt)
    def test_invalid_plan_rejected(self):
        with self.assertRaises(ValueError):fixed_batch_plan(torch.zeros(5,10))
        with self.assertRaises(ValueError):fixed_batch_plan(torch.full((6,10),float('nan')))


if __name__=='__main__':unittest.main()
