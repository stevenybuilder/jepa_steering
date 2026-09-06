import importlib.util
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/"scripts/geometry_map"))
SPEC=importlib.util.spec_from_file_location("capture_pusht_spatial",ROOT/"scripts/geometry_map/capture_pusht_spatial.py")
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
class SpatialTests(unittest.TestCase):
    def test_regions_and_global(self):
        grid=torch.zeros(1,16,16,1);grid[:,:8,:8]=1;grid[:,:8,8:]=2;grid[:,8:,:8]=3;grid[:,8:,8:]=4
        mean,spatial=M.summaries(grid.reshape(1,256,1))
        self.assertTrue(torch.equal(spatial,torch.tensor([[2.5,1,2,3,4]])))
        self.assertEqual(mean.item(),2.5)
    def test_shape_guard(self):
        with self.assertRaises(ValueError):M.summaries(torch.zeros(1,16,16,1))
    def test_confirmation_sealed(self):
        self.assertFalse(M.permitted({"source_id":125,"split":"confirmation"}))
        self.assertFalse(M.permitted({"source_id":99,"split":"validation"}))
        self.assertTrue(M.permitted({"source_id":124,"split":"validation"}))
if __name__=="__main__":unittest.main()
