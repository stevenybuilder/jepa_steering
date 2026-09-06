import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/"scripts/geometry_map"))
SPEC=importlib.util.spec_from_file_location("analyze_pusht_spatial",ROOT/"scripts/geometry_map/analyze_pusht_spatial.py")
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
class AnalysisTests(unittest.TestCase):
    def test_initial_identity_not_source_id(self):
        self.assertEqual(M.identity_key([1,2,3]),M.identity_key(np.array([1.,2.,3.])))
        self.assertNotEqual(M.identity_key([1,2,3]),M.identity_key([1,2,4]))
    def test_same_compression_rank(self):
        x=np.random.default_rng(1).normal(size=(100,80));scale,pca,z=M.compress_fit(x)
        self.assertEqual(z.shape,(100,64));self.assertEqual(M.compress_predict((scale,pca),x[:7]).shape,(7,64))
    def test_physical_persistence_units(self):
        y=np.array([[3.,4.,0.,0.,0.,1.]]);current=np.array([[0.,0.,0.,0.,0.,1.]])
        result=M.summarize(y,y,current,np.array([1]))
        self.assertEqual(result["world_endpoint_rmse_pixels"],0.)
        self.assertEqual(result["physical_persistence_rmse_pixels"],2.5)
if __name__=="__main__":unittest.main()
