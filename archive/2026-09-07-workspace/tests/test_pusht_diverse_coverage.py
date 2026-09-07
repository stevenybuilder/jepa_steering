import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
import pusht_diverse_coverage as m

class CoverageTests(unittest.TestCase):
    def test_sealed_sources_not_selected(self):
        self.assertEqual(len(m.SOURCES),20)
        self.assertFalse(set(m.SOURCES)&set(range(150)))
        self.assertEqual(m.SOURCES,(202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020,2121))
    def test_frozen_group_split(self):
        rows=m.rows();self.assertEqual(sum(r['split']=='fit' for r in rows),16)
        self.assertEqual(sum(r['split']=='development_validation' for r in rows),4)
    def test_identity_is_actual_state(self):
        self.assertEqual(m.fingerprint([1,2,3]),m.fingerprint([1.,2.,3.]))
        self.assertNotEqual(m.fingerprint([1,2,3]),m.fingerprint([1,2,4]))

if __name__=='__main__':unittest.main()
