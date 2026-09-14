import unittest
import numpy as np

from offline_study.tasks.droid.droid_fit_audit import fingerprints, compare_groups


class InputIdentityTests(unittest.TestCase):
    def test_float_precision_cannot_hide_exact_overlap(self):
        a=np.arange(70,dtype=np.float32).reshape(10,7)
        self.assertEqual(fingerprints(a),fingerprints(a.astype(np.float64)))
        with self.assertRaises(ValueError):compare_groups([{'priority_rank':1,**fingerprints(a)}],[fingerprints(a)])
        b=a+100
        result=compare_groups([{'priority_rank':i,**fingerprints(a)} for i in (1,2)],[fingerprints(b)])
        self.assertEqual(result['duplicate_fit_groups']['state_float32_sha256'],[[1,2]])


if __name__=='__main__':unittest.main()
