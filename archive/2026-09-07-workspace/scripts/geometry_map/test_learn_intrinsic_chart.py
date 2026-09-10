import unittest
import numpy as np
from learn_intrinsic_chart import DiffusionChart,distances,rbf_reconstruct,evaluate


class ChartTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(19)
        self.x=self.rng.normal(size=(40,6))

    def test_nystrom_reproduces_training_coordinates(self):
        for alpha in (0.,1.):
            chart=DiffusionChart(4,alpha).fit(self.x)
            np.testing.assert_allclose(chart.transform(self.x),chart.training,atol=1e-10)

    def test_rotation_invariant_coordinate_distances(self):
        q,_=np.linalg.qr(self.rng.normal(size=(6,6)))
        a=DiffusionChart(4,1).fit(self.x);b=DiffusionChart(4,1).fit(self.x@q)
        np.testing.assert_allclose(distances(a.training,a.training),distances(b.training,b.training),atol=1e-9)

    def test_degenerate_inputs_rejected(self):
        with self.assertRaises(ValueError):DiffusionChart(2,1).fit(np.zeros((20,5)))

    def test_same_decoder_can_reconstruct_smooth_mapping(self):
        coordinates=np.linspace(-1,1,30)[:,None];values=np.c_[coordinates,coordinates**2]
        pred=rbf_reconstruct(coordinates,coordinates,values)
        self.assertLess(np.mean((pred-values)**2),1e-4)

    def test_wrong_source_shape_rejected(self):
        with self.assertRaises(ValueError):evaluate(np.zeros((9,64)))


if __name__=='__main__':unittest.main()
