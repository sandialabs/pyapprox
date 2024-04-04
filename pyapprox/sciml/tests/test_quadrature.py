import unittest

import numpy as np

from pyapprox.sciml.quadrature import (
    Fixed1DGaussLegendreIOQuadRule, TensorProduct2DQuadRule)


class TestQuadrature(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_gauss_legendre_1d(self):
        quad_rule = Fixed1DGaussLegendreIOQuadRule(3)
        xx, ww = quad_rule.get_samples_weights()

        def fun(xx):
            return (xx.T)**2
        assert np.allclose(fun(xx).T@ww, 1/3)

    def test_tensor_product_quadrature_rule(self):
        quad_rule1 = Fixed1DGaussLegendreIOQuadRule(3)
        quad_rule2 = Fixed1DGaussLegendreIOQuadRule(4)
        quad_rule = TensorProduct2DQuadRule(quad_rule1, quad_rule2)
        xx, ww = quad_rule.get_samples_weights()
        assert xx.shape[1] == 3*4

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]
        assert np.allclose(fun(xx).T@ww, 2/3)


if __name__ == "__main__":
    quadrature_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestQuadrature)
    unittest.TextTestRunner(verbosity=2).run(quadrature_test_suite)
