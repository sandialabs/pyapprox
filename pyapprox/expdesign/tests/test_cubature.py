import unittest

import numpy as np
from scipy import stats

from pyapprox.expdesign.cubature import get_cubature_rule
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.basisexp import MonomialExpansion
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.variables.marginals import ContinuousScipyMarginal


class TestCubature:
    def setUp(self):
        np.random.seed(1)

    def _check_cubature_rule(self, nvars, degree):
        bkd = self.get_backend()
        marginals = [
            ContinuousScipyMarginal(stats.uniform(-1, 2), backend=bkd)
        ] * nvars
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_hyperbolic_indices(degree, 1.0)
        bexp = MonomialExpansion(basis, solver=None, nqoi=1)
        coefs = bkd.asarray(np.random.normal(0, 1, (basis.nterms(), 1)))
        bexp.set_coefficients(coefs)
        exact_integral = bexp.mean_iid_uniform_marginals(marginals)
        x_quad, w_quad = get_cubature_rule(nvars, degree, bkd=bkd)
        vals = bexp(x_quad)
        integral = vals.T @ w_quad
        assert np.allclose(exact_integral, integral)

    def test_cubature_rules(self):
        for degree in [2, 3, 5]:
            for nvars in np.arange(2, 10):
                self._check_cubature_rule(nvars, degree)

    def test_monomial_variance(self):
        bkd = self.get_backend()
        nvars, degree = 2, 2
        marginals = [
            ContinuousScipyMarginal(stats.uniform(-1, 2), backend=bkd)
        ] * nvars
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_hyperbolic_indices(degree, 1.0)
        bexp = MonomialExpansion(basis, solver=None, nqoi=1)
        coefs = bkd.asarray(np.random.normal(0, 1, (basis.nterms(), 1)))
        bexp.set_coefficients(coefs)
        exact_variance = bexp.variance_iid_uniform_marginals(marginals)
        x_quad, w_quad = get_cubature_rule(nvars, 5, bkd=bkd)
        vals = bexp(x_quad)
        mean = vals.T @ w_quad
        variance = (vals**2).T @ w_quad - mean**2
        assert np.allclose(exact_variance, variance)


class TestNumpyCubature(TestCubature, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchCubature(TestCubature, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
