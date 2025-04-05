import unittest

import numpy as np
from scipy import stats

from pyapprox.expdesign.cubature import get_cubature_rule
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.basisexp import MonomialExpansion
from pyapprox.surrogates.bases.univariate.base import Monomial1D
from pyapprox.surrogates.bases.basis import MultiIndexBasis
from pyapprox.variables.marginals import ContinuousScipyMarginal


class TestCubature:
    def setUp(self):
        np.random.seed(1)

    def _check_cubature_rule(self, nvars, degree):
        bkd = self.get_backend()
        marginals = [
            ContinuousScipyMarginal(stats.uniform(0, 1), backend=bkd)
        ] * (nvars - 1) + [
            ContinuousScipyMarginal(stats.uniform(0, 2), backend=bkd)
        ]
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_hyperbolic_indices(degree, 1.0)
        bexp = MonomialExpansion(basis, solver=None, nqoi=1)
        coefs = np.random.normal(0, 1, (basis.nterms(), 1))
        bexp.set_coefficients(coefs)
        exact_integral = bexp.mean_iid_uniform_marginals(marginals)
        x_quad, w_quad = get_cubature_rule(nvars, degree)
        vals = bexp(x_quad)
        integral = vals.T @ w_quad

        print((exact_integral, integral))
        assert np.allclose(exact_integral, integral)

    def test_cubature_rules(self):
        for degree in [2, 3, 5]:
            for nvars in np.arange(2, 10):
                self._check_cubature_rule(nvars, degree)


class TestNumpyCubature(TestCubature, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchCubature(TestCubature, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
