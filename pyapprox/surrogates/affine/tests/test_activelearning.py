import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.variables.marginals import ContinuousScipyMarginal
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.affine.activelearning import FeketeSampler
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.univariate.orthopoly import (
    LegendrePolynomial1D,
    AffineMarginalTransform,
)


class TestActiveLearning:
    def setUp(self):
        np.random.seed(1)

    def test_fekete_sampler(self):
        bkd = self.get_backend()
        nvars = 2
        degree = 3
        nqoi = 1
        variable = IndependentMarginalsVariable(
            [ContinuousScipyMarginal(stats.norm(0, 1), backend=bkd)] * nvars,
            backend=bkd,
        )
        polys_1d = [
            LegendrePolynomial1D(
                trans=AffineMarginalTransform(marginal, backend=bkd),
                backend=bkd,
            )
            for marginal in variable.marginals()
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_hyperbolic_indices(degree, 1.0)
        bexp = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        sampler = FeketeSampler(variable)
        sampler.set_surrogate(bexp)
        nsamples = 10
        samples = sampler(nsamples)

        values = bkd.sum(samples**degree, axis=0)[:, None]
        coefs = sampler.interpolatory_coefficients(values)
        bexp.set_coefficients(coefs)

        test_samples = variable.rvs(100)
        test_values = bkd.sum(test_samples**degree, axis=0)[:, None]
        assert bkd.allclose(bexp(test_samples), test_values)


class TestNumpyActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
