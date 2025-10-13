import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.variables.marginals import ContinuousScipyMarginal
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.affine.activelearning import (
    FeketeSampler,
    LejaSampler,
    AdaptivePolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.multiindex import (
    ExpandingMarginGenerator,
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
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=variable._bkd
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

    def test_leja_sampler(self):
        bkd = self.get_backend()
        nvars = 2
        degree = 3
        nqoi = 1
        variable = IndependentMarginalsVariable(
            [ContinuousScipyMarginal(stats.norm(0, 1), backend=bkd)] * nvars,
            backend=bkd,
        )
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=variable._bkd
            )
            for marginal in variable.marginals()
        ]
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_hyperbolic_indices(degree, 1.0)

        np.random.seed(1)
        gen = ExpandingMarginGenerator(basis.nvars(), degree, 1.0, 2)
        basis.set_indices(gen.get_indices())
        bexp = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        sampler = LejaSampler(variable)
        sampler.set_surrogate(bexp)
        sampler.set_candidate_samples(sampler.default_candidate_samples(2000))
        nsamples = 10
        samples = sampler(nsamples)

        values = bkd.sum(samples**degree, axis=0)[:, None]
        coefs = sampler.interpolatory_coefficients(values)
        bexp.set_coefficients(coefs)

        def fun(samples):
            return bkd.sum(samples**degree, axis=0)[:, None]

        test_samples = variable.rvs(100)
        test_values = fun(test_samples)
        assert bkd.allclose(bexp(test_samples), test_values)

    def test_adaptive_leja_sampler(self):
        bkd = self.get_backend()
        nvars = 2
        degree = 3
        nqoi = 1

        def fun(samples):
            return bkd.sum(samples**degree, axis=0)[:, None]

        variable = IndependentMarginalsVariable(
            [ContinuousScipyMarginal(stats.norm(0, 1), backend=bkd)] * nvars,
            backend=bkd,
        )
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=variable._bkd
            )
            for marginal in variable.marginals()
        ]

        # start with level 2 hyperbolic cross then add entire margin (4 terms)
        # this allows comparison with level 3 hyperbolic cross non-adaptive
        # factorization
        init_degree = 2
        basis1 = OrthonormalPolynomialBasis(bases_1d)
        gen1 = ExpandingMarginGenerator(basis1.nvars(), init_degree, 1.0, 4)
        basis1.set_indices(gen1.get_indices())

        np.random.seed(1)
        bexp1 = AdaptivePolynomialChaosExpansion(
            basis1,
            variable,
            gen1,
            max_nsamples=10,
            # max_nsamples=6,
            nqoi=nqoi,
        )
        bexp1.build(fun)

        init_degree = 3
        basis2 = OrthonormalPolynomialBasis(bases_1d)
        gen2 = ExpandingMarginGenerator(basis2.nvars(), init_degree, 1.0, 4)
        basis2.set_indices(gen1.get_indices())
        np.random.seed(
            1
        )  # use the same seed to make sure candidate samples are the same
        bexp2 = AdaptivePolynomialChaosExpansion(
            basis2,
            variable,
            gen2,
            max_nsamples=10,
            nqoi=nqoi,
        )
        bexp2.sampler().set_initial_pivots(
            bexp1.sampler()._LUFactorizer.pivots()
        )
        bexp2.build(fun)

        assert bkd.allclose(basis1.get_indices(), basis2.get_indices())
        assert bkd.allclose(
            bexp1._sampler._candidate_samples,
            bexp2._sampler._candidate_samples,
        )
        assert bkd.allclose(
            bexp1._sampler._candidate_weights,
            bexp2._sampler._candidate_weights,
        )
        assert bkd.allclose(bexp1._sampler._weights, bexp2._sampler._weights)
        assert bkd.allclose(bexp1._ctrain_samples, bexp2._ctrain_samples)
        assert bkd.allclose(
            bexp1._sampler._LUFactorizer._LU_factor,
            bexp2._sampler._LUFactorizer._LU_factor,
        )
        assert bkd.allclose(bexp1._sampler._L, bexp2._sampler._L)
        assert bkd.allclose(bexp1._sampler._U, bexp2._sampler._U)
        assert bkd.allclose(bexp1._ctrain_values, bexp2._ctrain_values)
        assert bkd.allclose(bexp1.get_coefficients(), bexp2.get_coefficients())

        test_samples = variable.rvs(100)
        test_values = fun(test_samples)
        assert bkd.allclose(bexp1(test_samples), test_values)


class TestNumpyActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
