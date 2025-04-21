import unittest
import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.misc import lists_of_arrays_equal
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    define_iid_random_variable,
    FiniteSamplesVariable,
    RejectionSamplingVariable,
)
from pyapprox.variables.marginals import BetaMarginal


class TestJoint:
    def setUp(self):
        np.random.seed(1)

    def test_define_mixed_tensor_product_random_variable_I(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable type
        the distribution parameters ARE the same
        """
        bkd = self.get_backend()
        marginals = [
            stats.uniform(-1, 2),
            stats.beta(1, 1, -1, 2),
            stats.norm(0, 1),
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.beta(1, 1, -1, 2),
        ]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        assert len(variable._unique_marginals) == 3
        assert lists_of_arrays_equal(
            variable._unique_indices, [[0, 3, 4], [1, 5], [2]]
        )

    def test_define_mixed_tensor_product_random_variable_II(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable
        type the distribution parameters ARE NOT the same
        """
        bkd = self.get_backend()
        marginals = [
            stats.uniform(-1, 2),
            stats.beta(1, 1, -1, 2),
            stats.norm(-1, 2),
            stats.uniform(),
            stats.uniform(-1, 2),
            stats.beta(2, 1, -2, 3),
        ]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)

        assert len(variable._unique_marginals) == 5
        assert lists_of_arrays_equal(
            variable._unique_indices, [[0, 4], [1], [2], [3], [5]]
        )

    def test_statistics(self):
        bkd = self.get_backend()
        marginals = [
            stats.uniform(2, 4),
            stats.beta(1, 1, -1, 2),
            stats.norm(0, 1),
        ]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        assert np.allclose(variable.mean()[:, 0], bkd.asarray([4, 0, 0]))
        assert np.allclose(
            variable.var()[:, 0], bkd.asarray([m.var() for m in marginals])
        )
        assert np.allclose(
            variable.std()[:, 0], bkd.asarray([m.std() for m in marginals])
        )
        assert np.allclose(
            variable.median()[:, 0], [m.median() for m in marginals]
        )

        intervals = variable.interval(1)
        assert np.allclose(
            intervals, np.array([[2, 6], [-1, 1], [-np.inf, np.inf]])
        )

    def test_define_iid_random_variable(self):
        """
        Construct a independent and identiically distributed (iid)
        multivariate random variable from the tensor-product of
        the same one-dimensional variable.
        """
        bkd = self.get_backend()
        var = stats.norm(loc=2, scale=3)
        nvars = 2
        iid_variable = define_iid_random_variable(var, nvars, backend=bkd)

        assert len(iid_variable._unique_marginals) == 1
        assert np.allclose(iid_variable._unique_indices, np.arange(nvars))

    def test_archived_data_model(self):
        bkd = self.get_backend()
        nvars, nsamples = 2, 100
        samples = np.random.normal(0, 1, (nvars, nsamples))

        model = FiniteSamplesVariable(samples, randomness="none", backend=bkd)
        valid_samples, II = model._rvs(nsamples)
        assert np.allclose(II, np.arange(nsamples))
        assert np.allclose(valid_samples, samples[:, II])

        # check error thrown if too many samples are requested
        # with randomnes=None
        self.assertRaises(ValueError, model.rvs, nsamples)

        # check repeated calling of deterministic rvs
        model = FiniteSamplesVariable(samples, randomness="none", backend=bkd)
        valid_samples = model.rvs(nsamples // 2)
        valid_samples, II = model._rvs(2)
        assert np.allclose(II, np.arange(nsamples // 2, nsamples // 2 + 2))

        # check repeated calling of rvs replacement passes when
        # more than existing samples are requested
        model = FiniteSamplesVariable(
            samples, randomness="replacement", backend=bkd
        )
        assert nsamples % 2 == 0
        valid_samples1, II = model._rvs(nsamples // 2)
        valid_samples0, JJ = model._rvs(nsamples)

    def independent_beta_marginals_variable(self):
        bkd = self.get_backend()
        a1, b1 = 2, 3
        a2, b2 = 3, 3
        bounds = [0, 1]
        marginals1 = [
            BetaMarginal(a1, b1, *bounds),
            BetaMarginal(a2, b2, *bounds),
        ]
        variable1 = IndependentMarginalsVariable(marginals1, backend=bkd)
        samples = variable1.rvs(10)
        assert bkd.allclose(
            variable1.pdf(samples)[:, 0],
            marginals1[0].pdf(samples[0]) * marginals1[1].pdf(samples[1]),
        )
        assert bkd.allclose(
            variable1.logpdf(samples)[:, 0],
            marginals1[0].logpdf(samples[0])
            * marginals1[1].logpdf(samples[1]),
        )
        assert bkd.allclose(
            bkd.mean(variable1.rvs(int(1e6)), axis=1),
            variable1.mean()[:, 0],
            rtol=1e-2,
        )

        marginals2 = [
            BetaMarginal(a2, b2, *bounds),
            BetaMarginal(a1, b1, *bounds),
        ]
        variable2 = IndependentMarginalsVariable(marginals2, backend=bkd)

        quadx, quadw = bkd.asarray(np.polynomial.legendre.leggauss(100))
        quadx_01 = bkd.asarray((quadx + 1) / 2)
        quadw_01 = bkd.asarray(quadw / 2)
        quadx = bkd.cartesian_product([quadx_01] * 2)
        quadw = bkd.outer_product([quadw_01] * 2)
        kl_div = (
            variable1.pdf(quadx)
            * (bkd.log(variable1.pdf(quadx) / variable2.pdf(quadx)))
        )[:, 0] @ quadw
        assert bkd.allclose(
            variable1.kl_divergence(variable2), kl_div, rtol=1e-5
        )

    def test_rejection_sampling(self):
        bkd = self.get_backend()
        proposal = IndependentMarginalsVariable(
            [stats.uniform(0, 1)], backend=bkd
        )
        target = IndependentMarginalsVariable(
            [stats.beta(2, 2, 0, 1)], backend=bkd
        )
        variable = RejectionSamplingVariable(target, proposal, 2.0)
        nsamples = 1e6
        assert bkd.allclose(
            bkd.mean(variable.rvs(nsamples)), target.mean(), rtol=1e-2
        )


class TestNumpyJoint(TestJoint, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchJoint(TestJoint, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
