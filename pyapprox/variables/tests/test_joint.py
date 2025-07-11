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
    DirichletVariable,
    IndependentGroupsVariable,
)
from pyapprox.variables.marginals import BetaMarginal, GaussianMarginal
from pyapprox.interface.model import ModelFromVectorizedCallable


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
        variable = IndependentMarginalsVariable(
            marginals, backend=bkd, compress=True
        )
        assert len(variable._unique_marginals) == 3
        list2 = [[0, 3, 4], [1, 5], [2]]
        list2 = [bkd.array(item) for item in list2]
        assert lists_of_arrays_equal(variable._unique_indices, list2)

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
        variable = IndependentMarginalsVariable(
            marginals, backend=bkd, compress=True
        )

        assert len(variable._unique_marginals) == 5
        list2 = [[0, 4], [1], [2], [3], [5]]
        list2 = [bkd.array(item) for item in list2]
        assert lists_of_arrays_equal(variable._unique_indices, list2)

    def test_independent_groups_variable(self):
        bkd = self.get_backend()
        group1_marginals = [
            BetaMarginal(2, 2, 0, 1, backend=bkd) for ii in range(2)
        ]
        group1_variable = IndependentMarginalsVariable(
            group1_marginals, backend=bkd
        )
        group2_marginals = [
            GaussianMarginal(0, 1, backend=bkd) for ii in range(2)
        ]
        group2_variable = IndependentMarginalsVariable(
            group2_marginals, backend=bkd
        )
        variable = IndependentGroupsVariable(
            [group1_variable, group2_variable]
        )

        # test rvs
        nsamples = int(1e6)
        samples = variable.rvs(nsamples)
        bkd.allclose(
            samples.mean(),
            bkd.vstack([group1_variable.mean(), group2_variable.mean()]),
            rtol=1e-2,
        )
        # test pdf
        nsamples = 10
        samples = variable.rvs(nsamples)
        group1_samples = samples[: group1_variable.nvars()]
        group2_samples = samples[group1_variable.nvars() :]
        assert bkd.allclose(
            variable.pdf(samples)[:, 0],
            bkd.prod(
                bkd.hstack(
                    [
                        group1_variable.pdf(group1_samples),
                        group2_variable.pdf(group2_samples),
                    ]
                ),
                axis=1,
            ),
        )
        # test logpdf
        assert bkd.allclose(
            variable.logpdf(samples)[:, 0],
            bkd.prod(
                bkd.hstack(
                    [
                        group1_variable.logpdf(group1_samples),
                        group2_variable.logpdf(group2_samples),
                    ]
                ),
                axis=1,
            ),
        )

        model = ModelFromVectorizedCallable(
            1,
            variable.nvars(),
            variable.pdf,
            jacobian=variable.pdf_jacobian,
            backend=bkd,
        )
        errors = model.check_apply_jacobian(variable.rvs(1))
        assert errors.min() / errors.max() < 1e-6

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
            BetaMarginal(a1, b1, *bounds, backend=bkd),
            BetaMarginal(a2, b2, *bounds, backend=bkd),
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
            + marginals1[1].logpdf(samples[1]),
        )
        assert bkd.allclose(
            bkd.mean(variable1.rvs(int(1e6)), axis=1),
            variable1.mean()[:, 0],
            rtol=1e-2,
        )

        marginals2 = [
            BetaMarginal(a2, b2, *bounds, backend=bkd),
            BetaMarginal(a1, b1, *bounds, backend=bkd),
        ]
        variable2 = IndependentMarginalsVariable(marginals2, backend=bkd)

        quadx, quadw = np.polynomial.legendre.leggauss(100)
        quadx = bkd.asarray(quadx)
        quadw = bkd.asarray(quadw)
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

        if not bkd.jacobian_implemented():
            return

        usamples = bkd.asarray(np.random.uniform(0, 1, (variable2.nvars(), 3)))
        samples = variable2.ppf(usamples)

        def wrap(param):
            a = param[::2]
            b = param[1::2]
            for ii, marginal in enumerate(variable2.marginals()):
                marginal.set_shapes(a[ii], b[ii])
            return variable2.ppf(usamples)

        eps = 1e-7
        param = bkd.array([3, 2, 2, 2])
        jac_fd = []
        for ii in range(param.shape[0]):
            param_eps = bkd.copy(param)
            param_eps[ii] += eps
            jac_fd.append(((wrap(param_eps) - wrap(param)) / eps))

        assert bkd.allclose(
            bkd.stack(jac_fd, axis=-1), bkd.jacobian(wrap, param)
        )

        def pdf_wrap(param):
            a = param[::2]
            b = param[1::2]
            for ii, marginal in enumerate(variable2.marginals()):
                marginal.set_shapes(a[ii], b[ii])
            return variable2.pdf(samples)[:, 0]

        assert bkd.allclose(
            variable2.pdf_shape_jacobian(samples),
            bkd.jacobian(pdf_wrap, param),
        )

        def ppf_wrap(param):
            a = param[::2]
            b = param[1::2]
            for ii, marginal in enumerate(variable2.marginals()):
                marginal.set_shapes(a[ii], b[ii])
            return variable2.ppf(usamples)

        a = param[::2]
        b = param[1::2]
        for ii, marginal in enumerate(variable2.marginals()):
            marginal.set_shapes(a[ii], b[ii])
        usamples = bkd.asarray(np.random.uniform(0, 1, (variable2.nvars(), 3)))
        samples = ppf_wrap(param)
        auto_jac = bkd.jacobian(ppf_wrap, param)
        jac = variable2.ppf_shape_jacobian(usamples, True)
        assert bkd.allclose(jac[0], auto_jac[0, :, :2])
        assert bkd.allclose(jac[1], auto_jac[1, :, 2:])

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

    def test_dirichlet_variable(self):
        bkd = self.get_backend()
        shapes = bkd.array([2, 3, 2])
        variable = DirichletVariable(shapes, backend=bkd)
        scipy_rv = stats.dirichlet(shapes)
        samples = variable.rvs(10000000)
        assert bkd.allclose(
            samples.mean(axis=1), bkd.asarray(scipy_rv.mean()), rtol=1e-2
        )
        assert bkd.allclose(
            variable.pdf(samples), bkd.asarray(scipy_rv.pdf(samples))[:, None]
        )
        assert bkd.allclose(
            variable.mean()[:, 0], bkd.asarray(scipy_rv.mean())
        )
        assert bkd.allclose(variable.var()[:, 0], bkd.asarray(scipy_rv.var()))
        other_shapes = bkd.array([3, 4, 3])
        other = DirichletVariable(other_shapes, backend=bkd)
        kl_divergence = (
            bkd.log(variable.pdf(samples)) - bkd.log(other.pdf(samples))
        ).mean()
        assert bkd.allclose(
            kl_divergence, variable.kl_divergence(other), rtol=1e-2
        )


class TestNumpyJoint(TestJoint, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchJoint(TestJoint, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
