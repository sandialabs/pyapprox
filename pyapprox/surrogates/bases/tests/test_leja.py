import unittest

import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt
import sympy as sp

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.bases.leja import (
    OnePointPDFLejaObjective, BetaMarginal, UniformMarginal, GaussianMarginal,
    LejaSequence, TwoPointPDFLejaObjective
)
from pyapprox.surrogates.bases.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal
)
from pyapprox.interface.model import ModelFromCallable
from pyapprox.optimization.pya_minimize import ScipyConstrainedOptimizer


def _sp_beta_pdf_01(alpha, beta, x):
    """Beta variable PDF on [0, 1] for evaluating with Sympy"""
    return x**(alpha-1)*(1-x)**(beta-1)/special.beta(alpha, beta)


def _sp_beta_pdf(alpha, beta, lb, ub, x):
    """Beta variable PDF on [lb, ub] for evaluating with Sympy"""
    return _sp_beta_pdf_01(alpha, beta, (x-lb)/(ub-lb))/(ub-lb)


class TestLeja:
    def setUp(self):
        np.random.seed(1)

    def test_marginals(self):
        bkd = self.get_backend()
        # use model infasttucture to test gradients
        marginal = BetaMarginal(stats.beta(2, 2), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = model.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1

        marginal = GaussianMarginal(stats.norm(0, 1), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        errors = model.check_apply_jacobian(iterate)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1

        marginal = UniformMarginal(stats.uniform(-1, 2), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = model.check_apply_jacobian(
            iterate, relative=False, fd_eps=fd_eps
        )
        # gradient is zero so check its small for large finite difference step
        assert errors[0] < 1e-14

    def test_one_point_pdf_leja_objective(self):
        bkd = self.get_backend()
        scipy_marginal = stats.beta(2, 2, -1, 2)
        poly = setup_univariate_orthogonal_polynomial_from_marginal(
            scipy_marginal, backend=bkd)
        marginal = BetaMarginal(scipy_marginal, backend=bkd)
        obj = OnePointPDFLejaObjective(marginal, poly)
        obj.set_sequence(bkd.array([[-.95, 0]]))
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])

        sequence = bkd.hstack((obj.sequence(), iterate))
        basis_mat = bkd.sqrt(
            obj._compute_weights(sequence)
        )*obj._poly(sequence)
        assert np.allclose(
            -np.linalg.det(basis_mat)**2,
            obj(iterate)*np.linalg.det(basis_mat[:-1, :-1])**2
        )

        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = obj.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.08

    def _check_two_point_pdf_leja_objective(self, scipy_marginal):
        bkd = self.get_backend()
        poly = setup_univariate_orthogonal_polynomial_from_marginal(
            scipy_marginal, backend=bkd)
        marginal = BetaMarginal(scipy_marginal, backend=bkd)
        obj = TwoPointPDFLejaObjective(marginal, poly)
        obj.set_sequence(bkd.array([[scipy_marginal.mean()]]))
        iterate = bkd.array(
            [
                [marginal._marginal.interval(0.75)[0]],
                [marginal._marginal.interval(0.75)[1]]
            ]
        )
        sequence = bkd.hstack((obj.sequence(), iterate[:1], iterate[1:2]))
        basis_mat = bkd.sqrt(
            obj._compute_weights(sequence)
        )*obj._poly(sequence)
        assert np.allclose(
            -np.linalg.det(basis_mat)**2,
            obj(iterate)*np.linalg.det(basis_mat[:-2, :-2])**2
        )

        # test plot works
        obj.plot(plt.figure().gca())
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = obj.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min()/errors.max() < 1e-6

    def test_two_point_pdf_leja_objective(self):
        test_cases = [
            [stats.beta(2, 2, -1, 2)],
            [stats.beta(1, 1, 0, 1)],
            [stats.beta(3, 1, -1, 2)]
        ]
        for test_case in test_cases:
            self._check_two_point_pdf_leja_objective(*test_case)

    def test_one_point_pdf_leja_sequence(self):
        bkd = self.get_backend()
        scipy_marginal = stats.beta(1, 1, -1, 2)
        poly = setup_univariate_orthogonal_polynomial_from_marginal(
            scipy_marginal, backend=bkd)
        marginal = BetaMarginal(scipy_marginal, backend=bkd)
        obj = OnePointPDFLejaObjective(marginal, poly)
        obj.set_sequence(bkd.array([[scipy_marginal.mean()]]))

        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8, maxiter=1000, method="trust-constr"
        )
        leja = LejaSequence(obj, optimizer)

        # test plotting of objective before a single new point is added
        ax = plt.figure().gca()
        leja.plot(ax)
        leja.step(nsamples=2)
        ax.axvline(x=leja.sequence()[0, -1])

        # test adding multiple samples
        leja.step(nsamples=10)

        # test quadrature weights
        quad_weights = leja.quadrature_weights(leja.sequence())
        integral = (leja.sequence()**4) @ quad_weights
        exact_integral = 1./5.
        assert np.allclose(integral, exact_integral)

    def _check_two_point_pdf_leja_sequence(
            self, scipy_marginal, exact_integral
    ):
        bkd = self.get_backend()
        poly = setup_univariate_orthogonal_polynomial_from_marginal(
            scipy_marginal, backend=bkd)
        marginal = BetaMarginal(scipy_marginal, backend=bkd)
        obj = TwoPointPDFLejaObjective(marginal, poly)
        obj.set_sequence(bkd.array([[scipy_marginal.mean()]]))

        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8, maxiter=1000, method="trust-constr"
        )
        leja = LejaSequence(obj, optimizer)

        # test plotting of objective before a two new points are added
        axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
        leja.plot(axs[0])
        leja.step(nsamples=3)
        axs[0].plot(*leja.sequence()[0, -2:], '*')
        axs[0].set_title("Objective")
        axs[1].plot(leja.sequence()[0], leja.sequence()[0]*0, 'o')
        axs[1].set_title("Leja sequence")
        # plt.show()

        # test adding multiple samples
        leja.step(nsamples=5)

        # test quadrature weights
        quad_weights = leja.quadrature_weights(leja.sequence())
        integral = (leja.sequence()**4) @ quad_weights
        assert np.allclose(integral, exact_integral)

    def _exact_beta_integral(self, alpha, beta, lb, ub):
        x = sp.Symbol("x")
        return float(
            sp.integrate(
                x**4*_sp_beta_pdf(alpha, beta, lb, ub, x), (x, lb, ub)
            )
        )

    def test_two_point_pdf_leja_sequence(self):
        test_cases = [
            [stats.beta(1, 1, -1, 2), self._exact_beta_integral(1, 1, -1, 1)],
            [stats.beta(2, 2, -1, 2), self._exact_beta_integral(2, 2, -1, 1)],
            [stats.beta(3, 1, 0, 1), self._exact_beta_integral(3, 1, 0, 1)],
        ]
        for test_case in test_cases:
            self._check_two_point_pdf_leja_sequence(*test_case)


class TestNumpyLeja(TestLeja, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
# class TestTorchLeja(TestLeja, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
