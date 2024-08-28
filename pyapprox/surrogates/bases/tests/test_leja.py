import unittest

import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt
import sympy as sp

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.bases.leja import (
    OnePointPDFLejaObjective,
    BetaMarginal,
    UniformMarginal,
    GaussianMarginal,
    TwoPointPDFLejaObjective,
    setup_univariate_leja_sequence,
    OnePointChristoffelLejaObjective,
    TwoPointChristoffelLejaObjective,
)
from pyapprox.interface.model import ModelFromCallable


def _sp_beta_pdf_01(alpha, beta, x):
    """Beta variable PDF on [0, 1] for evaluating with Sympy"""
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / special.beta(alpha, beta)


def _sp_beta_pdf(alpha, beta, lb, ub, x):
    """Beta variable PDF on [lb, ub] for evaluating with Sympy"""
    return _sp_beta_pdf_01(alpha, beta, (x - lb) / (ub - lb)) / (ub - lb)


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
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1

        marginal = GaussianMarginal(stats.norm(0, 1), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        errors = model.check_apply_jacobian(iterate)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1

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

    def _check_one_point_leja_objective(self, objective_class, marginal):
        bkd = self.get_backend()
        leja = setup_univariate_leja_sequence(
            marginal,
            objective_class,
            init_sequence=bkd.array([[0.1, 0.9]]),
            backend=bkd,
        )
        iterate = bkd.array([[marginal.interval(0.75)[1]]])

        sequence = bkd.hstack((leja._obj.sequence(), iterate))
        basis_mat = bkd.sqrt(
            leja._obj._compute_weights(sequence)
        ) * leja._obj._poly(sequence)
        assert np.allclose(
            -np.linalg.det(basis_mat) ** 2,
            leja._obj(iterate) * np.linalg.det(basis_mat[:-1, :-1]) ** 2,
        )

        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = leja._obj.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min() / errors.max() < 1e-6

    def test_one_point_leja_objective(self):
        test_cases = [
            [OnePointPDFLejaObjective, stats.norm(0, 1)],
            [OnePointChristoffelLejaObjective, stats.beta(3, 1, -1, 2)],
        ]
        for test_case in test_cases:
            self._check_one_point_leja_objective(*test_case)

    def _check_two_point_leja_objective(self, objective_class, marginal):
        bkd = self.get_backend()
        leja = setup_univariate_leja_sequence(
            marginal,
            objective_class,
            backend=bkd,
            init_sequence=bkd.array(
                [
                    [
                        marginal.median(),
                        marginal.interval(0.95)[0],
                        marginal.interval(0.95)[1],
                    ]
                ]
            ),
        )
        iterate = bkd.array(
            [[marginal.interval(0.75)[0]], [marginal.interval(0.75)[1]]]
        )
        sequence = bkd.hstack(
            (leja._obj.sequence(), iterate[:1], iterate[1:2])
        )
        basis_mat = bkd.sqrt(
            leja._obj._compute_weights(sequence)
        ) * leja._obj._poly(sequence)
        assert np.allclose(
            -np.linalg.det(basis_mat) ** 2,
            leja._obj(iterate) * np.linalg.det(basis_mat[:-2, :-2]) ** 2,
        )

        # test plot works
        leja._obj.plot(plt.figure().gca())
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = leja._obj.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min() / errors.max() < 1e-6

    def test_two_point_leja_objective(self):
        test_cases = [
            [TwoPointPDFLejaObjective, stats.beta(2, 2, -1, 2)],
            [TwoPointPDFLejaObjective, stats.beta(1, 1, 0, 1)],
            [TwoPointPDFLejaObjective, stats.beta(3, 1, -1, 2)],
            [TwoPointChristoffelLejaObjective, stats.beta(2, 2, -1, 2)],
            [TwoPointChristoffelLejaObjective, stats.beta(1, 1, 0, 1)],
            [TwoPointChristoffelLejaObjective, stats.beta(3, 1, -1, 2)],
        ]
        for test_case in test_cases:
            self._check_two_point_leja_objective(*test_case)

    def _check_leja_sequence(
        self, objective_class, marginal, exact_integral
    ):
        bkd = self.get_backend()
        leja = setup_univariate_leja_sequence(
            marginal, objective_class, backend=bkd
        )

        # test plotting of objective before a two new points are added
        axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
        leja.plot(axs[0])
        leja.step(nsamples=3)
        axs[0].plot(*leja.sequence()[0, -2:], "*")
        axs[0].set_title("Objective")
        axs[1].plot(leja.sequence()[0], leja.sequence()[0] * 0, "o")
        axs[1].set_title("Leja sequence")
        # plt.show()

        # test adding multiple samples
        leja.step(nsamples=5)

        # test quadrature weights
        quad_weights = leja.quadrature_weights(leja.sequence())
        integral = (leja.sequence() ** 4) @ quad_weights
        assert np.allclose(integral, exact_integral)

    def _exact_beta_integral(self, alpha, beta, lb, ub):
        x = sp.Symbol("x")
        return float(
            sp.integrate(
                x**4 * _sp_beta_pdf(alpha, beta, lb, ub, x), (x, lb, ub)
            )
        )

    def test_leja_sequence(self):
        test_cases = [
            [
                OnePointPDFLejaObjective,
                stats.beta(1, 1, -1, 2),
                self._exact_beta_integral(1, 1, -1, 1),
            ],
            [
                OnePointChristoffelLejaObjective,
                stats.beta(1, 1, -1, 2),
                self._exact_beta_integral(1, 1, -1, 1),
            ],
            [
                TwoPointPDFLejaObjective,
                stats.beta(1, 1, -1, 2),
                self._exact_beta_integral(1, 1, -1, 1),
            ],
            [
                TwoPointPDFLejaObjective,
                stats.beta(2, 2, -1, 2),
                self._exact_beta_integral(2, 2, -1, 1),
            ],
            [
                TwoPointChristoffelLejaObjective,
                stats.beta(3, 1, 0, 1),
                self._exact_beta_integral(3, 1, 0, 1),
            ],
        ]
        for test_case in test_cases:
            self._check_leja_sequence(*test_case)


class TestNumpyLeja(TestLeja, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
# class TestTorchLeja(TestLeja, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
