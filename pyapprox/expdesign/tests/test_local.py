import unittest

import numpy as np

from pyapprox.expdesign.local import (
    DOptimalLstSqCriterion,
    DOptimalQuantileCriterion,
    COptimalLstSqCriterion,
    COptimalQuantileCriterion,
    AOptimalLstSqCriterion,
    AOptimalQuantileCriterion,
    IOptimalLstSqCriterion,
    LocalOptimalExperimentalDesign,
    GOptimalLstSqCriterion,
)
from pyapprox.surrogates.bases.univariate import Monomial1D
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer,
    MiniMaxOptimizer,
)
from pyapprox.benchmarks.algebraic import MichaelisMentenModel
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestLocalOED:
    def setUp(self):
        np.random.seed(1)

    def _check_criterion_gradients(
        self, crit_cls, hetro, additional_checks, *args
    ):
        bkd = self.get_backend()
        poly_degree = 3
        ndesign_pts = 11
        design_samples = bkd.linspace(-1, 1, ndesign_pts)[None, :]
        monomial = Monomial1D(poly_degree + 1, backend=bkd)
        design_factors = monomial(design_samples)
        if hetro:
            noise_mult = 1 + design_samples[0, :] ** 2 + 1
        else:
            noise_mult = None
        crit = crit_cls(
            design_factors, *args, noise_mult=noise_mult, backend=bkd
        )
        iterate = bkd.array(np.random.uniform(0, 1, (ndesign_pts, 1)))
        if additional_checks is not None:
            additional_checks(crit, iterate)
        # print(crit.jacobian(iterate))
        # print(crit.approx_jacobian(iterate))
        errors = crit.check_apply_jacobian(iterate, disp=True)
        assert errors.min() / errors.max() < 1e-6

        if (
            not crit._apply_hessian_implemented
            and not crit._hessian_implemented
        ):
            return
        errors = crit.check_apply_hessian(iterate, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_criterion_gradients(self):
        bkd = self.get_backend()

        def check_aopt_obj(crit, iterate):
            M0 = crit._M0(iterate)
            M1inv = crit._M1inv(iterate)
            assert bkd.allclose(crit(iterate), bkd.trace(M1inv @ M0 @ M1inv))

        def check_iopt_obj(crit, iterate):
            M0 = crit._M0(iterate)
            M1inv = crit._M1inv(iterate)
            # objective from old code
            # u = M1inv @ crit._pred_factors.T
            # value = bkd.sum(u * crit._pred_prob_measure * (M0 @ u))
            value = bkd.trace(M1inv @ M0 @ M1inv @ crit._Bmat)
            assert bkd.allclose(crit(iterate), value)

        poly_degree = 3
        design_samples = bkd.linspace(-1, 1, 9)[None, :]
        monomial = Monomial1D(poly_degree + 1, backend=bkd)
        pred_factors = monomial(design_samples)

        test_cases = [
            [DOptimalLstSqCriterion, False, None],
            [DOptimalLstSqCriterion, True, None],
            [DOptimalQuantileCriterion, False, None],
            [DOptimalQuantileCriterion, True, None],
            [COptimalLstSqCriterion, False, None, bkd.ones(4)],
            [COptimalLstSqCriterion, True, None, bkd.ones(4)],
            [COptimalQuantileCriterion, False, None, bkd.ones(4)],
            [COptimalQuantileCriterion, True, None, bkd.ones(4)],
            [AOptimalLstSqCriterion, False, check_aopt_obj],
            [AOptimalLstSqCriterion, True, check_aopt_obj],
            [AOptimalQuantileCriterion, False, check_aopt_obj],
            [AOptimalQuantileCriterion, True, check_aopt_obj],
            [IOptimalLstSqCriterion, False, check_iopt_obj, pred_factors],
            [IOptimalLstSqCriterion, True, check_iopt_obj, pred_factors],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_criterion_gradients(*test_case)

    def test_homoscedastic_least_squares_doptimal_design(self):
        """
        Create D-optimal designs, for least squares regression with
        homoscedastic noise, and compare to known analytical solutions.
        See Section 5 of Wenjie Z, Computing Optimal Designs for Regression
        Modelsvia Convex Programming, Ph.D. Thesis, 2012
        """
        bkd = self.get_backend()
        poly_degree = 2
        ndesign_pts = 7
        design_samples = bkd.linspace(-1, 1, ndesign_pts)[None, :]
        monomial = Monomial1D(poly_degree + 1, backend=bkd)
        design_factors = monomial(design_samples)
        noise_mult = None
        crit = DOptimalLstSqCriterion(design_factors, noise_mult, backend=bkd)
        oed = LocalOptimalExperimentalDesign(crit)
        mu = oed.construct()
        II = bkd.where(mu > 1e-5)[0]
        assert bkd.allclose(II, bkd.array([0, 3, 6], dtype=int))
        assert bkd.allclose(bkd.ones(3) / 3, mu[II])

        # See J.E. Boon, Generating Exact D-Optimal Designs for Polynomial
        # Models 2007. For how to derive analytical solution for this test case
        poly_degree = 3
        ndesign_pts = 60
        design_samples = bkd.linspace(-1, 1, ndesign_pts)
        # include theoretical optima see Boon paper
        design_samples = bkd.sort(
            bkd.hstack(
                (design_samples, bkd.array((-1 / np.sqrt(5), 1 / np.sqrt(5))))
            )
        )[None, :]
        monomial = Monomial1D(poly_degree + 1, backend=bkd)
        design_factors = monomial(design_samples)
        noise_mult = None
        crit = DOptimalLstSqCriterion(design_factors, noise_mult, backend=bkd)
        oed = LocalOptimalExperimentalDesign(crit)
        opt = ScipyConstrainedOptimizer(opts={"gtol": 1e-12})
        oed.set_optimizer(opt)
        mu = oed.construct()
        II = bkd.where(mu > 1e-5)[0]
        print(design_samples[0, II])
        jj = bkd.where(design_samples == -1 / np.sqrt(5))[1][0]
        kk = bkd.where(design_samples == 1 / np.sqrt(5))[1][0]
        assert bkd.allclose(
            II, bkd.array([0, jj, kk, mu.shape[0] - 1], dtype=int)
        )
        assert bkd.allclose(0.25 * bkd.ones(4), mu[II], atol=1e-5)

    def test_heteroscedastic_least_squares_doptimal_design(self):
        """
        Create D-optimal designs, for least squares regression with
        hetroscedastic noise, and compare to known analytical solutions.
        See Theorem 4.3 in Dette & Trampisch, Optimal Designs for Quantile
        Regression Models https://doi.org/10.1080/01621459.2012.695665
        """
        bkd = self.get_backend()
        ndesign_pts = 17
        lb, ub = 2, 10
        design_samples = bkd.linspace(lb, ub, ndesign_pts)[None, :]
        model = MichaelisMentenModel(design_samples, backend=bkd)
        theta = bkd.array([1, 2])[:, None]
        errors = model.check_apply_jacobian(theta)
        assert errors.min() / errors.max() < 1e-7
        power = 1
        noise_mult = 1 / model(theta)[0] ** power
        design_factors = model.jacobian(theta)
        crit = DOptimalLstSqCriterion(design_factors, noise_mult, backend=bkd)
        oed = LocalOptimalExperimentalDesign(crit)
        opt = ScipyConstrainedOptimizer(opts={"gtol": 1e-12})
        oed.set_optimizer(opt)
        mu = oed.construct()
        exact_nonzero_design_samples = bkd.array(
            [
                max(
                    lb,
                    (power + 1)
                    * ub
                    * theta[1]
                    / ((power + 2) * theta[1] + ub),
                ),
                ub,
            ]
        )
        II = np.where(mu > 1e-5)[0]
        assert np.allclose(mu[II], bkd.array([0.5, 0.5]))
        assert np.allclose(design_samples[0, II], exact_nonzero_design_samples)

    def test_homoscedastic_least_squares_goptimal_design(self):
        """
        Check G gives same as D optimality. This holds due to equivalence
        theorem.
        """
        bkd = self.get_backend()
        poly_degree = 2
        ndesign_pts = 7
        design_samples = bkd.linspace(-1, 1, ndesign_pts)[None, :]
        monomial = Monomial1D(poly_degree + 1, backend=bkd)
        design_factors = monomial(design_samples)
        noise_mult = None
        opt = ScipyConstrainedOptimizer(opts={"gtol": 1e-12})

        # # construct d-optimal design
        # crit = DOptimalLstSqCriterion(design_factors, noise_mult, backend=bkd)
        # d_oed = LocalOptimalExperimentalDesign(crit)
        # d_oed.set_optimizer(opt)
        # d_mu = d_oed.construct()

        # construct g-optimal design
        pred_factors = bkd.copy(design_factors)
        crit = GOptimalLstSqCriterion(
            design_factors, pred_factors, noise_mult, backend=bkd
        )
        g_oed = LocalOptimalExperimentalDesign(crit)
        g_oed.set_optimizer(MiniMaxOptimizer(opt))
        g_mu = g_oed.construct()
        assert bkd.allclose(g_mu, d_mu)


# class TestNumpyLocalOED(TestLocalOED, unittest.TestCase):
#     def get_backend(self):
#         return NumpyLinAlgMixin


class TestTorchLocalOED(TestLocalOED, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
