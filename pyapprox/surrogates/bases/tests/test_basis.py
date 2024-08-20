import unittest

from scipy import stats
import numpy as np
import sympy as sp

from pyapprox.surrogates.bases.orthopoly import (
    LegendrePolynomial1D,
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.bases.univariate import (
    Monomial1D, setup_univariate_piecewise_polynomial_basis
)
from pyapprox.surrogates.bases.basis import (
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
    TensorProductInterpolatingBasis,
)
from pyapprox.surrogates.bases.basisexp import (
    MonomialExpansion,
    PolynomialChaosExpansion,
    TensorProductInterpolant,
)
from pyapprox.surrogates.bases.linearsystemsolvers import (
    LstSqSolver,
    OMPSolver,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.interp.indexing import sort_indices_lexiographically
from pyapprox.util.sys_utilities import package_available
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.density import beta_pdf_on_ab, gaussian_pdf


if package_available("jax"):
    from pyapprox.util.linearalgebra.jaxlinalg import JaxLinAlgMixin


class TestBasis:
    def setUp(self):
        np.random.seed(1)

    def _check_monomial_basis(self, nvars, nterms_1d):
        bkd = self.get_backend()
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_tensor_product_indices([nterms_1d]*nvars)
        samples = bkd._la_array(np.random.uniform(-1, 1, (nvars, 4)))
        basis_mat = basis(samples)
        for ii, index in enumerate(basis._indices.T):
            assert np.allclose(
                basis_mat[:, ii], bkd._la_prod(samples.T**index, axis=1)
            )

    def test_monomial_basis(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            self._check_monomial_basis(*test_case)

    def _check_monomial_jacobian(self, nvars, nterms_1d):
        bkd = self.get_backend()
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_tensor_product_indices([nterms_1d]*nvars)
        samples = bkd._la_array(np.random.uniform(-1, 1, (nvars, 4)))
        jac = basis.jacobian(samples)
        derivs = bkd._la_stack(
            [samples * 0, samples * 0 + 1]
            + [ii * samples ** (ii - 1) for ii in range(2, nterms_1d)]
        )
        indices = basis.get_indices()
        for ii in range(indices.shape[1]):
            for dd in range(basis.nvars()):
                index = bkd._la_copy(indices[:, ii : ii + 1])
                # evaluate basis that has constant in direction of derivative
                index = bkd._la_up(index, dd, 0)
                basis.set_indices(index)
                deriv_dd = derivs[indices[dd, ii], dd, :] * bkd._la_prod(
                    basis(samples), axis=1
                )
                assert bkd._la_allclose(deriv_dd, jac[:, ii, dd])

    def test_monomial_jacobian(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            # print(test_case)
            self._check_monomial_jacobian(*test_case)

    def _check_fit_monomial_expansion(self, nvars, solver, nqoi):
        bkd = self.get_backend()
        nterms_1d = 3
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_tensor_product_indices([nterms_1d] * nvars)
        basisexp = MonomialExpansion(basis, solver=solver, nqoi=nqoi)
        ntrain_samples = 2 * basis.nterms()
        train_samples = bkd._la_cos(
            bkd._la_array(np.random.uniform(0, np.pi, (nvars, ntrain_samples)))
        )

        # Attempt to recover coefficients of additive function
        def fun(samples):
            values = bkd._la_sum(samples**2 + samples, axis=0)[:, None] + 1.0
            # Create 2 QoI
            return bkd._la_hstack([(ii + 1) * values for ii in range(nqoi)])

        train_values = fun(train_samples)
        basisexp.fit(train_samples, train_values)
        coef = basisexp.get_coefficients()
        nonzero_indices = bkd._la_hstack(
            (
                bkd._la_where(
                    bkd._la_count_nonzero(basis._indices, axis=0) == 0
                )[0],
                bkd._la_where(
                    bkd._la_count_nonzero(basis._indices, axis=0) == 1
                )[0],
            )
        )
        true_coef = bkd._la_full((basis.nterms(), basisexp.nqoi()), 0)
        for ii in range(nqoi):
            # true_coef[nonzero_indices, ii] = ii+1
            true_coef = bkd._la_up(true_coef, (nonzero_indices, ii), ii + 1)
        assert bkd._la_allclose(coef, true_coef)
        samples = bkd._la_atleast2d(np.random.uniform(-1, 1, (nvars, 1000)))
        assert bkd._la_allclose(basisexp(samples), fun(samples))

    def test_fit_monomial_expansion(self):
        bkd = self.get_backend()
        test_cases = [
            [1, LstSqSolver(backend=bkd), 2],
            [2, LstSqSolver(backend=bkd), 2],
            [3, LstSqSolver(backend=bkd), 2],
            [1, OMPSolver(max_nonzeros=3, backend=bkd), 1],
            [2, OMPSolver(max_nonzeros=6, backend=bkd), 1],
            [3, OMPSolver(max_nonzeros=9, backend=bkd), 1],
        ]
        for test_case in test_cases:
            self._check_fit_monomial_expansion(*test_case)

    def test_orthonormal_polynomial_basis(self):
        bkd = self.get_backend()
        nvars, degree = 2, 2
        bases_1d = [LegendrePolynomial1D(backend=bkd) for ii in range(nvars)]
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_indices(
            bkd._la_array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
        )
        samples = bkd._la_array(np.random.uniform(-1, 1, (nvars, 101)))
        basis_mat = basis(samples)
        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(nvars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                bkd._la_stack(
                    [1 + 0.0 * x, x, 0.5 * (3.0 * x**2 - 1)], axis=0
                ).T
            )
            exact_basis_derivs_1d.append(
                bkd._la_stack([0.0 * x, 1.0 + 0.0 * x, 3.0 * x], axis=0).T
            )
            exact_basis_vals_1d[-1] /= bkd._la_sqrt(
                1.0 / (2 * bkd._la_arange(degree + 1) + 1)
            )
            exact_basis_derivs_1d[-1] /= bkd._la_sqrt(
                1.0 / (2 * bkd._la_arange(degree + 1) + 1)
            )

        exact_basis_mat = bkd._la_stack(
            [
                exact_basis_vals_1d[0][:, 0],
                exact_basis_vals_1d[0][:, 1],
                exact_basis_vals_1d[1][:, 1],
                exact_basis_vals_1d[0][:, 2],
                exact_basis_vals_1d[0][:, 1] * exact_basis_vals_1d[1][:, 1],
                exact_basis_vals_1d[1][:, 2],
            ],
            axis=0,
        ).T

        assert bkd._la_allclose(basis_mat, exact_basis_mat)

    def _check_multiply_expansion(self, bexp1, bexp2, nqoi):
        bkd = self.get_backend()
        coef1 = bkd._la_arange(bexp1.nterms() * nqoi, dtype=float).reshape(
            (bexp1.nterms(), nqoi)
        )
        coef2 = bkd._la_arange(bexp2.nterms() * nqoi, dtype=float).reshape(
            (bexp2.nterms(), nqoi)
        )
        bexp1.set_coefficients(coef1)
        bexp2.set_coefficients(coef2)

        bexp3 = bexp1 * bexp2
        samples = bkd._la_array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd._la_allclose(
            bexp3(samples), bexp1(samples) * bexp2(samples)
        )

        for order in range(4):
            bexp = bexp1**order
            assert bkd._la_allclose(bexp(samples), bexp1(samples) ** order)

    def _check_multiply_monomial_expansion(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        basis1 = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis1.set_tensor_product_indices([nterms_1d]*nvars)
        basis2 = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis2.set_tensor_product_indices([nterms_1d]*nvars)
        bexp1 = MonomialExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = MonomialExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_monomial_expansion(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_monomial_expansion(*test_case)

    def _check_multiply_pce(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        polys_1d = [LegendrePolynomial1D(backend=bkd) for ii in range(nvars)]
        basis1 = OrthonormalPolynomialBasis(polys_1d)
        basis1.set_tensor_product_indices([nterms_1d]*nvars)
        basis2 = OrthonormalPolynomialBasis(polys_1d)
        basis2.set_tensor_product_indices([nterms_1d+1]*nvars)
        bexp1 = PolynomialChaosExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = PolynomialChaosExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_pce(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_pce(*test_case)

    def _check_add_expansion(self, bexp1, bexp2, nqoi):
        bkd = self.get_backend()
        coef1 = bkd._la_arange(bexp1.nterms() * nqoi, dtype=float).reshape(
            (bexp1.nterms(), nqoi)
        )
        coef2 = bkd._la_arange(bexp2.nterms() * nqoi, dtype=float).reshape(
            (bexp2.nterms(), nqoi)
        )
        bexp1.set_coefficients(coef1)
        bexp2.set_coefficients(coef2)

        bexp3 = bexp1 + bexp2
        samples = bkd._la_array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd._la_allclose(
            bexp3(samples), bexp1(samples) + bexp2(samples)
        )

        bexp4 = 2*bexp1 - bexp2*3 + 1
        samples = bkd._la_array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd._la_allclose(
            bexp4(samples), 2*bexp1(samples) - bexp2(samples)*3 + 1
        )

    def _check_add_pce(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        polys_1d = [LegendrePolynomial1D(backend=bkd) for ii in range(nvars)]
        basis1 = OrthonormalPolynomialBasis(polys_1d)
        basis1.set_tensor_product_indices([nterms_1d]*nvars)
        basis2 = OrthonormalPolynomialBasis(polys_1d)
        basis2.set_tensor_product_indices([nterms_1d+1]*nvars)
        bexp1 = PolynomialChaosExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = PolynomialChaosExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_add_expansion(bexp1, bexp2, nqoi)

    def test_add_pce(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_add_pce(*test_case)

    def test_marginalize_pce(self):
        nvars, nqoi, nterms_1d = 4, 2, 3
        bkd = self.get_backend()
        polys_1d = [LegendrePolynomial1D(backend=bkd) for ii in range(nvars)]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_tensor_product_indices([nterms_1d]*nvars)
        pce = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        coef = bkd._la_arange(pce.nterms() * nqoi).reshape(
            (pce.nterms(), nqoi)
        )
        pce.set_coefficients(coef)
        inactive_idx = bkd._la_arange(nvars, dtype=int)[::2]
        mpce = pce.marginalize(inactive_idx)
        assert mpce.nterms() == (nterms_1d) ** (nvars - len(inactive_idx)) - 1
        assert bkd._la_allclose(
            sort_indices_lexiographically(mpce.basis.get_indices()),
            # delete first index which corresponds to constant term
            sort_indices_lexiographically(
                bkd._la_cartesian_product(
                    [bkd._la_arange(nterms_1d, dtype=int)] * mpce.nvars()
                )
            )[:, 1:],
        )
        indices = bkd._la_all(
            pce.basis.get_indices()[inactive_idx] == 0, axis=0
        )
        # delete first index which corresponds to constant term
        # indices[bkd._la_all(pce.basis.get_indices() == 0, axis=0)] = False
        indices = bkd._la_up(
            indices, bkd._la_all(pce.basis.get_indices() == 0, axis=0), False
        )
        assert bkd._la_allclose(
            mpce.get_coefficients(), pce.get_coefficients()[indices]
        )

    def _check_tensor_product_interpolation(
            self, basis_types, nnodes_1d, atol
    ):
        bkd = self.get_backend()
        bounds = [0, 1]
        nvars = len(basis_types)
        nnodes_1d = np.array(nnodes_1d)
        bases_1d = [
            setup_univariate_piecewise_polynomial_basis(
                bt, bounds, backend=bkd
            )
            for bt in basis_types
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        interp = TensorProductInterpolant(basis)
        basis.set_1d_nodes(
            [bkd._la_linspace(0, 1, N)[None, :] for N in nnodes_1d])

        def fun(samples):
            # when nnodes_1d is zero to test interpolation make sure
            # function is constant in that direction
            return bkd._la_sum(samples[nnodes_1d > 1]**3, axis=0)[:, None]

        train_samples = basis.tensor_product_grid()
        train_values = fun(train_samples)
        interp.fit(train_values)

        test_samples = bkd._la_asarray(np.random.uniform(0, 1, (nvars, 5)))
        approx_values = interp(test_samples)
        test_values = fun(test_samples)
        assert bkd._la_allclose(test_values, approx_values, atol=atol)

    def test_tensor_product_interpolation(self):
        test_cases = [
            [["linear", "linear"], [41, 43], 1e-3],
            [["quadratic", "quadratic"], [41, 43], 1e-5],
            [["cubic", "cubic"], [40, 40], 1e-15],
            [["linear", "quadratic"], [41, 43], 1e-3],
            # todo add lagrange once most common lagrange quadrature
            # combos have been added to setup_univariate_interpolating_basis
            # [["lagrange", "lagrange"], [4, 5], 1e-15],
            # [["linear", "quadratic", "lagrange"], [41, 23, 4], 1e-3],
            # [["cubic", "quadratic", "lagrange"], [25, 23, 4], 1e-4],
            # Following tests use of active vars when nnodes_1d[ii] = 0
            # [["linear", "quadratic", "lagrange"], [1, 23, 4], 1e-4],
        ]
        for test_case in test_cases:
            self._check_tensor_product_interpolation(*test_case)

    def _check_tensorproduct_interpolant_quadrature(
            self, name, nvars, degree, nnodes, tol):
        bkd = self.get_backend()
        bounds = [-1, 1]

        def fun(degree, xx):
            return bkd._la_sum(xx**degree, axis=0)[:, None]

        def integral(nvars, degree):
            if degree == 1:
                return 0
            if degree == 2:
                return nvars*2/3*2**(nvars-1)
            if degree == 3:
                return 0
            if degree == 4:
                return nvars*2/5*2**(nvars-1)

        bases_1d = [
            setup_univariate_piecewise_polynomial_basis(
                name, bounds, backend=bkd
            )
            for ii in range(nvars)
        ]
        nodes_1d = [
            bkd._la_linspace(*bounds, nnodes)[None, :] for ii in range(nvars)
        ]
        interp_basis = TensorProductInterpolatingBasis(bases_1d)
        interp_basis.set_1d_nodes(nodes_1d)
        samples, weights = interp_basis.quadrature_rule()
        assert weights.ndim == 2 and weights.shape[1] == 1
        assert np.allclose(
            fun(degree, samples).T @ weights, integral(nvars, degree),
            atol=tol)

    def test_tensorproduct_interpolant_quadrature(self):
        test_cases = [
            ["linear", 2, 1, 3, 1e-15],
            ["quadratic", 2, 2, 3,  1e-15],
            ["quadratic", 2, 4, 91, 1e-5],
            ["cubic", 2, 3, 4, 1e-15],
            ["cubic", 2, 4, 46, 1e-5],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_tensorproduct_interpolant_quadrature(*test_case)

    def test_orthpoly_with_transformation(self):
        bkd = self.get_backend()
        marginals = [stats.uniform(0, 1), stats.uniform(-2, 3)]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=bkd
            )
            for marginal in marginals
        ]
        nterms_1d = 3
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_tensor_product_indices(
            [nterms_1d]*variable.num_vars()
        )
        nqoi = 1
        bexp = PolynomialChaosExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=nqoi
        )
        ntrain_samples = 20

        def fun(sample):
            return (
                bkd._la_sum(sample**2, axis=0) +
                bkd._la_prod(sample, axis=0)
            )[:, None]

        def jac(sample):
            return 2*sample.T+bkd._la_flip(sample).T

        def hess(sample):
            return bkd._la_array([[2., 1.], [1., 2.]])

        train_samples = variable.rvs(ntrain_samples)
        train_values = fun(train_samples)
        bexp.fit(train_samples, train_values)
        ntest_samples = 10
        test_samples = variable.rvs(ntest_samples)
        assert bkd._la_allclose(fun(test_samples), bexp(test_samples))

        test_samples = bkd._la_array([[1, 1]]).T
        assert bkd._la_allclose(
            jac(test_samples[:, :1]),
            bexp.jacobian(test_samples[:, :1])[0]
        )
        assert bkd._la_allclose(
            hess(test_samples[:, :1]),
            bexp.hessian(test_samples[:, :1])[0, ..., 0]
        )

        # the following checks that transform of orthonormal basis
        # computes derivatives correctly
        nqoi = 2
        monomial_basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(variable.num_vars())]
        )
        monomial_basis.set_tensor_product_indices(
            [nterms_1d]*variable.num_vars()
        )
        fun = MonomialExpansion(monomial_basis, nqoi=nqoi)
        fun.set_coefficients(
            bkd._la_array(np.random.normal(0, 1, (fun.nterms(), nqoi)))
        )

        train_samples = variable.rvs(ntrain_samples)
        train_values = fun(train_samples)
        bexp = PolynomialChaosExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=nqoi
        )
        bexp.fit(train_samples, train_values)

        assert bkd._la_allclose(
            fun.jacobian(test_samples[:, :1]),
            bexp.jacobian(test_samples[:, :1])
        )
        assert bkd._la_allclose(
            fun.hessian(test_samples[:, :1]),
            bexp.hessian(test_samples[:, :1])
        )

    def test_pce_moments(self):
        alpha_stat, beta_stat, lb, ub = 2, 2, -3, 1
        bkd = self.get_backend()
        marginals = [
            stats.norm(0, 1), stats.beta(alpha_stat, beta_stat, lb, ub-lb)
        ]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=bkd
            )
            for marginal in marginals
        ]
        nterms_1d = 3
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_tensor_product_indices(
            [nterms_1d]*variable.num_vars()
        )
        nqoi = 1
        bexp = PolynomialChaosExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=nqoi
        )

        def fun(sample):
            return (
                bkd._la_sum(sample**2, axis=0) +
                bkd._la_prod(sample, axis=0)
            )[:, None]

        ntrain_samples = 20
        train_samples = variable.rvs(ntrain_samples)
        train_values = fun(train_samples)
        bexp.fit(train_samples, train_values)
        ntest_samples = 10
        test_samples = variable.rvs(ntest_samples)
        assert bkd._la_allclose(fun(test_samples), bexp(test_samples))

        # compute integral exactly with sympy
        x, y = sp.Symbol('x'), sp.Symbol('y')
        wfun_x = gaussian_pdf(0, 1, x, sp)
        wfun_y = beta_pdf_on_ab(alpha_stat, beta_stat, lb, ub, y)
        exact_mean = float(
            sp.integrate(
                wfun_x*wfun_y*(x**2+y**2+x*y),
                (x, -sp.oo, sp.oo), (y, lb, ub)
            )
        )
        exact_variance = float(
            sp.integrate(
                wfun_x*wfun_y*(x**2+y**2+x*y)**2,
                (x, -sp.oo, sp.oo), (y, lb, ub)
            )
        ) - exact_mean**2
        assert np.allclose(exact_mean, bexp.mean())
        assert np.allclose(exact_variance, bexp.variance())

    def test_pce_to_monomial(self):
        bkd = self.get_backend()
        marginals = [stats.uniform(0, 1)]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=bkd
            )
            for marginal in marginals
        ]
        nterms_1d = 3
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_tensor_product_indices(
            [nterms_1d]*variable.num_vars()
        )
        nqoi = 3
        pce = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        pce.set_coefficients(
            bkd._la_array(np.random.normal(0, 1, (pce.nterms(), nqoi)))
        )

        mon = pce.to_monomial_expansion()

        ntrain_samples = 10
        test_samples = variable.rvs(ntrain_samples)
        assert bkd._la_allclose(mon(test_samples), pce(test_samples))


class TestNumpyBasis(TestBasis, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchBasis(TestBasis, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


class TestJaxBasis(TestBasis, unittest.TestCase):
    def setUp(self):
        if not package_available("jax"):
            self.skipTest("jax not available")
        TestBasis.setUp(self)

    def get_backend(self):
        return JaxLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)
