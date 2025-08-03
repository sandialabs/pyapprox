import unittest

from scipy import stats
import numpy as np
import sympy as sp

from pyapprox.surrogates.univariate.orthopoly import (
    LegendrePolynomial1D,
    setup_univariate_orthogonal_polynomial_from_marginal,
    AffineMarginalTransform,
    GaussQuadratureRule,
    Chebyshev1stKindGaussLobattoQuadratureRule,
)
from pyapprox.surrogates.univariate.base import (
    Monomial1D,
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.univariate.local import (
    setup_univariate_piecewise_polynomial_basis,
)
from pyapprox.surrogates.affine.basis import (
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
    TensorProductInterpolatingBasis,
    TensorProductQuadratureRule,
    TrigonometricBasis,
    FourierBasis,
    setup_tensor_product_piecewise_poly_quadrature_rule,
    TriangleLebesqueQuadratureRule,
    LstSqSolveBasedRotatedOrthonormalPolynomialBasis,
    QRBasedRotatedOrthonormalPolynomialBasis,
    CholeskyBasedRotatedOrthonormalPolynomialBasis,
)
from pyapprox.surrogates.univariate.lagrange import setup_lagrange_basis
from pyapprox.surrogates.affine.basisexp import (
    MonomialExpansion,
    PolynomialChaosExpansion,
    TensorProductInterpolant,
    TrigonometricExpansion,
    FourierExpansion,
    setup_polynomial_chaos_expansion_from_variable,
    TensorProductLagrangeInterpolantToPolynomialChaosExpansionConverter,
    TensorProductMonomialExpansion,
)
from pyapprox.surrogates.affine.linearsystemsolvers import (
    LstSqSolver,
    OMPSolver,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.multiindex import sort_indices_lexiographically
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import beta_pdf_on_ab, gaussian_pdf

# from pyapprox.util.sys_utilities import package_available
# if package_available("jax"):
#     from pyapprox.util.backends.jax import JaxBackendMixin


class TestBasis:
    def setUp(self):
        np.random.seed(1)

    def _check_monomial_basis(self, nvars, nterms_1d):
        bkd = self.get_backend()
        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis.set_tensor_product_indices([nterms_1d] * nvars)
        samples = bkd.array(np.random.uniform(-1, 1, (nvars, 4)))
        basis_mat = basis(samples)
        for ii, index in enumerate(basis._indices.T):
            assert np.allclose(
                basis_mat[:, ii], bkd.prod(samples.T**index, axis=1)
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
        basis.set_tensor_product_indices([nterms_1d] * nvars)
        samples = bkd.array(np.random.uniform(-1, 1, (nvars, 4)))
        jac = basis.jacobian(samples)
        derivs = bkd.stack(
            [samples * 0, samples * 0 + 1]
            + [ii * samples ** (ii - 1) for ii in range(2, nterms_1d)]
        )
        indices = basis.get_indices()
        for ii in range(indices.shape[1]):
            for dd in range(basis.nvars()):
                index = bkd.copy(indices[:, ii : ii + 1])
                # evaluate basis that has constant in direction of derivative
                index = bkd.up(index, dd, 0)
                basis.set_indices(index)
                deriv_dd = derivs[indices[dd, ii], dd, :] * bkd.prod(
                    basis(samples), axis=1
                )
                assert bkd.allclose(deriv_dd, jac[:, ii, dd])

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
        train_samples = bkd.cos(
            bkd.array(np.random.uniform(0, np.pi, (nvars, ntrain_samples)))
        )

        # Attempt to recover coefficients of additive function
        def fun(samples):
            values = bkd.sum(samples**2 + samples, axis=0)[:, None] + 1.0
            # Create 2 QoI
            return bkd.hstack([(ii + 1) * values for ii in range(nqoi)])

        train_values = fun(train_samples)
        basisexp.fit(train_samples, train_values)
        coef = basisexp.get_coefficients()
        nonzero_indices = bkd.hstack(
            (
                bkd.where(bkd.count_nonzero(basis._indices, axis=0) == 0)[0],
                bkd.where(bkd.count_nonzero(basis._indices, axis=0) == 1)[0],
            )
        )
        true_coef = bkd.full((basis.nterms(), basisexp.nqoi()), 0)
        for ii in range(nqoi):
            # true_coef[nonzero_indices, ii] = ii+1
            true_coef = bkd.up(true_coef, (nonzero_indices, ii), ii + 1)
        assert bkd.allclose(coef, true_coef)
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1000)))
        assert bkd.allclose(basisexp(samples), fun(samples))

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
        trans = AffineMarginalTransform(
            stats.uniform(-1, 2), enforce_bounds=True, backend=bkd
        )
        bases_1d = [
            LegendrePolynomial1D(trans=trans, backend=bkd)
            for ii in range(nvars)
        ]
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_indices(
            bkd.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
        )
        samples = bkd.array(np.random.uniform(-1, 1, (nvars, 101)))
        basis_mat = basis(samples)
        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(nvars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                bkd.stack([1 + 0.0 * x, x, 0.5 * (3.0 * x**2 - 1)], axis=0).T
            )
            exact_basis_derivs_1d.append(
                bkd.stack([0.0 * x, 1.0 + 0.0 * x, 3.0 * x], axis=0).T
            )
            exact_basis_vals_1d[-1] /= bkd.sqrt(
                1.0 / (2 * bkd.arange(degree + 1) + 1)
            )
            exact_basis_derivs_1d[-1] /= bkd.sqrt(
                1.0 / (2 * bkd.arange(degree + 1) + 1)
            )

        exact_basis_mat = bkd.stack(
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

        assert bkd.allclose(basis_mat, exact_basis_mat)

    def _check_multiply_expansion(self, bexp1, bexp2, nqoi):
        bkd = self.get_backend()
        coef1 = bkd.arange(bexp1.nterms() * nqoi, dtype=float).reshape(
            (bexp1.nterms(), nqoi)
        )
        coef2 = bkd.arange(bexp2.nterms() * nqoi, dtype=float).reshape(
            (bexp2.nterms(), nqoi)
        )
        bexp1.set_coefficients(coef1)
        bexp2.set_coefficients(coef2)

        bexp3 = bexp1 * bexp2
        samples = bkd.array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd.allclose(bexp3(samples), bexp1(samples) * bexp2(samples))

        for order in range(4):
            bexp = bexp1**order
            assert bkd.allclose(bexp(samples), bexp1(samples) ** order)

    def _check_multiply_monomial_expansion(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        basis1 = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis1.set_tensor_product_indices([nterms_1d] * nvars)
        basis2 = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        basis2.set_tensor_product_indices([nterms_1d] * nvars)
        bexp1 = MonomialExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = MonomialExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_monomial_expansion(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_monomial_expansion(*test_case)

    def _check_multiply_pce(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        trans = AffineMarginalTransform(
            stats.uniform(-1, 2), enforce_bounds=True, backend=bkd
        )
        polys_1d = [
            LegendrePolynomial1D(trans=trans, backend=bkd)
            for ii in range(nvars)
        ]
        basis1 = OrthonormalPolynomialBasis(polys_1d)
        basis1.set_tensor_product_indices([nterms_1d] * nvars)
        basis2 = OrthonormalPolynomialBasis(polys_1d)
        basis2.set_tensor_product_indices([nterms_1d + 1] * nvars)
        bexp1 = PolynomialChaosExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = PolynomialChaosExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_pce(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_pce(*test_case)

    def _check_add_expansion(self, bexp1, bexp2, nqoi):
        bkd = self.get_backend()
        coef1 = bkd.arange(bexp1.nterms() * nqoi, dtype=float).reshape(
            (bexp1.nterms(), nqoi)
        )
        coef2 = bkd.arange(bexp2.nterms() * nqoi, dtype=float).reshape(
            (bexp2.nterms(), nqoi)
        )
        bexp1.set_coefficients(coef1)
        bexp2.set_coefficients(coef2)

        bexp3 = bexp1 + bexp2
        samples = bkd.array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd.allclose(bexp3(samples), bexp1(samples) + bexp2(samples))

        bexp4 = 2 * bexp1 - bexp2 * 3 + 1
        samples = bkd.array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd.allclose(
            bexp4(samples), 2 * bexp1(samples) - bexp2(samples) * 3 + 1
        )

    def _check_add_pce(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        trans = AffineMarginalTransform(
            stats.uniform(-1, 2), enforce_bounds=True, backend=bkd
        )
        polys_1d = [
            LegendrePolynomial1D(trans=trans, backend=bkd)
            for ii in range(nvars)
        ]
        basis1 = OrthonormalPolynomialBasis(polys_1d)
        basis1.set_tensor_product_indices([nterms_1d] * nvars)
        basis2 = OrthonormalPolynomialBasis(polys_1d)
        basis2.set_tensor_product_indices([nterms_1d + 1] * nvars)
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
        trans = AffineMarginalTransform(
            stats.uniform(-1, 2), enforce_bounds=True, backend=bkd
        )
        polys_1d = [
            LegendrePolynomial1D(trans=trans, backend=bkd)
            for ii in range(nvars)
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_tensor_product_indices([nterms_1d] * nvars)
        pce = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        coef = bkd.arange(pce.nterms() * nqoi).reshape((pce.nterms(), nqoi))
        pce.set_coefficients(coef)
        inactive_idx = bkd.arange(nvars, dtype=int)[::2]
        mpce = pce.marginalize(inactive_idx)
        assert mpce.nterms() == (nterms_1d) ** (nvars - len(inactive_idx)) - 1
        assert bkd.allclose(
            sort_indices_lexiographically(mpce.basis().get_indices()),
            # delete first index which corresponds to constant term
            sort_indices_lexiographically(
                bkd.cartesian_product(
                    [bkd.arange(nterms_1d, dtype=int)] * mpce.nvars()
                )
            )[:, 1:],
        )
        indices = bkd.all(pce.basis().get_indices()[inactive_idx] == 0, axis=0)
        # delete first index which corresponds to constant term
        # indices[bkd.all(pce.basis().get_indices() == 0, axis=0)] = False
        indices = bkd.up(
            indices, bkd.all(pce.basis().get_indices() == 0, axis=0), False
        )
        assert bkd.allclose(
            mpce.get_coefficients(), pce.get_coefficients()[indices]
        )

    def _check_tensor_product_piecewise_polynomial_interpolation(
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
        basis.set_tensor_product_indices(nnodes_1d)

        def fun(samples):
            # when nnodes_1d is zero to test interpolation make sure
            # function is constant in that direction
            return bkd.sum(samples[nnodes_1d > 1] ** 3, axis=0)[:, None]

        train_samples = basis.tensor_product_grid()
        train_values = fun(train_samples)
        interp.fit(train_values)

        test_samples = bkd.asarray(np.random.uniform(0, 1, (nvars, 5)))
        approx_values = interp(test_samples)
        test_values = fun(test_samples)
        assert bkd.allclose(test_values, approx_values, atol=atol)

        variable = IndependentMarginalsVariable(
            [stats.uniform(*bounds)] * 2, backend=bkd
        )
        quad_rule = setup_tensor_product_piecewise_poly_quadrature_rule(
            variable, basis_types, weighted=True
        )

        def fun(samples):
            return bkd.sum(samples**2, axis=0)[:, None]

        quadx, quadw = quad_rule(nnodes_1d)
        integral = fun(quadx).T @ quadw
        exact_integral = bkd.array(2.0 / 3.0)
        assert bkd.allclose(integral, exact_integral, atol=atol)

    def test_tensor_product_piecewise_polynomial_interpolation(self):
        test_cases = [
            [["linear", "linear"], [41, 43], 1e-3],
            [["quadratic", "quadratic"], [41, 43], 1e-5],
            [["cubic", "cubic"], [40, 40], 1e-15],
            [["linear", "quadratic"], [41, 43], 1e-3],
        ]
        for test_case in test_cases:
            self._check_tensor_product_piecewise_polynomial_interpolation(
                *test_case
            )

    def _check_tensor_product_lagrange_interpolation(
        self, basis_types, nnodes_1d, quad_rule, bounds
    ):
        bkd = self.get_backend()
        nvars = len(basis_types)
        nnodes_1d = np.array(nnodes_1d)
        bases_1d = [
            setup_lagrange_basis(bt, quad_rule, bounds, bkd)
            for bt in basis_types
        ]
        bases_1d[0].set_nterms(5)
        basis = TensorProductInterpolatingBasis(bases_1d)
        interp = TensorProductInterpolant(basis)
        basis.set_tensor_product_indices(nnodes_1d)

        assert bkd.allclose(
            bases_1d[0](bases_1d[0].quadrature_rule()[0]),
            bkd.eye(bases_1d[0].nterms()),
        )

        def fun(samples):
            # when nnodes_1d is zero to test interpolation make sure
            # function is constant in that direction
            return bkd.sum(samples[nnodes_1d > 1] ** 3, axis=0)[:, None]

        train_samples = basis.tensor_product_grid()
        train_values = fun(train_samples)
        interp.fit(train_values)

        test_samples = bkd.asarray(np.random.uniform(0, 1, (nvars, 5)))
        approx_values = interp(test_samples)
        test_values = fun(test_samples)
        assert bkd.allclose(test_values, approx_values, atol=1e-15)

    def test_tensor_product_lagrange_interpolation(self):
        # quad_rule = GaussQuadratureRule(stats.uniform(0, 1))
        quad_rule = Chebyshev1stKindGaussLobattoQuadratureRule(
            [-1, 1], backend=self.get_backend()
        )
        test_cases = [
            [["lagrange", "lagrange"], [4, 5], quad_rule, None],
            [["barycentric", "barycentric"], [4, 4], quad_rule, None],
            [["chebyhsev1", "chebyhsev1"], [4, 4], None, [-1, 1]],
        ]
        for test_case in test_cases:
            self._check_tensor_product_lagrange_interpolation(*test_case)

    def _check_tensorproduct_interpolant_quadrature(
        self, name, nvars, degree, nnodes, tol
    ):
        bkd = self.get_backend()
        bounds = [-1, 1]

        def fun(degree, xx):
            return bkd.sum(xx**degree, axis=0)[:, None]

        def integral(nvars, degree):
            if degree == 1:
                return 0
            if degree == 2:
                return nvars * 2 / 3 * 2 ** (nvars - 1)
            if degree == 3:
                return 0
            if degree == 4:
                return nvars * 2 / 5 * 2 ** (nvars - 1)

        bases_1d = [
            setup_univariate_piecewise_polynomial_basis(
                name, bounds, backend=bkd
            )
            for ii in range(nvars)
        ]
        interp_basis = TensorProductInterpolatingBasis(bases_1d)
        interp_basis.set_tensor_product_indices([nnodes] * nvars)
        samples, weights = interp_basis.quadrature_rule()
        assert weights.ndim == 2 and weights.shape[1] == 1
        assert np.allclose(
            fun(degree, samples).T @ weights, integral(nvars, degree), atol=tol
        )

    def test_tensorproduct_interpolant_quadrature(self):
        test_cases = [
            ["linear", 2, 1, 3, 1e-15],
            ["quadratic", 2, 2, 3, 1e-15],
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
        bexp = setup_polynomial_chaos_expansion_from_variable(variable, 1)
        nterms_1d = 3
        bexp.basis().set_tensor_product_indices([nterms_1d] * variable.nvars())
        ntrain_samples = 20

        def fun(sample):
            return (bkd.sum(sample**2, axis=0) + bkd.prod(sample, axis=0))[
                :, None
            ]

        def jac(sample):
            return 2 * sample.T + bkd.flip(sample).T

        def hess(sample):
            return bkd.array([[2.0, 1.0], [1.0, 2.0]])

        train_samples = variable.rvs(ntrain_samples)
        train_values = fun(train_samples)
        bexp.fit(train_samples, train_values)
        ntest_samples = 10
        test_samples = variable.rvs(ntest_samples)
        assert bkd.allclose(fun(test_samples), bexp(test_samples))

        test_samples = bkd.array([[0.5, 1]]).T
        assert bkd.allclose(
            jac(test_samples[:, :1]), bexp.jacobian(test_samples[:, :1])
        )
        assert bkd.allclose(
            hess(test_samples[:, :1]), bexp.hessian(test_samples[:, :1])[0]
        )

        # the following checks that transform of orthonormal basis
        # computes derivatives correctly
        nqoi = 2
        fun = TensorProductMonomialExpansion(
            [Monomial1D(backend=bkd) for ii in range(variable.nvars())],
            [nterms_1d] * variable.nvars(),
            nqoi=nqoi,
        )
        fun.set_coefficients(
            bkd.array(np.random.normal(0, 1, (fun.nterms(), nqoi)))
        )

        train_samples = variable.rvs(ntrain_samples)
        train_values = fun(train_samples)
        bexp = PolynomialChaosExpansion(
            bexp.basis(), solver=LstSqSolver(backend=bkd), nqoi=nqoi
        )
        bexp.fit(train_samples, train_values)

        assert bkd.allclose(
            fun.jacobian(test_samples[:, :1]),
            bexp.jacobian(test_samples[:, :1]),
        )
        assert bkd.allclose(
            fun.hessian(test_samples[:, :1]), bexp.hessian(test_samples[:, :1])
        )

    def test_pce_moments(self):
        alpha_stat, beta_stat, lb, ub = 2, 2, -3, 1
        bkd = self.get_backend()
        marginals = [
            stats.norm(0, 1),
            stats.beta(alpha_stat, beta_stat, lb, ub - lb),
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
        basis.set_tensor_product_indices([nterms_1d] * variable.nvars())
        nqoi = 1
        bexp = PolynomialChaosExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=nqoi
        )

        def fun(sample):
            return (bkd.sum(sample**2, axis=0) + bkd.prod(sample, axis=0))[
                :, None
            ]

        ntrain_samples = 20
        train_samples = variable.rvs(ntrain_samples)
        train_values = fun(train_samples)
        bexp.fit(train_samples, train_values)
        ntest_samples = 10
        test_samples = variable.rvs(ntest_samples)
        assert bkd.allclose(fun(test_samples), bexp(test_samples))

        # compute integral exactly with sympy
        x, y = sp.Symbol("x"), sp.Symbol("y")
        wfun_x = gaussian_pdf(0, 1, x, sp)
        wfun_y = beta_pdf_on_ab(alpha_stat, beta_stat, lb, ub, y)
        exact_mean = float(
            sp.integrate(
                wfun_x * wfun_y * (x**2 + y**2 + x * y),
                (x, -sp.oo, sp.oo),
                (y, lb, ub),
            )
        )
        exact_variance = (
            float(
                sp.integrate(
                    wfun_x * wfun_y * (x**2 + y**2 + x * y) ** 2,
                    (x, -sp.oo, sp.oo),
                    (y, lb, ub),
                )
            )
            - exact_mean**2
        )
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
        basis.set_tensor_product_indices([nterms_1d] * variable.nvars())
        nqoi = 3
        pce = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        pce.set_coefficients(
            bkd.array(np.random.normal(0, 1, (pce.nterms(), nqoi)))
        )

        mon = pce.to_monomial_expansion()

        ntrain_samples = 10
        test_samples = variable.rvs(ntrain_samples)
        assert bkd.allclose(mon(test_samples), pce(test_samples))

    def tensor_product_quadrature_rule(self):
        bkd = self.get_backend()
        nvars = 2
        alpha_stat, beta_stat, lb, ub = 2, 2, -3, 1
        marginals = [
            stats.norm(0, 1),
            stats.beta(alpha_stat, beta_stat, lb, ub - lb),
        ]
        quad_rule = TensorProductQuadratureRule(
            nvars, [GaussQuadratureRule(marginal) for marginal in marginals]
        )

        def fun(sample):
            return (bkd.sum(sample**2, axis=0) + bkd.prod(sample, axis=0))[
                :, None
            ]

        quad_samples, quad_weights = quad_rule(bkd.array([3, 3]))

        # compute integral exactly with sympy
        x, y = sp.Symbol("x"), sp.Symbol("y")
        wfun_x = gaussian_pdf(0, 1, x, sp)
        wfun_y = beta_pdf_on_ab(alpha_stat, beta_stat, lb, ub, y)
        exact_mean = float(
            sp.integrate(
                wfun_x * wfun_y * (x**2 + y**2 + x * y),
                (x, -sp.oo, sp.oo),
                (y, lb, ub),
            )
        )
        assert np.allclose(fun(quad_samples).T @ quad_weights, exact_mean)

    def test_trigonometric_polynomial(self):
        bkd = self.get_backend()
        bounds = [-np.pi, np.pi]
        nterms = 5
        trig_basis = TrigonometricBasis(bounds, backend=bkd)
        trig_basis.set_indices(bkd.arange(nterms)[None, :])
        trig_exp = TrigonometricExpansion(trig_basis)

        def fun(xx):
            return 1 - 4 * bkd.cos(xx.T) + 6 * bkd.sin(2 * xx.T)

        trig_coefs = bkd.array([1.0, -4.0, 0.0, 0.0, 6])[:, None]
        trig_exp.set_coefficients(trig_coefs)

        test_samples = bkd.linspace(*bounds, 11)[None, :]
        np.set_printoptions(linewidth=1000)
        assert bkd.allclose(trig_exp(test_samples), fun(test_samples))

        invfbasis = FourierBasis(bounds, inverse=True, backend=bkd)
        invfbasis.set_indices(bkd.arange(nterms)[None, :])
        invf_exp = FourierExpansion(invfbasis)
        fourier_coefs = invf_exp.fourier_coefficients_from_trig_coefficients(
            trig_coefs
        )
        assert bkd.allclose(
            fourier_coefs,
            bkd.array([3j, -2, 1, -2, -3j], dtype=bkd.complex_dtype())[
                :, None
            ],
        )

        recovered_trig_coefs = (
            trig_exp.trig_coefficients_from_fourier_coefficients(fourier_coefs)
        )
        assert bkd.allclose(recovered_trig_coefs, trig_coefs)

        fbasis = FourierBasis(bounds, inverse=False, backend=bkd)
        fbasis.set_indices(bkd.arange(nterms)[None, :])
        fexp = FourierExpansion(fbasis)
        quad_samples = fexp.quadrature_samples()
        vals = fun(quad_samples)
        fcoefs = fexp.compute_coefficients(vals)
        assert bkd.allclose(fcoefs, fourier_coefs)

        # check that coefficients computed using quadrature creates
        # expansion that interpolates the function values at those samples
        invf_exp.set_coefficients(fcoefs)
        assert bkd.allclose(invf_exp(quad_samples), vals)
        trig_quad_samples = trig_exp.quadrature_samples()
        assert bkd.allclose(
            trig_exp(trig_quad_samples), fun(trig_quad_samples)
        )

        # Compare coefficients computed with quadrature to those computed using
        # fft
        if not bkd.bkd_equal(bkd, NumpyMixin):
            return

        # numpy fft defined for samples on [0, 2*pi] so shift quad_samples
        # to that interval then swap left and right halves of transform to
        # recover fourier coefs computedon [-pi, pi]
        fft_coefs = np.fft.fft(
            fun(quad_samples + np.pi)[:, 0], norm="forward"
        )[:, None]
        fcoefs = bkd.vstack(
            (
                fft_coefs[fbasis._bases_1d[0]._Kmax + 1 :],
                fft_coefs[: fbasis._bases_1d[0]._Kmax + 1],
            )
        )
        assert bkd.allclose(fcoefs, fourier_coefs)

    def test_univariate_lagrange_polynomial_to_polynomial_chaos_expansion(
        self,
    ):
        nvars = 2
        bkd = self.get_backend()
        bounds = [[0, 1], [0, 2]]
        nnodes_1d = bkd.array([3] * nvars)
        bases_1d = [
            setup_lagrange_basis(
                "lagrange",
                ClenshawCurtisQuadratureRule(bounds=b, backend=bkd),
                b,
                bkd,
            )
            for b in bounds
        ]
        bases_1d[0].set_nterms(5)
        basis = TensorProductInterpolatingBasis(bases_1d)
        interp = TensorProductInterpolant(basis)
        basis.set_tensor_product_indices(nnodes_1d)

        def fun(samples):
            # when nnodes_1d is zero to test interpolation make sure
            # function is constant in that direction
            return bkd.sum(samples**2, axis=0)[:, None]

        train_samples = basis.tensor_product_grid()
        train_values = fun(train_samples)
        interp.fit(train_values)

        marginals = [stats.uniform(b[0], b[1] - b[0]) for b in bounds]
        pce_quad_rule = TensorProductQuadratureRule(
            nvars,
            [
                GaussQuadratureRule(marginal, backend=bkd)
                for marginal in marginals
            ],
        )
        converter = TensorProductLagrangeInterpolantToPolynomialChaosExpansionConverter(
            pce_quad_rule
        )
        pce = converter.convert(interp)

        test_samples = converter.variable().rvs(100)
        pce_vals = pce(test_samples)
        interp_vals = interp(test_samples)
        assert bkd.allclose(pce_vals, interp_vals)
        assert bkd.allclose(pce.mean(), bkd.array([1 / 3 + 4 / 3]))

    def test_lagrange_basis_derivatives(self):
        bkd = self.get_backend()
        nvars = 2
        bounds = [[0, 1], [-2, 1]]
        marginals = [stats.uniform(0, 1), stats.uniform(-2, 3)]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        nnodes_1d = bkd.array([5] * nvars)
        bases_1d = [
            setup_lagrange_basis(
                "lagrange",
                ClenshawCurtisQuadratureRule(bounds=b, backend=bkd),
                b,
                bkd,
            )
            for b in bounds
        ]
        # bases_1d[0].set_nterms(5)
        basis = TensorProductInterpolatingBasis(bases_1d)
        interp = TensorProductInterpolant(basis)
        basis.set_tensor_product_indices(nnodes_1d)

        def fun(sample):
            return (bkd.sum(sample**2, axis=0) + bkd.prod(sample, axis=0))[
                :, None
            ]

        def jac(sample):
            return 2 * sample.T + bkd.flip(sample).T

        def hess(sample):
            return bkd.array([[2.0, 1.0], [1.0, 2.0]])

        train_samples = basis.tensor_product_grid()
        train_values = fun(train_samples)
        interp.fit(train_values)

        ntest_samples = 10
        test_samples = variable.rvs(ntest_samples)
        assert bkd.allclose(fun(test_samples), interp(test_samples))

        test_samples = bkd.array([[0.5, 1]]).T
        assert bkd.allclose(
            jac(test_samples[:, :1]), interp.jacobian(test_samples[:, :1])
        )
        assert bkd.allclose(
            hess(test_samples[:, :1]), interp.hessian(test_samples[:, :1])[0]
        )

    @unittest.skip("Not implemented yet")
    def test_pce_product_of_beta_variables(self):
        bkd = self.get_backend()

        def fun(x):
            return bkd.sqrt(x.prod(axis=0))[:, None]

        dist_alpha1, dist_beta1 = 1, 1
        dist_alpha2, dist_beta2 = dist_alpha1 + 0.5, dist_beta1
        nvars = 2

        marginals = [
            stats.beta(dist_alpha1, dist_beta1),
            stats.beta(dist_alpha2, dist_beta2),
        ]
        quad_rule = TensorProductQuadratureRule(
            nvars, [GaussQuadratureRule(marginal) for marginal in marginals]
        )
        quad_samples, quad_weights = quad_rule([100] * nvars)

        mean = fun(quad_samples)[:, 0] @ quad_weights
        variance = (fun(quad_samples)[:, 0] ** 2) @ quad_weights - mean**2
        assert np.allclose(
            mean, stats.beta(dist_alpha1 * 2, dist_beta1 * 2).mean()
        )
        assert np.allclose(
            variance, stats.beta(dist_alpha1 * 2, dist_beta1 * 2).var()
        )

        raise NotImplementedError(
            "Replicate test in old version of pyapprox of the same name"
        )

    def _check_triangular_gauss_quadrature(
        self, fun, vertices, exact_integral
    ):
        bkd = self.get_backend()
        quad_rule = TriangleLebesqueQuadratureRule(vertices, bkd)
        tri_quadx, tri_quadw = quad_rule([5, 5])
        integral = fun(tri_quadx)[:, 0] @ tri_quadw[:, 0]
        assert bkd.allclose(integral, bkd.asarray(exact_integral))

    def test_triangular_gauss_quadrature(self):
        bkd = self.get_backend()
        test_cases = [
            [
                lambda x: 1 + 0 * x[0][:, None],
                bkd.array([[0, 1, 0], [0, 0, 1]]),
                0.5,
            ],
            [
                lambda x: (x**2).sum(axis=0)[:, None],
                bkd.array([[0, 0.5, 1], [0, 1, 0]]),
                0.0729167 + 0.15625,
            ],
            [
                lambda x: (x**3).sum(axis=0)[:, None],
                bkd.array([[0, 1, 0], [0, 0, 1]]),
                1.0 / 10.0,
            ],
        ]
        for test_case in test_cases:
            self._check_triangular_gauss_quadrature(*test_case)

    def _check_rotated_orthonormal_basis(self, rotated_basis_cls):
        bkd = self.get_backend()
        nvars, nterms_1d = 2, 3

        marginals = [stats.uniform(0, 1) for ii in range(nvars)]
        transforms = [
            AffineMarginalTransform(marginal, enforce_bounds=True, backend=bkd)
            for marginal in marginals
        ]
        polys_1d = [
            LegendrePolynomial1D(trans=transforms[ii], backend=bkd)
            for ii in range(nvars)
        ]

        # create independent target variable.
        # Does not test if correlation is captured.
        # will use another test for that
        target_marginals = [stats.beta(2, 2) for ii in range(nvars)]
        target_variable = IndependentMarginalsVariable(
            target_marginals, backend=bkd
        )

        rotated_basis = rotated_basis_cls(polys_1d)
        rotated_basis.set_tensor_product_indices(
            [nterms_1d for ii in range(nvars)]
        )

        nquad_samples = int(1e6)
        quad_samples = target_variable.rvs(nquad_samples)
        quad_weights = bkd.full((nquad_samples, 1), 1.0 / nquad_samples)
        rotated_basis.set_quadrature_rule_tuple(quad_samples, quad_weights)
        nsamples = 1e6
        samples = target_variable.rvs(nsamples)
        basis_mat = rotated_basis(samples)
        gram = basis_mat.T @ (quad_weights * basis_mat)
        assert bkd.allclose(gram, bkd.eye(basis_mat.shape[1]), atol=8e-3)

        # test computation of coefficients in unrotated basis
        coefs = bkd.asarray(
            np.random.normal(0.0, 1.0, (rotated_basis.nterms(), 1))
        )

        unrotated_coefs = rotated_basis.tensor_product_basis_coefficients(
            coefs
        )

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        true_values = rotated_basis(samples) @ coefs
        values = (
            rotated_basis.tensor_product_basis_matrix(samples)
            @ unrotated_coefs
        )
        assert bkd.allclose(values, true_values)

    def test_rotated_orthonormal_basis(self):
        rotated_basis_classes = [
            LstSqSolveBasedRotatedOrthonormalPolynomialBasis,
            QRBasedRotatedOrthonormalPolynomialBasis,
            CholeskyBasedRotatedOrthonormalPolynomialBasis,
        ]
        for rotated_basis_cls in rotated_basis_classes:
            np.random.seed(1)
            self._check_rotated_orthonormal_basis(rotated_basis_cls)


class TestNumpyBasis(TestBasis, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchBasis(TestBasis, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


# class TestJaxBasis(TestBasis, unittest.TestCase):
#     def setUp(self):
#         if not package_available("jax"):
#             self.skipTest("jax not available")
#         TestBasis.setUp(self)

#     def get_backend(self):
#         return JaxBackendMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
