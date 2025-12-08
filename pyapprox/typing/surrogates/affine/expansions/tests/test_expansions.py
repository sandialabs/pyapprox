"""Tests for basis expansions module."""

import unittest
from typing import Type

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Backend

from pyapprox.typing.surrogates.affine.univariate import (
    LegendrePolynomial1D,
    HermitePolynomial1D,
)
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.basis import (
    OrthonormalPolynomialBasis,
)
from pyapprox.typing.surrogates.affine.expansions import (
    BasisExpansion,
    PolynomialChaosExpansion,
    create_pce,
    LeastSquaresSolver,
    RidgeRegressionSolver,
    pce_statistics,
)


class _BaseExpansionTest:
    """Base class for expansion tests. Not run directly."""

    __test__ = False
    bkd_class: Type[Backend[NDArray]] = NumpyBkd

    def setUp(self):
        self.bkd = self.bkd_class()


class TestBasisExpansion(_BaseExpansionTest, unittest.TestCase):
    """Test BasisExpansion base class."""

    __test__ = True

    def _create_basis(self, nvars: int, max_level: int):
        """Helper to create a Legendre basis."""
        bkd = self.bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_basic_properties(self):
        """Test basic properties."""
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, self.bkd, nqoi=2)

        self.assertEqual(exp.nvars(), 2)
        self.assertGreater(exp.nterms(), 0)
        self.assertEqual(exp.nqoi(), 2)
        self.assertTrue(exp.jacobian_supported())
        self.assertTrue(exp.hessian_supported())

    def test_coefficient_shape(self):
        """Test coefficient storage."""
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, self.bkd, nqoi=3)

        coef = exp.get_coefficients()
        self.assertEqual(coef.shape, (exp.nterms(), 3))

    def test_set_coefficients(self):
        """Test setting coefficients."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        new_coef = bkd.asarray(np.random.randn(exp.nterms(), 2))
        exp.set_coefficients(new_coef)
        bkd.assert_allclose(exp.get_coefficients(), new_coef)

    def test_evaluation_shape(self):
        """Test evaluation shape."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        self.assertEqual(values.shape, (nsamples, 2))

    def test_jacobian_shape(self):
        """Test Jacobian shape."""
        bkd = self.bkd
        basis = self._create_basis(nvars=3, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        # Set random coefficients
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        jac = exp.jacobians(samples)
        self.assertEqual(jac.shape, (nsamples, 2, 3))

    def test_hessian_shape(self):
        """Test Hessian shape."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        hess = exp.hessians(samples)
        self.assertEqual(hess.shape, (nsamples, 1, 2, 2))

    def test_jacobian_finite_difference(self):
        """Test Jacobian accuracy via finite differences."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        jac = exp.jacobians(samples)

        eps = 1e-7
        for dd in range(2):
            samples_plus = bkd.copy(samples)
            samples_minus = bkd.copy(samples)
            samples_plus[dd, :] += eps
            samples_minus[dd, :] -= eps

            fd_jac = (exp(samples_plus) - exp(samples_minus)) / (2 * eps)
            bkd.assert_allclose(jac[:, 0, dd], fd_jac[:, 0], rtol=1e-5, atol=1e-7)


class TestBasisExpansionFitting(_BaseExpansionTest, unittest.TestCase):
    """Test basis expansion fitting."""

    __test__ = True

    def _create_basis(self, nvars: int, max_level: int):
        bkd = self.bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial(self):
        """Test fitting a polynomial function."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x,y) = x^2 + xy
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + x*y, (-1, 1))

        exp.fit(samples, values)

        # Test on new samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + x_test*y_test, (-1, 1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_with_solver(self):
        """Test fitting with explicit solver."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=2)
        solver = LeastSquaresSolver(bkd)
        exp = BasisExpansion(basis, bkd, nqoi=1, solver=solver)

        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y, (-1, 1))

        exp.fit(samples, values)

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test + y_test, (-1, 1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)


class TestPolynomialChaosExpansion(_BaseExpansionTest, unittest.TestCase):
    """Test PolynomialChaosExpansion class."""

    __test__ = True

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self.bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_mean_constant(self):
        """Test mean computation for constant function."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=2)

        # Set coefficient for constant term
        coef = bkd.zeros((pce.nterms(), 1))
        const_idx = pce._get_constant_index()
        coef[const_idx, 0] = 5.0
        pce.set_coefficients(coef)

        mean = pce.mean()
        bkd.assert_allclose(mean, bkd.asarray([5.0]))

    def test_variance_single_term(self):
        """Test variance computation for single non-constant term."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=2)

        coef = bkd.zeros((pce.nterms(), 1))
        # Set constant term to 2.0
        const_idx = pce._get_constant_index()
        coef[const_idx, 0] = 2.0
        # Set another term to 3.0
        other_idx = 1 if const_idx != 1 else 2
        coef[other_idx, 0] = 3.0
        pce.set_coefficients(coef)

        var = pce.variance()
        # Variance = 3^2 = 9
        bkd.assert_allclose(var, bkd.asarray([9.0]))

    def test_statistics_via_quadrature(self):
        """Test PCE statistics against Monte Carlo."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=3)

        # Fit to f(x,y) = 1 + x + y + xy for uniform on [-1,1]^2
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(1.0 + x + y + x*y, (-1, 1))

        pce.fit(samples, values)

        # Analytical statistics for this function:
        # Mean = E[1 + x + y + xy] = 1 + 0 + 0 + 0 = 1
        # Var = Var[x] + Var[y] + Var[xy] = 1/3 + 1/3 + 1/9 = 7/9
        mean = pce.mean()
        var = pce.variance()

        bkd.assert_allclose(mean, bkd.asarray([1.0]), atol=1e-10)
        bkd.assert_allclose(var, bkd.asarray([7.0/9.0]), atol=1e-10)

    def test_fit_via_projection(self):
        """Test spectral projection fitting."""
        bkd = self.bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(2)]
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        pce = PolynomialChaosExpansion(basis, bkd)

        # Get quadrature points
        quad_samples, quad_weights = basis.tensor_product_quadrature([8, 8])

        # Evaluate function at quadrature points
        x, y = quad_samples[0, :], quad_samples[1, :]
        values = bkd.reshape(1.0 + x + y, (-1, 1))

        # Fit via projection
        pce.fit_via_projection(quad_samples, values, quad_weights)

        # Test on new samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(1.0 + x_test + y_test, (-1, 1))
        predicted = pce(test_samples)

        bkd.assert_allclose(predicted, expected, atol=1e-10)


class TestPCESobolIndices(_BaseExpansionTest, unittest.TestCase):
    """Test PCE Sobol sensitivity indices."""

    __test__ = True

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self.bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_total_sobol_sum(self):
        """Test that total Sobol indices are valid."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=3)

        # Fit to f(x,y) = x + y + xy
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x*y, (-1, 1))
        pce.fit(samples, values)

        total_indices = pce.total_sobol_indices()

        # Total indices should each include main effect + interaction
        # Sum should be >= 1 (due to interactions counted twice)
        self.assertTrue(bkd.to_numpy(bkd.sum(total_indices)) >= 1.0 - 1e-10)

        # Each total index should be positive
        self.assertTrue(np.all(bkd.to_numpy(total_indices) >= -1e-10))

    def test_main_effect_sum(self):
        """Test that main effect Sobol indices sum to <= 1."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x*y, (-1, 1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()

        # Main effects should sum to <= 1
        self.assertTrue(bkd.to_numpy(bkd.sum(main_indices)) <= 1.0 + 1e-10)

    def test_additive_function(self):
        """Test Sobol indices for additive function (no interactions)."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=3)

        # f(x,y) = x + 2*y (additive, no interactions)
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + 2*y, (-1, 1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()
        total_indices = pce.total_sobol_indices()

        # For additive function, main = total
        bkd.assert_allclose(main_indices, total_indices, atol=1e-10)

        # Main effects should sum to 1
        bkd.assert_allclose(bkd.sum(main_indices), 1.0, atol=1e-10)


class TestPCEStatisticsFunctions(_BaseExpansionTest, unittest.TestCase):
    """Test standalone PCE statistics functions."""

    __test__ = True

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self.bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_functions_match_methods(self):
        """Test that standalone functions match PCE methods."""
        bkd = self.bkd
        pce = self._create_pce(nvars=2, max_level=3)

        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x*y, (-1, 1))
        pce.fit(samples, values)

        # Compare function outputs to method outputs
        bkd.assert_allclose(pce_statistics.mean(pce), pce.mean())
        bkd.assert_allclose(pce_statistics.variance(pce), pce.variance())
        bkd.assert_allclose(pce_statistics.std(pce), pce.std())
        bkd.assert_allclose(
            pce_statistics.total_sobol_indices(pce),
            pce.total_sobol_indices()
        )
        bkd.assert_allclose(
            pce_statistics.main_effect_sobol_indices(pce),
            pce.main_effect_sobol_indices()
        )


class TestSolvers(_BaseExpansionTest, unittest.TestCase):
    """Test linear system solvers."""

    __test__ = True

    def test_least_squares_overdetermined(self):
        """Test least squares on overdetermined system."""
        bkd = self.bkd
        solver = LeastSquaresSolver(bkd)

        # y = 2x + 1 with some noise
        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef = solver.solve(basis_matrix, y)
        bkd.assert_allclose(coef[0, 0], 1.0, atol=1e-10)
        bkd.assert_allclose(coef[1, 0], 2.0, atol=1e-10)

    def test_ridge_regression(self):
        """Test ridge regression."""
        bkd = self.bkd
        solver = RidgeRegressionSolver(bkd, alpha=0.1)

        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef = solver.solve(basis_matrix, y)
        # Ridge regression should still get reasonable coefficients
        self.assertTrue(0.5 < bkd.to_numpy(coef[0, 0]) < 1.5)
        self.assertTrue(1.5 < bkd.to_numpy(coef[1, 0]) < 2.5)


if __name__ == "__main__":
    unittest.main()
