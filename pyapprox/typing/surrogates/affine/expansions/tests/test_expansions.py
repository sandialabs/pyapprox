"""Tests for basis expansions module."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

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
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
    BatchDerivativeChecker,
)


class TestBasisExpansion(Generic[Array], unittest.TestCase):
    """Test BasisExpansion base class."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_basis(self, nvars: int, max_level: int):
        """Helper to create a Legendre basis."""
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_basic_properties(self):
        """Test basic properties."""
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, self._bkd, nqoi=2)

        self.assertEqual(exp.nvars(), 2)
        self.assertGreater(exp.nterms(), 0)
        self.assertEqual(exp.nqoi(), 2)
        # Check method availability via hasattr (dynamic binding pattern)
        self.assertTrue(hasattr(exp, "jacobian_batch"))
        # hessian_batch not available for nqoi != 1
        self.assertFalse(hasattr(exp, "hessian_batch"))

    def test_derivative_methods_available_for_nqoi_1(self):
        """Test derivative methods are available for nqoi=1."""
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, self._bkd, nqoi=1)

        # All derivative methods should be available
        self.assertTrue(hasattr(exp, "jacobian_batch"))
        self.assertTrue(hasattr(exp, "hessian_batch"))
        self.assertTrue(hasattr(exp, "jacobian"))
        self.assertTrue(hasattr(exp, "hessian"))
        self.assertTrue(hasattr(exp, "hvp"))
        self.assertTrue(hasattr(exp, "whvp"))

    def test_coefficient_shape(self):
        """Test coefficient storage."""
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, self._bkd, nqoi=3)

        coef = exp.get_coefficients()
        self.assertEqual(coef.shape, (exp.nterms(), 3))

    def test_set_coefficients(self):
        """Test setting coefficients."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        new_coef = bkd.asarray(np.random.randn(exp.nterms(), 2))
        exp.set_coefficients(new_coef)
        bkd.assert_allclose(exp.get_coefficients(), new_coef)

    def test_evaluation_shape(self):
        """Test evaluation shape returns (nqoi, nsamples)."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        # Shape should be (nqoi, nsamples) per CLAUDE.md convention
        self.assertEqual(values.shape, (2, nsamples))

    def test_jacobian_batch_shape(self):
        """Test Jacobian shape."""
        bkd = self._bkd
        basis = self._create_basis(nvars=3, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        # Set random coefficients
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        jac = exp.jacobian_batch(samples)
        self.assertEqual(jac.shape, (nsamples, 2, 3))

    def test_hessian_batch_shape(self):
        """Test Hessian shape for nqoi=1."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        hess = exp.hessian_batch(samples)
        # Shape should be (nsamples, nvars, nvars) per CLAUDE.md convention
        self.assertEqual(hess.shape, (nsamples, 2, 2))

    def test_hessian_batch_not_available_for_multi_qoi(self):
        """Test hessian_batch not available for nqoi > 1."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        # hessian_batch method should not be bound for nqoi > 1
        self.assertFalse(hasattr(exp, "hessian_batch"))

    def test_jacobian_batch_finite_difference(self):
        """Test Jacobian accuracy via finite differences."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))
        jac = exp.jacobian_batch(samples)

        eps = 1e-7
        for dd in range(2):
            samples_plus = bkd.copy(samples)
            samples_minus = bkd.copy(samples)
            samples_plus[dd, :] += eps
            samples_minus[dd, :] -= eps

            fd_jac = (exp(samples_plus) - exp(samples_minus)) / (2 * eps)
            # jac shape: (nsamples, nqoi, nvars), fd_jac shape: (nqoi, nsamples)
            bkd.assert_allclose(jac[:, 0, dd], fd_jac[0, :], rtol=1e-5, atol=1e-7)


class TestBasisExpansionNumpy(TestBasisExpansion[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBasisExpansionTorch(TestBasisExpansion[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestBasisExpansionFitting(Generic[Array], unittest.TestCase):
    """Test basis expansion fitting."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_basis(self, nvars: int, max_level: int):
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial(self):
        """Test fitting a polynomial function."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x,y) = x^2 + xy
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = bkd.reshape(x**2 + x*y, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + x_test*y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_with_solver(self):
        """Test fitting with explicit solver."""
        bkd = self._bkd
        basis = self._create_basis(nvars=2, max_level=2)
        solver = LeastSquaresSolver(bkd)
        exp = BasisExpansion(basis, bkd, nqoi=1, solver=solver)

        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y, (1, -1))

        exp.fit(samples, values)

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test + y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)


class TestBasisExpansionFittingNumpy(TestBasisExpansionFitting[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBasisExpansionFittingTorch(TestBasisExpansionFitting[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestPolynomialChaosExpansion(Generic[Array], unittest.TestCase):
    """Test PolynomialChaosExpansion class."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_mean_constant(self):
        """Test mean computation for constant function."""
        bkd = self._bkd
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
        bkd = self._bkd
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
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3)

        # Fit to f(x,y) = 1 + x + y + xy for uniform on [-1,1]^2
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = bkd.reshape(1.0 + x + y + x*y, (1, -1))

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
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(2)]
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        pce = PolynomialChaosExpansion(basis, bkd)

        # Get quadrature points
        quad_samples, quad_weights = basis.tensor_product_quadrature([8, 8])

        # Evaluate function at quadrature points
        x, y = quad_samples[0, :], quad_samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = bkd.reshape(1.0 + x + y, (1, -1))

        # Fit via projection
        pce.fit_via_projection(quad_samples, values, quad_weights)

        # Test on new samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(1.0 + x_test + y_test, (1, -1))
        predicted = pce(test_samples)

        bkd.assert_allclose(predicted, expected, atol=1e-10)


class TestPolynomialChaosExpansionNumpy(TestPolynomialChaosExpansion[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolynomialChaosExpansionTorch(TestPolynomialChaosExpansion[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestPCESobolIndices(Generic[Array], unittest.TestCase):
    """Test PCE Sobol sensitivity indices."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_total_sobol_sum(self):
        """Test that total Sobol indices are valid."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3)

        # Fit to f(x,y) = x + y + xy
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x*y, (1, -1))
        pce.fit(samples, values)

        total_indices = pce.total_sobol_indices()

        # Total indices should each include main effect + interaction
        # Sum should be >= 1 (due to interactions counted twice)
        self.assertTrue(bkd.to_numpy(bkd.sum(total_indices)) >= 1.0 - 1e-10)

        # Each total index should be positive
        self.assertTrue(np.all(bkd.to_numpy(total_indices) >= -1e-10))

    def test_main_effect_sum(self):
        """Test that main effect Sobol indices sum to <= 1."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x*y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()

        # Main effects should sum to <= 1
        self.assertTrue(bkd.to_numpy(bkd.sum(main_indices)) <= 1.0 + 1e-10)

    def test_additive_function(self):
        """Test Sobol indices for additive function (no interactions)."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3)

        # f(x,y) = x + 2*y (additive, no interactions)
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + 2*y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()
        total_indices = pce.total_sobol_indices()

        # For additive function, main = total
        bkd.assert_allclose(main_indices, total_indices, atol=1e-10)

        # Main effects should sum to 1
        bkd.assert_allclose(
            bkd.reshape(bkd.sum(main_indices), (1,)), bkd.asarray([1.0]), atol=1e-10
        )


class TestPCESobolIndicesNumpy(TestPCESobolIndices[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCESobolIndicesTorch(TestPCESobolIndices[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestPCEStatisticsFunctions(Generic[Array], unittest.TestCase):
    """Test standalone PCE statistics functions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_functions_match_methods(self):
        """Test that standalone functions match PCE methods."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3)

        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x*y, (1, -1))
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


class TestPCEStatisticsFunctionsNumpy(TestPCEStatisticsFunctions[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCEStatisticsFunctionsTorch(TestPCEStatisticsFunctions[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestSolvers(Generic[Array], unittest.TestCase):
    """Test linear system solvers."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_least_squares_overdetermined(self):
        """Test least squares on overdetermined system."""
        bkd = self._bkd
        solver = LeastSquaresSolver(bkd)

        # y = 2x + 1 with some noise
        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef = solver.solve(basis_matrix, y)
        bkd.assert_allclose(
            bkd.reshape(coef[0, 0], (1,)), bkd.asarray([1.0]), atol=1e-10
        )
        bkd.assert_allclose(
            bkd.reshape(coef[1, 0], (1,)), bkd.asarray([2.0]), atol=1e-10
        )

    def test_ridge_regression(self):
        """Test ridge regression."""
        bkd = self._bkd
        solver = RidgeRegressionSolver(bkd, alpha=0.1)

        x = bkd.asarray(np.linspace(0, 1, 20).reshape(-1, 1))
        basis_matrix = bkd.concatenate([bkd.ones((20, 1)), x], axis=1)
        y = 2 * x + 1

        coef = solver.solve(basis_matrix, y)
        # Ridge regression should still get reasonable coefficients
        self.assertTrue(0.5 < bkd.to_numpy(coef[0, 0]) < 1.5)
        self.assertTrue(1.5 < bkd.to_numpy(coef[1, 0]) < 2.5)


class TestSolversNumpy(TestSolvers[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSolversTorch(TestSolvers[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeChecker(Generic[Array], unittest.TestCase):
    """Test derivatives using DerivativeChecker per CLAUDE.md."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        bases_1d = [LegendrePolynomial1D(bkd) for _ in range(nvars)]
        return create_pce(bases_1d, max_level, bkd, nqoi=nqoi)

    def test_jacobian_with_derivative_checker(self):
        """Validate Jacobian using DerivativeChecker for single-sample methods."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=1)

        # Set random coefficients
        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Check derivatives using standard DerivativeChecker for single-sample methods
        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        # error_ratio should be ~1e-6 for correct derivatives
        # A small ratio indicates the minimum error is much smaller than maximum
        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_with_derivative_checker(self):
        """Validate Hessian using DerivativeChecker for single-sample methods."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=1)

        # Set random coefficients
        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Check derivatives
        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        # error_ratio should be ~1e-6 for correct derivatives
        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_jacobian_batch_with_batch_derivative_checker(self):
        """Validate jacobian_batch using BatchDerivativeChecker."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=2)

        # Set random coefficients
        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        # Create test samples
        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        # Check jacobian_batch using BatchDerivativeChecker
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        # error_ratio should be ~1e-6 for correct derivatives
        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)

    def test_hessian_batch_with_batch_derivative_checker(self):
        """Validate hessian_batch using BatchDerivativeChecker."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=1)

        # Set random coefficients
        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Create test samples
        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        # Check hessian_batch using BatchDerivativeChecker
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        # error_ratio should be ~1e-6 for correct derivatives
        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)


class TestDerivativeCheckerNumpy(TestDerivativeChecker[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerTorch(TestDerivativeChecker[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
