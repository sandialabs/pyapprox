"""Tests for basis expansions module."""

import numpy as np

from pyapprox.probability import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.basis import (
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.expansions import (
    BasisExpansion,
    PolynomialChaosExpansion,
    create_pce_from_marginals,
    pce_statistics,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.solvers import LeastSquaresSolver
from pyapprox.surrogates.affine.univariate import (
    MonomialBasis1D,
    create_bases_1d,
)

# Tests are organized into separate files:
# - test_solvers.py: Linear system solvers (LeastSquares, Ridge, OMP, etc.)
# - test_pce_statistics.py: PCE statistics and Sobol indices
# - test_derivative_checks.py: Derivative validation with DerivativeChecker
# This file: BasisExpansion tests with multiple basis types and dimensions


class TestBasisExpansion:
    """Test BasisExpansion base class."""

    def _create_basis(self, bkd, nvars: int, max_level: int):
        """Helper to create a Legendre basis for uniform marginals."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_basic_properties(self, bkd):
        """Test basic properties."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        assert exp.nvars() == 2
        assert exp.nterms() > 0
        assert exp.nqoi() == 2
        # Check method availability via hasattr (dynamic binding pattern)
        assert hasattr(exp, "jacobian_batch")
        # hessian_batch not available for nqoi != 1
        assert not hasattr(exp, "hessian_batch")

    def test_derivative_methods_available_for_nqoi_1(self, bkd):
        """Test derivative methods are available for nqoi=1."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # All derivative methods should be available
        assert hasattr(exp, "jacobian_batch")
        assert hasattr(exp, "hessian_batch")
        assert hasattr(exp, "jacobian")
        assert hasattr(exp, "hessian")
        assert hasattr(exp, "hvp")
        assert hasattr(exp, "whvp")

    def test_coefficient_shape(self, bkd):
        """Test coefficient storage."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=3)

        coef = exp.get_coefficients()
        assert coef.shape == (exp.nterms(), 3)

    def test_set_coefficients(self, bkd):
        """Test setting coefficients."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        new_coef = bkd.asarray(np.random.randn(exp.nterms(), 2))
        exp.set_coefficients(new_coef)
        bkd.assert_allclose(exp.get_coefficients(), new_coef)

    def test_evaluation_shape(self, bkd):
        """Test evaluation shape returns (nqoi, nsamples)."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        # Shape should be (nqoi, nsamples) per CLAUDE.md convention
        assert values.shape == (2, nsamples)

    def test_jacobian_batch_shape(self, bkd):
        """Test Jacobian shape."""
        basis = self._create_basis(bkd, nvars=3, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        # Set random coefficients
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        jac = exp.jacobian_batch(samples)
        assert jac.shape == (nsamples, 2, 3)

    def test_hessian_batch_shape(self, bkd):
        """Test Hessian shape for nqoi=1."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        hess = exp.hessian_batch(samples)
        # Shape should be (nsamples, nvars, nvars) per CLAUDE.md convention
        assert hess.shape == (nsamples, 2, 2)

    def test_hessian_batch_not_available_for_multi_qoi(self, bkd):
        """Test hessian_batch not available for nqoi > 1."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        # hessian_batch method should not be bound for nqoi > 1
        assert not hasattr(exp, "hessian_batch")

    def test_basis_matrix_shape(self, bkd):
        """Test basis_matrix() returns correct shape."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        Phi = exp.basis_matrix(samples)

        assert Phi.shape == (nsamples, exp.nterms())

    def test_basis_matrix_matches_basis_call(self, bkd):
        """Test basis_matrix() returns same result as calling basis directly."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        Phi = exp.basis_matrix(samples)
        expected = exp.get_basis()(samples)

        bkd.assert_allclose(Phi, expected)

    def test_with_params_creates_independent_copy(self, bkd):
        """Test with_params() creates independent copy, original unchanged."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        original_coef = bkd.copy(exp.get_coefficients())
        new_coef = bkd.asarray(np.random.randn(exp.nterms(), 1))

        new_exp = exp.with_params(new_coef)

        # Different objects
        assert new_exp is not exp
        # Original unchanged
        bkd.assert_allclose(exp.get_coefficients(), original_coef)
        # New has updated coefficients
        bkd.assert_allclose(new_exp.get_coefficients(), new_coef)

    def test_with_params_preserves_properties(self, bkd):
        """Test with_params() preserves nvars, nterms, nqoi."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        new_coef = bkd.asarray(np.random.randn(exp.nterms(), 2))
        new_exp = exp.with_params(new_coef)

        assert new_exp.nvars() == exp.nvars()
        assert new_exp.nterms() == exp.nterms()
        assert new_exp.nqoi() == exp.nqoi()

    def test_with_params_evaluates_correctly(self, bkd):
        """Test with_params() result evaluates correctly."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Fit to known function
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y, (1, -1))
        exp.fit(samples, values)

        # Create new expansion with same coefficients
        new_exp = exp.with_params(exp.get_coefficients())

        # Should evaluate identically
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        bkd.assert_allclose(new_exp(test_samples), exp(test_samples))

    def test_jacobian_batch_finite_difference(self, bkd):
        """Test Jacobian accuracy via finite differences."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
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


class TestBasisExpansionFitting:
    """Test basis expansion fitting."""

    def _create_basis(self, bkd, nvars: int, max_level: int):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial(self, bkd):
        """Test fitting a polynomial function."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x,y) = x^2 + xy
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = bkd.reshape(x**2 + x * y, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + x_test * y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_with_solver(self, bkd):
        """Test fitting with explicit solver."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
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


class TestHermiteBasisExpansion:
    """Test BasisExpansion with Hermite polynomials.

    Hermite polynomials are orthonormal with respect to the standard
    Gaussian distribution. Tests use samples from the normal distribution.
    """

    def _create_hermite_basis(self, bkd, nvars: int, max_level: int):
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial_1d(self, bkd):
        """Test fitting a quadratic polynomial in 1D Hermite basis."""
        basis = self._create_hermite_basis(bkd, nvars=1, max_level=4)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x) = x^2 - 1
        # E[x^2] = 1 for standard normal, so E[f] = 0
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.randn(1, nsamples))
        x = samples[0, :]
        values = bkd.reshape(x**2 - 1.0, (1, -1))

        exp.fit(samples, values)

        # Test on new samples from the same distribution
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.randn(1, 50))
        x_test = test_samples[0, :]
        expected = bkd.reshape(x_test**2 - 1.0, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_2d(self, bkd):
        """Test fitting a quadratic polynomial in 2D Hermite basis."""
        basis = self._create_hermite_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^2 + y^2 + x*y (quadratic with interaction)
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.randn(2, nsamples))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2 + x * y, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.randn(2, 30))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + y_test**2 + x_test * y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_cubic_polynomial(self, bkd):
        """Test fitting a cubic polynomial in Hermite basis."""
        basis = self._create_hermite_basis(bkd, nvars=2, max_level=4)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^3 + x^2*y + x*y^2 (cubic with mixed terms)
        nsamples = 150
        np.random.seed(42)
        samples = bkd.asarray(np.random.randn(2, nsamples))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**3 + x**2 * y + x * y**2, (1, -1))

        exp.fit(samples, values)

        # Test
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.randn(2, 30))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(
            x_test**3 + x_test**2 * y_test + x_test * y_test**2, (1, -1)
        )
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_evaluation_shape(self, bkd):
        """Test evaluation returns correct shape."""
        basis = self._create_hermite_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=3)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 3)))

        nsamples = 10
        samples = bkd.asarray(np.random.randn(2, nsamples))
        values = exp(samples)
        assert values.shape == (3, nsamples)


class TestLaguerreBasisExpansion:
    """Test BasisExpansion with Laguerre polynomials.

    Laguerre polynomials are orthonormal with respect to the exponential
    distribution (gamma with shape=1, scale=1). Support is [0, infinity).
    """

    def _create_laguerre_basis(self, bkd, nvars: int, max_level: int):
        # Gamma(1, 1) = exponential distribution
        marginals = [GammaMarginal(1.0, 1.0, bkd=bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial_1d(self, bkd):
        """Test fitting a quadratic polynomial in 1D Laguerre basis."""
        basis = self._create_laguerre_basis(bkd, nvars=1, max_level=4)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x) = x^2 - 2*x + 1
        # For exponential distribution: E[x] = 1, E[x^2] = 2
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.exponential(1.0, (1, nsamples)))
        x = samples[0, :]
        values = bkd.reshape(x**2 - 2.0 * x + 1.0, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.exponential(1.0, (1, 50)))
        x_test = test_samples[0, :]
        expected = bkd.reshape(x_test**2 - 2.0 * x_test + 1.0, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_2d(self, bkd):
        """Test fitting a quadratic polynomial in 2D Laguerre basis."""
        basis = self._create_laguerre_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^2 + y^2 + x*y - 2 (quadratic with interaction)
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.exponential(1.0, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2 + x * y - 2.0, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.exponential(1.0, (2, 30)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + y_test**2 + x_test * y_test - 2.0, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_evaluation_shape(self, bkd):
        """Test evaluation returns correct shape."""
        basis = self._create_laguerre_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 10
        samples = bkd.asarray(np.random.exponential(1.0, (2, nsamples)))
        values = exp(samples)
        assert values.shape == (2, nsamples)


class TestJacobiBasisExpansion:
    """Test BasisExpansion with Jacobi polynomials.

    Jacobi polynomials with parameters (alpha, beta) are orthonormal with
    respect to the beta distribution on [-1, 1].
    """

    def _create_jacobi_basis(
        self, bkd, nvars: int, max_level: int, alpha: float = 0.5, beta: float = 1.0
    ):
        # BetaMarginal(a, b) on [0, 1] -> Jacobi(b-1, a-1) on [-1, 1]
        # For Jacobi(alpha, beta), use Beta(beta+1, alpha+1)
        marginals = [BetaMarginal(beta + 1.0, alpha + 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial_1d(self, bkd):
        """Test fitting a quadratic polynomial in 1D Jacobi basis."""
        basis = self._create_jacobi_basis(
            bkd, nvars=1, max_level=4, alpha=0.5, beta=0.5
        )
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x) = x^2 + x
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        x = samples[0, :]
        values = bkd.reshape(x**2 + x, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        x_test = test_samples[0, :]
        expected = bkd.reshape(x_test**2 + x_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_2d(self, bkd):
        """Test fitting a quadratic polynomial in 2D Jacobi basis."""
        basis = self._create_jacobi_basis(
            bkd, nvars=2, max_level=3, alpha=1.0, beta=2.0
        )
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^2 + y^2 + x*y (quadratic)
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2 + x * y, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + y_test**2 + x_test * y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_different_alpha_beta(self, bkd):
        """Test Jacobi with various alpha, beta parameters on quadratic."""
        # Test several (alpha, beta) combinations
        test_cases = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5), (0.5, 2.0)]

        for alpha, beta in test_cases:
            basis = self._create_jacobi_basis(
                bkd, nvars=1, max_level=3, alpha=alpha, beta=beta
            )
            exp = BasisExpansion(basis, bkd, nqoi=1)

            # Fit quadratic function f(x) = x^2 - 0.5
            nsamples = 50
            np.random.seed(42)
            samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
            x = samples[0, :]
            values = bkd.reshape(x**2 - 0.5, (1, -1))

            exp.fit(samples, values)

            # Test
            np.random.seed(123)
            test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
            x_test = test_samples[0, :]
            expected = bkd.reshape(x_test**2 - 0.5, (1, -1))
            predicted = exp(test_samples)

            bkd.assert_allclose(
                predicted,
                expected,
                rtol=1e-10,
                atol=1e-10,
            )

    def test_evaluation_shape(self, bkd):
        """Test evaluation returns correct shape."""
        basis = self._create_jacobi_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        assert values.shape == (2, nsamples)


class TestChebyshevBasisExpansion:
    """Test BasisExpansion with Chebyshev polynomials.

    Chebyshev polynomials of the first kind are orthonormal with respect
    to the arcsine distribution on [-1, 1] (Beta(0.5, 0.5)).
    Chebyshev polynomials of the second kind use Beta(1.5, 1.5).
    """

    def _create_chebyshev_basis(self, bkd, nvars: int, max_level: int, kind: int = 1):
        # Chebyshev 1st kind: Beta(0.5, 0.5) -> Jacobi(-0.5, -0.5)
        # Chebyshev 2nd kind: Beta(1.5, 1.5) -> Jacobi(0.5, 0.5)
        if kind == 1:
            marginals = [BetaMarginal(0.5, 0.5, bkd) for _ in range(nvars)]
        else:
            marginals = [BetaMarginal(1.5, 1.5, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_fit_polynomial_1d_first_kind(self, bkd):
        """Test fitting a quadratic polynomial in 1D Chebyshev 1st kind basis."""
        basis = self._create_chebyshev_basis(bkd, nvars=1, max_level=4, kind=1)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x) = x^2 + 0.5*x
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        x = samples[0, :]
        values = bkd.reshape(x**2 + 0.5 * x, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        x_test = test_samples[0, :]
        expected = bkd.reshape(x_test**2 + 0.5 * x_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_2d_first_kind(self, bkd):
        """Test fitting a quadratic polynomial in 2D Chebyshev 1st kind basis."""
        basis = self._create_chebyshev_basis(bkd, nvars=2, max_level=3, kind=1)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^2 + y^2 + x*y (quadratic with interaction)
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2 + x * y, (1, -1))

        exp.fit(samples, values)

        # Test
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + y_test**2 + x_test * y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_1d_second_kind(self, bkd):
        """Test fitting a quadratic polynomial in 1D Chebyshev 2nd kind basis."""
        basis = self._create_chebyshev_basis(bkd, nvars=1, max_level=4, kind=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x) = x^2 - 0.3*x
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        x = samples[0, :]
        values = bkd.reshape(x**2 - 0.3 * x, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        x_test = test_samples[0, :]
        expected = bkd.reshape(x_test**2 - 0.3 * x_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_2d_second_kind(self, bkd):
        """Test fitting a quadratic polynomial in 2D Chebyshev 2nd kind basis."""
        basis = self._create_chebyshev_basis(bkd, nvars=2, max_level=3, kind=2)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^2 + x*y + y^2 (quadratic with interaction)
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + x * y + y**2, (1, -1))

        exp.fit(samples, values)

        # Test
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + x_test * y_test + y_test**2, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_evaluation_shape_first_kind(self, bkd):
        """Test evaluation returns correct shape for 1st kind."""
        basis = self._create_chebyshev_basis(bkd, nvars=2, max_level=3, kind=1)
        exp = BasisExpansion(basis, bkd, nqoi=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        assert values.shape == (2, nsamples)

    def test_evaluation_shape_second_kind(self, bkd):
        """Test evaluation returns correct shape for 2nd kind."""
        basis = self._create_chebyshev_basis(bkd, nvars=2, max_level=3, kind=2)
        exp = BasisExpansion(basis, bkd, nqoi=3)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 3)))

        nsamples = 8
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        assert values.shape == (3, nsamples)


class TestMonomialBasisExpansion:
    """Test BasisExpansion with Monomial polynomials.

    Monomial polynomials {1, x, x^2, ...} provide a simple non-orthonormal basis.
    Tests use MultiIndexBasis directly (not OrthonormalPolynomialBasis).
    """

    def _create_monomial_basis(self, bkd, nvars: int, max_level: int):
        """Create a MultiIndexBasis with MonomialBasis1D univariate bases."""
        bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        # MultiIndexBasis is marked ABC but has no abstract methods
        basis = MultiIndexBasis.__new__(MultiIndexBasis)
        MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
        return basis

    def test_fit_polynomial_1d(self, bkd):
        """Test fitting a quadratic polynomial in 1D monomial basis."""
        basis = self._create_monomial_basis(bkd, nvars=1, max_level=4)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # Generate training data for f(x) = x^2 + 2x + 1
        nsamples = 50
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        x = samples[0, :]
        values = bkd.reshape(x**2 + 2.0 * x + 1.0, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 30)))
        x_test = test_samples[0, :]
        expected = bkd.reshape(x_test**2 + 2.0 * x_test + 1.0, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_polynomial_2d(self, bkd):
        """Test fitting a quadratic polynomial in 2D monomial basis."""
        basis = self._create_monomial_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^2 + y^2 + xy (quadratic with interaction)
        nsamples = 100
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2 + x * y, (1, -1))

        exp.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test**2 + y_test**2 + x_test * y_test, (1, -1))
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_fit_cubic_polynomial(self, bkd):
        """Test fitting a cubic polynomial in monomial basis."""
        basis = self._create_monomial_basis(bkd, nvars=2, max_level=4)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # f(x,y) = x^3 + x^2y + xy^2 (cubic with mixed terms)
        nsamples = 150
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**3 + x**2 * y + x * y**2, (1, -1))

        exp.fit(samples, values)

        # Test
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(
            x_test**3 + x_test**2 * y_test + x_test * y_test**2, (1, -1)
        )
        predicted = exp(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_evaluation_shape(self, bkd):
        """Test evaluation returns correct shape."""
        basis = self._create_monomial_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=3)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 3)))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = exp(samples)
        assert values.shape == (3, nsamples)

    def test_derivative_methods_available(self, bkd):
        """Test that jacobian_batch and hessian_batch are available."""
        basis = self._create_monomial_basis(bkd, nvars=2, max_level=3)
        exp = BasisExpansion(basis, bkd, nqoi=1)

        # MonomialBasis1D implements jacobian_batch and hessian_batch
        assert hasattr(exp, "jacobian_batch")
        assert hasattr(exp, "hessian_batch")


class TestPolynomialChaosExpansion:
    """Test PolynomialChaosExpansion class."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_mean_constant(self, bkd):
        """Test mean computation for constant function."""
        pce = self._create_pce(bkd, nvars=2, max_level=2)

        # Set coefficient for constant term
        coef = bkd.zeros((pce.nterms(), 1))
        const_idx = pce._get_constant_index()
        coef[const_idx, 0] = 5.0
        pce.set_coefficients(coef)

        mean = pce.mean()
        bkd.assert_allclose(mean, bkd.asarray([5.0]))

    def test_variance_single_term(self, bkd):
        """Test variance computation for single non-constant term."""
        pce = self._create_pce(bkd, nvars=2, max_level=2)

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

    def test_statistics_via_quadrature(self, bkd):
        """Test PCE statistics against Monte Carlo."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # Fit to f(x,y) = 1 + x + y + xy for uniform on [-1,1]^2
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        values = bkd.reshape(1.0 + x + y + x * y, (1, -1))

        pce.fit(samples, values)

        # Analytical statistics for this function:
        # Mean = E[1 + x + y + xy] = 1 + 0 + 0 + 0 = 1
        # Var = Var[x] + Var[y] + Var[xy] = 1/3 + 1/3 + 1/9 = 7/9
        mean = pce.mean()
        var = pce.variance()

        bkd.assert_allclose(mean, bkd.asarray([1.0]), atol=1e-10)
        bkd.assert_allclose(var, bkd.asarray([7.0 / 9.0]), atol=1e-10)

    def test_fit_via_projection(self, bkd):
        """Test spectral projection fitting."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
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


class TestPCESobolIndices:
    """Test PCE Sobol sensitivity indices."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_total_sobol_sum(self, bkd):
        """Test that total Sobol indices are valid."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # Fit to f(x,y) = x + y + xy
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        total_indices = pce.total_sobol_indices()

        # Total indices should each include main effect + interaction
        # Sum should be >= 1 (due to interactions counted twice)
        assert bkd.to_numpy(bkd.sum(total_indices)) >= 1.0 - 1e-10

        # Each total index should be positive
        assert np.all(bkd.to_numpy(total_indices) >= -1e-10)

    def test_main_effect_sum(self, bkd):
        """Test that main effect Sobol indices sum to <= 1."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()

        # Main effects should sum to <= 1
        assert bkd.to_numpy(bkd.sum(main_indices)) <= 1.0 + 1e-10

    def test_additive_function(self, bkd):
        """Test Sobol indices for additive function (no interactions)."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        # f(x,y) = x + 2*y (additive, no interactions)
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + 2 * y, (1, -1))
        pce.fit(samples, values)

        main_indices = pce.main_effect_sobol_indices()
        total_indices = pce.total_sobol_indices()

        # For additive function, main = total
        bkd.assert_allclose(main_indices, total_indices, atol=1e-10)

        # Main effects should sum to 1
        bkd.assert_allclose(
            bkd.reshape(bkd.sum(main_indices), (1,)), bkd.asarray([1.0]), atol=1e-10
        )


class TestPCEStatisticsFunctions:
    """Test standalone PCE statistics functions."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_functions_match_methods(self, bkd):
        """Test that standalone functions match PCE methods."""
        pce = self._create_pce(bkd, nvars=2, max_level=3)

        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x + y + x * y, (1, -1))
        pce.fit(samples, values)

        # Compare function outputs to method outputs
        bkd.assert_allclose(pce_statistics.mean(pce), pce.mean())
        bkd.assert_allclose(pce_statistics.variance(pce), pce.variance())
        bkd.assert_allclose(pce_statistics.std(pce), pce.std())
        bkd.assert_allclose(
            pce_statistics.total_sobol_indices(pce), pce.total_sobol_indices()
        )
        bkd.assert_allclose(
            pce_statistics.main_effect_sobol_indices(pce),
            pce.main_effect_sobol_indices(),
        )


class TestMixedBasisExpansion:
    """Test BasisExpansion with mixed polynomial bases."""

    def _create_mixed_pce(self, bkd, nqoi: int = 1):
        """Create PCE with Legendre, Hermite, and Laguerre bases."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),  # var 0: uniform
            GaussianMarginal(0.0, 1.0, bkd),  # var 1: gaussian
            GammaMarginal(2.0, 1.0, bkd=bkd),  # var 2: gamma(2,1)
        ]
        return create_pce_from_marginals(marginals, max_level=3, bkd=bkd, nqoi=nqoi)

    def test_mixed_basis_evaluation(self, bkd):
        """Test evaluation with mixed polynomial bases."""
        pce = self._create_mixed_pce(bkd, nqoi=2)

        # Set random coefficients
        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        # Generate samples in appropriate domains
        nsamples = 10
        samples = bkd.zeros((3, nsamples))
        samples[0, :] = bkd.asarray(np.random.uniform(-1, 1, nsamples))  # uniform
        samples[1, :] = bkd.asarray(np.random.randn(nsamples))  # gaussian
        samples[2, :] = bkd.asarray(np.random.exponential(1.0, nsamples))  # gamma

        values = pce(samples)
        assert values.shape == (2, nsamples)

    def test_mixed_basis_fitting(self, bkd):
        """Test fitting with mixed polynomial bases."""
        pce = self._create_mixed_pce(bkd, nqoi=1)

        # Generate training data: f(x,y,z) = x^2 + y + z
        nsamples = 100
        np.random.seed(42)
        samples = bkd.zeros((3, nsamples))
        x = bkd.asarray(np.random.uniform(-1, 1, nsamples))
        y = bkd.asarray(np.random.randn(nsamples))
        z = bkd.asarray(np.random.exponential(1.0, nsamples))
        samples[0, :] = x
        samples[1, :] = y
        samples[2, :] = z
        values = bkd.reshape(x**2 + y + z, (1, -1))

        pce.fit(samples, values)

        # Test on new samples
        np.random.seed(123)
        test_samples = bkd.zeros((3, 20))
        x_test = bkd.asarray(np.random.uniform(-1, 1, 20))
        y_test = bkd.asarray(np.random.randn(20))
        z_test = bkd.asarray(np.random.exponential(1.0, 20))
        test_samples[0, :] = x_test
        test_samples[1, :] = y_test
        test_samples[2, :] = z_test
        expected = bkd.reshape(x_test**2 + y_test + z_test, (1, -1))
        predicted = pce(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_create_pce_from_marginals_continuous(self, bkd):
        """Test create_pce_from_marginals with continuous distributions."""
        from pyapprox.probability.univariate import (
            GammaMarginal,
            GaussianMarginal,
            UniformMarginal,
        )
        from pyapprox.surrogates.affine.expansions.pce import (
            create_pce_from_marginals,
        )

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
            GammaMarginal(2.0, 1.0, bkd=bkd),
        ]
        pce = create_pce_from_marginals(marginals, max_level=3, bkd=bkd)

        # Verify correct polynomial types selected
        assert pce.nvars() == 3
        assert pce.nterms() > 0

        # Check that we can evaluate
        nsamples = 5
        samples = bkd.zeros((3, nsamples))
        samples[0, :] = bkd.asarray(np.random.uniform(-1, 1, nsamples))
        samples[1, :] = bkd.asarray(np.random.randn(nsamples))
        samples[2, :] = bkd.asarray(np.random.exponential(1.0, nsamples))
        values = pce(samples)
        assert values.shape == (1, nsamples)

    def test_create_pce_from_marginals_discrete(self, bkd):
        """Test create_pce_from_marginals with discrete distributions."""
        from scipy import stats

        from pyapprox.probability.univariate import ScipyDiscreteMarginal
        from pyapprox.surrogates.affine.expansions.pce import (
            create_pce_from_marginals,
        )

        marginals = [
            ScipyDiscreteMarginal(
                stats.poisson(mu=3.0), bkd
            ),  # -> CharlierPolynomial1D
            ScipyDiscreteMarginal(stats.binom(n=10, p=0.3), bkd),  # -> DiscreteNumeric
        ]
        pce = create_pce_from_marginals(marginals, max_level=3, bkd=bkd)

        # Verify correct polynomial types
        assert pce.nvars() == 2
        assert pce.nterms() > 0

        # Check that we can evaluate
        nsamples = 5
        samples = bkd.zeros((2, nsamples))
        # Sample from Poisson and binomial
        samples[0, :] = bkd.asarray(np.random.poisson(3.0, nsamples).astype(float))
        samples[1, :] = bkd.asarray(np.random.binomial(10, 0.3, nsamples).astype(float))
        values = pce(samples)
        assert values.shape == (1, nsamples)

    def test_discrete_orthonormality_exact(self, bkd):
        """Test orthonormality of discrete polynomials using exact evaluation.

        For discrete distributions, evaluate at all mass points weighted by
        probabilities to verify orthonormality:
        sum_k p_i(x_k) * p_j(x_k) * P(X=x_k) ~ delta_ij

        Note: For unbounded distributions like Poisson, we can only capture
        a finite interval of mass points, so orthonormality degrades slightly
        for higher polynomial degrees. We test with fewer terms and use a
        more relaxed tolerance for Poisson.
        """
        from scipy import stats

        from pyapprox.probability.univariate import ScipyDiscreteMarginal
        from pyapprox.surrogates.affine.expansions.pce import (
            create_pce_from_marginals,
        )

        # Test Charlier polynomial orthonormality (Poisson)
        # For unbounded distributions, use fewer terms since we can't capture
        # all mass points, which causes orthonormality to degrade
        poisson_marginal = ScipyDiscreteMarginal(stats.poisson(mu=3.0), bkd)
        pce_poisson = create_pce_from_marginals(
            [poisson_marginal], max_level=3, bkd=bkd
        )
        basis_1d = pce_poisson._basis._bases_1d[0]
        nterms = 4  # Use fewer terms for better orthonormality with truncated support
        basis_1d.set_nterms(nterms)

        # Get all mass points for Poisson (use interval that captures 1-1e-8 of mass)
        interval = poisson_marginal.interval(1 - 1e-8)  # Shape: (1, 2)
        lo, hi = int(interval[0, 0]), int(interval[0, 1])
        xk = bkd.arange(lo, hi + 1, dtype=bkd.default_dtype())
        pk = poisson_marginal(bkd.reshape(xk, (1, -1)))[0, :]  # 1D probabilities

        # Evaluate polynomials at all mass points
        samples = bkd.reshape(xk, (1, -1))  # Shape: (1, nmasses)
        vals = basis_1d(samples)  # Shape: (nmasses, nterms)

        # Compute weighted Grammian: G_ij = sum_k vals[k,i] * vals[k,j] * pk[k]
        weighted_vals = vals * bkd.reshape(pk, (-1, 1))  # Broadcast weights
        grammian = vals.T @ weighted_vals
        expected = bkd.eye(nterms)
        # Use relaxed tolerance due to truncated support for unbounded distribution
        bkd.assert_allclose(grammian, expected, rtol=1e-3, atol=1e-4)

    def test_discrete_numeric_orthonormality_exact(self, bkd):
        """Test orthonormality of numeric discrete polynomials using exact evaluation.

        For distributions that use DiscreteNumericOrthonormalPolynomial1D (like
        binomial),
        verify orthonormality by evaluating at all mass points.
        """
        from scipy import stats

        from pyapprox.probability.univariate import ScipyDiscreteMarginal
        from pyapprox.surrogates.affine.expansions.pce import (
            create_pce_from_marginals,
        )

        # Test binomial (uses DiscreteNumericOrthonormalPolynomial1D)
        binom_marginal = ScipyDiscreteMarginal(stats.binom(n=10, p=0.3), bkd)
        pce_binom = create_pce_from_marginals([binom_marginal], max_level=4, bkd=bkd)
        basis_1d = pce_binom._basis._bases_1d[0]
        nterms = 5
        basis_1d.set_nterms(nterms)

        # For binomial(n=10), mass points are exactly {0, 1, ..., 10}
        xk = bkd.arange(0, 11, dtype=bkd.default_dtype())
        pk = binom_marginal(bkd.reshape(xk, (1, -1)))[0, :]  # 1D probabilities

        # Evaluate polynomials at all mass points
        samples = bkd.reshape(xk, (1, -1))  # Shape: (1, 11)
        vals = basis_1d(samples)  # Shape: (11, nterms)

        # Compute weighted Grammian
        weighted_vals = vals * bkd.reshape(pk, (-1, 1))
        grammian = vals.T @ weighted_vals
        expected = bkd.eye(nterms)
        bkd.assert_allclose(grammian, expected, rtol=1e-8, atol=1e-10)

    def test_mixed_basis_jacobian_batch(self, bkd):
        """Test jacobian_batch with mixed bases."""
        pce = self._create_mixed_pce(bkd, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        nsamples = 5
        samples = bkd.zeros((3, nsamples))
        samples[0, :] = bkd.asarray(np.random.uniform(-0.9, 0.9, nsamples))
        samples[1, :] = bkd.asarray(np.random.randn(nsamples))
        samples[2, :] = bkd.asarray(np.random.exponential(1.0, nsamples))

        jac = pce.jacobian_batch(samples)
        assert jac.shape == (nsamples, 2, 3)

    def test_mixed_basis_hessian_batch(self, bkd):
        """Test hessian_batch with mixed bases (nqoi=1)."""
        pce = self._create_mixed_pce(bkd, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        nsamples = 5
        samples = bkd.zeros((3, nsamples))
        samples[0, :] = bkd.asarray(np.random.uniform(-0.9, 0.9, nsamples))
        samples[1, :] = bkd.asarray(np.random.randn(nsamples))
        samples[2, :] = bkd.asarray(np.random.exponential(1.0, nsamples))

        hess = pce.hessian_batch(samples)
        assert hess.shape == (nsamples, 3, 3)
