"""Tests for BasisExpansion derivative implementations.

Tests validate Jacobian and Hessian computations for BasisExpansion
using DerivativeChecker per CLAUDE.md convention. Tests cover:
- Legendre polynomials (uniform distribution)
- Hermite polynomials (Gaussian distribution)
- Laguerre polynomials (gamma distribution)
- Jacobi polynomials (general beta distribution)
- Chebyshev polynomials (arcsine distribution)
- Multiple dimensions (1D, 2D, 3D)
- Single and batch derivative methods
"""

import numpy as np
import pytest
import torch

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    BatchDerivativeChecker,
    DerivativeChecker,
)
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
    create_pce_from_marginals,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import (
    MonomialBasis1D,
    create_bases_1d,
)
from pyapprox.util.backends.torch import TorchBkd
from tests._helpers.markers import slow_test


class TestDerivativeCheckerLegendre:
    """Test derivatives for Legendre polynomial expansion."""

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_jacobian_1d(self, bkd):
        """Test Jacobian for 1D Legendre expansion."""
        pce = self._create_pce(bkd, nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_2d(self, bkd):
        """Test Jacobian for 2D Legendre expansion."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_3d(self, bkd):
        """Test Jacobian for 3D Legendre expansion."""
        pce = self._create_pce(bkd, nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    @pytest.mark.slow_on("TorchBkd")
    def test_hessian_1d(self, bkd):
        """Test Hessian for 1D Legendre expansion."""
        pce = self._create_pce(bkd, nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_hessian_2d(self, bkd):
        """Test Hessian for 2D Legendre expansion."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_hessian_3d(self, bkd):
        """Test Hessian for 3D Legendre expansion."""
        pce = self._create_pce(bkd, nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_jacobian_batch(self, bkd):
        """Test jacobian_batch for Legendre expansion."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6

    def test_hessian_batch(self, bkd):
        """Test hessian_batch for Legendre expansion."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6


class TestDerivativeCheckerHermite:
    """Test derivatives for Hermite polynomial expansion.

    Hermite polynomials are orthonormal with respect to the standard
    normal distribution (Gaussian). Tests use samples from a narrower
    range to avoid numerical issues with large polynomial values.
    """

    def _create_hermite_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self, bkd):
        """Test Jacobian for 1D Hermite expansion."""
        exp = self._create_hermite_expansion(bkd, nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        # Use samples in narrower range for Hermite
        sample = bkd.asarray(np.random.uniform(-2.0, 2.0, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_2d(self, bkd):
        """Test Jacobian for 2D Hermite expansion."""
        exp = self._create_hermite_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-2.0, 2.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_2d(self, bkd):
        """Test Hessian for 2D Hermite expansion."""
        exp = self._create_hermite_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-2.0, 2.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_jacobian_batch(self, bkd):
        """Test jacobian_batch for Hermite expansion."""
        exp = self._create_hermite_expansion(bkd, nvars=2, max_level=3, nqoi=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-2.0, 2.0, (2, nsamples)))

        checker = BatchDerivativeChecker(exp, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6


class TestDerivativeCheckerLaguerre:
    """Test derivatives for Laguerre polynomial expansion.

    Laguerre polynomials are orthonormal with respect to the exponential
    distribution (gamma with shape=1). Tests use samples from [0.1, 5.0]
    to stay in the support while avoiding boundary issues.
    """

    def _create_laguerre_expansion(
        self, bkd, nvars: int, max_level: int, nqoi: int = 1
    ):
        marginals = [GammaMarginal(1.0, 1.0, bkd=bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self, bkd):
        """Test Jacobian for 1D Laguerre expansion."""
        exp = self._create_laguerre_expansion(bkd, nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        # Laguerre support is [0, inf), use positive samples
        sample = bkd.asarray(np.random.uniform(0.1, 5.0, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_2d(self, bkd):
        """Test Jacobian for 2D Laguerre expansion."""
        exp = self._create_laguerre_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(0.1, 5.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_2d(self, bkd):
        """Test Hessian for 2D Laguerre expansion."""
        exp = self._create_laguerre_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(0.1, 5.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6


class TestDerivativeCheckerJacobi:
    """Test derivatives for Jacobi polynomial expansion.

    Jacobi polynomials with parameters (alpha, beta) are orthonormal
    with respect to the beta distribution. Tests use alpha=0.5, beta=1.0.
    """

    def _create_jacobi_expansion(
        self, bkd, nvars: int, max_level: int, alpha: float, beta: float, nqoi: int = 1
    ):
        # BetaMarginal(a, b) on [0, 1] -> Jacobi(b-1, a-1) on [-1, 1]
        # For Jacobi(alpha, beta), use Beta(beta+1, alpha+1)
        marginals = [BetaMarginal(beta + 1.0, alpha + 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self, bkd):
        """Test Jacobian for 1D Jacobi expansion."""
        exp = self._create_jacobi_expansion(
            bkd, nvars=1, max_level=5, alpha=0.5, beta=1.0, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_2d(self, bkd):
        """Test Jacobian for 2D Jacobi expansion."""
        exp = self._create_jacobi_expansion(
            bkd, nvars=2, max_level=3, alpha=0.5, beta=1.0, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_2d(self, bkd):
        """Test Hessian for 2D Jacobi expansion."""
        exp = self._create_jacobi_expansion(
            bkd, nvars=2, max_level=3, alpha=0.5, beta=1.0, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_different_alpha_beta(self, bkd):
        """Test Jacobian for Jacobi with different alpha, beta."""
        exp = self._create_jacobi_expansion(
            bkd, nvars=2, max_level=3, alpha=2.0, beta=0.5, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6


class TestDerivativeCheckerChebyshev:
    """Test derivatives for Chebyshev polynomial expansion.

    Chebyshev polynomials (1st kind) are orthonormal with respect to
    the arcsine distribution on [-1, 1] (Beta(0.5, 0.5)).
    """

    def _create_chebyshev_expansion(
        self, bkd, nvars: int, max_level: int, nqoi: int = 1, kind: int = 1
    ):
        # Chebyshev 1st kind: Beta(0.5, 0.5) -> Jacobi(-0.5, -0.5)
        # Chebyshev 2nd kind: Beta(1.5, 1.5) -> Jacobi(0.5, 0.5)
        if kind == 1:
            marginals = [BetaMarginal(0.5, 0.5, bkd) for _ in range(nvars)]
        else:
            marginals = [BetaMarginal(1.5, 1.5, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d_first_kind(self, bkd):
        """Test Jacobian for 1D Chebyshev 1st kind expansion."""
        exp = self._create_chebyshev_expansion(
            bkd, nvars=1, max_level=5, nqoi=1, kind=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_2d_first_kind(self, bkd):
        """Test Jacobian for 2D Chebyshev 1st kind expansion."""
        exp = self._create_chebyshev_expansion(
            bkd, nvars=2, max_level=3, nqoi=1, kind=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_2d_first_kind(self, bkd):
        """Test Hessian for 2D Chebyshev 1st kind expansion."""
        exp = self._create_chebyshev_expansion(
            bkd, nvars=2, max_level=3, nqoi=1, kind=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_jacobian_2d_second_kind(self, bkd):
        """Test Jacobian for 2D Chebyshev 2nd kind expansion."""
        exp = self._create_chebyshev_expansion(
            bkd, nvars=2, max_level=3, nqoi=1, kind=2
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_2d_second_kind(self, bkd):
        """Test Hessian for 2D Chebyshev 2nd kind expansion."""
        exp = self._create_chebyshev_expansion(
            bkd, nvars=2, max_level=3, nqoi=1, kind=2
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6


class TestDerivativeCheckerMultiQoi:
    """Test derivatives for multi-QoI expansions.

    These tests verify derivatives work correctly for expansions with
    multiple quantities of interest (nqoi > 1).
    """

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_jacobian_batch_multi_qoi(self, bkd):
        """Test jacobian_batch for multi-QoI expansion."""
        pce = self._create_pce(bkd, nvars=2, max_level=3, nqoi=3)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 3)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6

    def test_jacobian_batch_3d_multi_qoi(self, bkd):
        """Test jacobian_batch for 3D multi-QoI expansion."""
        pce = self._create_pce(bkd, nvars=3, max_level=2, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        nsamples = 4
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6


class TestDerivativeCheckerMonomial:
    """Test derivatives for Monomial basis expansion.

    Monomial polynomials {1, x, x^2, ...} provide a non-orthonormal basis.
    Tests use MultiIndexBasis directly (not OrthonormalPolynomialBasis).
    """

    def _create_monomial_expansion(
        self, bkd, nvars: int, max_level: int, nqoi: int = 1
    ):
        """Create a BasisExpansion with MonomialBasis1D univariate bases."""
        bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        # MultiIndexBasis is marked ABC but has no abstract methods
        basis = MultiIndexBasis.__new__(MultiIndexBasis)
        MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self, bkd):
        """Test Jacobian for 1D Monomial expansion."""
        exp = self._create_monomial_expansion(bkd, nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_2d(self, bkd):
        """Test Jacobian for 2D Monomial expansion."""
        exp = self._create_monomial_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_3d(self, bkd):
        """Test Jacobian for 3D Monomial expansion."""
        exp = self._create_monomial_expansion(bkd, nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_1d(self, bkd):
        """Test Hessian for 1D Monomial expansion."""
        exp = self._create_monomial_expansion(bkd, nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_hessian_2d(self, bkd):
        """Test Hessian for 2D Monomial expansion."""
        exp = self._create_monomial_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_hessian_3d(self, bkd):
        """Test Hessian for 3D Monomial expansion."""
        exp = self._create_monomial_expansion(bkd, nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_jacobian_batch(self, bkd):
        """Test batch Jacobian via BatchDerivativeChecker."""
        exp = self._create_monomial_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(exp, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6

    def test_hessian_batch(self, bkd):
        """Test batch Hessian via BatchDerivativeChecker."""
        exp = self._create_monomial_expansion(bkd, nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(exp, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6


class TestMixedBasisDerivatives:
    """Test derivative checking with mixed polynomial bases.

    Tests validate Jacobian and Hessian computations for PCE with
    different polynomial types per variable (e.g., Legendre + Hermite).
    """

    def _create_mixed_pce(self, bkd, nqoi: int = 1):
        """Create PCE with Legendre and Hermite bases."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        return create_pce_from_marginals(marginals, max_level=3, bkd=bkd, nqoi=nqoi)

    def test_jacobian_with_derivative_checker(self, bkd):
        """Validate jacobian for mixed basis using DerivativeChecker."""
        pce = self._create_mixed_pce(bkd, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        # Legendre: [-1, 1], Hermite: unbounded but use moderate range
        sample = bkd.asarray([[0.3], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_hessian_with_derivative_checker(self, bkd):
        """Validate hessian for mixed basis using DerivativeChecker."""
        pce = self._create_mixed_pce(bkd, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray([[0.3], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        assert float(hess_error) < 1e-6

    def test_jacobian_batch_with_derivative_checker(self, bkd):
        """Validate jacobian_batch for mixed basis using BatchDerivativeChecker."""
        pce = self._create_mixed_pce(bkd, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 5)))
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6

    def test_hessian_batch_with_derivative_checker(self, bkd):
        """Validate hessian_batch for mixed basis using BatchDerivativeChecker."""
        pce = self._create_mixed_pce(bkd, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 5)))
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6

    def test_jacobian_batch_multi_qoi(self, bkd):
        """Validate jacobian_batch for mixed basis with multiple QoIs."""
        pce = self._create_mixed_pce(bkd, nqoi=3)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 3)))

        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 5)))
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        assert float(error_ratio) < 1e-6


class TestJacobianWrtParams:
    """Test jacobian_wrt_params for BasisExpansion.

    Tests validate that jacobian_wrt_params (derivatives w.r.t. active coefficients)
    matches finite differences using DerivativeChecker.

    Note: jacobian_wrt_params returns Jacobian w.r.t. ACTIVE params only.
    By default all params are active, so nactive_params == nparams.
    """

    def _create_pce(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    @slow_test
    def test_jacobian_wrt_params_1d(self, bkd):
        """Test jacobian_wrt_params for 1D expansion with nqoi=1."""
        pce = self._create_pce(bkd, nvars=1, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Evaluate at a fixed sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 3)))

        # Create wrapper: params -> output values
        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        # By default all params are active
        nactive = pce.nactive_params()

        def fun(params):
            # params shape: (nactive_params, 1)
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            return pce(samples).T  # (nsamples, nqoi)

        def jacobian_func(params):
            # params shape: (nactive_params, 1)
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            # jacobian_wrt_params returns (nsamples, nqoi, nactive_params)
            jac = pce.jacobian_wrt_params(samples)
            # Sum over samples to get (nqoi, nactive_params)
            return bkd.sum(jac, axis=0)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=pce.nqoi(),
            nvars=nactive,
            fun=lambda p: bkd.sum(fun(p), axis=0, keepdims=True).T,  # (nqoi, 1)
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = pce.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_wrt_params_2d(self, bkd):
        """Test jacobian_wrt_params for 2D expansion with nqoi=1."""
        pce = self._create_pce(bkd, nvars=2, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Evaluate at a fixed sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        nactive = pce.nactive_params()

        def fun(params):
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            return pce(samples).T  # (nsamples, nqoi)

        def jacobian_func(params):
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            jac = pce.jacobian_wrt_params(samples)  # (nsamples, nqoi, nactive_params)
            return jac[0, :, :]  # (nqoi, nactive_params) for single sample

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=pce.nqoi(),
            nvars=nactive,
            fun=lambda p: fun(p),
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = pce.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_wrt_params_multi_qoi(self, bkd):
        """Test jacobian_wrt_params for expansion with nqoi > 1."""
        pce = self._create_pce(bkd, nvars=2, max_level=2, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        # Evaluate at a single sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        nactive = pce.nactive_params()

        def fun(params):
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            # pce(samples) returns (nqoi, nsamples)
            # For FunctionFromCallable we need (nqoi, 1) output
            return pce(samples)  # (nqoi, nsamples=1)

        def jacobian_func(params):
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            jac = pce.jacobian_wrt_params(samples)  # (1, nqoi, nactive_params)
            return jac[0, :, :]  # (nqoi, nactive_params)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=pce.nqoi(),
            nvars=nactive,
            fun=fun,
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = pce.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_wrt_params_with_fixed_params(self, bkd):
        """Test jacobian_wrt_params when some params are fixed (inactive).

        Verifies that jacobian only includes active parameters and that
        the derivative check passes for the reduced parameter space.
        """
        pce = self._create_pce(bkd, nvars=2, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Fix the first 2 parameters (make them inactive)
        # Only optimize the remaining parameters
        nparams_total = pce.nparams()
        active_indices = bkd.arange(2, nparams_total)
        pce.hyp_list().set_active_indices(active_indices)

        nactive = pce.nactive_params()
        assert nactive == nparams_total - 2

        # Evaluate at a single sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        def fun(params):
            # params shape: (nactive_params, 1)
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            return pce(samples).T  # (nsamples, nqoi)

        def jacobian_func(params):
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            jac = pce.jacobian_wrt_params(samples)  # (nsamples, nqoi, nactive_params)
            return jac[0, :, :]  # (nqoi, nactive_params) for single sample

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=pce.nqoi(),
            nvars=nactive,
            fun=lambda p: fun(p),
            jacobian=jacobian_func,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        params = pce.hyp_list().get_active_values()
        sample_params = bkd.reshape(params, (nactive, 1))
        errors = checker.check_derivatives(sample_params, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

        # Verify the jacobian shape is correct (only active params)
        jac = pce.jacobian_wrt_params(samples)
        assert jac.shape == (1, 1, nactive)


class TestJacobianWrtParamsAutograd:
    """Torch-only tests for jacobian_wrt_params using torch.autograd."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_jacobian_wrt_params_autograd(self):
        """Verify jacobian_wrt_params matches torch autograd."""
        from torch.autograd.functional import jacobian as torch_jacobian

        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Evaluate at multiple samples
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))

        # Get analytical jacobian (w.r.t. active params)
        analytical_jac = pce.jacobian_wrt_params(samples)  # (nsamples, nqoi, nactive)
        nactive = pce.nactive_params()

        # Get autograd jacobian
        def output_from_params(params: torch.Tensor) -> torch.Tensor:
            # params shape: (nactive_params,)
            pce.hyp_list().set_active_values(params)
            pce._sync_from_hyp_list()
            return pce(samples).flatten()  # (nsamples * nqoi,)

        params = pce.hyp_list().get_active_values()
        autograd_jac = torch_jacobian(output_from_params, params)
        # autograd_jac shape: (nsamples * nqoi, nactive_params)

        # Reshape analytical to match autograd
        nsamples = samples.shape[1]
        analytical_flat = bkd.reshape(analytical_jac, (nsamples * pce.nqoi(), nactive))

        bkd.assert_allclose(analytical_flat, autograd_jac, rtol=1e-10)

    def test_jacobian_wrt_params_autograd_multi_qoi(self):
        """Verify jacobian_wrt_params matches torch autograd for multi-QoI."""
        from torch.autograd.functional import jacobian as torch_jacobian

        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=2, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        # Evaluate at multiple samples
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 3)))

        # Get analytical jacobian (w.r.t. active params)
        analytical_jac = pce.jacobian_wrt_params(samples)  # (nsamples, nqoi, nactive)
        nactive = pce.nactive_params()

        # Get autograd jacobian
        def output_from_params(params: torch.Tensor) -> torch.Tensor:
            # params shape: (nactive_params,)
            pce.hyp_list().set_active_values(params)
            pce._sync_from_hyp_list()
            # PCE output is (nqoi, nsamples), need to flatten properly
            # Order should match analytical: sample-major, then qoi
            return pce(samples).T.flatten()  # (nsamples, nqoi) flattened

        params = pce.hyp_list().get_active_values()
        autograd_jac = torch_jacobian(output_from_params, params)
        # autograd_jac shape: (nsamples * nqoi, nactive_params)

        # Reshape analytical to match autograd
        nsamples = samples.shape[1]
        analytical_flat = bkd.reshape(analytical_jac, (nsamples * pce.nqoi(), nactive))

        bkd.assert_allclose(analytical_flat, autograd_jac, rtol=1e-10)
