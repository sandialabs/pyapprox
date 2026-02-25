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

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestDerivativeCheckerLegendre(Generic[Array], unittest.TestCase):
    """Test derivatives for Legendre polynomial expansion."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_jacobian_1d(self):
        """Test Jacobian for 1D Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_2d(self):
        """Test Jacobian for 2D Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_3d(self):
        """Test Jacobian for 3D Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_1d(self):
        """Test Hessian for 1D Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_hessian_2d(self):
        """Test Hessian for 2D Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_hessian_3d(self):
        """Test Hessian for 3D Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_jacobian_batch(self):
        """Test jacobian_batch for Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)

    def test_hessian_batch(self):
        """Test hessian_batch for Legendre expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)


class TestDerivativeCheckerLegendreNumpy(TestDerivativeCheckerLegendre[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerLegendreTorch(TestDerivativeCheckerLegendre[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeCheckerHermite(Generic[Array], unittest.TestCase):
    """Test derivatives for Hermite polynomial expansion.

    Hermite polynomials are orthonormal with respect to the standard
    normal distribution (Gaussian). Tests use samples from a narrower
    range to avoid numerical issues with large polynomial values.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_hermite_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self):
        """Test Jacobian for 1D Hermite expansion."""
        bkd = self._bkd
        exp = self._create_hermite_expansion(nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        # Use samples in narrower range for Hermite
        sample = bkd.asarray(np.random.uniform(-2.0, 2.0, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_2d(self):
        """Test Jacobian for 2D Hermite expansion."""
        bkd = self._bkd
        exp = self._create_hermite_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-2.0, 2.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_2d(self):
        """Test Hessian for 2D Hermite expansion."""
        bkd = self._bkd
        exp = self._create_hermite_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-2.0, 2.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_jacobian_batch(self):
        """Test jacobian_batch for Hermite expansion."""
        bkd = self._bkd
        exp = self._create_hermite_expansion(nvars=2, max_level=3, nqoi=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 2)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-2.0, 2.0, (2, nsamples)))

        checker = BatchDerivativeChecker(exp, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)


class TestDerivativeCheckerHermiteNumpy(TestDerivativeCheckerHermite[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerHermiteTorch(TestDerivativeCheckerHermite[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeCheckerLaguerre(Generic[Array], unittest.TestCase):
    """Test derivatives for Laguerre polynomial expansion.

    Laguerre polynomials are orthonormal with respect to the exponential
    distribution (gamma with shape=1). Tests use samples from [0.1, 5.0]
    to stay in the support while avoiding boundary issues.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_laguerre_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [GammaMarginal(1.0, 1.0, bkd=bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self):
        """Test Jacobian for 1D Laguerre expansion."""
        bkd = self._bkd
        exp = self._create_laguerre_expansion(nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        # Laguerre support is [0, inf), use positive samples
        sample = bkd.asarray(np.random.uniform(0.1, 5.0, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_2d(self):
        """Test Jacobian for 2D Laguerre expansion."""
        bkd = self._bkd
        exp = self._create_laguerre_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(0.1, 5.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_2d(self):
        """Test Hessian for 2D Laguerre expansion."""
        bkd = self._bkd
        exp = self._create_laguerre_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(0.1, 5.0, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)


class TestDerivativeCheckerLaguerreNumpy(TestDerivativeCheckerLaguerre[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerLaguerreTorch(TestDerivativeCheckerLaguerre[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeCheckerJacobi(Generic[Array], unittest.TestCase):
    """Test derivatives for Jacobi polynomial expansion.

    Jacobi polynomials with parameters (alpha, beta) are orthonormal
    with respect to the beta distribution. Tests use alpha=0.5, beta=1.0.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_jacobi_expansion(
        self, nvars: int, max_level: int, alpha: float, beta: float, nqoi: int = 1
    ):
        bkd = self._bkd
        # BetaMarginal(a, b) on [0, 1] -> Jacobi(b-1, a-1) on [-1, 1]
        # For Jacobi(alpha, beta), use Beta(beta+1, alpha+1)
        marginals = [BetaMarginal(beta + 1.0, alpha + 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self):
        """Test Jacobian for 1D Jacobi expansion."""
        bkd = self._bkd
        exp = self._create_jacobi_expansion(
            nvars=1, max_level=5, alpha=0.5, beta=1.0, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_2d(self):
        """Test Jacobian for 2D Jacobi expansion."""
        bkd = self._bkd
        exp = self._create_jacobi_expansion(
            nvars=2, max_level=3, alpha=0.5, beta=1.0, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_2d(self):
        """Test Hessian for 2D Jacobi expansion."""
        bkd = self._bkd
        exp = self._create_jacobi_expansion(
            nvars=2, max_level=3, alpha=0.5, beta=1.0, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_different_alpha_beta(self):
        """Test Jacobian for Jacobi with different alpha, beta."""
        bkd = self._bkd
        exp = self._create_jacobi_expansion(
            nvars=2, max_level=3, alpha=2.0, beta=0.5, nqoi=1
        )

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)


class TestDerivativeCheckerJacobiNumpy(TestDerivativeCheckerJacobi[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerJacobiTorch(TestDerivativeCheckerJacobi[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeCheckerChebyshev(Generic[Array], unittest.TestCase):
    """Test derivatives for Chebyshev polynomial expansion.

    Chebyshev polynomials (1st kind) are orthonormal with respect to
    the arcsine distribution on [-1, 1] (Beta(0.5, 0.5)).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_chebyshev_expansion(
        self, nvars: int, max_level: int, nqoi: int = 1, kind: int = 1
    ):
        bkd = self._bkd
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

    def test_jacobian_1d_first_kind(self):
        """Test Jacobian for 1D Chebyshev 1st kind expansion."""
        bkd = self._bkd
        exp = self._create_chebyshev_expansion(nvars=1, max_level=5, nqoi=1, kind=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_2d_first_kind(self):
        """Test Jacobian for 2D Chebyshev 1st kind expansion."""
        bkd = self._bkd
        exp = self._create_chebyshev_expansion(nvars=2, max_level=3, nqoi=1, kind=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_2d_first_kind(self):
        """Test Hessian for 2D Chebyshev 1st kind expansion."""
        bkd = self._bkd
        exp = self._create_chebyshev_expansion(nvars=2, max_level=3, nqoi=1, kind=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_jacobian_2d_second_kind(self):
        """Test Jacobian for 2D Chebyshev 2nd kind expansion."""
        bkd = self._bkd
        exp = self._create_chebyshev_expansion(nvars=2, max_level=3, nqoi=1, kind=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_2d_second_kind(self):
        """Test Hessian for 2D Chebyshev 2nd kind expansion."""
        bkd = self._bkd
        exp = self._create_chebyshev_expansion(nvars=2, max_level=3, nqoi=1, kind=2)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)


class TestDerivativeCheckerChebyshevNumpy(TestDerivativeCheckerChebyshev[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerChebyshevTorch(TestDerivativeCheckerChebyshev[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeCheckerMultiQoi(Generic[Array], unittest.TestCase):
    """Test derivatives for multi-QoI expansions.

    These tests verify derivatives work correctly for expansions with
    multiple quantities of interest (nqoi > 1).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_jacobian_batch_multi_qoi(self):
        """Test jacobian_batch for multi-QoI expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=3, nqoi=3)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 3)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)

    def test_jacobian_batch_3d_multi_qoi(self):
        """Test jacobian_batch for 3D multi-QoI expansion."""
        bkd = self._bkd
        pce = self._create_pce(nvars=3, max_level=2, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        nsamples = 4
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, nsamples)))

        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)


class TestDerivativeCheckerMultiQoiNumpy(TestDerivativeCheckerMultiQoi[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerMultiQoiTorch(TestDerivativeCheckerMultiQoi[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDerivativeCheckerMonomial(Generic[Array], unittest.TestCase):
    """Test derivatives for Monomial basis expansion.

    Monomial polynomials {1, x, x², ...} provide a non-orthonormal basis.
    Tests use MultiIndexBasis directly (not OrthonormalPolynomialBasis).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_monomial_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create a BasisExpansion with MonomialBasis1D univariate bases."""
        bkd = self._bkd
        bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        # MultiIndexBasis is marked ABC but has no abstract methods
        basis = MultiIndexBasis.__new__(MultiIndexBasis)
        MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_jacobian_1d(self):
        """Test Jacobian for 1D Monomial expansion."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_2d(self):
        """Test Jacobian for 2D Monomial expansion."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_3d(self):
        """Test Jacobian for 3D Monomial expansion."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_1d(self):
        """Test Hessian for 1D Monomial expansion."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=1, max_level=5, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_hessian_2d(self):
        """Test Hessian for 2D Monomial expansion."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_hessian_3d(self):
        """Test Hessian for 3D Monomial expansion."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=3, max_level=2, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        checker = DerivativeChecker(exp)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (3, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_jacobian_batch(self):
        """Test batch Jacobian via BatchDerivativeChecker."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(exp, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)

    def test_hessian_batch(self):
        """Test batch Hessian via BatchDerivativeChecker."""
        bkd = self._bkd
        exp = self._create_monomial_expansion(nvars=2, max_level=3, nqoi=1)

        np.random.seed(42)
        exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))

        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        checker = BatchDerivativeChecker(exp, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)


class TestDerivativeCheckerMonomialNumpy(TestDerivativeCheckerMonomial[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDerivativeCheckerMonomialTorch(TestDerivativeCheckerMonomial[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestMixedBasisDerivatives(Generic[Array], unittest.TestCase):
    """Test derivative checking with mixed polynomial bases.

    Tests validate Jacobian and Hessian computations for PCE with
    different polynomial types per variable (e.g., Legendre + Hermite).
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_mixed_pce(self, nqoi: int = 1):
        """Create PCE with Legendre and Hermite bases."""
        bkd = self._bkd
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        return create_pce_from_marginals(marginals, max_level=3, bkd=bkd, nqoi=nqoi)

    def test_jacobian_with_derivative_checker(self):
        """Validate jacobian for mixed basis using DerivativeChecker."""
        bkd = self._bkd
        pce = self._create_mixed_pce(nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        # Legendre: [-1, 1], Hermite: unbounded but use moderate range
        sample = bkd.asarray([[0.3], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)

        jac_error = checker.error_ratio(errors[0])
        self.assertLess(float(jac_error), 1e-6)

    def test_hessian_with_derivative_checker(self):
        """Validate hessian for mixed basis using DerivativeChecker."""
        bkd = self._bkd
        pce = self._create_mixed_pce(nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        checker = DerivativeChecker(pce)
        sample = bkd.asarray([[0.3], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)

        hess_error = checker.error_ratio(errors[1])
        self.assertLess(float(hess_error), 1e-6)

    def test_jacobian_batch_with_derivative_checker(self):
        """Validate jacobian_batch for mixed basis using BatchDerivativeChecker."""
        bkd = self._bkd
        pce = self._create_mixed_pce(nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 5)))
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)

    def test_hessian_batch_with_derivative_checker(self):
        """Validate hessian_batch for mixed basis using BatchDerivativeChecker."""
        bkd = self._bkd
        pce = self._create_mixed_pce(nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 5)))
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_hessian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)

    def test_jacobian_batch_multi_qoi(self):
        """Validate jacobian_batch for mixed basis with multiple QoIs."""
        bkd = self._bkd
        pce = self._create_mixed_pce(nqoi=3)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 3)))

        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 5)))
        checker = BatchDerivativeChecker(pce, samples)
        errors = checker.check_jacobian_batch(verbosity=0)

        error_ratio = checker.error_ratio(errors)
        self.assertLess(float(error_ratio), 1e-6)


class TestMixedBasisDerivativesNumpy(TestMixedBasisDerivatives[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMixedBasisDerivativesTorch(TestMixedBasisDerivatives[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestJacobianWrtParams(Generic[Array], unittest.TestCase):
    """Test jacobian_wrt_params for BasisExpansion.

    Tests validate that jacobian_wrt_params (derivatives w.r.t. active coefficients)
    matches finite differences using DerivativeChecker.

    Note: jacobian_wrt_params returns Jacobian w.r.t. ACTIVE params only.
    By default all params are active, so nactive_params == nparams.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        return create_pce_from_marginals(marginals, max_level, bkd, nqoi=nqoi)

    def test_jacobian_wrt_params_1d(self):
        """Test jacobian_wrt_params for 1D expansion with nqoi=1."""
        bkd = self._bkd
        pce = self._create_pce(nvars=1, max_level=3, nqoi=1)

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

        def fun(params: Array) -> Array:
            # params shape: (nactive_params, 1)
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            return pce(samples).T  # (nsamples, nqoi)

        def jacobian_func(params: Array) -> Array:
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
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_wrt_params_2d(self):
        """Test jacobian_wrt_params for 2D expansion with nqoi=1."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Evaluate at a fixed sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        nactive = pce.nactive_params()

        def fun(params: Array) -> Array:
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            return pce(samples).T  # (nsamples, nqoi)

        def jacobian_func(params: Array) -> Array:
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
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_wrt_params_multi_qoi(self):
        """Test jacobian_wrt_params for expansion with nqoi > 1."""
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=2, nqoi=2)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 2)))

        # Evaluate at a single sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        nactive = pce.nactive_params()

        def fun(params: Array) -> Array:
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            # pce(samples) returns (nqoi, nsamples)
            # For FunctionFromCallable we need (nqoi, 1) output
            return pce(samples)  # (nqoi, nsamples=1)

        def jacobian_func(params: Array) -> Array:
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
        self.assertLess(float(jac_error), 1e-6)

    def test_jacobian_wrt_params_with_fixed_params(self):
        """Test jacobian_wrt_params when some params are fixed (inactive).

        Verifies that jacobian only includes active parameters and that
        the derivative check passes for the reduced parameter space.
        """
        bkd = self._bkd
        pce = self._create_pce(nvars=2, max_level=2, nqoi=1)

        np.random.seed(42)
        pce.set_coefficients(bkd.asarray(np.random.randn(pce.nterms(), 1)))

        # Fix the first 2 parameters (make them inactive)
        # Only optimize the remaining parameters
        nparams_total = pce.nparams()
        active_indices = bkd.arange(2, nparams_total)
        pce.hyp_list().set_active_indices(active_indices)

        nactive = pce.nactive_params()
        self.assertEqual(nactive, nparams_total - 2)

        # Evaluate at a single sample
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        def fun(params: Array) -> Array:
            # params shape: (nactive_params, 1)
            pce.hyp_list().set_active_values(params[:, 0])
            pce._sync_from_hyp_list()
            return pce(samples).T  # (nsamples, nqoi)

        def jacobian_func(params: Array) -> Array:
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
        self.assertLess(float(jac_error), 1e-6)

        # Verify the jacobian shape is correct (only active params)
        jac = pce.jacobian_wrt_params(samples)
        self.assertEqual(jac.shape, (1, 1, nactive))


class TestJacobianWrtParamsNumpy(TestJacobianWrtParams[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestJacobianWrtParamsTorch(TestJacobianWrtParams[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()

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


if __name__ == "__main__":
    unittest.main()
