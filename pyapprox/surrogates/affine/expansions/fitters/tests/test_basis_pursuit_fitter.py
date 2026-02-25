"""Tests for BasisPursuitFitter (Basis Pursuit L1 minimization).

Tests focus on:
- Returning SparseResult
- Sparse coefficient recovery
- Support identification
- Multi-QoI rejection
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.affine.expansions.fitters.basis_pursuit import (
    BasisPursuitFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    SparseResult,
)

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal


class TestBasisPursuitFitter(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_sparse_result(self) -> None:
        """Fit returns SparseResult."""
        bkd = self._bkd
        nvars, max_level = 1, 5

        # Create a sparse target
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data (BP requires exact data)
        nsamples = nterms + 5  # slightly overdetermined
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        self.assertIsInstance(result, SparseResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        bkd = self._bkd
        nvars, max_level = 1, 5

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        self.assertEqual(result.params().shape, (nterms, 1))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        bkd = self._bkd
        nvars, max_level = 1, 3

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values_2d = target_expansion(samples)
        values_1d = values_2d[0, :]  # flatten to 1D

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values_1d)

        self.assertEqual(result.params().shape[1], 1)

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        bkd = self._bkd
        nvars, max_level = 1, 5

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_multi_qoi_raises(self) -> None:
        """nqoi > 1 raises ValueError."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        fitter = BasisPursuitFitter(bkd)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_recovers_sparse_signal(self) -> None:
        """BP recovers a known sparse signal accurately.

        Basis Pursuit recovers the sparsest solution when data is noiseless
        and the system is sufficiently overdetermined.
        """
        bkd = self._bkd
        nvars = 1
        max_level = 9  # 10 terms

        # Create target with exactly 3 nonzero coefficients
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = -0.5
        true_coef[5, 0] = 0.3
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data (more samples than terms)
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at test points
        ntest = 50
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Should predict accurately
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        self.assertLess(float(rel_error), 1e-8)

    def test_support_identification(self) -> None:
        """BP correctly identifies support of sparse signal."""
        bkd = self._bkd
        nvars = 1
        max_level = 9  # 10 terms

        # Create target with known support
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_support = [0, 3, 7]  # indices of nonzero coefficients
        for idx in true_support:
            true_coef[idx, 0] = 1.0 / (idx + 1)
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Check support
        self.assertEqual(result.n_nonzero(), len(true_support))

        # Support should contain the true indices
        recovered_support = sorted([int(s) for s in result.support()])
        self.assertEqual(recovered_support, true_support)

    def test_n_nonzero_correct(self) -> None:
        """n_nonzero matches support length."""
        bkd = self._bkd
        nvars = 1
        max_level = 5

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[1, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        self.assertEqual(result.n_nonzero(), len(result.support()))

    def test_regularization_strength_zero(self) -> None:
        """regularization_strength is 0 for BP (no regularization)."""
        bkd = self._bkd
        nvars = 1
        max_level = 3

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        fitter = BasisPursuitFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values)

        bkd.assert_allclose(
            bkd.asarray([result.regularization_strength()]),
            bkd.asarray([0.0]),
        )

    def test_coefficient_recovery(self) -> None:
        """BP recovers exact coefficients for sparse signal."""
        bkd = self._bkd
        nvars = 1
        max_level = 9  # 10 terms

        # Create target with 2 nonzero coefficients
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 2.0
        true_coef[4, 0] = -1.5
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Check coefficient recovery
        bkd.assert_allclose(result.params(), true_coef, atol=1e-8)


    def test_recovers_1d_expansion_random_sparse(self) -> None:
        """BP recovers a 1D expansion with random sparse coefficients.

        Replicates legacy test_basis_pursuit: creates random sparse
        coefficients and verifies exact recovery.
        """
        bkd = self._bkd
        nvars = 1
        max_level = 7  # 8 terms (degree 7 polynomial)
        sparsity = 2

        # Create expansion
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()

        # Create random sparse coefficients (like legacy test)
        true_coef = bkd.zeros((nterms, 1))
        np.random.seed(42)  # ensure reproducibility
        sparse_indices = np.random.permutation(nterms)[:sparsity]
        for idx in sparse_indices:
            true_coef[idx, 0] = 1.0
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data (overdetermined)
        nsamples = nterms + 2
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        fitter = BasisPursuitFitter(bkd)
        result = fitter.fit(fit_expansion, samples, values)

        # Should recover exactly
        bkd.assert_allclose(result.params(), true_coef, atol=1e-8)


class TestBasisPursuitFitterNumpy(TestBasisPursuitFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBasisPursuitFitterTorch(TestBasisPursuitFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
