"""Tests for multivariate Leja and Fekete sampling."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests


class TestFeketeSampler(Generic[Array], unittest.TestCase):
    """Tests for Fekete point sampling."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_fekete_sample_count(self) -> None:
        """Test that Fekete sampler returns correct number of points."""
        from pyapprox.typing.surrogates.affine.leja import FeketeSampler
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )

        # Create a simple 1D basis
        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)

        # Create multivariate basis wrapper
        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self._bkd)
        basis = MultiIndexBasis([poly], self._bkd, idx_gen.get_indices())

        # Generate candidates
        candidates = self._bkd.linspace(-1, 1, 20)[None, :]

        sampler = FeketeSampler(self._bkd, basis, candidates)
        selected = sampler.sample(5)

        self.assertEqual(selected.shape, (1, 5))

    def test_fekete_selected_indices(self) -> None:
        """Test that selected indices are valid."""
        from pyapprox.typing.surrogates.affine.leja import FeketeSampler
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )

        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self._bkd)
        basis = MultiIndexBasis([poly], self._bkd, idx_gen.get_indices())

        candidates = self._bkd.linspace(-1, 1, 20)[None, :]
        ncandidates = candidates.shape[1]

        sampler = FeketeSampler(self._bkd, basis, candidates)
        sampler.sample(5)

        indices = sampler.get_selected_indices()
        self.assertEqual(len(indices), 5)
        for idx in indices:
            self.assertTrue(0 <= int(idx) < ncandidates)


class TestLejaSampler(Generic[Array], unittest.TestCase):
    """Tests for multivariate Leja sampling."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_leja_sample_count(self) -> None:
        """Test that Leja sampler returns correct number of points."""
        from pyapprox.typing.surrogates.affine.leja import LejaSampler
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )

        # Create a simple 1D basis
        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)

        # Create multivariate basis wrapper
        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self._bkd)
        basis = MultiIndexBasis([poly], self._bkd, idx_gen.get_indices())

        # Generate candidates
        candidates = self._bkd.linspace(-1, 1, 20)[None, :]

        sampler = LejaSampler(self._bkd, basis, candidates)
        selected = sampler.sample(5)

        self.assertEqual(selected.shape, (1, 5))

    def test_leja_incremental(self) -> None:
        """Test incremental Leja sampling."""
        from pyapprox.typing.surrogates.affine.leja import LejaSampler
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )

        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self._bkd)
        basis = MultiIndexBasis([poly], self._bkd, idx_gen.get_indices())

        candidates = self._bkd.linspace(-1, 1, 20)[None, :]

        sampler = LejaSampler(self._bkd, basis, candidates)

        # Sample 3 points
        sampler.sample(3)
        self.assertEqual(sampler.nsamples(), 3)

        # Add 2 more incrementally
        new_samples = sampler.sample_incremental(2)
        self.assertEqual(new_samples.shape, (1, 2))
        self.assertEqual(sampler.nsamples(), 5)

    def test_leja_selected_indices(self) -> None:
        """Test that selected indices are valid and consistent."""
        from pyapprox.typing.surrogates.affine.leja import LejaSampler
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )

        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self._bkd)
        basis = MultiIndexBasis([poly], self._bkd, idx_gen.get_indices())

        candidates = self._bkd.linspace(-1, 1, 20)[None, :]
        ncandidates = candidates.shape[1]

        sampler = LejaSampler(self._bkd, basis, candidates)
        sampler.sample(5)

        indices = sampler.get_selected_indices()
        self.assertEqual(len(indices), 5)
        for idx in indices:
            self.assertTrue(0 <= int(idx) < ncandidates)

    def test_1d_multivariate_matches_univariate_with_optimal_candidates(
        self,
    ) -> None:
        """Test 1D multivariate Leja matches univariate when candidates include optimal points.

        This test verifies that when the candidate set includes the optimal
        Leja points from the gradient-optimized univariate LejaSequence1D,
        and the initial point is the same, the multivariate LejaSampler
        recovers the same sequence.
        """
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSampler,
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )

        npts = 5

        # First, generate optimal Leja sequence using univariate optimizer
        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        univariate_leja = LejaSequence1D(
            self._bkd, poly, weighting, bounds=(-1.0, 1.0)
        )
        optimal_samples, _ = univariate_leja.quadrature_rule(npts)

        # Create candidate set that INCLUDES the optimal points plus extra
        # Make sure optimal points are in the candidate set
        extra_candidates = self._bkd.linspace(-1, 1, 50)[None, :]
        candidates = self._bkd.hstack([optimal_samples, extra_candidates])

        # Set up multivariate sampler with nvars=1
        poly_mv = LegendrePolynomial1D(self._bkd)
        poly_mv.set_nterms(npts)
        idx_gen = HyperbolicIndexGenerator(1, npts - 1, 1.0, self._bkd)
        basis = MultiIndexBasis([poly_mv], self._bkd, idx_gen.get_indices())

        # Use the first optimal point as initial point for multivariate sampler
        initial_idx = self._bkd.asarray([0], dtype=self._bkd.int64_dtype())
        sampler = LejaSampler(self._bkd, basis, candidates)
        sampler.set_initial_pivots(initial_idx)
        multivariate_samples = sampler.sample(npts)

        # The multivariate sampler should recover the same points
        # (possibly in different order after the first point)
        # Check that all optimal points are selected
        for i in range(npts):
            opt_pt = optimal_samples[0, i]
            # Check if this point is in the multivariate samples
            found = False
            for j in range(npts):
                if self._bkd.allclose(
                    self._bkd.asarray([opt_pt]),
                    self._bkd.asarray([multivariate_samples[0, j]]),
                    atol=1e-10,
                ):
                    found = True
                    break
            self.assertTrue(
                found,
                f"Optimal point {opt_pt} not found in multivariate samples",
            )


# NumPy backend tests
class TestFeketeSamplerNumpy(TestFeketeSampler[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLejaSamplerNumpy(TestLejaSampler[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestFeketeSamplerTorch(TestFeketeSampler[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestLejaSamplerTorch(TestLejaSampler[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
