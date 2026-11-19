"""Tests for multivariate Leja and Fekete sampling."""


class TestFeketeSampler:
    """Tests for Fekete point sampling."""

    def test_fekete_sample_count(self, bkd) -> None:
        """Test that Fekete sampler returns correct number of points."""
        from pyapprox.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )
        from pyapprox.surrogates.affine.leja import FeketeSampler
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        # Create a simple 1D basis
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)

        # Create multivariate basis wrapper
        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, bkd)
        basis = MultiIndexBasis([poly], bkd, idx_gen.get_indices())

        # Generate candidates
        candidates = bkd.linspace(-1, 1, 20)[None, :]

        sampler = FeketeSampler(bkd, basis, candidates)
        selected = sampler.sample(5)

        assert selected.shape == (1, 5)

    def test_fekete_selected_indices(self, bkd) -> None:
        """Test that selected indices are valid."""
        from pyapprox.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )
        from pyapprox.surrogates.affine.leja import FeketeSampler
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, bkd)
        basis = MultiIndexBasis([poly], bkd, idx_gen.get_indices())

        candidates = bkd.linspace(-1, 1, 20)[None, :]
        ncandidates = candidates.shape[1]

        sampler = FeketeSampler(bkd, basis, candidates)
        sampler.sample(5)

        indices = sampler.get_selected_indices()
        assert len(indices) == 5
        for idx in indices:
            assert 0 <= int(idx) < ncandidates


class TestLejaSampler:
    """Tests for multivariate Leja sampling."""

    def test_leja_sample_count(self, bkd) -> None:
        """Test that Leja sampler returns correct number of points."""
        from pyapprox.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )
        from pyapprox.surrogates.affine.leja import LejaSampler
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        # Create a simple 1D basis
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)

        # Create multivariate basis wrapper
        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, bkd)
        basis = MultiIndexBasis([poly], bkd, idx_gen.get_indices())

        # Generate candidates
        candidates = bkd.linspace(-1, 1, 20)[None, :]

        sampler = LejaSampler(bkd, basis, candidates)
        selected = sampler.sample(5)

        assert selected.shape == (1, 5)

    def test_leja_incremental(self, bkd) -> None:
        """Test incremental Leja sampling."""
        from pyapprox.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )
        from pyapprox.surrogates.affine.leja import LejaSampler
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, bkd)
        basis = MultiIndexBasis([poly], bkd, idx_gen.get_indices())

        candidates = bkd.linspace(-1, 1, 20)[None, :]

        sampler = LejaSampler(bkd, basis, candidates)

        # Sample 3 points
        sampler.sample(3)
        assert sampler.nsamples() == 3

        # Add 2 more incrementally
        new_samples = sampler.sample_incremental(2)
        assert new_samples.shape == (1, 2)
        assert sampler.nsamples() == 5

    def test_leja_selected_indices(self, bkd) -> None:
        """Test that selected indices are valid and consistent."""
        from pyapprox.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )
        from pyapprox.surrogates.affine.leja import LejaSampler
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, bkd)
        basis = MultiIndexBasis([poly], bkd, idx_gen.get_indices())

        candidates = bkd.linspace(-1, 1, 20)[None, :]
        ncandidates = candidates.shape[1]

        sampler = LejaSampler(bkd, basis, candidates)
        sampler.sample(5)

        indices = sampler.get_selected_indices()
        assert len(indices) == 5
        for idx in indices:
            assert 0 <= int(idx) < ncandidates

    def test_1d_multivariate_matches_univariate_with_optimal_candidates(
        self, bkd,
    ) -> None:
        """Test 1D multivariate Leja matches univariate when candidates include optimal
        points.

        This test verifies that when the candidate set includes the optimal
        Leja points from the gradient-optimized univariate LejaSequence1D,
        and the initial point is the same, the multivariate LejaSampler
        recovers the same sequence.
        """
        from pyapprox.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.surrogates.affine.indices import (
            HyperbolicIndexGenerator,
        )
        from pyapprox.surrogates.affine.leja import (
            ChristoffelWeighting,
            LejaSampler,
            LejaSequence1D,
        )
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        npts = 5

        # First, generate optimal Leja sequence using univariate optimizer
        poly = LegendrePolynomial1D(bkd)
        weighting = ChristoffelWeighting(bkd)
        univariate_leja = LejaSequence1D(bkd, poly, weighting, bounds=(-1.0, 1.0))
        optimal_samples, _ = univariate_leja.quadrature_rule(npts)

        # Create candidate set that INCLUDES the optimal points plus extra
        # Make sure optimal points are in the candidate set
        extra_candidates = bkd.linspace(-1, 1, 50)[None, :]
        candidates = bkd.hstack([optimal_samples, extra_candidates])

        # Set up multivariate sampler with nvars=1
        poly_mv = LegendrePolynomial1D(bkd)
        poly_mv.set_nterms(npts)
        idx_gen = HyperbolicIndexGenerator(1, npts - 1, 1.0, bkd)
        basis = MultiIndexBasis([poly_mv], bkd, idx_gen.get_indices())

        # Use the first optimal point as initial point for multivariate sampler
        initial_idx = bkd.asarray([0], dtype=bkd.int64_dtype())
        sampler = LejaSampler(bkd, basis, candidates)
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
                if bkd.allclose(
                    bkd.asarray([opt_pt]),
                    bkd.asarray([multivariate_samples[0, j]]),
                    atol=1e-10,
                ):
                    found = True
                    break
            assert found, (
                f"Optimal point {opt_pt} not found in multivariate samples"
            )
