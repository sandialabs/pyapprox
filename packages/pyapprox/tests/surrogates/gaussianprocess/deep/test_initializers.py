"""Tests for inducing point initialization strategies."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.deep.initializers import (
    CustomInitializer,
    InducingInitializer,
    RandomUniformInitializer,
    SobolInitializer,
)


class TestRandomUniformInitializer:
    def test_shape(self, bkd):
        init = RandomUniformInitializer((-2.0, 2.0))
        rng = np.random.RandomState(0)
        locs = init.initialize(10, 3, bkd, rng)
        assert locs.shape == (3, 10)

    def test_within_bounds(self, bkd):
        init = RandomUniformInitializer((-1.5, 3.0))
        rng = np.random.RandomState(42)
        locs = bkd.to_numpy(init.initialize(20, 2, bkd, rng))
        assert np.all(locs >= -1.5)
        assert np.all(locs <= 3.0)

    def test_deterministic_with_same_seed(self, bkd):
        init = RandomUniformInitializer((-1.0, 1.0))
        locs1 = init.initialize(5, 2, bkd, np.random.RandomState(7))
        locs2 = init.initialize(5, 2, bkd, np.random.RandomState(7))
        bkd.assert_allclose(locs1, locs2)

    def test_satisfies_protocol(self):
        assert isinstance(RandomUniformInitializer(), InducingInitializer)


class TestSobolInitializer:
    def test_shape(self, bkd):
        init = SobolInitializer((-1.0, 1.0))
        rng = np.random.RandomState(0)
        locs = init.initialize(8, 2, bkd, rng)
        assert locs.shape == (2, 8)

    def test_within_bounds(self, bkd):
        init = SobolInitializer((-3.0, 3.0))
        rng = np.random.RandomState(42)
        locs = bkd.to_numpy(init.initialize(16, 3, bkd, rng))
        assert np.all(locs >= -3.0)
        assert np.all(locs <= 3.0)

    def test_better_coverage_than_random(self, numpy_bkd):
        """Sobol should fill the space more uniformly than random."""
        bkd = numpy_bkd
        M = 32
        sobol = SobolInitializer((0.0, 1.0))
        rand = RandomUniformInitializer((0.0, 1.0))
        rng_s = np.random.RandomState(0)
        rng_r = np.random.RandomState(0)
        locs_sobol = bkd.to_numpy(sobol.initialize(M, 2, bkd, rng_s))
        locs_rand = bkd.to_numpy(rand.initialize(M, 2, bkd, rng_r))

        def max_min_dist(pts):
            from scipy.spatial.distance import cdist
            d = cdist(pts.T, pts.T)
            np.fill_diagonal(d, np.inf)
            return np.min(d, axis=1).max()

        mmd_sobol = max_min_dist(locs_sobol)
        mmd_rand = max_min_dist(locs_rand)
        assert mmd_sobol <= mmd_rand * 1.5

    def test_satisfies_protocol(self):
        assert isinstance(SobolInitializer(), InducingInitializer)


class TestCustomInitializer:
    def test_returns_exact_locations(self, bkd):
        locs = bkd.array(np.arange(6).reshape(2, 3).astype(float))
        init = CustomInitializer(locs)
        result = init.initialize(3, 2, bkd, np.random.RandomState(0))
        bkd.assert_allclose(result, locs)

    def test_shape_mismatch_raises(self, bkd):
        locs = bkd.array(np.zeros((2, 3)))
        init = CustomInitializer(locs)
        with pytest.raises(ValueError, match="does not match"):
            init.initialize(5, 2, bkd, np.random.RandomState(0))

    def test_satisfies_protocol(self, numpy_bkd):
        locs = numpy_bkd.array(np.zeros((2, 3)))
        assert isinstance(CustomInitializer(locs), InducingInitializer)


class TestBuildersWithInitializer:
    def test_custom_initializer_used(self, bkd):
        """Builder should use the provided initializer."""
        from pyapprox.surrogates.gaussianprocess.deep.builders import (
            build_single_fidelity_dgp,
        )
        from pyapprox.surrogates.kernels.matern import Matern52Kernel

        def mf(nvars, bkd):
            return Matern52Kernel(
                lenscale=[1.0] * nvars, lenscale_bounds=(0.1, 10.0),
                nvars=nvars, bkd=bkd,
            )

        init = SobolInitializer((-2.0, 2.0))
        dgp = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=8,
            kernel_factory=mf, bkd=bkd,
            initializer=init, seed=0,
        )
        locs = bkd.to_numpy(dgp.layers()[0].inducing_points().get_samples())
        assert np.all(locs >= -2.0)
        assert np.all(locs <= 2.0)
