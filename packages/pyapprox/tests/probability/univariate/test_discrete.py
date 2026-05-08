"""
Tests for discrete univariate distributions.
"""

import numpy as np
import pytest

from pyapprox.probability.univariate.discrete import (
    CustomDiscreteMarginal,
    DiscreteChebyshevMarginal,
)


class TestCustomDiscreteMarginal:
    """Tests for CustomDiscreteMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        self._xk = self._bkd.asarray([0.0, 1.0, 2.0, 3.0])
        self._pk = self._bkd.asarray([0.1, 0.4, 0.3, 0.2])
        self._dist = CustomDiscreteMarginal(self._xk, self._pk, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        assert self._dist.nvars() == 1

    def test_nmasses(self) -> None:
        """Test nmasses returns correct count."""
        assert self._dist.nmasses() == 4

    def test_probability_masses(self) -> None:
        """Test probability_masses returns correct values."""
        xk, pk = self._dist.probability_masses()
        assert self._bkd.allclose(xk, self._xk, atol=1e-10)
        assert self._bkd.allclose(pk, self._pk, atol=1e-10)

    def test_pmf_at_mass_points(self) -> None:
        """Test PMF at mass points."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0]])
        pmf_vals = self._dist.pmf(samples)
        expected = self._bkd.reshape(self._pk, (1, -1))
        assert self._bkd.allclose(pmf_vals, expected, atol=1e-10)

    def test_pmf_off_mass_points(self) -> None:
        """Test PMF returns 0 at non-mass points."""
        samples = self._bkd.asarray([[0.5, 1.5, 2.5, 4.0]])
        pmf_vals = self._dist.pmf(samples)
        expected = self._bkd.zeros((1, 4))
        assert self._bkd.allclose(pmf_vals, expected, atol=1e-10)

    def test_pdf_alias(self) -> None:
        """Test pdf is alias for pmf."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0]])
        assert self._bkd.allclose(
            self._dist.pdf(samples), self._dist.pmf(samples), atol=1e-10
        )

    def test_logpmf(self) -> None:
        """Test logpmf at mass points."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0]])
        logpmf_vals = self._dist.logpmf(samples)
        expected = self._bkd.reshape(self._bkd.log(self._pk), (1, -1))
        assert self._bkd.allclose(logpmf_vals, expected, atol=1e-10)

    def test_cdf_at_mass_points(self) -> None:
        """Test CDF at mass points."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0]])
        cdf_vals = self._dist.cdf(samples)
        # CDF should be cumulative sum: [0.1, 0.5, 0.8, 1.0]
        expected = self._bkd.asarray([[0.1, 0.5, 0.8, 1.0]])
        assert self._bkd.allclose(cdf_vals, expected, atol=1e-10)

    def test_cdf_between_mass_points(self) -> None:
        """Test CDF between mass points."""
        samples = self._bkd.asarray([[0.5, 1.5, 2.5]])
        cdf_vals = self._dist.cdf(samples)
        # CDF is step function: [0.1, 0.5, 0.8]
        expected = self._bkd.asarray([[0.1, 0.5, 0.8]])
        assert self._bkd.allclose(cdf_vals, expected, atol=1e-10)

    def test_cdf_below_support(self) -> None:
        """Test CDF below support is 0."""
        samples = self._bkd.asarray([[-1.0, -0.5]])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.zeros((1, 2))
        assert self._bkd.allclose(cdf_vals, expected, atol=1e-10)

    def test_cdf_above_support(self) -> None:
        """Test CDF above support is 1."""
        samples = self._bkd.asarray([[3.5, 4.0, 10.0]])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.ones((1, 3))
        assert self._bkd.allclose(cdf_vals, expected, atol=1e-10)

    def test_invcdf(self) -> None:
        """Test inverse CDF."""
        # At cumulative probabilities, should return mass points
        probs = self._bkd.asarray([[0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]])
        invcdf_vals = self._dist.invcdf(probs)
        # 0.05 -> 0, 0.1 -> 0, 0.3 -> 1, 0.5 -> 1, 0.7 -> 2, 0.8 -> 2, 0.9 -> 3, 1.0 ->
        # 3
        expected = self._bkd.asarray([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])
        assert self._bkd.allclose(invcdf_vals, expected, atol=1e-10)

    def test_ppf_alias(self) -> None:
        """Test ppf is alias for invcdf."""
        probs = self._bkd.asarray([[0.25, 0.5, 0.75]])
        assert self._bkd.allclose(
            self._dist.ppf(probs), self._dist.invcdf(probs), atol=1e-10
        )

    def test_cdf_invcdf_roundtrip(self) -> None:
        """Test invcdf(cdf(x)) returns valid mass point."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0]])
        recovered = self._dist.invcdf(self._dist.cdf(samples))
        # For discrete, recovered should be same as input
        assert self._bkd.allclose(recovered, samples, atol=1e-10)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self._dist.rvs(100)
        assert samples.shape == (1, 100)

    def test_rvs_values_in_support(self) -> None:
        """Test rvs produces values at mass points."""
        samples = self._dist.rvs(1000)
        samples_flat = self._bkd.flatten(samples)
        samples_np = self._bkd.to_numpy(samples_flat)

        # All samples should be at mass points
        for s in samples_np:
            assert np.any(np.abs(self._bkd.to_numpy(self._xk) - s) < 1e-10)

    def test_mean_value(self) -> None:
        """Test mean matches analytical formula."""
        expected_mean = 0.0 * 0.1 + 1.0 * 0.4 + 2.0 * 0.3 + 3.0 * 0.2
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.mean_value()]),
            self._bkd.asarray([expected_mean]),
            atol=1e-10,
        )

    def test_variance(self) -> None:
        """Test variance matches analytical formula."""
        mean = self._dist.mean_value()
        second_moment = 0.0**2 * 0.1 + 1.0**2 * 0.4 + 2.0**2 * 0.3 + 3.0**2 * 0.2
        expected_var = second_moment - mean**2
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.variance()]),
            self._bkd.asarray([expected_var]),
            atol=1e-10,
        )

    def test_std(self) -> None:
        """Test std is sqrt of variance."""
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.std()]),
            self._bkd.asarray([np.sqrt(self._dist.variance())]),
            atol=1e-10,
        )

    def test_moment(self) -> None:
        """Test moment computation."""
        # First moment should equal mean
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.moment(1)]),
            self._bkd.asarray([self._dist.mean_value()]),
            atol=1e-10,
        )
        # Second moment
        second_moment = 0.0**2 * 0.1 + 1.0**2 * 0.4 + 2.0**2 * 0.3 + 3.0**2 * 0.2
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.moment(2)]),
            self._bkd.asarray([second_moment]),
            atol=1e-10,
        )

    def test_is_bounded(self) -> None:
        """Test discrete distributions are bounded."""
        assert self._dist.is_bounded()

    def test_bounds(self) -> None:
        """Test bounds are min/max of mass locations."""
        lower, upper = self._dist.bounds()
        assert self._bkd.allclose(
            self._bkd.asarray([lower]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([upper]),
            self._bkd.asarray([3.0]),
            atol=1e-10,
        )

    def test_interval(self) -> None:
        """Test interval returns valid bounds."""
        interval = self._dist.interval(0.8)
        # interval returns shape (1, 2) with [lower, upper] bounds
        assert interval.shape == (1, 2)
        lower, upper = float(interval[0, 0]), float(interval[0, 1])
        assert lower <= upper

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = CustomDiscreteMarginal(self._xk, self._pk, self._bkd)
        dist3 = CustomDiscreteMarginal(
            self._bkd.asarray([0.0, 1.0]), self._bkd.asarray([0.5, 0.5]), self._bkd
        )
        assert self._dist == dist2
        assert self._dist != dist3

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        assert "CustomDiscreteMarginal" in repr_str
        assert "4" in repr_str  # nmasses

    def test_invalid_xk_shape(self) -> None:
        """Test invalid xk shape raises error."""
        with pytest.raises(ValueError):
            CustomDiscreteMarginal(
                self._bkd.asarray([[0.0, 1.0]]),  # 2D
                self._bkd.asarray([0.5, 0.5]),
                self._bkd,
            )

    def test_invalid_pk_shape(self) -> None:
        """Test invalid pk shape raises error."""
        with pytest.raises(ValueError):
            CustomDiscreteMarginal(
                self._bkd.asarray([0.0, 1.0]),
                self._bkd.asarray([[0.5, 0.5]]),  # 2D
                self._bkd,
            )

    def test_mismatched_lengths(self) -> None:
        """Test mismatched xk/pk lengths raise error."""
        with pytest.raises(ValueError):
            CustomDiscreteMarginal(
                self._bkd.asarray([0.0, 1.0, 2.0]),
                self._bkd.asarray([0.5, 0.5]),
                self._bkd,
            )

    def test_pk_not_sum_to_one(self) -> None:
        """Test pk not summing to 1 raises error."""
        with pytest.raises(ValueError):
            CustomDiscreteMarginal(
                self._bkd.asarray([0.0, 1.0]), self._bkd.asarray([0.3, 0.3]), self._bkd
            )

    def test_negative_probability(self) -> None:
        """Test negative probability raises error."""
        with pytest.raises(ValueError):
            CustomDiscreteMarginal(
                self._bkd.asarray([0.0, 1.0]), self._bkd.asarray([-0.5, 1.5]), self._bkd
            )

    def test_empty_arrays(self) -> None:
        """Test empty arrays raise error."""
        with pytest.raises(ValueError):
            CustomDiscreteMarginal(
                self._bkd.asarray([]), self._bkd.asarray([]), self._bkd
            )

    def test_unsorted_input(self) -> None:
        """Test unsorted input is handled correctly."""
        xk = self._bkd.asarray([2.0, 0.0, 3.0, 1.0])
        pk = self._bkd.asarray([0.3, 0.1, 0.2, 0.4])
        dist = CustomDiscreteMarginal(xk, pk, self._bkd)

        # Should be sorted internally
        xk_sorted, pk_sorted = dist.probability_masses()
        expected_xk = self._bkd.asarray([0.0, 1.0, 2.0, 3.0])
        expected_pk = self._bkd.asarray([0.1, 0.4, 0.3, 0.2])
        assert self._bkd.allclose(xk_sorted, expected_xk, atol=1e-10)
        assert self._bkd.allclose(pk_sorted, expected_pk, atol=1e-10)


class TestDiscreteChebyshevMarginal:
    """Tests for DiscreteChebyshevMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        self._nmasses = 5
        self._dist = DiscreteChebyshevMarginal(self._nmasses, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        assert self._dist.nvars() == 1

    def test_nmasses(self) -> None:
        """Test nmasses returns correct count."""
        assert self._dist.nmasses() == self._nmasses

    def test_probability_masses_uniform(self) -> None:
        """Test probability masses are uniform."""
        xk, pk = self._dist.probability_masses()
        expected_xk = self._bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        expected_pk = self._bkd.asarray([0.2, 0.2, 0.2, 0.2, 0.2])
        assert self._bkd.allclose(xk, expected_xk, atol=1e-10)
        assert self._bkd.allclose(pk, expected_pk, atol=1e-10)

    def test_pmf_uniform(self) -> None:
        """Test PMF is uniform at all mass points."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]])
        pmf_vals = self._dist.pmf(samples)
        expected = self._bkd.ones((1, 5)) / 5.0
        assert self._bkd.allclose(pmf_vals, expected, atol=1e-10)

    def test_cdf_linear(self) -> None:
        """Test CDF is step function with uniform jumps."""
        samples = self._bkd.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.asarray([[0.2, 0.4, 0.6, 0.8, 1.0]])
        assert self._bkd.allclose(cdf_vals, expected, atol=1e-10)

    def test_mean_value(self) -> None:
        """Test mean is (nmasses-1)/2."""
        expected_mean = (self._nmasses - 1) / 2.0
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.mean_value()]),
            self._bkd.asarray([expected_mean]),
            atol=1e-10,
        )

    def test_bounds(self) -> None:
        """Test bounds are [0, nmasses-1]."""
        lower, upper = self._dist.bounds()
        assert self._bkd.allclose(
            self._bkd.asarray([lower]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([upper]),
            self._bkd.asarray([float(self._nmasses - 1)]),
            atol=1e-10,
        )

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = DiscreteChebyshevMarginal(self._nmasses, self._bkd)
        dist3 = DiscreteChebyshevMarginal(3, self._bkd)
        assert self._dist == dist2
        assert self._dist != dist3

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        assert "DiscreteChebyshevMarginal" in repr_str
        assert "5" in repr_str

    def test_invalid_nmasses(self) -> None:
        """Test invalid nmasses raises error."""
        with pytest.raises(ValueError):
            DiscreteChebyshevMarginal(0, self._bkd)
        with pytest.raises(ValueError):
            DiscreteChebyshevMarginal(-1, self._bkd)

    def test_single_mass(self) -> None:
        """Test single mass point distribution."""
        dist = DiscreteChebyshevMarginal(1, self._bkd)
        assert dist.nmasses() == 1
        assert self._bkd.allclose(
            self._bkd.asarray([dist.mean_value()]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([dist.variance()]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )
