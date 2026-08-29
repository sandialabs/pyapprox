"""Tests for deciding how many modes of a spectrum to keep.

Each policy is a few lines, so the risk is not that one is wrong in
isolation but that they disagree about the boundary cases: whether a
fraction is of energy or of amplitude, whether a mode exactly at a
threshold is in or out, whether "no modes qualify" is zero or an error.
Those are what is pinned here.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kle.truncation import (
    by_count,
    by_eigenvalue_floor,
    by_numerical_rank,
    by_variance_fraction,
    resolve_nterms,
)


def _spectrum(bkd, values=(10.0, 5.0, 3.0, 1.0)):
    return bkd.array(np.asarray(values, dtype=float))


def _with_noise_tail(bkd):
    """A spectrum whose last mode is rounding rather than variance."""
    return _spectrum(bkd, (10.0, 5.0, 3.0, 1.0, 1e-18))


class TestByCount:
    def test_returns_the_requested_count(self, bkd) -> None:
        assert by_count(_spectrum(bkd), 3, bkd) == 3

    def test_accepts_the_whole_spectrum(self, bkd) -> None:
        assert by_count(_spectrum(bkd), 4, bkd) == 4

    def test_rejects_more_than_available(self, bkd) -> None:
        with pytest.raises(ValueError, match="exceeds the 4 modes"):
            by_count(_spectrum(bkd), 5, bkd)

    def test_rejects_nonpositive(self, bkd) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            by_count(_spectrum(bkd), 0, bkd)


class TestByVarianceFraction:
    """The total is 19, so the cumulative ratios are .526 .789 .947 1."""

    @pytest.mark.parametrize(
        "fraction,expected",
        [
            (0.1, 1),
            (0.526, 1),  # the first mode alone reaches it
            (0.53, 2),
            (0.79, 3),
            (0.95, 4),
            (1.0, 4),
        ],
    )
    def test_smallest_count_reaching_the_fraction(
        self, bkd, fraction, expected
    ) -> None:
        assert by_variance_fraction(_spectrum(bkd), fraction, bkd) == expected

    def test_is_of_energy_not_amplitude(self, bkd) -> None:
        """Squares matter: the same numbers read as singular values give
        a different answer, which is the bug this centralizes away."""
        eig_vals = _spectrum(bkd)
        as_amplitudes = bkd.sqrt(eig_vals)
        assert by_variance_fraction(
            eig_vals, 0.9, bkd
        ) != by_variance_fraction(as_amplitudes, 0.9, bkd)

    def test_never_returns_zero(self, bkd) -> None:
        """A basis of no modes is not a basis."""
        assert by_variance_fraction(_spectrum(bkd), 1e-12, bkd) == 1

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_rejects_fractions_outside_the_unit_interval(
        self, bkd, fraction
    ) -> None:
        with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
            by_variance_fraction(_spectrum(bkd), fraction, bkd)

    def test_rejects_a_spectrum_with_no_variance(self, bkd) -> None:
        with pytest.raises(ValueError, match="no variance"):
            by_variance_fraction(bkd.zeros((4,)), 0.9, bkd)


class TestByNumericalRank:
    def test_drops_a_rounding_tail(self, bkd) -> None:
        assert by_numerical_rank(_with_noise_tail(bkd), bkd) == 4

    def test_keeps_a_full_rank_spectrum(self, bkd) -> None:
        assert by_numerical_rank(_spectrum(bkd), bkd) == 4

    def test_returns_zero_for_no_variance(self, bkd) -> None:
        """Zero rather than an error: an empty spectrum is a fact about
        the operator, and the caller decides whether it is fatal."""
        assert by_numerical_rank(bkd.zeros((4,)), bkd) == 0

    def test_threshold_scales_with_the_largest_eigenvalue(self, bkd) -> None:
        """Relative, not absolute: scaling the spectrum cannot change
        which modes count as usable."""
        base = _with_noise_tail(bkd)
        assert by_numerical_rank(base * 1e8, bkd) == by_numerical_rank(
            base, bkd
        )


class TestByEigenvalueFloor:
    def test_counts_modes_above_the_floor(self, bkd) -> None:
        assert by_eigenvalue_floor(_spectrum(bkd), 0.2, bkd) == 3

    def test_floor_is_relative_to_the_largest(self, bkd) -> None:
        base = _spectrum(bkd)
        assert by_eigenvalue_floor(base * 1e6, 0.2, bkd) == 3

    def test_differs_from_numerical_rank_by_who_sets_the_cut(
        self, bkd
    ) -> None:
        """A caller-set floor is stricter than machine precision, which
        is why both exist rather than one."""
        base = _with_noise_tail(bkd)
        assert by_eigenvalue_floor(base, 0.2, bkd) == 3
        assert by_numerical_rank(base, bkd) == 4

    def test_rejects_a_floor_that_discards_everything(self, bkd) -> None:
        with pytest.raises(ValueError, match="every mode would be"):
            by_eigenvalue_floor(_spectrum(bkd), 1.5, bkd)

    def test_rejects_a_negative_floor(self, bkd) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            by_eigenvalue_floor(_spectrum(bkd), -0.1, bkd)

    def test_rejects_a_spectrum_with_no_variance(self, bkd) -> None:
        with pytest.raises(ValueError, match="no variance"):
            by_eigenvalue_floor(bkd.zeros((4,)), 0.1, bkd)


class TestResolveNterms:
    """The exactly-one rule, in one place so entry points cannot differ."""

    def test_dispatches_to_count(self, bkd) -> None:
        assert resolve_nterms(_spectrum(bkd), bkd, nterms=2) == 2

    def test_dispatches_to_fraction(self, bkd) -> None:
        assert resolve_nterms(
            _spectrum(bkd), bkd, variance_fraction=0.9
        ) == 3

    def test_rejects_neither(self, bkd) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            resolve_nterms(_spectrum(bkd), bkd)

    def test_rejects_both(self, bkd) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            resolve_nterms(
                _spectrum(bkd), bkd, nterms=2, variance_fraction=0.9
            )

    def test_propagates_the_chosen_policys_validation(self, bkd) -> None:
        with pytest.raises(ValueError, match="exceeds the 4 modes"):
            resolve_nterms(_spectrum(bkd), bkd, nterms=99)
