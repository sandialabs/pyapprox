"""Tests for the backward box and incremental Smolyak coefficients.

The incremental update is only useful if it agrees with the
inclusion-exclusion formula exactly, so most of these tests compare it
against ``compute_smolyak_coefficients`` after every single add rather
than only at the end.
"""

import random
from typing import List

import pytest
from pyapprox.surrogates.sparsegrids.smolyak import (
    IncrementalSmolyakCoefficients,
    SubspaceKey,
    _unit_box_shifts,
    backward_box,
    compute_smolyak_coefficients,
)


def _from_scratch(keys: List[SubspaceKey], bkd) -> List[int]:
    """Coefficients via inclusion-exclusion, aligned with ``keys``."""
    nvars = len(keys[0])
    indices = bkd.asarray([[k[d] for k in keys] for d in range(nvars)])
    return [round(float(c)) for c in compute_smolyak_coefficients(indices, bkd)]


def _grow_randomly(
    nvars: int, seed: int, nadds: int
) -> List[SubspaceKey]:
    """Pick a random admissible growth sequence for a downward-closed set."""
    random.seed(seed)
    inc = IncrementalSmolyakCoefficients(nvars)
    inc.add(tuple([0] * nvars))
    order: List[SubspaceKey] = []
    for _ in range(nadds):
        candidates = set()
        for key in inc.keys():
            for dim in range(nvars):
                forward = key[:dim] + (key[dim] + 1,) + key[dim + 1 :]
                if inc.can_add(forward):
                    candidates.add(forward)
        if not candidates:
            break
        chosen = random.choice(sorted(candidates))
        inc.add(chosen)
        order.append(chosen)
    return order


class TestUnitBoxShifts:
    """The {0,1}^n corner table shared by the box and the batch formula."""

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
    def test_shape_and_signs(self, n: int) -> None:
        """Corners are the binary expansions, signed by parity."""
        shifts, signs = _unit_box_shifts(n)
        assert shifts.shape == (n, 2**n)
        assert signs.shape == (2**n,)
        for corner in range(2**n):
            bits = [int(shifts[d, corner]) for d in range(n)]
            assert bits == [(corner >> d) & 1 for d in range(n)]
            assert signs[corner] == pytest.approx((-1.0) ** sum(bits))


class TestBackwardBox:
    """Structure of the backward box."""

    @pytest.mark.parametrize(
        "key", [(0, 0), (2, 0), (1, 3), (2, 2, 1), (0, 0, 0), (4,)]
    )
    def test_size_signs_and_first_entry(self, key: SubspaceKey) -> None:
        """2^nnz distinct entries, correct signs, (+1, key) first."""
        box = backward_box(key)
        nnz = sum(1 for level in key if level > 0)
        assert len(box) == 2**nnz
        assert box[0] == (1, key)
        assert len({entry for _, entry in box}) == 2**nnz
        for sign, entry in box:
            drop = sum(a - b for a, b in zip(key, entry))
            assert sign == (-1) ** drop
            # Only dimensions that were nonzero may decrease, by at most 1.
            for a, b in zip(key, entry):
                assert b in (a, a - 1)
                assert b >= 0

    def test_unit_level_entries_are_backward_neighbours(self) -> None:
        """The |e| = 1 entries are exactly the backward neighbours."""
        key = (2, 1, 3)
        box = backward_box(key)
        singles = {
            entry
            for _, entry in box
            if sum(a - b for a, b in zip(key, entry)) == 1
        }
        expected = {
            key[:d] + (key[d] - 1,) + key[d + 1 :]
            for d in range(len(key))
            if key[d] > 0
        }
        assert singles == expected


class TestIncrementalMatchesFromScratch:
    """The incremental update reproduces inclusion-exclusion exactly."""

    @pytest.mark.parametrize("nvars", [1, 2, 3, 4])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_growth(self, numpy_bkd, nvars: int, seed: int) -> None:
        """Coefficients agree after every add, and always sum to one."""
        order = _grow_randomly(nvars, seed * 10 + nvars, 30)
        inc = IncrementalSmolyakCoefficients(nvars)
        inc.add(tuple([0] * nvars))
        for key in order:
            inc.add(key)
            keys = inc.keys()
            assert inc.coefficient_list(keys) == _from_scratch(keys, numpy_bkd)
            assert sum(inc.coefficient_list(keys)) == 1

    def test_1d_telescopes(self) -> None:
        """In 1D only the highest level keeps a nonzero coefficient."""
        inc = IncrementalSmolyakCoefficients(1)
        for level in range(6):
            inc.add((level,))
        assert inc.nonzero_items() == [((5,), 1)]


class TestDelta:
    """delta() previews the update without applying it."""

    def test_matches_after_minus_before(self, numpy_bkd) -> None:
        """Each reported change equals the realized coefficient change."""
        inc = IncrementalSmolyakCoefficients(2)
        for key in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            inc.add(key)
        candidate = (2, 0)
        before = {key: inc.coefficient(key) for key in inc.keys()}
        changes = inc.delta(candidate)
        inc.add(candidate)
        for key, sign in changes:
            assert inc.coefficient(key) - before.get(key, 0) == sign

    def test_does_not_mutate(self) -> None:
        """Previewing leaves the object untouched."""
        inc = IncrementalSmolyakCoefficients(2)
        inc.add((0, 0))
        inc.add((1, 0))
        snapshot = {key: inc.coefficient(key) for key in inc.keys()}
        inc.delta((2, 0))
        assert {key: inc.coefficient(key) for key in inc.keys()} == snapshot
        assert inc.nterms() == 2

    def test_raises_when_not_admissible(self) -> None:
        """A missing backward neighbour, or a repeat, is an error."""
        inc = IncrementalSmolyakCoefficients(2)
        inc.add((0, 0))
        with pytest.raises(ValueError, match="cannot be added"):
            inc.delta((2, 0))  # (1, 0) is missing
        with pytest.raises(ValueError, match="cannot be added"):
            inc.delta((0, 0))  # already present

    def test_raises_on_wrong_length(self) -> None:
        """A key of the wrong length is rejected before anything else."""
        inc = IncrementalSmolyakCoefficients(2)
        with pytest.raises(ValueError, match="expected 2"):
            inc.delta((0, 0, 0))

    def test_raises_on_negative_level(self) -> None:
        """Negative levels are not valid subspace keys."""
        inc = IncrementalSmolyakCoefficients(2)
        with pytest.raises(ValueError, match="negative level"):
            inc.can_add((-1, 0))


class TestCanAdd:
    """Admissibility, kept distinct from AdmissibilityCriteria."""

    def test_requires_all_backward_neighbours(self) -> None:
        inc = IncrementalSmolyakCoefficients(2)
        inc.add((0, 0))
        inc.add((1, 0))
        assert inc.can_add((2, 0))
        assert inc.can_add((0, 1))
        # (1, 1) needs both (0, 1) and (1, 0); only (1, 0) is present.
        assert not inc.can_add((1, 1))
        inc.add((0, 1))
        assert inc.can_add((1, 1))

    def test_present_key_cannot_be_readded(self) -> None:
        inc = IncrementalSmolyakCoefficients(2)
        inc.add((0, 0))
        assert not inc.can_add((0, 0))


class TestAccessors:
    """keys(), nonzero_items(), coefficient*(), with_added(), counts."""

    def test_keys_are_insertion_ordered(self) -> None:
        inc = IncrementalSmolyakCoefficients(2)
        order = [(0, 0), (1, 0), (0, 1), (2, 0)]
        for key in order:
            inc.add(key)
        assert inc.keys() == order

    def test_nonzero_items_skips_zeros(self) -> None:
        inc = IncrementalSmolyakCoefficients(1)
        inc.add((0,))
        inc.add((1,))
        assert inc.coefficient((0,)) == 0
        assert ((0,), 0) not in inc.nonzero_items()
        assert inc.nonzero_items() == [((1,), 1)]

    def test_coefficient_of_absent_key_is_zero(self) -> None:
        inc = IncrementalSmolyakCoefficients(2)
        inc.add((0, 0))
        assert inc.coefficient((7, 7)) == 0

    def test_coefficient_list_aligns_with_keys(self) -> None:
        inc = IncrementalSmolyakCoefficients(2)
        for key in [(0, 0), (1, 0), (0, 1)]:
            inc.add(key)
        keys = inc.keys()
        assert inc.coefficient_list(keys) == [
            inc.coefficient(key) for key in keys
        ]

    def test_with_added_leaves_original_unchanged(self, numpy_bkd) -> None:
        """The copy holds the extra keys; the original does not."""
        inc = IncrementalSmolyakCoefficients(2)
        for key in [(0, 0), (1, 0), (0, 1)]:
            inc.add(key)
        original = {key: inc.coefficient(key) for key in inc.keys()}

        extended = inc.with_added([(1, 1)])
        assert {key: inc.coefficient(key) for key in inc.keys()} == original
        assert inc.nterms() == 3
        assert extended.nterms() == 4

        keys = extended.keys()
        assert extended.coefficient_list(keys) == _from_scratch(
            keys, numpy_bkd
        )

    def test_nterms_counts_zero_coefficient_keys(self) -> None:
        inc = IncrementalSmolyakCoefficients(1)
        inc.add((0,))
        inc.add((1,))
        assert inc.nterms() == 2
        assert len(inc.nonzero_items()) == 1

    def test_nvars(self) -> None:
        assert IncrementalSmolyakCoefficients(3).nvars() == 3

    def test_rejects_nonpositive_nvars(self) -> None:
        with pytest.raises(ValueError, match="nvars must be positive"):
            IncrementalSmolyakCoefficients(0)
