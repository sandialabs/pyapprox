"""Tests for forming the ambient basis from its small right factor.

The oracle throughout is the resident path: a lift is correct when it
produces what ``MethodOfSnapshotsSolver`` produces from the same data,
to roundoff. That is a sharper test than any property asserted about the
blocked computation alone, and it is the comparison that would catch a
sign convention or a block boundary being handled differently.

The sign tests use a right factor that has *not* been canonicalized, so
several columns are genuinely negative. A factor taken straight from the
solver has already been signed, and a test built on one would exercise
the sign path without ever flipping anything.
"""

from typing import Any, Tuple

import numpy as np
import pytest
from pyapprox.surrogates.kle.basis_lift import (
    lift_basis,
    pivot_signs,
)
from pyapprox.surrogates.kle.basis_sinks import ArrayBasisSink
from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    MethodOfSnapshotsSolver,
)
from pyapprox.surrogates.kle.snapshot_sources import ArraySnapshotSource
from pyapprox.surrogates.kle.utils import eigenvector_signs
from pyapprox.util.backends.protocols import Backend

NSTATES, NSAMPLES, NTERMS = 400, 30, 8


def _snapshots(bkd: Backend, seed: int = 0) -> Any:
    """Low-rank with a decaying spectrum, as real snapshots are."""
    rng = np.random.RandomState(seed)
    left = rng.normal(size=(NSTATES, 16))
    right = rng.normal(size=(16, NSAMPLES))
    return bkd.array(left @ np.diag(np.logspace(0, -3, 16)) @ right)


def _raw_right_factor(
    bkd: Backend, snapshots: Any, nterms: int = NTERMS
) -> Any:
    """``Q / sqrt(lambda)`` from the Gram, with signs left alone.

    Deliberately not run through the convention: canonicalized columns
    would all come back positive and the sign tests would pass without
    flipping anything.
    """
    raw = bkd.to_numpy(snapshots)
    vals, vecs = np.linalg.eigh(raw.T @ raw)
    vals = vals[::-1][:nterms]
    vecs = vecs[:, ::-1][:, :nterms]
    return bkd.array(vecs / np.sqrt(vals))


def _pair(bkd: Backend, seed: int = 0) -> Tuple[Any, Any]:
    """Snapshots and an unsigned right factor for them."""
    snapshots = _snapshots(bkd, seed)
    return snapshots, _raw_right_factor(bkd, snapshots)


def _lift(
    bkd: Backend, snapshots: Any, factor: Any, **kwargs: Any
) -> Any:
    """Lift into a fresh array sink, since a sink is single-use."""
    nterms = int(factor.shape[1])
    return lift_basis(
        ArraySnapshotSource(snapshots, bkd),
        factor,
        ArrayBasisSink(int(snapshots.shape[0]), nterms, bkd),
        bkd,
        **kwargs,
    )


class TestTheLiftMatchesTheResidentSolver:
    """The comparison that makes everything else secondary."""

    def test_reproduces_the_solver_basis(self, bkd: Backend) -> None:
        """Same data, same eigenvectors, to roundoff.

        The solver signs its own factor, so this goes through
        ``coordinates`` to recover the very factor it used -- otherwise
        the two would differ by a sign and the comparison would be
        testing the convention rather than the lift.
        """
        snapshots = _snapshots(bkd)
        decomposition = MethodOfSnapshotsSolver(bkd).solve(
            snapshots, nterms=NTERMS
        )
        sqrt_vals = bkd.sqrt(decomposition.eigenvalues)
        factor = (
            decomposition.coordinates.T / sqrt_vals
        ) / sqrt_vals
        bkd.assert_allclose(
            _lift(bkd, snapshots, factor).to_array(),
            decomposition.eigenvectors,
            atol=1e-13,
        )

    def test_the_lifted_basis_is_orthonormal(
        self, bkd: Backend
    ) -> None:
        """``V^T V = I`` for the unweighted metric used here."""
        snapshots, factor = _pair(bkd)
        basis = _lift(bkd, snapshots, factor).to_array()
        bkd.assert_allclose(
            bkd.dot(basis.T, basis),
            bkd.eye(NTERMS),
            atol=1e-12,
        )


class TestBlockingChangesNothing:
    """Blocking re-associates sums; it must not move the answer.

    A result that depended on the budget would mean a caller's memory
    limit silently selected which numbers they got.
    """

    @pytest.mark.parametrize("max_bytes", [8, 300, 1 << 20])
    def test_the_basis_is_independent_of_the_block_size(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        snapshots, factor = _pair(bkd)
        bkd.assert_allclose(
            _lift(bkd, snapshots, factor, max_bytes=max_bytes).to_array(),
            _lift(bkd, snapshots, factor).to_array(),
            atol=1e-14,
        )

    @pytest.mark.parametrize("max_bytes", [8, 300, 1 << 20])
    def test_the_signs_are_independent_of_the_block_size(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        """A running max compares unmodified elements, so exactly so.

        Tighter than the basis comparison above deliberately: this one
        should be bit-identical rather than merely close, because no
        arithmetic is performed on the candidates.
        """
        snapshots, factor = _pair(bkd)
        source = ArraySnapshotSource(snapshots, bkd)
        bkd.assert_allclose(
            pivot_signs(source, factor, bkd, max_bytes=max_bytes),
            pivot_signs(source, factor, bkd),
            atol=0.0,
        )


class TestTheSignConvention:
    """Matching the resident rule, computed without the resident array."""

    def test_the_test_data_actually_has_negative_signs(
        self, bkd: Backend
    ) -> None:
        """Otherwise the rest of this class proves nothing.

        An assertion about the fixture rather than the code: a right
        factor whose columns are all positive would let a lift that
        ignored signs entirely pass every test below.
        """
        snapshots, factor = _pair(bkd)
        signs = pivot_signs(
            ArraySnapshotSource(snapshots, bkd), factor, bkd
        )
        assert int((bkd.to_numpy(signs) < 0).sum()) > 0

    def test_signs_match_the_resident_convention(
        self, bkd: Backend
    ) -> None:
        """The streaming pivot and ``argmax`` pick the same entry."""
        snapshots, factor = _pair(bkd)
        _, expected = eigenvector_signs(
            bkd.dot(snapshots, factor), bkd
        )
        bkd.assert_allclose(
            pivot_signs(
                ArraySnapshotSource(snapshots, bkd), factor, bkd
            ),
            expected,
            atol=0.0,
        )

    def test_a_signed_lift_equals_the_signed_dense_product(
        self, bkd: Backend
    ) -> None:
        snapshots, factor = _pair(bkd)
        dense, signs = eigenvector_signs(
            bkd.dot(snapshots, factor), bkd
        )
        bkd.assert_allclose(
            _lift(bkd, snapshots, factor).to_array(),
            dense,
            atol=1e-14,
        )
        assert int((bkd.to_numpy(signs) < 0).sum()) > 0

    def test_opting_out_gives_the_unsigned_product(
        self, bkd: Backend
    ) -> None:
        """The flag's whole contract: ``S R`` with nothing applied."""
        snapshots, factor = _pair(bkd)
        bkd.assert_allclose(
            _lift(
                bkd, snapshots, factor, apply_sign_convention=False
            ).to_array(),
            bkd.dot(snapshots, factor),
            atol=1e-14,
        )

    def test_opting_out_visibly_differs_when_signs_are_negative(
        self, bkd: Backend
    ) -> None:
        """So the default is doing something, not merely costing a pass."""
        snapshots, factor = _pair(bkd)
        signed = _lift(bkd, snapshots, factor).to_array()
        unsigned = _lift(
            bkd, snapshots, factor, apply_sign_convention=False
        ).to_array()
        difference = float(
            np.abs(
                bkd.to_numpy(signed) - bkd.to_numpy(unsigned)
            ).max()
        )
        assert difference > 1e-3

    def test_the_two_paths_differ_only_by_the_signs(
        self, bkd: Backend
    ) -> None:
        """Nothing else changes when the convention is applied."""
        snapshots, factor = _pair(bkd)
        source = ArraySnapshotSource(snapshots, bkd)
        signs = pivot_signs(source, factor, bkd)
        unsigned = _lift(
            bkd, snapshots, factor, apply_sign_convention=False
        ).to_array()
        bkd.assert_allclose(
            _lift(bkd, snapshots, factor).to_array(),
            unsigned * signs,
            atol=1e-14,
        )

    def test_a_zero_column_keeps_a_positive_sign(
        self, bkd: Backend
    ) -> None:
        """``sign(0)`` is 0, which would erase the column rather than flip it.

        Not reachable through a solver, which rejects a zero mode, but
        reachable by a caller assembling a factor themselves -- and the
        failure would be a silently blanked basis column.
        """
        snapshots, factor = _pair(bkd)
        raw = bkd.to_numpy(factor).copy()
        raw[:, 2] = 0.0
        signs = pivot_signs(
            ArraySnapshotSource(snapshots, bkd), bkd.array(raw), bkd
        )
        bkd.assert_allclose(
            signs[2:3], bkd.array([1.0]), atol=0.0
        )


class TestTheNumberOfPassesIsTheCostPromised:
    """The opt-out exists to save a read; it has to actually save one."""

    class _CountingSource:
        """Wraps a source, recording how often it is iterated."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.passes = 0

        def bkd(self) -> Any:
            return self._inner.bkd()

        def nstates(self) -> int:
            return self._inner.nstates()

        def nsamples(self) -> int:
            return self._inner.nsamples()

        def row_blocks(self, max_bytes: Any = None) -> Any:
            self.passes += 1
            return self._inner.row_blocks(max_bytes)

    def test_signing_costs_two_passes(self, bkd: Backend) -> None:
        snapshots, factor = _pair(bkd)
        source = self._CountingSource(
            ArraySnapshotSource(snapshots, bkd)
        )
        lift_basis(
            source,
            factor,
            ArrayBasisSink(NSTATES, NTERMS, bkd),
            bkd,
        )
        assert source.passes == 2

    def test_opting_out_costs_one(self, bkd: Backend) -> None:
        snapshots, factor = _pair(bkd)
        source = self._CountingSource(
            ArraySnapshotSource(snapshots, bkd)
        )
        lift_basis(
            source,
            factor,
            ArrayBasisSink(NSTATES, NTERMS, bkd),
            bkd,
            apply_sign_convention=False,
        )
        assert source.passes == 1


class TestMismatchedArgumentsAreRejected:
    """Each of these would otherwise fail mid-loop, after real I/O."""

    def test_a_right_factor_of_the_wrong_height(
        self, bkd: Backend
    ) -> None:
        """Snapshots and factor from different datasets."""
        snapshots, _ = _pair(bkd)
        wrong = bkd.array(
            np.random.RandomState(1).standard_normal(
                (NSAMPLES + 3, NTERMS)
            )
        )
        with pytest.raises(ValueError, match="snapshots"):
            _lift(bkd, snapshots, wrong)

    def test_a_1d_right_factor(self, bkd: Backend) -> None:
        snapshots, _ = _pair(bkd)
        with pytest.raises(ValueError, match="2D"):
            lift_basis(
                ArraySnapshotSource(snapshots, bkd),
                bkd.array(np.zeros(NSAMPLES)),
                ArrayBasisSink(NSTATES, 1, bkd),
                bkd,
            )

    def test_a_sink_sized_for_the_wrong_ambient_dimension(
        self, bkd: Backend
    ) -> None:
        snapshots, factor = _pair(bkd)
        with pytest.raises(ValueError, match="sized for"):
            lift_basis(
                ArraySnapshotSource(snapshots, bkd),
                factor,
                ArrayBasisSink(NSTATES + 5, NTERMS, bkd),
                bkd,
            )

    def test_a_sink_sized_for_the_wrong_number_of_terms(
        self, bkd: Backend
    ) -> None:
        snapshots, factor = _pair(bkd)
        with pytest.raises(ValueError, match="terms"):
            lift_basis(
                ArraySnapshotSource(snapshots, bkd),
                factor,
                ArrayBasisSink(NSTATES, NTERMS + 2, bkd),
                bkd,
            )

    def test_a_bare_array_where_a_source_was_wanted(
        self, bkd: Backend
    ) -> None:
        """Caught at the boundary, naming the protocol."""
        snapshots, factor = _pair(bkd)
        with pytest.raises(TypeError, match="SnapshotSourceProtocol"):
            lift_basis(
                snapshots,
                factor,
                ArrayBasisSink(NSTATES, NTERMS, bkd),
                bkd,
            )

    def test_a_bare_array_where_a_sink_was_wanted(
        self, bkd: Backend
    ) -> None:
        snapshots, factor = _pair(bkd)
        with pytest.raises(TypeError, match="BasisSinkProtocol"):
            lift_basis(
                ArraySnapshotSource(snapshots, bkd),
                factor,
                bkd.zeros((NSTATES, NTERMS)),
                bkd,
            )
