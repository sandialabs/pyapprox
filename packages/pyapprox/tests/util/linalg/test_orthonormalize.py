"""Tests for orthonormalization as an injected choice.

The dense implementation is one LAPACK call, so testing that it
orthonormalizes would be testing LAPACK. What is worth testing is the
seam: that an alternative can be supplied, that it is actually used at
every site, and that the argument checks catch the shapes a randomized
decomposition can produce by accident.
"""

from typing import Any

import numpy as np
import pytest
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.orthonormalize import (
    CholeskyQR,
    HouseholderQR,
    OrthonormalizerProtocol,
)
from pyapprox.util.linalg.randomized import (
    DenseMatVecOperator,
    TwoPassRandomizedSVD,
    randomized_symmetric_eigendecomposition,
)


def _tall(bkd: Backend, nrows: int = 60, ncols: int = 8) -> Any:
    return bkd.array(
        np.random.RandomState(0).standard_normal((nrows, ncols))
    )


def _conditioned(bkd: Backend, cond: float, nrows: int = 400,
                 ncols: int = 10) -> Any:
    """A tall array with a prescribed condition number.

    Built from an explicit singular-value spread rather than left to a
    random draw, because the property under test *is* the conditioning:
    a random tall array is well conditioned and would exercise none of
    the behaviour that separates these implementations.
    """
    rng = np.random.RandomState(0)
    left = np.linalg.qr(rng.standard_normal((nrows, ncols)))[0]
    right = np.linalg.qr(rng.standard_normal((ncols, ncols)))[0]
    spread = np.logspace(0, -np.log10(cond), ncols)
    return bkd.array(left @ np.diag(spread) @ right)


class _CountingOrthonormalizer:
    """Delegates to the dense one, recording how often it was called."""

    def __init__(self, bkd: Backend) -> None:
        self._inner = HouseholderQR(bkd)
        self.calls = 0

    def __call__(self, array: Any) -> Any:
        self.calls += 1
        return self._inner(array)


class TestTheDenseImplementation:
    def test_satisfies_the_protocol(self, bkd: Backend) -> None:
        """So a boundary can reject something that is merely callable."""
        assert isinstance(HouseholderQR(bkd), OrthonormalizerProtocol)

    def test_returns_orthonormal_columns(self, bkd: Backend) -> None:
        factor = HouseholderQR(bkd)(_tall(bkd))
        bkd.assert_allclose(
            bkd.dot(factor.T, factor), bkd.eye(8), atol=1e-13
        )

    def test_preserves_the_span(self, bkd: Backend) -> None:
        """Orthonormal columns of the wrong space would still pass above.

        Checked by projecting the input onto the result and requiring
        the input back, which fails for any factor spanning something
        else.
        """
        array = _tall(bkd)
        factor = HouseholderQR(bkd)(array)
        projected = bkd.dot(
            factor, bkd.dot(factor.T, array)
        )
        bkd.assert_allclose(projected, array, atol=1e-12)

    def test_is_reduced_not_full(self, bkd: Backend) -> None:
        """A full QR would return an (nrows, nrows) factor.

        At the ambient dimensions this exists for, that factor cannot be
        formed at all, so the mode is part of the contract rather than a
        default worth inheriting.
        """
        assert HouseholderQR(bkd)(_tall(bkd, 60, 8)).shape == (60, 8)

    def test_rejects_a_1d_array(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="2D"):
            HouseholderQR(bkd)(bkd.array([1.0, 2.0, 3.0]))

    def test_rejects_a_wide_array(self, bkd: Backend) -> None:
        """More columns than rows means the rank exceeds the dimension.

        ``qr`` accepts this and silently returns fewer columns than it
        was given, which surfaces later as a basis with modes missing.
        """
        with pytest.raises(ValueError, match="tall"):
            HouseholderQR(bkd)(_tall(bkd, 5, 9))


class TestTheBlockedImplementation:
    """Cholesky-QR, which never holds the whole array.

    The oracle is :class:`HouseholderQR`, which is backward stable and
    so gives the answer conditioning cannot spoil. What is tested is
    where the blocked one matches it, where repeating is what makes it
    match, and that it refuses rather than approximating beyond that.
    """

    def _orthonormality(self, bkd: Backend, factor: Any) -> float:
        ncols = int(factor.shape[1])
        gram = bkd.dot(factor.T, factor)
        return float(
            np.abs(
                bkd.to_numpy(gram) - np.eye(ncols)
            ).max()
        )

    def test_satisfies_the_protocol(self, bkd: Backend) -> None:
        assert isinstance(CholeskyQR(bkd), OrthonormalizerProtocol)

    def test_returns_orthonormal_columns(self, bkd: Backend) -> None:
        factor = CholeskyQR(bkd)(_tall(bkd, 400, 10))
        assert self._orthonormality(bkd, factor) < 1e-13

    def test_preserves_the_span(self, bkd: Backend) -> None:
        """Orthonormal columns of the wrong space would still pass above."""
        array = _tall(bkd, 400, 10)
        factor = CholeskyQR(bkd)(array)
        bkd.assert_allclose(
            bkd.dot(factor, bkd.dot(factor.T, array)),
            array,
            atol=1e-12,
        )

    @pytest.mark.parametrize("max_bytes", [64, 4096, None])
    def test_the_block_size_does_not_change_the_result(
        self, bkd: Backend, max_bytes: int
    ) -> None:
        """Blocking re-associates a sum; it must not move the answer."""
        array = _conditioned(bkd, 1e3)
        factor = CholeskyQR(bkd, max_bytes=max_bytes)(array)
        assert self._orthonormality(bkd, factor) < 1e-13

    @pytest.mark.parametrize("cond", [1e2, 1e5, 1e8])
    def test_repeating_is_what_makes_it_accurate(
        self, bkd: Backend, cond: float
    ) -> None:
        """One pass loses half the digits; two get them back.

        Forming the Gram squares the condition number, so a single
        pass degrades as the input worsens while the repeated one does
        not. Asserted as a *comparison* rather than as two thresholds,
        because the gap is the point: at cond 1e8 one pass leaves no
        orthogonality at all and two match Householder.
        """
        array = _conditioned(bkd, cond)
        once = self._orthonormality(
            bkd, CholeskyQR(bkd, npasses=1)(array)
        )
        twice = self._orthonormality(bkd, CholeskyQR(bkd)(array))
        assert twice <= once
        assert twice < 1e-13

    def test_two_passes_match_householder(self, bkd: Backend) -> None:
        """At a conditioning where one pass visibly does not."""
        array = _conditioned(bkd, 1e8)
        dense = self._orthonormality(bkd, HouseholderQR(bkd)(array))
        blocked = self._orthonormality(bkd, CholeskyQR(bkd)(array))
        assert blocked < 1e-13
        assert blocked < 1e4 * max(dense, 1e-16)

    def test_a_hopeless_conditioning_raises(self, bkd: Backend) -> None:
        """Naming the cause, not numpy's "not positive definite".

        The Gram *is* positive definite in exact arithmetic; what
        failed is that squaring the condition number exhausted the
        precision, and a caller needs to be told that rather than left
        to doubt their data.
        """
        array = _conditioned(bkd, 1e12)
        with pytest.raises(np.linalg.LinAlgError, match="squares"):
            CholeskyQR(bkd)(array)

    def test_rejects_a_non_positive_pass_count(
        self, bkd: Backend
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CholeskyQR(bkd, npasses=0)

    def test_rejects_a_wide_array(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="tall"):
            CholeskyQR(bkd)(_tall(bkd, 5, 9))

    def test_is_injectable_into_a_decomposition(
        self, bkd: Backend
    ) -> None:
        """The reason the seam exists: swap it in, same factorization.

        Every factor is compared elementwise, not merely the subspace
        or the singular values. A blocked QR returning a *different*
        orthonormal basis for the same span would leave the
        reconstruction and the spectrum intact while changing every
        mode the decomposition reports -- so a subspace-level check
        would pass over exactly the substitution that must not happen.

        The tolerance is tight because the agreement is: measured at
        5e-16 on ``U`` and 4.5e-16 relative on the singular values,
        which is roundoff rather than an algorithmic difference. The
        two QRs are not bitwise identical, so this is `atol`, not
        equality.
        """
        rng = np.random.RandomState(0)
        matrix = rng.standard_normal((200, 30)) @ np.diag(
            np.logspace(0, -3, 30)
        )
        operator = DenseMatVecOperator(bkd.array(matrix), bkd)
        kwargs = dict(noversampling=4, npower_iters=1, seed=1)
        dense = TwoPassRandomizedSVD(operator, **kwargs).compute(5)
        blocked = TwoPassRandomizedSVD(
            operator, orthonormalizer=CholeskyQR(bkd), **kwargs
        ).compute(5)
        for got, expected in zip(blocked, dense):
            bkd.assert_allclose(got, expected, atol=1e-12)


class TestTheSeamIsUsed:
    """An injected implementation must reach every call site.

    A decomposition that used the injected one in some places and
    ``numpy.linalg.qr`` in others would still produce the right answer
    with the dense default, and would hold the whole sketch the moment
    a blocked implementation was supplied -- the exact failure the seam
    exists to prevent, invisible to a correctness test.
    """

    def _operator(self, bkd: Backend) -> Any:
        rng = np.random.RandomState(0)
        matrix = rng.standard_normal((80, 30)) @ np.diag(
            np.logspace(0, -3, 30)
        )
        return DenseMatVecOperator(bkd.array(matrix), bkd)

    @pytest.mark.parametrize("npower_iters", [0, 1, 3])
    def test_two_pass_svd_calls_it_once_per_iteration_plus_once(
        self, bkd: Backend, npower_iters: int
    ) -> None:
        """One orthonormalization per power iteration, one at the end."""
        counter = _CountingOrthonormalizer(bkd)
        TwoPassRandomizedSVD(
            self._operator(bkd),
            noversampling=4,
            npower_iters=npower_iters,
            seed=1,
            orthonormalizer=counter,
        ).compute(5)
        assert counter.calls == npower_iters + 1

    def test_injecting_the_default_changes_nothing(
        self, bkd: Backend
    ) -> None:
        """The wrapper must be transparent, or later comparisons lie."""
        operator = self._operator(bkd)
        kwargs = dict(noversampling=4, npower_iters=1, seed=1)
        expected = TwoPassRandomizedSVD(operator, **kwargs).compute(5)
        got = TwoPassRandomizedSVD(
            operator,
            orthonormalizer=_CountingOrthonormalizer(bkd),
            **kwargs,
        ).compute(5)
        for got_factor, expected_factor in zip(got, expected):
            bkd.assert_allclose(
                got_factor, expected_factor, atol=0.0
            )

    def test_the_symmetric_eigendecomposition_uses_it(
        self, bkd: Backend
    ) -> None:
        """The free function takes it as an argument, having no self."""
        rng = np.random.RandomState(0)
        raw = rng.standard_normal((40, 40))
        matrix = bkd.array(raw @ raw.T)
        counter = _CountingOrthonormalizer(bkd)
        randomized_symmetric_eigendecomposition(
            lambda vecs: bkd.dot(matrix, vecs),
            nvars=40,
            rank=5,
            bkd=bkd,
            noversampling=4,
            npower_iters=2,
            seed=1,
            orthonormalizer=counter,
        )
        assert counter.calls == 3

    def test_a_non_orthonormalizer_is_rejected(
        self, bkd: Backend
    ) -> None:
        """Named at construction rather than at the first sketch."""
        with pytest.raises(
            TypeError, match="OrthonormalizerProtocol"
        ):
            TwoPassRandomizedSVD(
                self._operator(bkd), orthonormalizer=object()
            )
