"""Tests for the sign convention imposed on an eigenbasis.

An eigenvector is determined only up to sign, so LAPACK's choice is not
portable and a convention has to be imposed. The convention is only
worth anything if it is *stable*: the same data must give the same
signs regardless of how many modes were asked for, or two callers
comparing bases -- or one reloading a stored basis -- see spurious
differences.

That stability is what these pin. The mathematical invariance under a
per-column flip is asserted first, since it is what licenses touching
the signs at all.
"""

import numpy as np
from pyapprox.surrogates.kle.utils import adjust_sign_eig


def _orthonormal(bkd, nrows=9, ncols=5, seed=0):
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.standard_normal((nrows, ncols)))
    return bkd.array(q)


def _pivot_values(bkd, basis):
    """The entry in each column that the convention makes positive."""
    arr = np.asarray(bkd.to_numpy(basis))
    return np.array(
        [arr[int(np.argmax(np.abs(arr[:, j]))), j]
         for j in range(arr.shape[1])]
    )


class TestFlippingIsMathematicallyFree:
    """Why the signs may be changed at all."""

    def test_eigen_equation_survives_a_per_column_flip(self, bkd) -> None:
        rng = np.random.RandomState(0)
        b = bkd.array(rng.standard_normal((6, 6)))
        matrix = bkd.dot(b, b.T)
        eig_vals, eig_vecs = bkd.eigh(matrix)
        flipped = adjust_sign_eig(bkd.copy(eig_vecs), bkd)
        bkd.assert_allclose(
            bkd.dot(matrix, flipped), flipped * eig_vals, atol=1e-12
        )

    def test_orthonormality_survives(self, bkd) -> None:
        basis = adjust_sign_eig(bkd.copy(_orthonormal(bkd)), bkd)
        bkd.assert_allclose(
            bkd.dot(basis.T, basis), bkd.eye(5), atol=1e-12
        )

    def test_spectral_reconstruction_survives(self, bkd) -> None:
        """The covariance a KLE represents is unchanged by the flip."""
        rng = np.random.RandomState(0)
        b = bkd.array(rng.standard_normal((6, 6)))
        matrix = bkd.dot(b, b.T)
        eig_vals, eig_vecs = bkd.eigh(matrix)
        flipped = adjust_sign_eig(bkd.copy(eig_vecs), bkd)
        bkd.assert_allclose(
            bkd.dot(flipped * eig_vals, flipped.T), matrix, atol=1e-10
        )


class TestSliceStability:
    """The property the previous convention lacked."""

    def test_truncating_commutes_with_canonicalizing(self, bkd) -> None:
        """Ten modes canonicalized then cut to three must equal three
        canonicalized directly. Otherwise two callers asking for
        different term counts disagree about the same data."""
        raw = _orthonormal(bkd, nrows=12, ncols=8)
        whole = adjust_sign_eig(bkd.copy(raw), bkd)
        for nterms in (1, 2, 5, 8):
            part = adjust_sign_eig(bkd.copy(raw[:, :nterms]), bkd)
            bkd.assert_allclose(
                whole[:, :nterms], part, rtol=0.0, atol=0.0
            )

    def test_a_column_is_unaffected_by_its_neighbours(self, bkd) -> None:
        """Reordering the other columns cannot change this one's sign."""
        raw = _orthonormal(bkd, nrows=10, ncols=4)
        forward = adjust_sign_eig(bkd.copy(raw), bkd)
        reversed_cols = adjust_sign_eig(
            bkd.copy(bkd.flip(raw, axis=(1,))), bkd
        )
        bkd.assert_allclose(
            forward[:, 0], bkd.flip(reversed_cols, axis=(1,))[:, 0],
            rtol=0.0, atol=0.0,
        )


class TestConvention:
    def test_largest_magnitude_entry_is_positive(self, bkd) -> None:
        basis = adjust_sign_eig(bkd.copy(_orthonormal(bkd)), bkd)
        assert bool((_pivot_values(bkd, basis) > 0.0).all())

    def test_is_idempotent(self, bkd) -> None:
        once = adjust_sign_eig(bkd.copy(_orthonormal(bkd)), bkd)
        twice = adjust_sign_eig(bkd.copy(once), bkd)
        bkd.assert_allclose(twice, once, rtol=0.0, atol=0.0)

    def test_opposite_inputs_reach_the_same_output(self, bkd) -> None:
        """A basis and its negation are the same basis."""
        raw = _orthonormal(bkd)
        bkd.assert_allclose(
            adjust_sign_eig(bkd.copy(raw), bkd),
            adjust_sign_eig(bkd.copy(-raw), bkd),
            rtol=0.0,
            atol=0.0,
        )

    def test_symmetric_column_is_handled_deterministically(
        self, bkd
    ) -> None:
        """Ties are the common case, not a corner: KLE eigenfunctions
        are symmetric or antisymmetric about the domain centre, so their
        extreme values match at both ends."""
        col = bkd.array(np.array([[1.0], [0.5], [-0.5], [-1.0]]))
        bkd.assert_allclose(
            adjust_sign_eig(bkd.copy(col), bkd),
            adjust_sign_eig(bkd.copy(-col), bkd),
            rtol=0.0,
            atol=0.0,
        )

    def test_a_leading_node_does_not_decide_the_sign(self, bkd) -> None:
        """A mode with a node at the start of the domain -- routine under
        Dirichlet conditions -- would otherwise be signed by noise."""
        col = bkd.array(np.array([[1e-17], [0.3], [-0.9]]))
        signed = adjust_sign_eig(bkd.copy(col), bkd)
        bkd.assert_allclose(signed, -col, rtol=1e-12)

    def test_zero_column_is_left_alone(self, bkd) -> None:
        zeros = bkd.zeros((4, 2))
        bkd.assert_allclose(
            adjust_sign_eig(bkd.copy(zeros), bkd), zeros, atol=0.0
        )
