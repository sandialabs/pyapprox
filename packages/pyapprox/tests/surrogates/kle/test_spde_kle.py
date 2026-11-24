"""Tests for spde_matern_kle_from_precision_eigenpairs.

The eigenpairs are synthetic: an orthonormal basis of R^n stands for the
M-orthonormal eigenvectors (M = I), so the expected KLE follows from the
definitions alone and no discretization enters.
"""

from typing import Any, Tuple

import numpy as np
import pytest

from pyapprox.surrogates.kle.spde_kle import (
    spde_matern_kle_from_precision_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend

_NNODES, _NTERMS = 9, 4
_VARIANCE, _SIGMA = 0.37, 1.8
_GAMMA, _DELTA, _XI = 0.04, 1.0, 0.1


def _eigenpairs(bkd: Backend[Array]) -> Tuple[Array, Array]:
    """Unsorted precision eigenvalues and orthonormal eigenvectors."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((_NNODES, _NTERMS)))
    return bkd.asarray(np.array([2.5, 0.7, 4.0, 1.3])), bkd.asarray(q)


def _build(bkd: Backend[Array], mu: Array, phi: Array, **kwargs: Any) -> Any:
    return spde_matern_kle_from_precision_eigenpairs(
        mu,
        phi,
        _VARIANCE,
        sigma=_SIGMA,
        bkd=bkd,
        gamma=_GAMMA,
        delta=_DELTA,
        xi=_XI,
        **kwargs,
    )


class TestSPDEMaternKLEFromPrecisionEigenpairs:
    def test_eigenvalues_are_scaled_inverse_squares_descending(
        self, bkd: Backend[Array]
    ) -> None:
        mu, phi = _eigenpairs(bkd)
        kle = _build(bkd, mu, phi)
        expected = -np.sort(-1.0 / (_VARIANCE * bkd.to_numpy(mu) ** 2))
        bkd.assert_allclose(kle.eigenvalues(), bkd.asarray(expected), rtol=1e-14)

    def test_covariance_is_scaled_inverse_precision_squared(
        self, bkd: Backend[Array]
    ) -> None:
        # With M = I the field covariance is sigma^2 / v * Phi mu^-2 Phi^T,
        # whatever order or signs the eigenpairs arrive in.
        mu, phi = _eigenpairs(bkd)
        kle = _build(bkd, mu, phi)
        w = kle.weighted_eigenvectors()
        expected = (_SIGMA**2 / _VARIANCE) * (phi / mu**2) @ phi.T
        bkd.assert_allclose(w @ w.T, expected, rtol=1e-13)

    def test_input_order_and_signs_do_not_change_the_kle(
        self, bkd: Backend[Array]
    ) -> None:
        mu, phi = _eigenpairs(bkd)
        reference = _build(bkd, mu, phi)
        perm = bkd.asarray(np.array([3, 0, 2, 1]), dtype=bkd.index_dtype())
        signs = bkd.asarray(np.array([-1.0, 1.0, -1.0, -1.0]))
        other = _build(bkd, mu[perm], phi[:, perm] * signs)
        bkd.assert_allclose(
            other.eigenvectors(), reference.eigenvectors(), rtol=0.0, atol=0.0
        )
        bkd.assert_allclose(
            other.eigenvalues(), reference.eigenvalues(), rtol=0.0, atol=0.0
        )

    def test_scalar_mean_is_broadcast(self, bkd: Backend[Array]) -> None:
        mu, phi = _eigenpairs(bkd)
        kle = _build(bkd, mu, phi, mean_field=0.25)
        bkd.assert_allclose(kle.mean_field(), bkd.full((_NNODES,), 0.25))
        bkd.assert_allclose(
            kle(bkd.zeros((_NTERMS, 2))), bkd.full((_NNODES, 2), 0.25)
        )

    def test_records_the_spde_coefficients(self, bkd: Backend[Array]) -> None:
        mu, phi = _eigenpairs(bkd)
        kle = _build(bkd, mu, phi)
        assert kle.correlation_length() == pytest.approx(np.sqrt(_GAMMA / _DELTA))

    def test_rejects_two_dimensional_eigenvalues(self, bkd: Backend[Array]) -> None:
        mu, phi = _eigenpairs(bkd)
        with pytest.raises(ValueError, match="1D"):
            _build(bkd, bkd.reshape(mu, (_NTERMS, 1)), phi)

    def test_rejects_mismatched_eigenvector_count(
        self, bkd: Backend[Array]
    ) -> None:
        mu, phi = _eigenpairs(bkd)
        with pytest.raises(ValueError, match="eigenvectors must have shape"):
            _build(bkd, mu, phi[:, :3])

    def test_rejects_nonpositive_eigenvalue(self, bkd: Backend[Array]) -> None:
        mu, phi = _eigenpairs(bkd)
        bad = bkd.asarray(np.array([2.5, 0.0, 4.0, 1.3]))
        with pytest.raises(ValueError, match="positive"):
            _build(bkd, bad, phi)


class TestSPDEMaternKLEFromPrecisionEigenpairsAutograd:
    def test_eigenvalue_gradient_flows_to_precision_eigenvalues(
        self, torch_bkd: Any
    ) -> None:
        import torch

        bkd = torch_bkd
        mu, phi = _eigenpairs(bkd)
        mu = mu.clone().requires_grad_(True)
        kle = _build(bkd, mu, phi)
        kle.eigenvalues().sum().backward()
        assert mu.grad is not None
        expected = -2.0 / (_VARIANCE * mu.detach() ** 3)
        torch.testing.assert_close(mu.grad, expected, rtol=1e-12, atol=0.0)
