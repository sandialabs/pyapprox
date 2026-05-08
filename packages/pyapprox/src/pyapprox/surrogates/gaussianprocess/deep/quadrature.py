"""Quadrature rules for Deep GP propagation and mini-batching."""

from typing import Generic, List, Optional, Tuple

import numpy as np

from pyapprox.probability import GaussianMarginal
from pyapprox.surrogates.affine.univariate import create_basis_1d
from pyapprox.util.backends.protocols import Array, Backend


def _gauss_hermite_1d(
    order: int, bkd: Backend[Array],
) -> Tuple[Array, Array]:
    """1D Gauss-Hermite quadrature for N(0, 1) using pyapprox basis."""
    basis = create_basis_1d(GaussianMarginal(0.0, 1.0, bkd), bkd)
    basis.set_nterms(order)
    pts, wts = basis.gauss_quadrature_rule(order)
    return pts[0, :], bkd.flatten(wts)


# ---------------------------------------------------------------------------
# Propagation quadrature (for uncertainty through layers)
# ---------------------------------------------------------------------------


class MonteCarloRule(Generic[Array]):
    """IID standard normal samples with equal weights 1/S."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(
        self, n_samples: int, dim: int, seed: Optional[int] = None
    ) -> Tuple[Array, Array]:
        """Return (nodes, weights) where nodes ~ N(0,I).

        Parameters
        ----------
        n_samples : int
            Number of samples S.
        dim : int
            Dimensionality of each sample.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        nodes : Array, shape (n_samples, dim)
        weights : Array, shape (n_samples,)
        """
        bkd = self._bkd
        rng = np.random.RandomState(seed)
        nodes = bkd.array(rng.randn(n_samples, dim))
        weights = bkd.full((n_samples,), 1.0 / n_samples)
        return nodes, weights


class GaussHermiteRule(Generic[Array]):
    """Tensor-product Gauss-Hermite quadrature for Gaussian expectations."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(
        self, order: int, dim: int
    ) -> Tuple[Array, Array]:
        """Return (nodes, weights) for E_{N(0,I)}[f].

        Parameters
        ----------
        order : int
            Number of quadrature points per dimension.
        dim : int
            Dimensionality.

        Returns
        -------
        nodes : Array, shape (order^dim, dim)
        weights : Array, shape (order^dim,)
        """
        bkd = self._bkd
        nodes_1d, weights_1d = _gauss_hermite_1d(order, bkd)

        if dim == 1:
            return bkd.reshape(nodes_1d, (-1, 1)), weights_1d

        # Tensor product via backend meshgrid
        grids: List[Array] = list(
            bkd.meshgrid(*([nodes_1d] * dim), indexing="ij")
        )
        nodes = bkd.stack(
            [bkd.ravel(g) for g in grids], axis=1
        )

        w_grids: List[Array] = list(
            bkd.meshgrid(*([weights_1d] * dim), indexing="ij")
        )
        weights = bkd.ravel(w_grids[0])
        for w in w_grids[1:]:
            weights = weights * bkd.ravel(w)

        return nodes, weights


class UnscentedRule(Generic[Array]):
    """Unscented transform: 2d+1 sigma points for mean/covariance propagation."""

    def __init__(self, bkd: Backend[Array], kappa: float = 0.0) -> None:
        self._bkd = bkd
        self._kappa = kappa

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, dim: int) -> Tuple[Array, Array]:
        """Return (nodes, weights) for E_{N(0,I)}[f].

        Parameters
        ----------
        dim : int
            Dimensionality.

        Returns
        -------
        nodes : Array, shape (2*dim+1, dim)
        weights : Array, shape (2*dim+1,)
        """
        bkd = self._bkd
        lam = self._kappa
        n = dim
        n_plus_lam = n + lam

        c = bkd.sqrt(bkd.asarray([float(n_plus_lam)]))[0]
        n_pts = 2 * n + 1

        nodes = bkd.zeros((n_pts, n))
        weights = bkd.zeros((n_pts,))

        w0 = lam / n_plus_lam if n_plus_lam != 0 else 1.0
        weights[0] = w0

        w_sigma = 1.0 / (2.0 * n_plus_lam)
        for i in range(n):
            nodes[1 + i, i] = c
            nodes[1 + n + i, i] = -c
            weights[1 + i] = w_sigma
            weights[1 + n + i] = w_sigma

        return nodes, weights


# ---------------------------------------------------------------------------
# Data quadrature (mini-batching)
# ---------------------------------------------------------------------------


class IndexBatchRule(Generic[Array]):
    """Random mini-batch indices with importance weights N/B."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(
        self, n_total: int, batch_size: int, seed: Optional[int] = None
    ) -> Tuple[Array, Array]:
        """Return (indices, weights).

        Parameters
        ----------
        n_total : int
            Total number of data points N.
        batch_size : int
            Mini-batch size B.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        indices : Array, shape (batch_size,)
        weights : Array, shape (batch_size,)
            Weights = N/B.
        """
        bkd = self._bkd
        rng = np.random.RandomState(seed)
        idx_np = rng.choice(n_total, size=batch_size, replace=False)
        indices = bkd.array(idx_np.astype(np.int64))
        weight_val = float(n_total) / float(batch_size)
        weights = bkd.full((batch_size,), weight_val)
        return indices, weights


class FullEnumerationRule(Generic[Array]):
    """All data points with weight 1."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, n_total: int) -> Tuple[Array, Array]:
        """Return (indices, weights) for all data.

        Returns
        -------
        indices : Array, shape (n_total,)
        weights : Array, shape (n_total,)
        """
        bkd = self._bkd
        indices = bkd.arange(n_total)
        weights = bkd.ones((n_total,))
        return indices, weights


class WeightedSampleRule(Generic[Array]):
    """User-supplied (nodes, weights) passed through directly."""

    def __init__(
        self, nodes: Array, weights: Array, bkd: Backend[Array]
    ) -> None:
        self._nodes = nodes
        self._weights = weights
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self) -> Tuple[Array, Array]:
        return self._nodes, self._weights
