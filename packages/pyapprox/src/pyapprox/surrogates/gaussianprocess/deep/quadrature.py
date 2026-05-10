"""Quadrature rules for Deep GP propagation and mini-batching.

Propagation rules integrate over L-dimensional standard-normal
reparameterization noise for DSVI. Each row of the returned
nodes array is one chain path through all L stochastic layers.

Data rules handle mini-batching for the ELBO data term.
"""

from typing import Generic, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm, qmc

from pyapprox.util.backends.protocols import Array, Backend


# ---------------------------------------------------------------------------
# Propagation quadrature (L-dim joint protocol)
# ---------------------------------------------------------------------------


@runtime_checkable
class PropagationRule(Protocol, Generic[Array]):
    """Joint L-dimensional quadrature for DSVI propagation noise.

    Returns (nodes, weights) where nodes has shape (S, L) and
    weights has shape (S,) summing to 1.
    """

    def __call__(
        self,
        n_samples: int,
        n_layers: int,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array]:
        ...


class MonteCarloRule:
    """IID standard-normal draws with equal weights 1/S."""

    def __init__(
        self, rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        n_samples: int,
        n_layers: int,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array]:
        nodes = self._rng.standard_normal((n_samples, n_layers))
        weights = np.full(n_samples, 1.0 / n_samples)
        return bkd.array(nodes), bkd.array(weights)


class SobolRule:
    """Quasi-Monte Carlo via scrambled Sobol in the joint L-dim space."""

    def __init__(
        self, scramble: bool = True, seed: Optional[int] = None,
    ) -> None:
        self._scramble = scramble
        self._seed = seed

    def __call__(
        self,
        n_samples: int,
        n_layers: int,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array]:
        sampler = qmc.Sobol(
            d=n_layers, scramble=self._scramble, seed=self._seed,
        )
        u = sampler.random(n_samples)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        nodes = norm.ppf(u)
        weights = np.full(n_samples, 1.0 / n_samples)
        return bkd.array(nodes), bkd.array(weights)


class TensorProductGHRule:
    """Deterministic tensor-product Gauss-Hermite in L dimensions.

    At order Q, produces Q^L nodes. Cost grows exponentially in L;
    practical for L <= 3 with moderate Q.
    """

    def __init__(self, order: int) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self._order = order
        self._cache: dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _get_numpy_nodes_weights(
        self, n_layers: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = self._order
        key = (Q, n_layers)
        if key in self._cache:
            return self._cache[key]

        nodes_1d, weights_1d = hermegauss(Q)
        weights_1d = weights_1d / np.sum(weights_1d)

        if n_layers == 1:
            nodes_np = nodes_1d.reshape(-1, 1)
            weights_np = weights_1d
        else:
            grids = np.meshgrid(*[nodes_1d] * n_layers, indexing="ij")
            nodes_np = np.stack([g.ravel() for g in grids], axis=1)
            weight_grids = np.meshgrid(
                *[weights_1d] * n_layers, indexing="ij",
            )
            weights_np = np.prod(
                np.stack([w.ravel() for w in weight_grids], axis=1),
                axis=1,
            )

        self._cache[key] = (nodes_np, weights_np)
        return nodes_np, weights_np

    def __call__(
        self,
        n_samples: int,
        n_layers: int,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array]:
        Q = self._order
        expected = Q ** n_layers
        if n_samples != expected:
            raise ValueError(
                f"TensorProductGHRule(order={Q}) with n_layers={n_layers} "
                f"produces exactly {expected} nodes; "
                f"got n_samples={n_samples}"
            )
        nodes_np, weights_np = self._get_numpy_nodes_weights(n_layers)
        return bkd.array(nodes_np), bkd.array(weights_np)


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
