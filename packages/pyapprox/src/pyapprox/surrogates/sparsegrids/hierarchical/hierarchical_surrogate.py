"""Hierarchical sparse grid surrogate (snapshot-based evaluation)."""

from typing import Generic, List, Tuple

from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)
from pyapprox.surrogates.sparsegrids.hierarchical.point_manager import (
    PointKey,
)
from pyapprox.util.backends.protocols import Array, Backend


class HierarchicalSurrogate(Generic[Array]):
    """Evaluates f̂(x) = Σ_{(l,j)} v_{l,j} · Ψ_{l,j}(x).

    A frozen snapshot of point keys, surpluses, and quadrature weights,
    decoupled from the mutable PointManager. Both active and redundant
    points contribute surpluses.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis_nd : HierarchicalBasisND[Array]
        ND basis for evaluating hat functions.
    point_keys : list of PointKey
        (subspace_level, point_index) for each contributing point.
    surpluses : Array
        Shape (nqoi, n_points). Column i is the surplus for point i.
    quad_weights : Array
        Shape (n_points,). Quadrature weight for each point.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis_nd: HierarchicalBasisND[Array],
        point_keys: List[PointKey],
        surpluses: Array,
        quad_weights: Array,
    ) -> None:
        self._bkd = bkd
        self._basis_nd = basis_nd
        self._point_keys = list(point_keys)
        self._surpluses = bkd.copy(surpluses)
        self._quad_weights = bkd.copy(quad_weights)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._basis_nd.nvars()

    def nqoi(self) -> int:
        return int(self._surpluses.shape[0])

    def n_points(self) -> int:
        return len(self._point_keys)

    def __call__(self, samples: Array) -> Array:
        """Evaluate the surrogate at multiple sample points.

        Parameters
        ----------
        samples : Array
            Shape (nvars, npts).

        Returns
        -------
        Array
            Shape (nqoi, npts).
        """
        if not self._point_keys:
            bkd = self._bkd
            return bkd.zeros(
                (self.nqoi(), samples.shape[1]),
                dtype=bkd.double_dtype(),
            )
        basis_matrix = self._basis_nd.evaluate_batch(
            samples, self._point_keys
        )
        return self._bkd.dot(self._surpluses, basis_matrix)

    def evaluate(self, sample: Array) -> Array:
        """Evaluate at a single sample point.

        Parameters
        ----------
        sample : Array
            Shape (nvars, 1).

        Returns
        -------
        Array
            Shape (nqoi, 1).
        """
        return self.__call__(sample)

    def mean(self) -> Array:
        """Hierarchical quadrature mean: Σ v_{l,j,q} · w_{l,j}.

        Returns
        -------
        Array
            Shape (nqoi,).
        """
        bkd = self._bkd
        # surpluses: (nqoi, n_points), quad_weights: (n_points,)
        return bkd.dot(self._surpluses, self._quad_weights)

    def add_point(
        self, key: PointKey, surplus: Array, quad_weight: float
    ) -> None:
        """Incrementally add a single point to the surrogate (in place)."""
        self.add_points([key], [surplus], [quad_weight])

    def add_points(
        self,
        keys: List[PointKey],
        surpluses: List[Array],
        quad_weights: List[float],
    ) -> None:
        """Batch-add multiple points to the surrogate (in place).

        Parameters
        ----------
        keys : list of PointKey
        surpluses : list of Array, each shape (nqoi,)
        quad_weights : list of float
        """
        if not keys:
            return
        bkd = self._bkd
        self._point_keys.extend(keys)
        new_cols = bkd.stack(surpluses, axis=1) if len(surpluses) > 1 else surpluses[0].reshape(-1, 1)
        if self._surpluses.shape[1] == 0:
            self._surpluses = new_cols
        else:
            self._surpluses = bkd.hstack([self._surpluses, new_cols])
        new_w = bkd.asarray(quad_weights, dtype=bkd.double_dtype())
        self._quad_weights = bkd.concatenate([self._quad_weights, new_w])

    def point_keys(self) -> List[PointKey]:
        return list(self._point_keys)

    def surpluses(self) -> Array:
        return self._surpluses

    def quad_weights(self) -> Array:
        return self._quad_weights
