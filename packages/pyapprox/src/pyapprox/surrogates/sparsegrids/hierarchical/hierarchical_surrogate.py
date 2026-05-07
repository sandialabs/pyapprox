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
        bkd = self._bkd
        npts = samples.shape[1]
        nqoi = self.nqoi()
        result = bkd.zeros((nqoi, npts), dtype=bkd.double_dtype())

        for i, key in enumerate(self._point_keys):
            basis_vals = self._basis_nd.evaluate(
                samples, key[0], key[1]
            )
            # basis_vals shape: (npts,)
            # surpluses[:, i] shape: (nqoi,)
            surplus_col = self._surpluses[:, i].reshape(nqoi, 1)
            result = result + surplus_col * basis_vals.reshape(1, npts)

        return result

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

    def point_keys(self) -> List[PointKey]:
        return list(self._point_keys)

    def surpluses(self) -> Array:
        return self._surpluses

    def quad_weights(self) -> Array:
        return self._quad_weights
