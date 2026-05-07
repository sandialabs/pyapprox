"""Hierarchical sparse grid surrogate (snapshot-based evaluation)."""

from itertools import product as cartesian_product
from typing import Dict, Generic, List, Set, Tuple

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
        self._key_to_col: Dict[PointKey, int] = {
            k: i for i, k in enumerate(self._point_keys)
        }
        self._selected_subspaces: Set[Tuple[int, ...]] = {
            k[0] for k in self._point_keys
        }

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._basis_nd.nvars()

    def nqoi(self) -> int:
        return int(self._surpluses.shape[0])

    def n_points(self) -> int:
        return len(self._point_keys)

    _MAX_BASIS_MATRIX_BYTES = 500 * 1024 * 1024  # 500 MB

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
        if not self._point_keys:
            return bkd.zeros(
                (self.nqoi(), samples.shape[1]),
                dtype=bkd.double_dtype(),
            )
        n_basis = len(self._point_keys)
        npts = int(samples.shape[1])
        max_pts_per_chunk = max(
            1, self._MAX_BASIS_MATRIX_BYTES // (n_basis * 8)
        )
        if npts <= max_pts_per_chunk:
            basis_matrix = self._basis_nd.evaluate_batch(
                samples, self._point_keys
            )
            return bkd.dot(self._surpluses, basis_matrix)
        chunks = []
        for start in range(0, npts, max_pts_per_chunk):
            end = min(start + max_pts_per_chunk, npts)
            basis_chunk = self._basis_nd.evaluate_batch(
                samples[:, start:end], self._point_keys
            )
            chunks.append(bkd.dot(self._surpluses, basis_chunk))
        return bkd.hstack(chunks)

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
        base_col = len(self._point_keys)
        self._point_keys.extend(keys)
        for i, k in enumerate(keys):
            self._key_to_col[k] = base_col + i
            self._selected_subspaces.add(k[0])
        if len(surpluses) > 1:
            new_cols = bkd.stack(surpluses, axis=1)
        else:
            new_cols = bkd.reshape(surpluses[0], (-1, 1))
        if self._surpluses.shape[1] == 0:
            self._surpluses = new_cols
        else:
            self._surpluses = bkd.hstack([self._surpluses, new_cols])
        new_w = bkd.asarray(quad_weights, dtype=bkd.double_dtype())
        self._quad_weights = bkd.concatenate([self._quad_weights, new_w])

    def compute_surpluses_at_grid_points(
        self,
        new_points: List[PointKey],
        f_values: Array,
    ) -> Array:
        """Compute hierarchical surpluses using componentwise-ancestor evaluation.

        Instead of evaluating the full surrogate at each new grid point
        (cost O(N * npts * d)), this exploits the fact that only
        componentwise-ancestor subspaces contribute nonzero basis values
        at a grid point (cost O(|ancestors_avg| * npts * d)).

        Parameters
        ----------
        new_points : list of PointKey
            (subspace_level, point_index) for each new grid point.
        f_values : Array
            Shape (nqoi, K). Function values at the new grid points.

        Returns
        -------
        Array
            Shape (nqoi, K). Hierarchical surpluses.
        """
        bkd = self._bkd
        n_new = len(new_points)
        nqoi = int(f_values.shape[0])
        nvars = self._basis_nd.nvars()
        bases_1d = self._basis_nd._bases_1d

        predictions: List[Array] = []

        for k in range(n_new):
            level_k, index_k = new_points[k]
            x_coords = [
                bases_1d[d].node(level_k[d], index_k[d])
                for d in range(nvars)
            ]

            f_hat = bkd.zeros((nqoi,), dtype=bkd.double_dtype())
            for ancestor in cartesian_product(
                *[range(level_k[d] + 1) for d in range(nvars)]
            ):
                if ancestor == level_k:
                    continue
                if ancestor not in self._selected_subspaces:
                    continue

                j_star = tuple(
                    bases_1d[d].unique_nonzero_index(
                        x_coords[d], ancestor[d]
                    )
                    for d in range(nvars)
                )
                key = (ancestor, j_star)
                col = self._key_to_col.get(key)
                if col is None:
                    raise ValueError(
                        f"Ancestor {key} of new point "
                        f"({level_k}, {index_k}) is not stored. "
                        "Downward-closed invariant violated."
                    )

                psi = 1.0
                for d in range(nvars):
                    psi *= bases_1d[d].evaluate_scalar(
                        x_coords[d], ancestor[d], j_star[d]
                    )

                f_hat = f_hat + self._surpluses[:, col] * psi

            predictions.append(f_hat)

        if n_new == 0:
            return bkd.zeros((nqoi, 0), dtype=bkd.double_dtype())
        pred_matrix = bkd.stack(predictions, axis=1)
        return f_values - pred_matrix

    def point_keys(self) -> List[PointKey]:
        return list(self._point_keys)

    def surpluses(self) -> Array:
        return self._surpluses

    def quad_weights(self) -> Array:
        return self._quad_weights
