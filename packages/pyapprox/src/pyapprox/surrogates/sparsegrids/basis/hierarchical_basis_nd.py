"""N-dimensional hierarchical basis (tensor product of 1D bases)."""

from itertools import product as cartesian_product
from typing import Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend

from .hierarchical_basis_1d import HierarchicalBasis1D


class HierarchicalBasisND(Generic[Array]):
    """Tensor product of 1D hierarchical bases within a subspace.

    A subspace is identified by a level multi-index (l_1, ..., l_d).
    Within a subspace, each point is identified by a point multi-index
    (j_1, ..., j_d) where j_k is a new-point index at level l_k for
    the k-th 1D basis.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    bases_1d : list of HierarchicalBasis1D
        One 1D basis per dimension.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        bases_1d: List[HierarchicalBasis1D[Array]],
    ) -> None:
        self._bkd = bkd
        self._bases_1d = bases_1d

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return len(self._bases_1d)

    def node(
        self,
        subspace_level: Tuple[int, ...],
        point_index: Tuple[int, ...],
    ) -> Array:
        """Physical-domain coordinates of a single grid point.

        Returns shape (nvars, 1).
        """
        coords = [
            self._bases_1d[d].node(subspace_level[d], point_index[d])
            for d in range(self.nvars())
        ]
        return self._bkd.asarray(coords, dtype=self._bkd.double_dtype()).reshape(
            self.nvars(), 1
        )

    def evaluate(
        self,
        samples: Array,
        subspace_level: Tuple[int, ...],
        point_index: Tuple[int, ...],
    ) -> Array:
        """Evaluate the ND basis function at sample points.

        The ND basis function is the product of 1D hat functions:
        Psi_{l,j}(x) = prod_d psi_{l_d, j_d}(x_d).

        Parameters
        ----------
        samples : Array
            Shape (nvars, npts).
        subspace_level : tuple of int
            Level multi-index.
        point_index : tuple of int
            Point multi-index within the subspace.

        Returns
        -------
        Array
            Shape (npts,).
        """
        nvars = self.nvars()
        result = self._bases_1d[0].evaluate(
            samples[0:1, :].reshape(-1),
            subspace_level[0],
            point_index[0],
        )
        for d in range(1, nvars):
            val_d = self._bases_1d[d].evaluate(
                samples[d : d + 1, :].reshape(-1),
                subspace_level[d],
                point_index[d],
            )
            result = result * val_d
        return result

    def points_in_subspace(
        self, subspace_level: Tuple[int, ...]
    ) -> List[Tuple[int, ...]]:
        """All new-point multi-indices in a subspace.

        Returns the Cartesian product of new_points_at_level for each
        dimension.
        """
        per_dim = [
            self._bases_1d[d].new_points_at_level(subspace_level[d])
            for d in range(self.nvars())
        ]
        if any(len(p) == 0 for p in per_dim):
            return []
        return list(cartesian_product(*per_dim))

    def children_of_point(
        self,
        subspace_level: Tuple[int, ...],
        point_index: Tuple[int, ...],
        direction: int,
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Children of a point when refining in one direction.

        Refinement in direction d increments the subspace level in
        dimension d by 1 and replaces the 1D index in dimension d
        with each of its 1D children. Other dimensions are unchanged.

        Returns list of (child_subspace_level, child_point_index).
        """
        basis_1d = self._bases_1d[direction]
        l_d = subspace_level[direction]
        j_d = point_index[direction]
        children_1d = basis_1d.children(l_d, j_d)

        child_sub = list(subspace_level)
        child_sub[direction] += 1
        child_sub_tuple = tuple(child_sub)

        result: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        for _, child_j in children_1d:
            child_pt = list(point_index)
            child_pt[direction] = child_j
            result.append((child_sub_tuple, tuple(child_pt)))
        return result

    def ancestors_of_point(
        self,
        subspace_level: Tuple[int, ...],
        point_index: Tuple[int, ...],
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """All hierarchical ancestors of a point (excluding self).

        An ancestor is any point whose ND basis function is nonzero at
        this point's node. These are exactly the Cartesian product of
        each dimension's 1D ancestor chain (including self), minus the
        point itself.
        """
        chains = []
        for d in range(self.nvars()):
            chain_d = [(subspace_level[d], point_index[d])]
            chain_d.extend(
                self._bases_1d[d].ancestors(subspace_level[d], point_index[d])
            )
            chains.append(chain_d)

        result = []
        for combo in cartesian_product(*chains):
            sub = tuple(c[0] for c in combo)
            idx = tuple(c[1] for c in combo)
            if sub == subspace_level and idx == point_index:
                continue
            result.append((sub, idx))
        return result

    def quadrature_weight(
        self,
        subspace_level: Tuple[int, ...],
        point_index: Tuple[int, ...],
    ) -> float:
        """ND quadrature weight (product of 1D weights)."""
        w = 1.0
        for d in range(self.nvars()):
            w *= self._bases_1d[d].quadrature_weight(
                subspace_level[d], point_index[d]
            )
        return w
