"""One-dimensional hierarchical hat-function basis for sparse grids."""

from typing import Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class HierarchicalBasis1D(Generic[Array]):
    """Hierarchical piecewise-linear (hat) basis on a 1D interval.

    Grid convention on canonical [0, 1]:
      - Level 0: single point at 0.5 (index 1, mapped via j / 2).
      - Level 1, boundary_mode="include": boundary points at 0 (index 0)
        and 1 (index 2). No new interior points.
      - Level l >= 2: new interior points at j / 2^l for odd j,
        1 <= j <= 2^l - 1, that do not appear at lower levels.
      - Children of (l, j): (l+1, 2j-1) and (l+1, 2j+1), filtered
        to only include valid new points at the child level.

    Node mapping: x = bounds[0] + (bounds[1] - bounds[0]) * j / 2^max(l, 1).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    bounds : tuple of float
        Physical domain (a, b). Default (0.0, 1.0).
    p_max : int
        Maximum polynomial degree. Only p_max=1 is implemented.
    boundary_mode : str
        "include" adds boundary points at level 1;
        "exclude" omits them (open domain).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        bounds: Tuple[float, float] = (0.0, 1.0),
        p_max: int = 1,
        boundary_mode: str = "include",
    ) -> None:
        if boundary_mode not in ("include", "exclude"):
            raise ValueError(
                f"boundary_mode must be 'include' or 'exclude', "
                f"got '{boundary_mode}'"
            )
        self._bkd = bkd
        self._a = float(bounds[0])
        self._b = float(bounds[1])
        self._width = self._b - self._a
        self._p_max = p_max
        self._boundary_mode = boundary_mode

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def bounds(self) -> Tuple[float, float]:
        return (self._a, self._b)

    def p_max(self) -> int:
        return self._p_max

    def boundary_mode(self) -> str:
        return self._boundary_mode

    def is_new_point(self, level: int, index: int) -> bool:
        """Whether (level, index) is a new hierarchical point at this level.

        Each level introduces a specific set of new nodes. Points that
        coincide positionally with lower-level nodes are not new.
        """
        if level == 0:
            return index == 1
        if level == 1:
            if self._boundary_mode == "include":
                return index in (0, 2)
            return False
        return index % 2 == 1 and 1 <= index <= 2**level - 1

    def _canonical_node(self, level: int, index: int) -> float:
        """Node position in [0, 1]."""
        denom = max(2**level, 2)
        return index / denom

    def node(self, level: int, index: int) -> float:
        """Physical-domain node coordinate."""
        return self._a + self._width * self._canonical_node(level, index)

    def support(self, level: int, index: int) -> Tuple[float, float]:
        """Support interval [left, right] of the hat function.

        The hat function is nonzero on (left, right) and equals 1 at its
        node. At boundaries the support extends to the domain edge.
        """
        x = self._canonical_node(level, index)
        if level == 0:
            h = 0.5
        else:
            h = 1.0 / 2**level

        left = max(x - h, 0.0)
        right = min(x + h, 1.0)
        return (self._a + self._width * left, self._a + self._width * right)

    def children(self, level: int, index: int) -> List[Tuple[int, int]]:
        """Children of node (level, index) in the dyadic tree.

        Only valid new points at the child level are returned.
        Precondition: (level, index) must satisfy is_new_point.
        """
        if level == 0:
            # Level-0 midpoint's children are the level-1 boundary
            # points (if included) and/or the level-2 interior points.
            # In include mode: boundaries {0, 2} at level 1.
            # In exclude mode: no level-1 points exist, so children
            # are at level 2: {1, 3} (the two interior points).
            if self._boundary_mode == "include":
                return [(1, 0), (1, 2)]
            return [(2, 1), (2, 3)]

        child_level = level + 1
        left_idx = 2 * index - 1
        right_idx = 2 * index + 1

        result: List[Tuple[int, int]] = []
        for ci in (left_idx, right_idx):
            if self.is_new_point(child_level, ci):
                result.append((child_level, ci))
        return result

    def new_points_at_level(self, level: int) -> List[int]:
        """Indices of nodes that are new at this level.

        Level 0: [1] (the midpoint).
        Level 1: boundary indices [0, 2] if boundary_mode="include",
                 else [] (no new interior points).
        Level l >= 2: odd indices in [1, 2^l - 1].
        """
        if level == 0:
            return [1]
        if level == 1:
            if self._boundary_mode == "include":
                return [0, 2]
            return []
        return list(range(1, 2**level, 2))

    def _parent(self, level: int, index: int) -> Tuple[int, int]:
        """Parent node in the dyadic tree.

        Walks up levels until a valid new point is found among the
        candidates {(j-1)//2, (j+1)//2} at each level. In include mode
        the parent is always exactly one level up. In exclude mode,
        level-2 points skip level 1 (which has no new points) and
        parent directly to (0, 1).
        """
        if level <= 0:
            raise ValueError("Level-0 node has no parent")
        l, j = level, index
        while l > 0:
            if l == 1:
                return (0, 1)
            parent_level = l - 1
            p1 = (j + 1) // 2
            p2 = (j - 1) // 2
            if self.is_new_point(parent_level, p1):
                return (parent_level, p1)
            if self.is_new_point(parent_level, p2):
                return (parent_level, p2)
            # Neither candidate is a new point at parent_level (e.g.,
            # exclude mode skipping level 1). Walk up one more level
            # using the midpoint candidate.
            l = parent_level
            j = p1 if p1 <= 2 ** parent_level else p2
        raise RuntimeError(
            f"No valid parent found for ({level}, {index})"
        )

    def ancestors(self, level: int, index: int) -> List[Tuple[int, int]]:
        """Ancestor chain walking up the dyadic tree.

        Returns ancestors from parent to root, excluding the node itself.
        """
        result: List[Tuple[int, int]] = []
        l, j = level, index
        while l > 0:
            l, j = self._parent(l, j)
            result.append((l, j))
        return result

    def degree_at(self, level: int) -> int:
        """Polynomial degree of the basis function at this level."""
        return min(self._p_max, max(level, 1))

    def _is_left_boundary(self, level: int, index: int) -> bool:
        """Whether this is a left-boundary hat (node at domain left edge)."""
        return (
            self._boundary_mode == "include"
            and level == 1
            and index == 0
        )

    def _is_right_boundary(self, level: int, index: int) -> bool:
        """Whether this is a right-boundary hat (node at domain right edge)."""
        return (
            self._boundary_mode == "include"
            and level == 1
            and index == 2
        )

    def _mesh_size(self, level: int) -> float:
        """Physical-domain mesh spacing at this level."""
        if level == 0:
            return self._width * 0.5
        return self._width / 2**level

    def evaluate(self, x: Array, level: int, index: int) -> Array:
        """Evaluate the hierarchical basis function at sample points.

        Parameters
        ----------
        x : Array
            Sample points, shape (npts,).
        level : int
            Level of the basis function.
        index : int
            Index of the basis function at this level.

        Returns
        -------
        Array
            Values of the hat function, shape (npts,).
        """
        if self._p_max > 1:
            raise NotImplementedError(
                f"p_max={self._p_max} not yet implemented; only p_max=1"
            )

        bkd = self._bkd
        node_x = self.node(level, index)
        h = self._mesh_size(level)
        zero = bkd.zeros_like(x)

        if level == 0 and self._boundary_mode == "include":
            return bkd.ones_like(x)

        if self._is_left_boundary(level, index):
            # One-sided hat: 1 at node (left edge), 0 at node + h
            right = bkd.asarray([node_x + h], dtype=bkd.double_dtype())
            h_val = bkd.asarray([h], dtype=bkd.double_dtype())
            ramp = (right - x) / h_val
            return bkd.where(
                x <= right,
                bkd.maximum(ramp, zero),
                zero,
            )

        if self._is_right_boundary(level, index):
            # One-sided hat: 1 at node (right edge), 0 at node - h
            left = bkd.asarray([node_x - h], dtype=bkd.double_dtype())
            h_val = bkd.asarray([h], dtype=bkd.double_dtype())
            ramp = (x - left) / h_val
            return bkd.where(
                x >= left,
                bkd.maximum(ramp, zero),
                zero,
            )

        # Interior hat: symmetric tent on [node - h, node + h]
        node_val = bkd.asarray([node_x], dtype=bkd.double_dtype())
        h_val = bkd.asarray([h], dtype=bkd.double_dtype())
        left_branch = (x - (node_val - h_val)) / h_val
        right_branch = ((node_val + h_val) - x) / h_val
        hat = bkd.minimum(left_branch, right_branch)
        return bkd.maximum(hat, zero)

    def quadrature_weight(self, level: int, index: int) -> float:
        """Quadrature weight (integral of the hat function over the domain).

        Level 0 is constant=1, so its integral is the domain width.
        For interior hat functions the integral equals half the support
        width (triangle area with unit height). Boundary hats have
        half the usual support width.
        """
        if level == 0 and self._boundary_mode == "include":
            return self._width
        left, right = self.support(level, index)
        return (right - left) / 2.0
