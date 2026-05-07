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

    def unique_nonzero_index(self, x_phys: float, level: int) -> int:
        """Find the unique index at this level whose basis function is nonzero at x.

        At each level, the supports tile the domain so exactly one basis
        function is nonzero at any interior point. Returns that index.
        """
        if level == 0:
            return 1
        if level == 1 and self._boundary_mode == "include":
            mid = self._a + self._width * 0.5
            if x_phys <= mid:
                return 0
            return 2
        x_canon = (x_phys - self._a) / self._width
        h = 1.0 / 2**level
        candidate = int(x_canon / h)
        if candidate % 2 == 1:
            return candidate
        return candidate + 1

    def evaluate_scalar(self, x: float, level: int, index: int) -> float:
        """Evaluate basis function at a single scalar point (no array overhead)."""
        if level == 0 and self._boundary_mode == "include":
            return 1.0

        node_x = self._a + self._width * (index / max(2**level, 2))
        h = self._width * 0.5 if level == 0 else self._width / 2**level

        is_left_bnd = (
            self._boundary_mode == "include" and level == 1 and index == 0
        )
        is_right_bnd = (
            self._boundary_mode == "include" and level == 1 and index == 2
        )

        if self._p_max == 1:
            if is_left_bnd:
                right = node_x + h
                if x > right:
                    return 0.0
                return max((right - x) / h, 0.0)
            if is_right_bnd:
                left = node_x - h
                if x < left:
                    return 0.0
                return max((x - left) / h, 0.0)
            left_val = (x - (node_x - h)) / h
            right_val = ((node_x + h) - x) / h
            return max(min(left_val, right_val), 0.0)

        if self._p_max == 2:
            if is_left_bnd or is_right_bnd:
                if is_left_bnd:
                    right = node_x + h
                    if x > right:
                        return 0.0
                    return max((right - x) / h, 0.0)
                left = node_x - h
                if x < left:
                    return 0.0
                return max((x - left) / h, 0.0)
            left = node_x - h
            right = node_x + h
            if x < left or x > right:
                return 0.0
            denom = (node_x - left) * (node_x - right)
            return (x - left) * (x - right) / denom

        raise NotImplementedError(f"p_max={self._p_max}")

    def _mesh_size(self, level: int) -> float:
        """Physical-domain mesh spacing at this level."""
        if level == 0:
            return self._width * 0.5
        return self._width / 2**level

    def _eval_hat(self, x: Array, level: int, index: int) -> Array:
        """Evaluate the piecewise-linear hat function (p_max=1)."""
        bkd = self._bkd
        node_x = self.node(level, index)
        h = self._mesh_size(level)
        zero = bkd.zeros_like(x)

        if level == 0 and self._boundary_mode == "include":
            return bkd.ones_like(x)

        if self._is_left_boundary(level, index):
            right = bkd.asarray([node_x + h], dtype=bkd.double_dtype())
            h_val = bkd.asarray([h], dtype=bkd.double_dtype())
            ramp = (right - x) / h_val
            return bkd.where(
                x <= right,
                bkd.maximum(ramp, zero),
                zero,
            )

        if self._is_right_boundary(level, index):
            left = bkd.asarray([node_x - h], dtype=bkd.double_dtype())
            h_val = bkd.asarray([h], dtype=bkd.double_dtype())
            ramp = (x - left) / h_val
            return bkd.where(
                x >= left,
                bkd.maximum(ramp, zero),
                zero,
            )

        node_val = bkd.asarray([node_x], dtype=bkd.double_dtype())
        h_val = bkd.asarray([h], dtype=bkd.double_dtype())
        left_branch = (x - (node_val - h_val)) / h_val
        right_branch = ((node_val + h_val) - x) / h_val
        hat = bkd.minimum(left_branch, right_branch)
        return bkd.maximum(hat, zero)

    def _eval_quadratic(self, x: Array, level: int, index: int) -> Array:
        """Evaluate the quadratic hierarchical basis function.

        The basis function is the quadratic Lagrange interpolant through
        (left, node, right) where left and right are the support
        endpoints.  It equals 1 at the node and 0 at both endpoints,
        forming a downward-opening parabola on the support and zero
        outside.
        """
        bkd = self._bkd

        if level == 0 and self._boundary_mode == "include":
            return bkd.ones_like(x)

        left, right = self.support(level, index)
        node_x = self.node(level, index)
        denom = (node_x - left) * (node_x - right)

        if self._is_left_boundary(level, index):
            # node == left, so (x - left) is zero at node → use
            # linear ramp: 1 at left edge, 0 at right edge
            return self._eval_hat(x, level, index)

        if self._is_right_boundary(level, index):
            return self._eval_hat(x, level, index)

        left_arr = bkd.asarray([left], dtype=bkd.double_dtype())
        right_arr = bkd.asarray([right], dtype=bkd.double_dtype())
        denom_arr = bkd.asarray([denom], dtype=bkd.double_dtype())

        poly = (x - left_arr) * (x - right_arr) / denom_arr
        mask = (x >= left_arr) & (x <= right_arr)
        return bkd.where(mask, poly, bkd.zeros_like(x))

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
            Values of the basis function, shape (npts,).
        """
        if self._p_max == 1:
            return self._eval_hat(x, level, index)
        if self._p_max == 2:
            return self._eval_quadratic(x, level, index)
        raise NotImplementedError(
            f"p_max={self._p_max} not yet implemented; only p_max<=2"
        )

    def evaluate_batch(
        self,
        x: Array,
        levels: List[int],
        indices: List[int],
    ) -> Array:
        """Evaluate multiple basis functions at all sample points.

        Parameters
        ----------
        x : Array
            Sample points, shape (npts,).
        levels : list of int
            Level for each basis function.
        indices : list of int
            Index for each basis function.

        Returns
        -------
        Array
            Shape (n_basis, npts). Row i is the i-th basis function
            evaluated at all sample points.
        """
        bkd = self._bkd
        n_basis = len(levels)
        npts = x.shape[0]
        a = self._a
        w = self._width
        bdy_include = self._boundary_mode == "include"

        nodes_list = [0.0] * n_basis
        lefts_list = [0.0] * n_basis
        rights_list = [0.0] * n_basis
        is_left_bdy = [0.0] * n_basis
        is_right_bdy = [0.0] * n_basis
        is_lev0 = [0.0] * n_basis
        for i in range(n_basis):
            lv = levels[i]
            ix = indices[i]
            denom = max(2**lv, 2)
            cn = ix / denom
            nodes_list[i] = a + w * cn
            if lv == 0:
                h = 0.5
            else:
                h = 1.0 / (2**lv)
            lefts_list[i] = a + w * max(cn - h, 0.0)
            rights_list[i] = a + w * min(cn + h, 1.0)
            if lv == 0:
                is_lev0[i] = 1.0
            if bdy_include and lv == 1 and ix == 0:
                is_left_bdy[i] = 1.0
            if bdy_include and lv == 1 and ix == 2:
                is_right_bdy[i] = 1.0

        nodes = bkd.asarray(nodes_list, dtype=bkd.double_dtype())
        lefts = bkd.asarray(lefts_list, dtype=bkd.double_dtype())
        rights = bkd.asarray(rights_list, dtype=bkd.double_dtype())

        x_row = x.reshape(1, npts)
        nodes_col = nodes.reshape(n_basis, 1)
        lefts_col = lefts.reshape(n_basis, 1)
        rights_col = rights.reshape(n_basis, 1)
        zero = bkd.zeros((n_basis, npts), dtype=bkd.double_dtype())
        support_width = rights_col - lefts_col
        safe_width = bkd.maximum(
            support_width,
            bkd.asarray([[1e-30]], dtype=bkd.double_dtype()),
        )

        is_left_col = bkd.asarray(
            is_left_bdy, dtype=bkd.double_dtype()
        ).reshape(n_basis, 1)
        is_right_col = bkd.asarray(
            is_right_bdy, dtype=bkd.double_dtype()
        ).reshape(n_basis, 1)
        is_bdy_col = is_left_col + is_right_col

        left_ramp = (rights_col - x_row) / safe_width
        right_ramp = (x_row - lefts_col) / safe_width
        in_support = (x_row >= lefts_col) & (x_row <= rights_col)

        bdy_hat = bkd.where(
            is_left_col > 0.5,
            left_ramp,
            right_ramp,
        )
        bdy_result = bkd.where(in_support, bdy_hat, zero)
        bdy_result = bkd.maximum(bdy_result, zero)

        h_left = nodes_col - lefts_col
        h_right = rights_col - nodes_col
        safe_h_left = bkd.maximum(
            h_left,
            bkd.asarray([[1e-30]], dtype=bkd.double_dtype()),
        )
        safe_h_right = bkd.maximum(
            h_right,
            bkd.asarray([[1e-30]], dtype=bkd.double_dtype()),
        )

        if self._p_max <= 1:
            left_branch = (x_row - lefts_col) / safe_h_left
            right_branch = (rights_col - x_row) / safe_h_right
            hat = bkd.minimum(left_branch, right_branch)
            interior_result = bkd.maximum(hat, zero)
        else:
            denom = h_left * (-h_right)
            safe_denom = bkd.where(
                denom == 0,
                bkd.ones_like(denom),
                denom,
            )
            poly = (x_row - lefts_col) * (x_row - rights_col) / safe_denom
            interior_result = bkd.where(in_support, poly, zero)

        result = bkd.where(is_bdy_col > 0.5, bdy_result, interior_result)

        if bdy_include and any(v > 0.5 for v in is_lev0):
            is_lev0_col = bkd.asarray(
                is_lev0, dtype=bkd.double_dtype()
            ).reshape(n_basis, 1)
            ones = bkd.ones(
                (n_basis, npts), dtype=bkd.double_dtype()
            )
            result = bkd.where(is_lev0_col > 0.5, ones, result)

        return result

    def _weight_hat(self, level: int, index: int) -> float:
        """Quadrature weight for hat function: half the support width."""
        left, right = self.support(level, index)
        return (right - left) / 2.0

    def _weight_quadratic(self, level: int, index: int) -> float:
        """Quadrature weight for quadratic basis: 4h/3.

        For the symmetric quadratic Lagrange interpolant through
        (node-h, node, node+h), the integral over the support is 4h/3.
        Boundary nodes (level 1) fall back to linear weights.
        """
        if self._is_left_boundary(level, index):
            return self._weight_hat(level, index)
        if self._is_right_boundary(level, index):
            return self._weight_hat(level, index)
        h = self._mesh_size(level)
        return 4.0 * h / 3.0

    def quadrature_weight(self, level: int, index: int) -> float:
        """Quadrature weight (integral of the basis function over the domain).

        For p_max=1 (hat): weight = h (half the support width 2h).
        For p_max=2 (quadratic): weight = 4h/3 for interior nodes,
        linear weight for boundary nodes.
        """
        if level == 0 and self._boundary_mode == "include":
            return self._width
        if self._p_max <= 1:
            return self._weight_hat(level, index)
        return self._weight_quadratic(level, index)
