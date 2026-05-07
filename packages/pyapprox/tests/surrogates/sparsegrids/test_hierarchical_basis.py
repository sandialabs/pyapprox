"""Tests for hierarchical 1D and ND basis functions."""

import numpy as np
import pytest

from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)


def _collect_points(basis_1d, max_level):
    """Collect all (level, index) pairs up to max_level."""
    pts = []
    for l in range(max_level + 1):
        for j in basis_1d.new_points_at_level(l):
            pts.append((l, j))
    return pts


def _hierarchical_surpluses_1d(basis_1d, f, max_level, bkd):
    """Compute hierarchical surpluses for a scalar function f."""
    pts = _collect_points(basis_1d, max_level)
    surpluses = {}
    for l, j in pts:
        x_node = basis_1d.node(l, j)
        pred = 0.0
        for (lp, jp), v in surpluses.items():
            val = bkd.to_numpy(
                basis_1d.evaluate(bkd.asarray([x_node]), lp, jp)
            )
            pred += v * float(val[0])
        surpluses[(l, j)] = f(x_node) - pred
    return pts, surpluses


class TestHierarchicalBasis1DInclude:
    """Tests for HierarchicalBasis1D with boundary_mode='include'."""

    @pytest.fixture()
    def basis(self, bkd):
        return HierarchicalBasis1D(bkd, bounds=(0.0, 1.0), boundary_mode="include")

    def test_new_points_at_level(self, basis):
        assert basis.new_points_at_level(0) == [1]
        assert basis.new_points_at_level(1) == [0, 2]
        assert basis.new_points_at_level(2) == [1, 3]
        assert basis.new_points_at_level(3) == [1, 3, 5, 7]

    def test_node_positions(self, basis):
        assert basis.node(0, 1) == pytest.approx(0.5)
        assert basis.node(1, 0) == pytest.approx(0.0)
        assert basis.node(1, 2) == pytest.approx(1.0)
        assert basis.node(2, 1) == pytest.approx(0.25)
        assert basis.node(2, 3) == pytest.approx(0.75)

    def test_is_new_point(self, basis):
        assert basis.is_new_point(0, 1)
        assert not basis.is_new_point(0, 0)
        assert basis.is_new_point(1, 0)
        assert basis.is_new_point(1, 2)
        assert not basis.is_new_point(1, 1)
        assert basis.is_new_point(2, 1)
        assert basis.is_new_point(2, 3)
        assert not basis.is_new_point(2, 0)
        assert not basis.is_new_point(2, 2)
        assert not basis.is_new_point(2, 4)

    def test_children_level_0(self, basis):
        assert basis.children(0, 1) == [(1, 0), (1, 2)]

    def test_children_level_1_boundary(self, basis):
        assert basis.children(1, 0) == [(2, 1)]
        assert basis.children(1, 2) == [(2, 3)]

    def test_children_level_2(self, basis):
        assert basis.children(2, 1) == [(3, 1), (3, 3)]
        assert basis.children(2, 3) == [(3, 5), (3, 7)]

    def test_children_within_parent_support(self, basis):
        for l in range(4):
            for j in basis.new_points_at_level(l):
                left, right = basis.support(l, j)
                for cl, cj in basis.children(l, j):
                    cx = basis.node(cl, cj)
                    assert left <= cx <= right, (
                        f"child ({cl},{cj}) at x={cx} outside "
                        f"parent ({l},{j}) support [{left},{right}]"
                    )

    def test_parent_include(self, basis):
        assert basis._parent(1, 0) == (0, 1)
        assert basis._parent(1, 2) == (0, 1)
        assert basis._parent(2, 1) == (1, 0)
        assert basis._parent(2, 3) == (1, 2)
        assert basis._parent(3, 1) == (2, 1)
        assert basis._parent(3, 3) == (2, 1)
        assert basis._parent(3, 5) == (2, 3)
        assert basis._parent(3, 7) == (2, 3)

    def test_ancestors_include(self, basis):
        assert basis.ancestors(3, 1) == [(2, 1), (1, 0), (0, 1)]
        assert basis.ancestors(3, 7) == [(2, 3), (1, 2), (0, 1)]
        assert basis.ancestors(2, 3) == [(1, 2), (0, 1)]
        assert basis.ancestors(1, 0) == [(0, 1)]

    def test_parent_child_consistency(self, basis):
        """Every child's parent is the original node."""
        for l in range(4):
            for j in basis.new_points_at_level(l):
                for cl, cj in basis.children(l, j):
                    assert basis._parent(cl, cj) == (l, j)

    def test_property_a_same_level_interpolation(self, bkd, basis):
        """psi_{l,j}(node(l,j')) = delta_{j,j'} for j,j' new at level l."""
        for l in range(5):
            indices = basis.new_points_at_level(l)
            if not indices:
                continue
            nodes = bkd.asarray(
                [basis.node(l, j) for j in indices],
                dtype=bkd.double_dtype(),
            )
            for i, ja in enumerate(indices):
                vals = bkd.to_numpy(basis.evaluate(nodes, l, ja))
                for k, jb in enumerate(indices):
                    expected = 1.0 if i == k else 0.0
                    assert abs(vals[k] - expected) < 1e-14, (
                        f"psi_{l},{ja}(node({l},{jb})) = {vals[k]}, "
                        f"expected {expected}"
                    )

    def test_property_b_vanishing_at_coarser_nodes(self, bkd, basis):
        """psi_{l,j}(node(l',j')) = 0 for l' < l."""
        max_level = 4
        all_pts = _collect_points(basis, max_level)
        for l_fine in range(1, max_level + 1):
            fine_pts = [(l, j) for l, j in all_pts if l == l_fine]
            for l_coarse in range(l_fine):
                coarse_pts = [(l, j) for l, j in all_pts if l == l_coarse]
                if not coarse_pts:
                    continue
                coarse_nodes = bkd.asarray(
                    [basis.node(l, j) for l, j in coarse_pts],
                    dtype=bkd.double_dtype(),
                )
                for lf, jf in fine_pts:
                    vals = bkd.to_numpy(basis.evaluate(coarse_nodes, lf, jf))
                    for k, (lc, jc) in enumerate(coarse_pts):
                        assert abs(vals[k]) < 1e-14, (
                            f"psi_{lf},{jf}(node({lc},{jc})) = {vals[k]}, "
                            f"expected 0"
                        )

    def test_hierarchical_interpolant_matches_nodal(self, bkd, basis):
        """Hierarchical surplus interpolant equals nodal PW-linear interp."""
        def f(x):
            return np.sin(2 * np.pi * x) + x**2

        max_level = 3
        pts, surpluses = _hierarchical_surpluses_1d(basis, f, max_level, bkd)
        nodes_sorted = sorted(basis.node(l, j) for l, j in pts)
        f_sorted = [f(n) for n in nodes_sorted]

        x_test = np.linspace(0, 1, 201)
        x_arr = bkd.asarray(x_test, dtype=bkd.double_dtype())

        hier_vals = np.zeros_like(x_test)
        for (l, j), v in surpluses.items():
            basis_vals = bkd.to_numpy(basis.evaluate(x_arr, l, j))
            hier_vals += v * basis_vals

        nodal_vals = np.interp(x_test, nodes_sorted, f_sorted)
        bkd.assert_allclose(
            bkd.asarray(hier_vals),
            bkd.asarray(nodal_vals),
            rtol=0,
            atol=1e-14,
        )

    def test_hierarchical_quadrature_constant(self, bkd, basis):
        """Hierarchical integral of f(x)=1 equals domain width."""
        _, surpluses = _hierarchical_surpluses_1d(
            basis, lambda x: 1.0, 3, bkd
        )
        integral = sum(
            v * basis.quadrature_weight(l, j)
            for (l, j), v in surpluses.items()
        )
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([1.0]),
            rtol=0,
            atol=1e-14,
        )

    def test_hierarchical_quadrature_linear(self, bkd, basis):
        """Hierarchical integral of f(x)=x equals 0.5."""
        _, surpluses = _hierarchical_surpluses_1d(
            basis, lambda x: x, 3, bkd
        )
        integral = sum(
            v * basis.quadrature_weight(l, j)
            for (l, j), v in surpluses.items()
        )
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([0.5]),
            rtol=0,
            atol=1e-14,
        )

    def test_support_width(self, basis):
        """Interior hats have support 2h, boundary hats have support h."""
        h2 = 1.0 / 4  # mesh size at level 2
        for j in [1, 3]:
            left, right = basis.support(2, j)
            assert right - left == pytest.approx(2 * h2)

        # Boundary hats at level 1
        left, right = basis.support(1, 0)
        assert right - left == pytest.approx(0.5)
        left, right = basis.support(1, 2)
        assert right - left == pytest.approx(0.5)

    def test_degree_at(self, basis):
        assert basis.degree_at(0) == 1
        assert basis.degree_at(1) == 1
        assert basis.degree_at(5) == 1

    def test_p_max_gt_1_raises(self, bkd):
        b = HierarchicalBasis1D(bkd, p_max=2)
        with pytest.raises(NotImplementedError, match="p_max=2"):
            b.evaluate(bkd.asarray([0.5]), 0, 1)

    def test_custom_bounds(self, bkd):
        b = HierarchicalBasis1D(bkd, bounds=(-1.0, 3.0), boundary_mode="include")
        assert b.node(0, 1) == pytest.approx(1.0)
        assert b.node(1, 0) == pytest.approx(-1.0)
        assert b.node(1, 2) == pytest.approx(3.0)
        assert b.node(2, 1) == pytest.approx(0.0)

        # Quadrature of f(x)=1 over [-1, 3] should give 4.0
        _, surpluses = _hierarchical_surpluses_1d(b, lambda x: 1.0, 2, bkd)
        integral = sum(
            v * b.quadrature_weight(l, j)
            for (l, j), v in surpluses.items()
        )
        bkd.assert_allclose(
            bkd.asarray([integral]), bkd.asarray([4.0]), rtol=0, atol=1e-14
        )


class TestHierarchicalBasis1DExclude:
    """Tests for HierarchicalBasis1D with boundary_mode='exclude'."""

    @pytest.fixture()
    def basis(self, bkd):
        return HierarchicalBasis1D(bkd, bounds=(0.0, 1.0), boundary_mode="exclude")

    def test_new_points_at_level(self, basis):
        assert basis.new_points_at_level(0) == [1]
        assert basis.new_points_at_level(1) == []
        assert basis.new_points_at_level(2) == [1, 3]
        assert basis.new_points_at_level(3) == [1, 3, 5, 7]

    def test_children_level_0_skips_to_level_2(self, basis):
        assert basis.children(0, 1) == [(2, 1), (2, 3)]

    def test_parent_level_2_skips_to_level_0(self, basis):
        assert basis._parent(2, 1) == (0, 1)
        assert basis._parent(2, 3) == (0, 1)

    def test_ancestors_exclude(self, basis):
        assert basis.ancestors(2, 1) == [(0, 1)]
        assert basis.ancestors(3, 1) == [(2, 1), (0, 1)]
        assert basis.ancestors(3, 7) == [(2, 3), (0, 1)]

    def test_property_a_same_level(self, bkd, basis):
        for l in range(5):
            indices = basis.new_points_at_level(l)
            if not indices:
                continue
            nodes = bkd.asarray(
                [basis.node(l, j) for j in indices],
                dtype=bkd.double_dtype(),
            )
            for i, ja in enumerate(indices):
                vals = bkd.to_numpy(basis.evaluate(nodes, l, ja))
                for k in range(len(indices)):
                    expected = 1.0 if i == k else 0.0
                    assert abs(vals[k] - expected) < 1e-14

    def test_property_b_vanishing(self, bkd, basis):
        max_level = 4
        all_pts = _collect_points(basis, max_level)
        for l_fine in range(1, max_level + 1):
            fine_pts = [(l, j) for l, j in all_pts if l == l_fine]
            for l_coarse in range(l_fine):
                coarse_pts = [(l, j) for l, j in all_pts if l == l_coarse]
                if not coarse_pts:
                    continue
                coarse_nodes = bkd.asarray(
                    [basis.node(l, j) for l, j in coarse_pts],
                    dtype=bkd.double_dtype(),
                )
                for lf, jf in fine_pts:
                    vals = bkd.to_numpy(basis.evaluate(coarse_nodes, lf, jf))
                    for k in range(len(coarse_pts)):
                        assert abs(vals[k]) < 1e-14

    def test_hierarchical_interpolant_matches_nodal(self, bkd, basis):
        def f(x):
            return np.sin(2 * np.pi * x) + x**2

        max_level = 3
        pts, surpluses = _hierarchical_surpluses_1d(basis, f, max_level, bkd)
        nodes_sorted = sorted(basis.node(l, j) for l, j in pts)
        f_sorted = [f(n) for n in nodes_sorted]

        # Test within the convex hull of grid nodes only (exclude mode
        # has no boundary nodes, so np.interp extrapolates outside the
        # node range while the hierarchical interpolant goes to zero).
        x_min, x_max = nodes_sorted[0], nodes_sorted[-1]
        x_test = np.linspace(x_min, x_max, 199)
        x_arr = bkd.asarray(x_test, dtype=bkd.double_dtype())

        hier_vals = np.zeros_like(x_test)
        for (l, j), v in surpluses.items():
            basis_vals = bkd.to_numpy(basis.evaluate(x_arr, l, j))
            hier_vals += v * basis_vals

        nodal_vals = np.interp(x_test, nodes_sorted, f_sorted)
        bkd.assert_allclose(
            bkd.asarray(hier_vals),
            bkd.asarray(nodal_vals),
            rtol=0,
            atol=1e-14,
        )

    def test_invalid_boundary_mode_raises(self, bkd):
        with pytest.raises(ValueError, match="boundary_mode"):
            HierarchicalBasis1D(bkd, boundary_mode="invalid")


class TestHierarchicalBasisND:
    """Tests for HierarchicalBasisND."""

    @pytest.fixture()
    def basis_2d(self, bkd):
        b1 = HierarchicalBasis1D(bkd, bounds=(0.0, 1.0), boundary_mode="include")
        b2 = HierarchicalBasis1D(bkd, bounds=(0.0, 1.0), boundary_mode="include")
        return HierarchicalBasisND(bkd, [b1, b2])

    def test_nvars(self, basis_2d):
        assert basis_2d.nvars() == 2

    def test_points_in_subspace_root(self, basis_2d):
        pts = basis_2d.points_in_subspace((0, 0))
        assert pts == [(1, 1)]

    def test_points_in_subspace_level_1_0(self, basis_2d):
        pts = basis_2d.points_in_subspace((1, 0))
        # Level 1 in dim 0 has indices {0, 2}; level 0 in dim 1 has {1}
        assert set(pts) == {(0, 1), (2, 1)}

    def test_points_in_subspace_empty(self, bkd):
        b1 = HierarchicalBasis1D(bkd, boundary_mode="exclude")
        b2 = HierarchicalBasis1D(bkd, boundary_mode="include")
        basis = HierarchicalBasisND(bkd, [b1, b2])
        # Level (1, 0): dim 0 exclude mode has no level-1 points
        assert basis.points_in_subspace((1, 0)) == []

    def test_node_2d(self, bkd, basis_2d):
        node = basis_2d.node((0, 0), (1, 1))
        node_np = bkd.to_numpy(node)
        assert node_np.shape == (2, 1)
        bkd.assert_allclose(node, bkd.asarray([[0.5], [0.5]]))

    def test_property_a_2d(self, bkd, basis_2d):
        """Same-subspace interpolation: Psi_{l,j}(node(l,j')) = delta."""
        for sub in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]:
            pts = basis_2d.points_in_subspace(sub)
            if not pts:
                continue
            # Build (2, npts) array of node coordinates
            node_coords = bkd.hstack(
                [basis_2d.node(sub, p) for p in pts]
            )
            for i, pi in enumerate(pts):
                vals = bkd.to_numpy(basis_2d.evaluate(node_coords, sub, pi))
                for k in range(len(pts)):
                    expected = 1.0 if i == k else 0.0
                    assert abs(vals[k] - expected) < 1e-14, (
                        f"sub={sub} Psi_{pi}(node_{pts[k]}) = "
                        f"{vals[k]}, expected {expected}"
                    )

    def test_property_b_2d(self, bkd, basis_2d):
        """Finer-subspace basis vanishes at coarser-subspace nodes.

        For a multi-index l_fine that is component-wise >= l_coarse with
        at least one component strictly greater, evaluate all basis
        functions from l_fine at all nodes from l_coarse.
        """
        subspaces = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
        for l_fine in subspaces:
            fine_pts = basis_2d.points_in_subspace(l_fine)
            if not fine_pts:
                continue
            for l_coarse in subspaces:
                # l_coarse must be strictly dominated by l_fine
                if not all(
                    l_coarse[d] <= l_fine[d] for d in range(2)
                ):
                    continue
                if l_coarse == l_fine:
                    continue
                coarse_pts = basis_2d.points_in_subspace(l_coarse)
                if not coarse_pts:
                    continue
                coarse_nodes = bkd.hstack(
                    [basis_2d.node(l_coarse, p) for p in coarse_pts]
                )
                for jf in fine_pts:
                    vals = bkd.to_numpy(
                        basis_2d.evaluate(coarse_nodes, l_fine, jf)
                    )
                    for k, jc in enumerate(coarse_pts):
                        assert abs(vals[k]) < 1e-14, (
                            f"Psi_{l_fine},{jf}(node_{l_coarse},{jc})"
                            f" = {vals[k]}, expected 0"
                        )

    def test_children_of_point_2d(self, basis_2d):
        # Children of root (0,0),(1,1) in direction 0
        children = basis_2d.children_of_point((0, 0), (1, 1), direction=0)
        assert set(children) == {
            ((1, 0), (0, 1)),
            ((1, 0), (2, 1)),
        }
        # Children in direction 1
        children = basis_2d.children_of_point((0, 0), (1, 1), direction=1)
        assert set(children) == {
            ((0, 1), (1, 0)),
            ((0, 1), (1, 2)),
        }

    def test_quadrature_weight_2d(self, basis_2d):
        """ND weight is product of 1D weights."""
        sub = (2, 1)
        pts = basis_2d.points_in_subspace(sub)
        for p in pts:
            w_nd = basis_2d.quadrature_weight(sub, p)
            w1 = basis_2d._bases_1d[0].quadrature_weight(sub[0], p[0])
            w2 = basis_2d._bases_1d[1].quadrature_weight(sub[1], p[1])
            assert w_nd == pytest.approx(w1 * w2)

    def test_hierarchical_interpolant_2d(self, bkd, basis_2d):
        """2D hierarchical interpolant of f(x,y) = x + y is exact."""
        def f_2d(x, y):
            return x + y

        # Collect all subspaces up to total level 2
        subspaces = []
        for l0 in range(3):
            for l1 in range(3):
                subspaces.append((l0, l1))

        # Compute surpluses level-by-level (lower subspaces first)
        subspaces.sort(key=lambda s: (sum(s), s))
        surpluses = {}
        evaluated_keys = []
        for sub in subspaces:
            pts = basis_2d.points_in_subspace(sub)
            for p in pts:
                node = basis_2d.node(sub, p)
                x, y = float(bkd.to_numpy(node[0, 0])), float(
                    bkd.to_numpy(node[1, 0])
                )
                pred = 0.0
                for (sl, sp), v in surpluses.items():
                    val = bkd.to_numpy(
                        basis_2d.evaluate(node, sl, sp)
                    )
                    pred += v * float(val[0])
                surpluses[(sub, p)] = f_2d(x, y) - pred
                evaluated_keys.append((sub, p))

        # Evaluate on random test points
        np.random.seed(42)
        x_test = np.random.rand(2, 50)
        x_arr = bkd.asarray(x_test, dtype=bkd.double_dtype())

        hier_vals = np.zeros(50)
        for (sub, p), v in surpluses.items():
            vals = bkd.to_numpy(basis_2d.evaluate(x_arr, sub, p))
            hier_vals += v * vals

        exact = x_test[0] + x_test[1]
        bkd.assert_allclose(
            bkd.asarray(hier_vals),
            bkd.asarray(exact),
            rtol=0,
            atol=1e-13,
        )

    def test_hierarchical_quadrature_2d(self, bkd, basis_2d):
        """Hierarchical quadrature of f(x,y) = x*y over [0,1]^2."""
        def f_2d(x, y):
            return x * y

        subspaces = []
        for l0 in range(3):
            for l1 in range(3):
                subspaces.append((l0, l1))
        subspaces.sort(key=lambda s: (sum(s), s))

        surpluses = {}
        for sub in subspaces:
            pts = basis_2d.points_in_subspace(sub)
            for p in pts:
                node = basis_2d.node(sub, p)
                x = float(bkd.to_numpy(node[0, 0]))
                y = float(bkd.to_numpy(node[1, 0]))
                pred = 0.0
                for (sl, sp), v in surpluses.items():
                    val = bkd.to_numpy(basis_2d.evaluate(node, sl, sp))
                    pred += v * float(val[0])
                surpluses[(sub, p)] = f_2d(x, y) - pred

        integral = sum(
            v * basis_2d.quadrature_weight(sub, p)
            for (sub, p), v in surpluses.items()
        )
        # int_0^1 int_0^1 x*y dx dy = 0.25
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([0.25]),
            rtol=0,
            atol=1e-14,
        )
