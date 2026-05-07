"""Tests for HierarchicalSurrogate."""

import numpy as np
import pytest
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_surrogate import (
    HierarchicalSurrogate,
)


def _build_1d_surrogate(bkd, f, max_level):
    """Build a 1D hierarchical surrogate for f on [0,1]."""
    b1d = HierarchicalBasis1D(bkd, boundary_mode="include")
    basis_nd = HierarchicalBasisND(bkd, [b1d])

    all_pts = []
    for lv in range(max_level + 1):
        for j in b1d.new_points_at_level(lv):
            all_pts.append(((lv,), (j,)))

    # Compute surpluses sequentially
    keys = []
    surplus_list = []
    weight_list = []
    surpluses_dict = {}
    for key in all_pts:
        sub, pt = key
        x = b1d.node(sub[0], pt[0])
        pred = 0.0
        for prev_key, v in surpluses_dict.items():
            val = bkd.to_numpy(
                basis_nd.evaluate(
                    bkd.asarray([[x]]), prev_key[0], prev_key[1]
                )
            )
            pred += v * float(val[0])
        surplus = f(x) - pred
        surpluses_dict[key] = surplus
        keys.append(key)
        surplus_list.append(surplus)
        weight_list.append(basis_nd.quadrature_weight(sub, pt))

    nqoi = 1
    surpluses_arr = bkd.asarray(surplus_list, dtype=bkd.double_dtype()).reshape(
        nqoi, len(keys)
    )
    weights_arr = bkd.asarray(weight_list, dtype=bkd.double_dtype())

    surrogate = HierarchicalSurrogate(
        bkd, basis_nd, keys, surpluses_arr, weights_arr
    )
    return surrogate, basis_nd


class TestHierarchicalSurrogate:
    def test_1d_known_surpluses(self, bkd):
        """Build with known surpluses and verify evaluation."""
        b1d = HierarchicalBasis1D(bkd, boundary_mode="include")
        basis_nd = HierarchicalBasisND(bkd, [b1d])

        # f(x) = 1: level-0 constant basis already captures it exactly,
        # so boundary surpluses at level 1 are zero.
        keys = [((0,), (1,)), ((1,), (0,)), ((1,), (2,))]
        surpluses = bkd.asarray([[1.0, 0.0, 0.0]])
        weights = bkd.asarray(
            [basis_nd.quadrature_weight(k[0], k[1]) for k in keys]
        )
        surr = HierarchicalSurrogate(bkd, basis_nd, keys, surpluses, weights)

        x_test = bkd.asarray([[0.0, 0.25, 0.5, 0.75, 1.0]])
        vals = surr(x_test)
        expected = bkd.ones((1, 5))
        bkd.assert_allclose(vals, expected, atol=1e-14)

    def test_1d_linear_interpolation(self, bkd):
        """f(x) = 3x + 1 should be reproduced exactly."""
        surr, _ = _build_1d_surrogate(bkd, lambda x: 3 * x + 1, max_level=2)

        x_test = bkd.asarray(
            np.linspace(0, 1, 51).reshape(1, -1), dtype=bkd.double_dtype()
        )
        vals = surr(x_test)
        expected = 3 * x_test + 1
        bkd.assert_allclose(vals, expected, atol=1e-13)

    def test_1d_evaluate_single(self, bkd):
        """evaluate() on a single sample gives shape (nqoi, 1)."""
        surr, _ = _build_1d_surrogate(bkd, lambda x: x**2, max_level=2)
        val = surr.evaluate(bkd.asarray([[0.25]]))
        assert bkd.to_numpy(val).shape == (1, 1)

    def test_2d_linear_exact(self, bkd):
        """f(x,y) = x + y is reproduced exactly on full grid."""
        b1 = HierarchicalBasis1D(bkd, boundary_mode="include")
        b2 = HierarchicalBasis1D(bkd, boundary_mode="include")
        basis_nd = HierarchicalBasisND(bkd, [b1, b2])

        def f_2d(x, y):
            return x + y

        # Collect subspaces up to level 2
        subspaces = []
        for l0 in range(3):
            for l1 in range(3):
                subspaces.append((l0, l1))
        subspaces.sort(key=lambda s: (sum(s), s))

        keys = []
        surplus_list = []
        weight_list = []
        surpluses_dict = {}

        for sub in subspaces:
            pts = basis_nd.points_in_subspace(sub)
            for pt in pts:
                node = basis_nd.node(sub, pt)
                x = float(bkd.to_numpy(node[0, 0]))
                y = float(bkd.to_numpy(node[1, 0]))
                pred = 0.0
                for prev_key, v in surpluses_dict.items():
                    val = bkd.to_numpy(
                        basis_nd.evaluate(node, prev_key[0], prev_key[1])
                    )
                    pred += v * float(val[0])
                surplus = f_2d(x, y) - pred
                key = (sub, pt)
                surpluses_dict[key] = surplus
                keys.append(key)
                surplus_list.append(surplus)
                weight_list.append(basis_nd.quadrature_weight(sub, pt))

        surpluses_arr = bkd.asarray(surplus_list).reshape(1, len(keys))
        weights_arr = bkd.asarray(weight_list)
        surr = HierarchicalSurrogate(
            bkd, basis_nd, keys, surpluses_arr, weights_arr
        )

        np.random.seed(42)
        x_test = bkd.asarray(np.random.rand(2, 30), dtype=bkd.double_dtype())
        vals = surr(x_test)
        expected = x_test[0:1, :] + x_test[1:2, :]
        bkd.assert_allclose(vals, expected, atol=1e-13)

    def test_mean_constant(self, bkd):
        """Mean of f(x)=1 over [0,1] is 1."""
        surr, _ = _build_1d_surrogate(bkd, lambda x: 1.0, max_level=2)
        m = surr.mean()
        bkd.assert_allclose(m, bkd.asarray([1.0]), atol=1e-14)

    def test_mean_linear(self, bkd):
        """Mean of f(x)=x over [0,1] is 0.5."""
        surr, _ = _build_1d_surrogate(bkd, lambda x: x, max_level=2)
        m = surr.mean()
        bkd.assert_allclose(m, bkd.asarray([0.5]), atol=1e-14)

    def test_snapshot_independence(self, bkd):
        """Modifying source data after construction does not affect surrogate."""
        b1d = HierarchicalBasis1D(bkd, boundary_mode="include")
        basis_nd = HierarchicalBasisND(bkd, [b1d])
        keys = [((0,), (1,))]
        surpluses = bkd.asarray([[5.0]])
        weights = bkd.asarray([0.5])

        surr = HierarchicalSurrogate(
            bkd, basis_nd, keys, surpluses, weights
        )

        # Modify source arrays
        surpluses_np = bkd.to_numpy(surpluses)
        surpluses_np[0, 0] = 999.0

        # Surrogate should still return original value
        val = surr(bkd.asarray([[0.5]]))
        bkd.assert_allclose(val, bkd.asarray([[5.0]]), atol=1e-14)

    def test_nvars_nqoi(self, bkd):
        surr, _ = _build_1d_surrogate(bkd, lambda x: x, max_level=1)
        assert surr.nvars() == 1
        assert surr.nqoi() == 1
        assert surr.n_points() == 3  # level 0 + level 1 boundary


class TestComputeSurplusesAtGridPoints:
    """Verify ancestor-based surplus computation matches batch evaluation."""

    def _build_surrogate_incrementally(self, bkd, basis_nd, subspaces, f_nd):
        """Build surrogate level-by-level, returning it plus all keys."""
        all_keys = []
        surr = None

        subspaces.sort(key=lambda s: (sum(s), s))
        for sub in subspaces:
            pts = basis_nd.points_in_subspace(sub)
            if not pts:
                continue
            new_keys = [(sub, pt) for pt in pts]
            nodes = bkd.hstack([basis_nd.node(sub, pt) for pt in pts])
            f_vals = f_nd(nodes)

            if surr is not None and surr.n_points() > 0:
                surpluses_ancestor = surr.compute_surpluses_at_grid_points(
                    new_keys, f_vals
                )
                surpluses_batch = f_vals - surr(nodes)
                bkd.assert_allclose(
                    surpluses_ancestor, surpluses_batch, atol=1e-12
                )
            else:
                surpluses_ancestor = bkd.copy(f_vals)

            new_weights = [
                basis_nd.quadrature_weight(sub, pt) for pt in pts
            ]
            new_surplus_list = [
                surpluses_ancestor[:, k] for k in range(len(new_keys))
            ]
            if surr is None:
                surpluses_arr = surpluses_ancestor
                weights_arr = bkd.asarray(
                    new_weights, dtype=bkd.double_dtype()
                )
                surr = HierarchicalSurrogate(
                    bkd, basis_nd, new_keys, surpluses_arr, weights_arr
                )
            else:
                surr.add_points(new_keys, new_surplus_list, new_weights)

            all_keys.extend(new_keys)

        return surr

    @pytest.mark.parametrize("p_max", [1, 2])
    def test_1d_agreement(self, bkd, p_max):
        b1d = HierarchicalBasis1D(bkd, p_max=p_max, boundary_mode="include")
        basis_nd = HierarchicalBasisND(bkd, [b1d])
        subspaces = [(lv,) for lv in range(5)]

        def f(x):
            return bkd.sin(x * 3.0) + x * x

        self._build_surrogate_incrementally(bkd, basis_nd, subspaces, f)

    @pytest.mark.parametrize("p_max", [1, 2])
    def test_2d_agreement(self, bkd, p_max):
        bases = [
            HierarchicalBasis1D(bkd, p_max=p_max, boundary_mode="include")
            for _ in range(2)
        ]
        basis_nd = HierarchicalBasisND(bkd, bases)
        subspaces = [
            (l0, l1) for l0 in range(4) for l1 in range(4)
        ]

        def f(x):
            return bkd.sin(x[0:1, :]) * bkd.cos(x[1:2, :]) + x[0:1, :] ** 2

        self._build_surrogate_incrementally(bkd, basis_nd, subspaces, f)

    @pytest.mark.parametrize("p_max", [1, 2])
    def test_3d_agreement(self, bkd, p_max):
        bases = [
            HierarchicalBasis1D(bkd, p_max=p_max, boundary_mode="include")
            for _ in range(3)
        ]
        basis_nd = HierarchicalBasisND(bkd, bases)
        subspaces = [
            (l0, l1, l2)
            for l0 in range(3) for l1 in range(3) for l2 in range(3)
        ]

        def f(x):
            return x[0:1, :] ** 2 + x[1:2, :] * x[2:3, :]

        self._build_surrogate_incrementally(bkd, basis_nd, subspaces, f)

    def test_1d_exclude_mode(self, bkd):
        b1d = HierarchicalBasis1D(bkd, p_max=1, boundary_mode="exclude")
        basis_nd = HierarchicalBasisND(bkd, [b1d])
        subspaces = [(lv,) for lv in range(5)]

        def f(x):
            return bkd.sin(x * 3.14159)

        self._build_surrogate_incrementally(bkd, basis_nd, subspaces, f)

    def test_missing_ancestor_raises(self, bkd):
        """ValueError when a required ancestor point is not stored."""
        bases = [
            HierarchicalBasis1D(bkd, p_max=1, boundary_mode="include")
            for _ in range(2)
        ]
        basis_nd = HierarchicalBasisND(bkd, bases)

        # Store (0,0)/(1,1), (1,0)/(0,1), and (0,1)/(1,2).
        # Subspace (0,1) is in _selected_subspaces via the last key.
        # New point ((1,1),(0,0)) at node (0.0,0.0) needs ancestor
        # ((0,1),(1,0)) — subspace (0,1) is selected but point (1,0)
        # is not stored (only (1,2) is).
        keys = [
            ((0, 0), (1, 1)),
            ((1, 0), (0, 1)),
            ((0, 1), (1, 2)),
        ]
        surpluses = bkd.asarray([[1.0, 0.5, 0.3]])
        weights = bkd.asarray([0.25, 0.25, 0.25])
        surr = HierarchicalSurrogate(bkd, basis_nd, keys, surpluses, weights)

        new_keys = [((1, 1), (0, 0))]
        f_vals = bkd.asarray([[0.5]])
        with pytest.raises(ValueError, match="not stored"):
            surr.compute_surpluses_at_grid_points(new_keys, f_vals)

    def test_cost_scaling(self, numpy_bkd):
        """Ancestor lookups per point grow sublinearly vs total grid size."""
        from itertools import product as cartesian_product

        bkd = numpy_bkd
        nvars = 3
        bases = [
            HierarchicalBasis1D(bkd, p_max=1, boundary_mode="include")
            for _ in range(nvars)
        ]
        basis_nd = HierarchicalBasisND(bkd, bases)

        ancestor_counts = []
        grid_sizes = []
        for max_l in [2, 3, 4, 5]:
            subspaces = [
                tuple(ls) for ls in cartesian_product(
                    *[range(max_l + 1)] * nvars
                )
            ]
            n_points = sum(
                len(basis_nd.points_in_subspace(s)) for s in subspaces
            )
            top_sub = (max_l,) * nvars
            top_pts = basis_nd.points_in_subspace(top_sub)
            if not top_pts:
                continue

            n_ancestors = 0
            selected = set(subspaces)
            for pt in top_pts:
                level_k = top_sub
                for anc in cartesian_product(
                    *[range(level_k[d] + 1) for d in range(nvars)]
                ):
                    if anc == level_k:
                        continue
                    if anc in selected:
                        n_ancestors += 1

            avg_ancestors = n_ancestors / len(top_pts)
            ancestor_counts.append(avg_ancestors)
            grid_sizes.append(n_points)

        ratio_grid = grid_sizes[-1] / grid_sizes[0]
        ratio_anc = ancestor_counts[-1] / ancestor_counts[0]
        assert ratio_anc < ratio_grid, (
            f"Ancestors grew {ratio_anc:.1f}x but grid grew "
            f"{ratio_grid:.1f}x — should be sublinear"
        )

    @pytest.mark.parametrize("p_max", [1, 2])
    def test_polynomial_reproduction(self, bkd, p_max):
        """Linear function produces zero surpluses past level 1."""
        b1d = HierarchicalBasis1D(bkd, p_max=p_max, boundary_mode="include")
        basis_nd = HierarchicalBasisND(bkd, [b1d])

        def f(x):
            return 2.0 * x + 1.0

        subspaces = [(lv,) for lv in range(5)]
        subspaces.sort(key=lambda s: (sum(s), s))

        surr = None
        for sub in subspaces:
            pts = basis_nd.points_in_subspace(sub)
            if not pts:
                continue
            new_keys = [(sub, pt) for pt in pts]
            nodes = bkd.hstack([basis_nd.node(sub, pt) for pt in pts])
            f_vals = f(nodes)

            if surr is not None and surr.n_points() > 0:
                surplus = surr.compute_surpluses_at_grid_points(
                    new_keys, f_vals
                )
            else:
                surplus = bkd.copy(f_vals)

            if sum(sub) >= 2:
                bkd.assert_allclose(
                    surplus,
                    bkd.zeros_like(surplus),
                    atol=1e-13,
                )

            new_surplus_list = [
                surplus[:, k] for k in range(len(new_keys))
            ]
            new_weights = [
                basis_nd.quadrature_weight(sub, pt) for pt in pts
            ]
            if surr is None:
                surr = HierarchicalSurrogate(
                    bkd, basis_nd, new_keys, surplus,
                    bkd.asarray(new_weights, dtype=bkd.double_dtype()),
                )
            else:
                surr.add_points(new_keys, new_surplus_list, new_weights)
