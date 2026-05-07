"""Tests for HierarchicalSurrogate."""

import numpy as np

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
    for l in range(max_level + 1):
        for j in b1d.new_points_at_level(l):
            all_pts.append(((l,), (j,)))

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
