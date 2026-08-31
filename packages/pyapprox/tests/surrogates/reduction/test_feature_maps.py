"""Tests for polynomial feature maps used in quadratic/cubic manifolds."""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.reduction.feature_maps import (
    DifferentiableFeatureMap,
    FeatureMap,
    MonomialFeatureMap,
    SparseMonomialFeatureMap,
    build_feature_map,
)


class TestMonomialFeatureMap:
    """Values, term counts, and degree selection."""

    def test_quadratic_nterms(self, bkd) -> None:
        # r=2 quadratic monomials: x0^2, x0 x1, x1^2 -> p=3
        fm = build_feature_map(2, bkd, degrees=(2,))
        assert fm.nterms() == 3

    def test_quadratic_values(self, bkd) -> None:
        fm = build_feature_map(2, bkd, degrees=(2,))
        rng = np.random.RandomState(0)
        z = bkd.array(rng.uniform(-1, 1, (2, 4)))
        h = fm(z)
        # The three quadratic monomials in some order; verify each column is a
        # permutation of {z0^2, z0 z1, z1^2}.
        z_np = bkd.to_numpy(z)
        expected = np.sort(
            np.vstack([z_np[0] ** 2, z_np[0] * z_np[1], z_np[1] ** 2]), axis=0
        )
        got = np.sort(bkd.to_numpy(h), axis=0)
        bkd.assert_allclose(bkd.array(got), bkd.array(expected))

    def test_cubic_adds_terms(self, bkd) -> None:
        # r=2: quadratic p=3; quadratic+cubic adds the 4 cubic monomials -> 7
        fm = build_feature_map(2, bkd, degrees=(2, 3))
        assert fm.nterms() == 7
        degrees = bkd.to_numpy(bkd.sum(fm.indices(), axis=0)).astype(int)
        assert sorted(degrees.tolist()) == [2, 2, 2, 3, 3, 3, 3]

    def test_degree_band_3d(self, bkd) -> None:
        # r=3 quadratic: C(3+1,2) = 6 monomials of degree exactly 2
        fm = build_feature_map(3, bkd, degrees=(2,))
        assert fm.nterms() == 6
        degrees = bkd.to_numpy(bkd.sum(fm.indices(), axis=0)).astype(int)
        assert np.all(degrees == 2)

    def test_rejects_low_degree(self, bkd) -> None:
        with pytest.raises(ValueError):
            build_feature_map(2, bkd, degrees=(1, 2))

    def test_accessors(self, bkd) -> None:
        fm = build_feature_map(3, bkd, degrees=(2, 3))
        assert fm.nreduced() == 3
        assert fm.degrees() == (2, 3)
        assert fm.bkd() is bkd
        assert fm.indices().shape == (3, fm.nterms())


class TestSparseMonomialFeatureMap:
    """An arbitrary index set, not a degree band."""

    def test_values_match_indices(self, bkd) -> None:
        # h(z) = [z0^2 z1, z1^3] -- anisotropic, not a degree band.
        indices = bkd.asarray(np.array([[2, 0], [1, 3]]))
        fm = SparseMonomialFeatureMap(indices, bkd)
        assert fm.nterms() == 2
        assert fm.nreduced() == 2
        z = bkd.array(np.random.RandomState(3).uniform(-1, 1, (2, 5)))
        z_np = bkd.to_numpy(z)
        expected = np.vstack([z_np[0] ** 2 * z_np[1], z_np[1] ** 3])
        bkd.assert_allclose(fm(z), bkd.array(expected))

    def test_from_index_set_splits_linear_and_correction(self, bkd) -> None:
        # Columns: constant, linear z0, linear z1, quadratic z0^2, z0 z1.
        # Dimension 2 never appears, so it is not an active dimension.
        full = bkd.asarray(
            np.array(
                [
                    [0, 1, 0, 2, 1],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                ]
            )
        )
        active_dims, fm = SparseMonomialFeatureMap.from_index_set(full, bkd)
        assert active_dims == [0, 1]
        # Only the two degree->=2 columns become correction features.
        assert fm.nterms() == 2
        assert fm.nreduced() == 2
        degrees = bkd.to_numpy(bkd.sum(fm.indices(), axis=0)).astype(int)
        assert np.all(degrees >= 2)


class TestFeatureMapProtocols:
    """Evaluation and differentiation are separate contracts."""

    def test_implementations_satisfy_both_protocols(self, bkd) -> None:
        monomial = build_feature_map(2, bkd, degrees=(2,))
        sparse = SparseMonomialFeatureMap(
            bkd.asarray(np.array([[2], [1]])), bkd
        )
        for fm in (monomial, sparse):
            assert isinstance(fm, FeatureMap)
            assert isinstance(fm, DifferentiableFeatureMap)

    def test_evaluation_only_map_is_not_differentiable(self, bkd) -> None:
        """A map without a jacobian still satisfies the weaker protocol.

        This is what the split buys: a consumer needing only evaluation can
        accept such a map, while one needing derivatives rejects it.
        """

        class EvaluationOnlyFeatureMap:
            def nreduced(self) -> int:
                return 2

            def nterms(self) -> int:
                return 1

            def __call__(self, codes):
                return codes[:1, :] ** 2

        fm = EvaluationOnlyFeatureMap()
        assert isinstance(fm, FeatureMap)
        assert not isinstance(fm, DifferentiableFeatureMap)


class TestMonomialFeatureMapJacobian:
    """Validate the analytic Jacobian against finite differences.

    Uses pyapprox's DerivativeChecker, which sweeps a range of finite-
    difference step sizes.  A correct Jacobian produces a V-shaped error
    curve (error decreases with step size until round-off dominates), so
    ``error_ratio = min/max`` is small.  A wrong Jacobian gives a flat
    curve with ratio near one.  Checking a single small FD error is not
    enough; the ratio over the sweep is the real test.
    """

    def _check(self, bkd, nreduced, degrees, seed) -> None:
        fm = MonomialFeatureMap(nreduced, bkd, degrees=degrees)
        p = fm.nterms()

        def fun(sample):
            # sample: (r, 1) -> features (p, 1)
            return fm(sample)

        def jacobian(sample):
            # fm.jacobian: (p, r, 1) -> single-sample (p, r)
            return fm.jacobian(sample)[:, :, 0]

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=p,
            nvars=nreduced,
            fun=fun,
            jacobian=jacobian,
            bkd=bkd,
        )

        checker = DerivativeChecker(function_obj)
        rng = np.random.RandomState(seed)
        sample = bkd.array(rng.uniform(-0.9, 0.9, (nreduced, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)
        jac_error = checker.error_ratio(errors[0])
        assert float(jac_error) < 2e-6

    def test_jacobian_quadratic_2d(self, bkd) -> None:
        self._check(bkd, nreduced=2, degrees=(2,), seed=42)

    def test_jacobian_quadratic_3d(self, bkd) -> None:
        self._check(bkd, nreduced=3, degrees=(2,), seed=7)

    def test_jacobian_cubic_2d(self, bkd) -> None:
        self._check(bkd, nreduced=2, degrees=(2, 3), seed=11)

    def test_jacobian_cubic_3d(self, bkd) -> None:
        self._check(bkd, nreduced=3, degrees=(2, 3), seed=13)

    def test_jacobian_quartic_2d(self, bkd) -> None:
        # Higher order to confirm extensibility of the analytic Jacobian.
        self._check(bkd, nreduced=2, degrees=(2, 3, 4), seed=17)

    def test_jacobian_sparse_map(self, bkd) -> None:
        # A single anisotropic monomial z0^2 z1: dh/dz0 = 2 z0 z1,
        # dh/dz1 = z0^2. Verified against the closed form rather than FD.
        fm = SparseMonomialFeatureMap(bkd.asarray(np.array([[2], [1]])), bkd)
        z = bkd.array(np.random.RandomState(5).uniform(-1, 1, (2, 4)))
        z_np = bkd.to_numpy(z)
        jac = fm.jacobian(z)
        assert jac.shape == (1, 2, 4)
        bkd.assert_allclose(jac[0, 0, :], bkd.array(2 * z_np[0] * z_np[1]))
        bkd.assert_allclose(jac[0, 1, :], bkd.array(z_np[0] ** 2))


class TestFeatureMapAutograd:
    """The torch computation graph must survive feature evaluation.

    Comparing the analytic Jacobian against
    ``torch.autograd.functional.jacobian`` checks two things at once: that
    the graph reaches the output, and that the hand-derived Jacobian is
    the derivative torch computes for the same expression. The finite
    difference tests above establish the second only to FD accuracy.
    """

    def _check(self, torch_bkd, nreduced, degrees, seed) -> None:
        import torch

        fm = MonomialFeatureMap(nreduced, torch_bkd, degrees=degrees)
        rng = np.random.RandomState(seed)
        z = torch_bkd.array(rng.uniform(-0.9, 0.9, (nreduced, 1)))

        # autograd jacobian of z -> h(z) has shape (p, 1, r, 1).
        auto = torch.autograd.functional.jacobian(fm, z)
        torch_bkd.assert_allclose(
            auto[:, 0, :, 0], fm.jacobian(z)[:, :, 0], rtol=1e-10
        )

    def test_autograd_quadratic_2d(self, torch_bkd) -> None:
        self._check(torch_bkd, nreduced=2, degrees=(2,), seed=42)

    def test_autograd_cubic_3d(self, torch_bkd) -> None:
        self._check(torch_bkd, nreduced=3, degrees=(2, 3), seed=13)

    def test_autograd_multiple_columns(self, torch_bkd) -> None:
        """Several coordinates at once, to exercise the column loop."""
        import torch

        fm = MonomialFeatureMap(2, torch_bkd, degrees=(2, 3))
        z = torch_bkd.array(
            np.random.RandomState(23).uniform(-0.9, 0.9, (2, 3))
        )
        auto = torch.autograd.functional.jacobian(fm, z)
        analytic = fm.jacobian(z)
        # Columns are independent: the autograd jacobian is nonzero only
        # where the input column matches the output column.
        for i in range(3):
            torch_bkd.assert_allclose(
                auto[:, i, :, i], analytic[:, :, i], rtol=1e-10
            )
