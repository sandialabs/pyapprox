"""Dual-backend tests for sparse grid derivatives using DerivativeChecker.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

from typing import List

import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


class TestSparseGridDerivatives:
    """Tests for sparse grid derivatives using DerivativeChecker."""

    def _build_surrogate(self, nvars, level, func, bkd):
        """Build an isotropic sparse grid surrogate for the given function."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories: List[BasisFactoryProtocol] = [
            GaussLagrangeFactory(marginal, bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
        samples = fitter.get_samples()
        values = func(samples)
        result = fitter.fit(values)
        return result.surrogate

    @pytest.mark.slow_on("TorchBkd")
    def test_jacobian_linear_function(self, bkd) -> None:
        """Test Jacobian of linear function is constant."""

        def func(s):
            return bkd.reshape(s[0, :] + 2 * s[1, :], (1, -1))

        surrogate = self._build_surrogate(2, 2, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian should be [1, 2]
        jac = surrogate.derivatives().jacobian(test_pt)
        expected_jac = bkd.asarray([[1.0, 2.0]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-6)

        jac_error = float(checker.error_ratio(errors[0]).item())
        assert jac_error < 1e-6

    def test_jacobian_quadratic_function(self, bkd) -> None:
        """Test Jacobian of quadratic function."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.reshape(x**2 + x * y, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian at (0.3, 0.4) should be [2*0.3 + 0.4, 0.3] = [1.0, 0.3]
        jac = surrogate.derivatives().jacobian(test_pt)
        expected_jac = bkd.asarray([[1.0, 0.3]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-6)

        jac_error = float(checker.error_ratio(errors[0]).item())
        assert jac_error < 1e-6

    def test_jacobian_3d_function(self, bkd) -> None:
        """Test Jacobian of 3D function."""

        def func(s):
            return bkd.reshape(s[0, :] + s[1, :] + s[2, :], (1, -1))

        surrogate = self._build_surrogate(3, 2, func, bkd)

        test_pt = bkd.asarray([[0.1], [0.2], [0.3]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian should be [1, 1, 1]
        jac = surrogate.derivatives().jacobian(test_pt)
        expected_jac = bkd.asarray([[1.0, 1.0, 1.0]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-6)

        jac_error = float(checker.error_ratio(errors[0]).item())
        assert jac_error < 1e-6

    def test_derivative_checker_passes(self, bkd) -> None:
        """Test that DerivativeChecker passes for sparse grid function."""

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        for x_val, y_val in [(0.0, 0.0), (0.3, 0.4), (-0.5, 0.2)]:
            test_pt = bkd.asarray([[x_val], [y_val]])
            checker = DerivativeChecker(surrogate)
            errors = checker.check_derivatives(test_pt, verbosity=0)

            jac_error = float(checker.error_ratio(errors[0]).item())
            assert jac_error < 1e-6

    def test_hessian_vector_product(self, bkd) -> None:
        """Test HVP computation."""

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        vec = bkd.asarray([[1.0], [0.0]])

        hvp = surrogate.derivatives().hvp(test_pt, vec)

        # Hessian is [[2, 0], [0, 2]]
        # HVP with [1, 0] should give [2, 0]
        expected_hvp = bkd.asarray([[2.0], [0.0]])
        bkd.assert_allclose(hvp, expected_hvp, rtol=1e-6, atol=1e-14)

    def test_weighted_hvp(self, bkd) -> None:
        """Test weighted HVP computation."""

        def func(s):
            return bkd.reshape(s[0, :] ** 2 + s[1, :] ** 2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        vec = bkd.asarray([[1.0], [1.0]])
        weights = bkd.asarray([[0.5]])

        whvp = surrogate.derivatives().whvp(test_pt, vec, weights)

        # Hessian is [[2, 0], [0, 2]]
        # WHVP with [1, 1] and weight 0.5 should give 0.5 * [2, 2] = [1, 1]
        expected_whvp = bkd.asarray([[1.0], [1.0]])
        bkd.assert_allclose(whvp, expected_whvp, rtol=1e-6)

    def test_hessian_via_derivative_checker(self, bkd) -> None:
        """Test Hessian via DerivativeChecker errors[1] for nqoi=1."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.reshape(x**2 + x * y + y**2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        assert len(errors) == 2
        hessian_error = float(checker.error_ratio(errors[1]).item())
        assert hessian_error < 1e-6

    @pytest.mark.parametrize(
        "basis_type", ["gauss", "leja", "clenshaw_curtis"]
    )
    def test_subspace_bundle_declares_what_it_can_do(
        self, bkd, basis_type: str
    ) -> None:
        """A declared field must work; the combination relies on this.

        The surrogate composes its bundle by intersecting the
        subspaces', so a subspace that over-declares would produce a
        surrogate field that raises on first use.
        """
        from pyapprox.surrogates.affine.indices import (
            ClenshawCurtisGrowthRule,
        )
        from pyapprox.surrogates.sparsegrids import create_basis_factories

        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        growth = (
            ClenshawCurtisGrowthRule()
            if basis_type == "clenshaw_curtis"
            else LinearGrowthRule(scale=1, shift=1)
        )
        tp_factory = TensorProductSubspaceFactory(
            bkd, create_basis_factories(marginals, bkd, basis_type), growth
        )
        subspace = tp_factory(
            bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        )
        samples = subspace.get_samples()
        subspace.set_values(
            bkd.reshape(bkd.sum(samples**2, axis=0), (1, -1))
        )

        derivs = subspace.derivatives()
        point = bkd.asarray([[0.3], [0.6]])
        vec = bkd.asarray([[1.0], [0.5]])

        assert derivs.jacobian is not None
        assert derivs.jacobian(point).shape == (1, 2)
        if derivs.hessian is not None:
            assert derivs.hessian(point).shape == (2, 2)
        if derivs.hvp is not None:
            assert derivs.hvp(point, vec).shape == (2, 1)
        if derivs.whvp is not None:
            assert derivs.whvp(
                point, vec, bkd.asarray([[1.0]])
            ).shape == (2, 1)

    def test_piecewise_subspace_declares_no_derivatives(self, bkd) -> None:
        """Absence is the honest answer when the basis has none."""
        from pyapprox.surrogates.affine.indices import (
            ClenshawCurtisGrowthRule,
        )
        from pyapprox.surrogates.sparsegrids import create_basis_factories

        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        tp_factory = TensorProductSubspaceFactory(
            bkd,
            create_basis_factories(marginals, bkd, "piecewise_linear"),
            ClenshawCurtisGrowthRule(),
        )
        subspace = tp_factory(
            bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        )
        samples = subspace.get_samples()
        subspace.set_values(
            bkd.reshape(bkd.sum(samples**2, axis=0), (1, -1))
        )
        derivs = subspace.derivatives()
        assert derivs.jacobian is None
        assert derivs.hessian is None
        assert derivs.hvp is None
        assert derivs.whvp is None

    def test_hessian_field_is_declared_and_correct(self, bkd) -> None:
        """The bundle's hessian field, not the checker's hvp route.

        Every subspace offers a hessian at nqoi=1, and the Hessian is
        linear in the subspaces, so the combination can offer one too.
        """

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.reshape(3 * x**2 + 2 * x * y + 5 * y**2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)
        hessian_fn = surrogate.derivatives().hessian
        assert hessian_fn is not None

        # Constant for a quadratic: d2f = [[6, 2], [2, 10]].
        expected = bkd.asarray([[6.0, 2.0], [2.0, 10.0]])
        for point in ([[0.3], [0.4]], [[-0.8], [0.1]]):
            got = hessian_fn(bkd.asarray(point))
            assert got.shape == (2, 2)
            bkd.assert_allclose(got, expected, atol=1e-9)

    def test_hessian_matches_hvp(self, bkd) -> None:
        """H @ v must equal the hvp the same bundle reports."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.reshape(x**2 + x * y + 2 * y**2, (1, -1))

        surrogate = self._build_surrogate(2, 3, func, bkd)
        derivs = surrogate.derivatives()
        assert derivs.hessian is not None and derivs.hvp is not None

        point = bkd.asarray([[0.25], [-0.5]])
        vec = bkd.asarray([[1.0], [0.5]])
        bkd.assert_allclose(
            derivs.hessian(point) @ vec, derivs.hvp(point, vec), atol=1e-9
        )

    def test_hessian_withheld_for_multiple_qoi(self, bkd) -> None:
        """A Hessian is only defined for nqoi == 1."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.stack([x**2 + y, x * y], axis=0)

        surrogate = self._build_surrogate(2, 3, func, bkd)
        derivs = surrogate.derivatives()
        assert derivs.hessian is None
        assert derivs.hvp is None
        # whvp is the multi-QoI route and stays available.
        assert derivs.whvp is not None

    def test_whvp_via_derivative_checker(self, bkd) -> None:
        """Test WHVP via DerivativeChecker for nqoi=2 with weights."""

        def func(s):
            x, y = s[0, :], s[1, :]
            return bkd.stack([x**2 + y, x * y], axis=0)

        surrogate = self._build_surrogate(2, 3, func, bkd)

        test_pt = bkd.asarray([[0.3], [0.4]])
        weights = bkd.asarray([[0.6], [0.4]])

        checker = DerivativeChecker(surrogate)
        errors = checker.check_derivatives(test_pt, verbosity=0, weights=weights)

        assert len(errors) == 2
        whvp_error = float(checker.error_ratio(errors[1]).item())
        assert whvp_error < 1e-6
