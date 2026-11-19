"""Tests for univariate domain transforms.

Tests verify that transforms correctly map between user and canonical domains,
and that they satisfy the Univariate1DTransformProtocol.
"""

import pytest

from pyapprox.surrogates.affine.univariate.transforms import (
    BoundedAffineTransform1D,
    IdentityTransform1D,
    UnboundedAffineTransform1D,
    Univariate1DTransformProtocol,
)


class TestIdentityTransform1D:
    """Tests for IdentityTransform1D."""

    def test_protocol_compliance(self, bkd) -> None:
        """IdentityTransform1D satisfies Univariate1DTransformProtocol."""
        transform = IdentityTransform1D(bkd)
        assert isinstance(transform, Univariate1DTransformProtocol)

    def test_map_to_canonical_unchanged(self, bkd) -> None:
        """map_to_canonical returns input unchanged."""
        transform = IdentityTransform1D(bkd)
        samples = bkd.asarray([[0.0, 0.5, 1.0, -1.0]])
        result = transform.map_to_canonical(samples)
        bkd.assert_allclose(result, samples)

    def test_map_from_canonical_unchanged(self, bkd) -> None:
        """map_from_canonical returns input unchanged."""
        transform = IdentityTransform1D(bkd)
        samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        result = transform.map_from_canonical(samples)
        bkd.assert_allclose(result, samples)

    def test_jacobian_factor_is_one(self, bkd) -> None:
        """jacobian_factor returns 1.0 for identity transform."""
        transform = IdentityTransform1D(bkd)
        bkd.assert_allclose(
            bkd.asarray([transform.jacobian_factor()]),
            bkd.asarray([1.0]),
        )

    def test_roundtrip(self, bkd) -> None:
        """Roundtrip through both maps returns original."""
        transform = IdentityTransform1D(bkd)
        samples = bkd.asarray([[-2.0, 0.0, 3.5]])
        canonical = transform.map_to_canonical(samples)
        roundtrip = transform.map_from_canonical(canonical)
        bkd.assert_allclose(roundtrip, samples)

    def test_repr(self, bkd) -> None:
        """repr returns expected string."""
        transform = IdentityTransform1D(bkd)
        assert repr(transform) == "IdentityTransform1D()"


class TestBoundedAffineTransform1D:
    """Tests for BoundedAffineTransform1D."""

    def test_protocol_compliance(self, bkd) -> None:
        """BoundedAffineTransform1D satisfies Univariate1DTransformProtocol."""
        transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
        assert isinstance(transform, Univariate1DTransformProtocol)

    def test_invalid_bounds_raises(self, bkd) -> None:
        """Raises ValueError if lb >= ub."""
        with pytest.raises(ValueError):
            BoundedAffineTransform1D(bkd, lb=1.0, ub=0.0)
        with pytest.raises(ValueError):
            BoundedAffineTransform1D(bkd, lb=1.0, ub=1.0)

    def test_unit_interval_to_canonical(self, bkd) -> None:
        """[0, 1] maps to [-1, 1] correctly."""
        transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
        samples = bkd.asarray([[0.0, 0.5, 1.0]])
        expected = bkd.asarray([[-1.0, 0.0, 1.0]])
        result = transform.map_to_canonical(samples)
        bkd.assert_allclose(result, expected)

    def test_unit_interval_from_canonical(self, bkd) -> None:
        """[-1, 1] maps to [0, 1] correctly."""
        transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
        canonical = bkd.asarray([[-1.0, 0.0, 1.0]])
        expected = bkd.asarray([[0.0, 0.5, 1.0]])
        result = transform.map_from_canonical(canonical)
        bkd.assert_allclose(result, expected)

    def test_arbitrary_bounds_to_canonical(self, bkd) -> None:
        """[2, 6] maps to [-1, 1] correctly."""
        transform = BoundedAffineTransform1D(bkd, lb=2.0, ub=6.0)
        # lb=2, ub=6, midpoint=4, half_width=2
        samples = bkd.asarray([[2.0, 4.0, 6.0]])  # lb, mid, ub
        expected = bkd.asarray([[-1.0, 0.0, 1.0]])
        result = transform.map_to_canonical(samples)
        bkd.assert_allclose(result, expected)

    def test_arbitrary_bounds_from_canonical(self, bkd) -> None:
        """[-1, 1] maps to [2, 6] correctly."""
        transform = BoundedAffineTransform1D(bkd, lb=2.0, ub=6.0)
        canonical = bkd.asarray([[-1.0, 0.0, 1.0]])
        expected = bkd.asarray([[2.0, 4.0, 6.0]])
        result = transform.map_from_canonical(canonical)
        bkd.assert_allclose(result, expected)

    def test_jacobian_factor_unit_interval(self, bkd) -> None:
        """jacobian_factor for [0, 1] is 2.0."""
        transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
        # half_width = 0.5, jacobian = 1/0.5 = 2.0
        bkd.assert_allclose(
            bkd.asarray([transform.jacobian_factor()]),
            bkd.asarray([2.0]),
        )

    def test_jacobian_factor_arbitrary_bounds(self, bkd) -> None:
        """jacobian_factor for [2, 6] is 0.5."""
        transform = BoundedAffineTransform1D(bkd, lb=2.0, ub=6.0)
        # half_width = 2.0, jacobian = 1/2.0 = 0.5
        bkd.assert_allclose(
            bkd.asarray([transform.jacobian_factor()]),
            bkd.asarray([0.5]),
        )

    def test_roundtrip(self, bkd) -> None:
        """Roundtrip through both maps returns original."""
        transform = BoundedAffineTransform1D(bkd, lb=-3.0, ub=7.0)
        samples = bkd.asarray([[-3.0, 0.0, 2.0, 7.0]])
        canonical = transform.map_to_canonical(samples)
        roundtrip = transform.map_from_canonical(canonical)
        bkd.assert_allclose(roundtrip, samples)

    def test_lb_ub_accessors(self, bkd) -> None:
        """lb() and ub() return correct bounds."""
        transform = BoundedAffineTransform1D(bkd, lb=-2.0, ub=5.0)
        bkd.assert_allclose(
            bkd.asarray([transform.lb()]),
            bkd.asarray([-2.0]),
        )
        bkd.assert_allclose(
            bkd.asarray([transform.ub()]),
            bkd.asarray([5.0]),
        )

    def test_repr(self, bkd) -> None:
        """repr returns expected string."""
        transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
        assert repr(transform) == "BoundedAffineTransform1D(lb=0.0, ub=1.0)"


class TestUnboundedAffineTransform1D:
    """Tests for UnboundedAffineTransform1D."""

    def test_protocol_compliance(self, bkd) -> None:
        """UnboundedAffineTransform1D satisfies Univariate1DTransformProtocol."""
        transform = UnboundedAffineTransform1D(bkd, loc=0.0, scale=1.0)
        assert isinstance(transform, Univariate1DTransformProtocol)

    def test_invalid_scale_raises(self, bkd) -> None:
        """Raises ValueError if scale <= 0."""
        with pytest.raises(ValueError):
            UnboundedAffineTransform1D(bkd, loc=0.0, scale=0.0)
        with pytest.raises(ValueError):
            UnboundedAffineTransform1D(bkd, loc=0.0, scale=-1.0)

    def test_standard_normal_unchanged(self, bkd) -> None:
        """N(0, 1) to N(0, 1) leaves samples unchanged."""
        transform = UnboundedAffineTransform1D(bkd, loc=0.0, scale=1.0)
        samples = bkd.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        result = transform.map_to_canonical(samples)
        bkd.assert_allclose(result, samples)

    def test_shifted_normal_to_canonical(self, bkd) -> None:
        """N(5, 4) = N(mean=5, std=2) maps to N(0, 1) correctly."""
        transform = UnboundedAffineTransform1D(bkd, loc=5.0, scale=2.0)
        # mean, mean+std, mean-std, mean+2std
        samples = bkd.asarray([[5.0, 7.0, 3.0, 9.0]])
        expected = bkd.asarray([[0.0, 1.0, -1.0, 2.0]])
        result = transform.map_to_canonical(samples)
        bkd.assert_allclose(result, expected)

    def test_shifted_normal_from_canonical(self, bkd) -> None:
        """N(0, 1) maps to N(5, 4) correctly."""
        transform = UnboundedAffineTransform1D(bkd, loc=5.0, scale=2.0)
        canonical = bkd.asarray([[0.0, 1.0, -1.0, 2.0]])
        expected = bkd.asarray([[5.0, 7.0, 3.0, 9.0]])
        result = transform.map_from_canonical(canonical)
        bkd.assert_allclose(result, expected)

    def test_jacobian_factor_standard(self, bkd) -> None:
        """jacobian_factor for N(0, 1) is 1.0."""
        transform = UnboundedAffineTransform1D(bkd, loc=0.0, scale=1.0)
        bkd.assert_allclose(
            bkd.asarray([transform.jacobian_factor()]),
            bkd.asarray([1.0]),
        )

    def test_jacobian_factor_scaled(self, bkd) -> None:
        """jacobian_factor for scale=2 is 0.5."""
        transform = UnboundedAffineTransform1D(bkd, loc=5.0, scale=2.0)
        bkd.assert_allclose(
            bkd.asarray([transform.jacobian_factor()]),
            bkd.asarray([0.5]),
        )

    def test_roundtrip(self, bkd) -> None:
        """Roundtrip through both maps returns original."""
        transform = UnboundedAffineTransform1D(bkd, loc=-3.0, scale=0.5)
        samples = bkd.asarray([[-5.0, -3.0, 0.0, 2.5]])
        canonical = transform.map_to_canonical(samples)
        roundtrip = transform.map_from_canonical(canonical)
        bkd.assert_allclose(roundtrip, samples)

    def test_loc_scale_accessors(self, bkd) -> None:
        """loc() and scale() return correct values."""
        transform = UnboundedAffineTransform1D(bkd, loc=-2.0, scale=3.0)
        bkd.assert_allclose(
            bkd.asarray([transform.loc()]),
            bkd.asarray([-2.0]),
        )
        bkd.assert_allclose(
            bkd.asarray([transform.scale()]),
            bkd.asarray([3.0]),
        )

    def test_repr(self, bkd) -> None:
        """repr returns expected string."""
        transform = UnboundedAffineTransform1D(bkd, loc=5.0, scale=2.0)
        assert repr(transform) == "UnboundedAffineTransform1D(loc=5.0, scale=2.0)"
