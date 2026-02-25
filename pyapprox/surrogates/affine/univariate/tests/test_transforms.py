"""Tests for univariate domain transforms.

Tests verify that transforms correctly map between user and canonical domains,
and that they satisfy the Univariate1DTransformProtocol.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests

from pyapprox.surrogates.affine.univariate.transforms import (
    Univariate1DTransformProtocol,
    IdentityTransform1D,
    BoundedAffineTransform1D,
    UnboundedAffineTransform1D,
)


class TestIdentityTransform1D(Generic[Array], unittest.TestCase):
    """Tests for IdentityTransform1D."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """IdentityTransform1D satisfies Univariate1DTransformProtocol."""
        transform = IdentityTransform1D(self._bkd)
        self.assertIsInstance(transform, Univariate1DTransformProtocol)

    def test_map_to_canonical_unchanged(self) -> None:
        """map_to_canonical returns input unchanged."""
        transform = IdentityTransform1D(self._bkd)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0, -1.0]])
        result = transform.map_to_canonical(samples)
        self._bkd.assert_allclose(result, samples)

    def test_map_from_canonical_unchanged(self) -> None:
        """map_from_canonical returns input unchanged."""
        transform = IdentityTransform1D(self._bkd)
        samples = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        result = transform.map_from_canonical(samples)
        self._bkd.assert_allclose(result, samples)

    def test_jacobian_factor_is_one(self) -> None:
        """jacobian_factor returns 1.0 for identity transform."""
        transform = IdentityTransform1D(self._bkd)
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.jacobian_factor()]),
            self._bkd.asarray([1.0]),
        )

    def test_roundtrip(self) -> None:
        """Roundtrip through both maps returns original."""
        transform = IdentityTransform1D(self._bkd)
        samples = self._bkd.asarray([[-2.0, 0.0, 3.5]])
        canonical = transform.map_to_canonical(samples)
        roundtrip = transform.map_from_canonical(canonical)
        self._bkd.assert_allclose(roundtrip, samples)

    def test_repr(self) -> None:
        """repr returns expected string."""
        transform = IdentityTransform1D(self._bkd)
        self.assertEqual(repr(transform), "IdentityTransform1D()")


class TestBoundedAffineTransform1D(Generic[Array], unittest.TestCase):
    """Tests for BoundedAffineTransform1D."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """BoundedAffineTransform1D satisfies Univariate1DTransformProtocol."""
        transform = BoundedAffineTransform1D(self._bkd, lb=0.0, ub=1.0)
        self.assertIsInstance(transform, Univariate1DTransformProtocol)

    def test_invalid_bounds_raises(self) -> None:
        """Raises ValueError if lb >= ub."""
        with self.assertRaises(ValueError):
            BoundedAffineTransform1D(self._bkd, lb=1.0, ub=0.0)
        with self.assertRaises(ValueError):
            BoundedAffineTransform1D(self._bkd, lb=1.0, ub=1.0)

    def test_unit_interval_to_canonical(self) -> None:
        """[0, 1] maps to [-1, 1] correctly."""
        transform = BoundedAffineTransform1D(self._bkd, lb=0.0, ub=1.0)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0]])
        expected = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        result = transform.map_to_canonical(samples)
        self._bkd.assert_allclose(result, expected)

    def test_unit_interval_from_canonical(self) -> None:
        """[-1, 1] maps to [0, 1] correctly."""
        transform = BoundedAffineTransform1D(self._bkd, lb=0.0, ub=1.0)
        canonical = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        expected = self._bkd.asarray([[0.0, 0.5, 1.0]])
        result = transform.map_from_canonical(canonical)
        self._bkd.assert_allclose(result, expected)

    def test_arbitrary_bounds_to_canonical(self) -> None:
        """[2, 6] maps to [-1, 1] correctly."""
        transform = BoundedAffineTransform1D(self._bkd, lb=2.0, ub=6.0)
        # lb=2, ub=6, midpoint=4, half_width=2
        samples = self._bkd.asarray([[2.0, 4.0, 6.0]])  # lb, mid, ub
        expected = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        result = transform.map_to_canonical(samples)
        self._bkd.assert_allclose(result, expected)

    def test_arbitrary_bounds_from_canonical(self) -> None:
        """[-1, 1] maps to [2, 6] correctly."""
        transform = BoundedAffineTransform1D(self._bkd, lb=2.0, ub=6.0)
        canonical = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        expected = self._bkd.asarray([[2.0, 4.0, 6.0]])
        result = transform.map_from_canonical(canonical)
        self._bkd.assert_allclose(result, expected)

    def test_jacobian_factor_unit_interval(self) -> None:
        """jacobian_factor for [0, 1] is 2.0."""
        transform = BoundedAffineTransform1D(self._bkd, lb=0.0, ub=1.0)
        # half_width = 0.5, jacobian = 1/0.5 = 2.0
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.jacobian_factor()]),
            self._bkd.asarray([2.0]),
        )

    def test_jacobian_factor_arbitrary_bounds(self) -> None:
        """jacobian_factor for [2, 6] is 0.5."""
        transform = BoundedAffineTransform1D(self._bkd, lb=2.0, ub=6.0)
        # half_width = 2.0, jacobian = 1/2.0 = 0.5
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.jacobian_factor()]),
            self._bkd.asarray([0.5]),
        )

    def test_roundtrip(self) -> None:
        """Roundtrip through both maps returns original."""
        transform = BoundedAffineTransform1D(self._bkd, lb=-3.0, ub=7.0)
        samples = self._bkd.asarray([[-3.0, 0.0, 2.0, 7.0]])
        canonical = transform.map_to_canonical(samples)
        roundtrip = transform.map_from_canonical(canonical)
        self._bkd.assert_allclose(roundtrip, samples)

    def test_lb_ub_accessors(self) -> None:
        """lb() and ub() return correct bounds."""
        transform = BoundedAffineTransform1D(self._bkd, lb=-2.0, ub=5.0)
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.lb()]),
            self._bkd.asarray([-2.0]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.ub()]),
            self._bkd.asarray([5.0]),
        )

    def test_repr(self) -> None:
        """repr returns expected string."""
        transform = BoundedAffineTransform1D(self._bkd, lb=0.0, ub=1.0)
        self.assertEqual(repr(transform), "BoundedAffineTransform1D(lb=0.0, ub=1.0)")


class TestUnboundedAffineTransform1D(Generic[Array], unittest.TestCase):
    """Tests for UnboundedAffineTransform1D."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance(self) -> None:
        """UnboundedAffineTransform1D satisfies Univariate1DTransformProtocol."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=0.0, scale=1.0)
        self.assertIsInstance(transform, Univariate1DTransformProtocol)

    def test_invalid_scale_raises(self) -> None:
        """Raises ValueError if scale <= 0."""
        with self.assertRaises(ValueError):
            UnboundedAffineTransform1D(self._bkd, loc=0.0, scale=0.0)
        with self.assertRaises(ValueError):
            UnboundedAffineTransform1D(self._bkd, loc=0.0, scale=-1.0)

    def test_standard_normal_unchanged(self) -> None:
        """N(0, 1) to N(0, 1) leaves samples unchanged."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=0.0, scale=1.0)
        samples = self._bkd.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        result = transform.map_to_canonical(samples)
        self._bkd.assert_allclose(result, samples)

    def test_shifted_normal_to_canonical(self) -> None:
        """N(5, 4) = N(mean=5, std=2) maps to N(0, 1) correctly."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=5.0, scale=2.0)
        # mean, mean+std, mean-std, mean+2std
        samples = self._bkd.asarray([[5.0, 7.0, 3.0, 9.0]])
        expected = self._bkd.asarray([[0.0, 1.0, -1.0, 2.0]])
        result = transform.map_to_canonical(samples)
        self._bkd.assert_allclose(result, expected)

    def test_shifted_normal_from_canonical(self) -> None:
        """N(0, 1) maps to N(5, 4) correctly."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=5.0, scale=2.0)
        canonical = self._bkd.asarray([[0.0, 1.0, -1.0, 2.0]])
        expected = self._bkd.asarray([[5.0, 7.0, 3.0, 9.0]])
        result = transform.map_from_canonical(canonical)
        self._bkd.assert_allclose(result, expected)

    def test_jacobian_factor_standard(self) -> None:
        """jacobian_factor for N(0, 1) is 1.0."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=0.0, scale=1.0)
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.jacobian_factor()]),
            self._bkd.asarray([1.0]),
        )

    def test_jacobian_factor_scaled(self) -> None:
        """jacobian_factor for scale=2 is 0.5."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=5.0, scale=2.0)
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.jacobian_factor()]),
            self._bkd.asarray([0.5]),
        )

    def test_roundtrip(self) -> None:
        """Roundtrip through both maps returns original."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=-3.0, scale=0.5)
        samples = self._bkd.asarray([[-5.0, -3.0, 0.0, 2.5]])
        canonical = transform.map_to_canonical(samples)
        roundtrip = transform.map_from_canonical(canonical)
        self._bkd.assert_allclose(roundtrip, samples)

    def test_loc_scale_accessors(self) -> None:
        """loc() and scale() return correct values."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=-2.0, scale=3.0)
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.loc()]),
            self._bkd.asarray([-2.0]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([transform.scale()]),
            self._bkd.asarray([3.0]),
        )

    def test_repr(self) -> None:
        """repr returns expected string."""
        transform = UnboundedAffineTransform1D(self._bkd, loc=5.0, scale=2.0)
        self.assertEqual(
            repr(transform), "UnboundedAffineTransform1D(loc=5.0, scale=2.0)"
        )


# NumPy backend test classes


class TestIdentityTransform1DNumpy(TestIdentityTransform1D[NDArray[Any]]):
    """NumPy backend tests for IdentityTransform1D."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBoundedAffineTransform1DNumpy(TestBoundedAffineTransform1D[NDArray[Any]]):
    """NumPy backend tests for BoundedAffineTransform1D."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestUnboundedAffineTransform1DNumpy(TestUnboundedAffineTransform1D[NDArray[Any]]):
    """NumPy backend tests for UnboundedAffineTransform1D."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend test classes


class TestIdentityTransform1DTorch(TestIdentityTransform1D[torch.Tensor]):
    """PyTorch backend tests for IdentityTransform1D."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestBoundedAffineTransform1DTorch(TestBoundedAffineTransform1D[torch.Tensor]):
    """PyTorch backend tests for BoundedAffineTransform1D."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestUnboundedAffineTransform1DTorch(TestUnboundedAffineTransform1D[torch.Tensor]):
    """PyTorch backend tests for UnboundedAffineTransform1D."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
