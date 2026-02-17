"""Tests for TensorProductQuadratureFactory."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.surrogates.quadrature.tensor_product_factory import (
    TensorProductQuadratureFactory,
)


class TestTensorProductQuadratureFactory(
    Generic[Array], unittest.TestCase
):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_affine_mapping_points_in_range(self) -> None:
        """Verify mapped points lie within [lb, ub]."""
        bkd = self._bkd
        domain = bkd.asarray([[2.0, 5.0], [10.0, 20.0]])
        factory = TensorProductQuadratureFactory(
            [5, 5], domain, bkd
        )
        quad = factory([0, 1])
        samples, weights = quad()
        self.assertEqual(samples.shape[0], 2)
        s_np = bkd.to_numpy(samples)
        self.assertTrue(np.all(s_np[0] >= 2.0 - 1e-14))
        self.assertTrue(np.all(s_np[0] <= 5.0 + 1e-14))
        self.assertTrue(np.all(s_np[1] >= 10.0 - 1e-14))
        self.assertTrue(np.all(s_np[1] <= 20.0 + 1e-14))

    def test_integrate_constant_gives_volume(self) -> None:
        """Integral of 1 over [a,b] should give b-a."""
        bkd = self._bkd
        domain = bkd.asarray([[2.0, 5.0]])
        factory = TensorProductQuadratureFactory(
            [5], domain, bkd
        )
        quad = factory([0])
        samples, weights = quad()
        integral = bkd.sum(weights)
        bkd.assert_allclose(
            bkd.asarray([integral]), bkd.asarray([3.0]), rtol=1e-12
        )

    def test_integrate_constant_2d(self) -> None:
        """Integral of 1 over [2,5] x [10,20] = 3 * 10 = 30."""
        bkd = self._bkd
        domain = bkd.asarray([[2.0, 5.0], [10.0, 20.0]])
        factory = TensorProductQuadratureFactory(
            [3, 3], domain, bkd
        )
        quad = factory([0, 1])
        _, weights = quad()
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([30.0]),
            rtol=1e-12,
        )

    def test_subset_selection(self) -> None:
        """3-variable factory, call with [1] -> 1D rule for var 1."""
        bkd = self._bkd
        domain = bkd.asarray([[0.0, 1.0], [2.0, 4.0], [5.0, 10.0]])
        factory = TensorProductQuadratureFactory(
            [3, 5, 7], domain, bkd
        )
        quad = factory([1])
        samples, weights = quad()
        self.assertEqual(samples.shape[0], 1)
        self.assertEqual(samples.shape[1], 5)
        # Points should be in [2, 4]
        s_np = bkd.to_numpy(samples)
        self.assertTrue(np.all(s_np >= 2.0 - 1e-14))
        self.assertTrue(np.all(s_np <= 4.0 + 1e-14))
        # Integral of 1 = 4 - 2 = 2
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.asarray([2.0]),
            rtol=1e-12,
        )

    def test_polynomial_integration(self) -> None:
        """Integrate x^2 on [0, 3] exactly."""
        bkd = self._bkd
        domain = bkd.asarray([[0.0, 3.0]])
        factory = TensorProductQuadratureFactory(
            [5], domain, bkd
        )
        quad = factory([0])
        samples, weights = quad()
        # int_0^3 x^2 dx = 9
        values = samples[0] ** 2
        integral = bkd.sum(weights * values)
        bkd.assert_allclose(
            bkd.asarray([integral]), bkd.asarray([9.0]), rtol=1e-12
        )

    def test_integrate_method(self) -> None:
        """Test the integrate() method of _AffinelyMappedQuadratureRule."""
        bkd = self._bkd
        domain = bkd.asarray([[0.0, 2.0], [0.0, 3.0]])
        factory = TensorProductQuadratureFactory(
            [5, 5], domain, bkd
        )
        quad = factory([0, 1])

        def func(samples):
            # f(x,y) = x*y, returns (nsamples, 1)
            return bkd.reshape(samples[0] * samples[1], (-1, 1))

        result = quad.integrate(func)
        # int_0^2 int_0^3 x*y dy dx = (2^2/2) * (3^2/2) = 2 * 4.5 = 9
        bkd.assert_allclose(result, bkd.asarray([9.0]), rtol=1e-12)


class TestTensorProductQuadratureFactoryNumpy(
    TestTensorProductQuadratureFactory[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTensorProductQuadratureFactoryTorch(
    TestTensorProductQuadratureFactory[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
