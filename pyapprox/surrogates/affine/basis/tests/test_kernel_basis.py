"""Tests for KernelBasis."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestKernelBasis(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_construction_and_shapes(self) -> None:
        bkd = self._bkd
        nvars = 1
        ncenters = 5
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            nvars,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(-2.0, 2.0, ncenters), (1, -1))
        basis = KernelBasis(kernel, centers)

        bkd.assert_allclose(
            bkd.asarray([basis.nbasis()]),
            bkd.asarray([ncenters]),
        )
        bkd.assert_allclose(
            bkd.asarray([basis.nvars()]),
            bkd.asarray([nvars]),
        )

    def test_evaluate_shape(self) -> None:
        bkd = self._bkd
        nvars = 1
        ncenters = 5
        npts = 10
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            nvars,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(-2.0, 2.0, ncenters), (1, -1))
        basis = KernelBasis(kernel, centers)

        points = bkd.reshape(bkd.linspace(-3.0, 3.0, npts), (1, -1))
        vals = basis(points)

        bkd.assert_allclose(
            bkd.asarray([vals.shape[0]]),
            bkd.asarray([npts]),
        )
        bkd.assert_allclose(
            bkd.asarray([vals.shape[1]]),
            bkd.asarray([ncenters]),
        )

    def test_evaluate_values_1d(self) -> None:
        """Verify basis values match direct kernel evaluation."""
        bkd = self._bkd
        lenscale = 0.5
        kernel = SquaredExponentialKernel(
            bkd.asarray([lenscale]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.asarray([[0.0, 1.0, 2.0]])
        basis = KernelBasis(kernel, centers)

        points = bkd.asarray([[-1.0, 0.5, 1.5, 3.0]])
        vals = basis(points)

        # Compute expected: K(x, mu_j) = exp(-0.5*(x-mu_j)^2/l^2)
        expected = bkd.zeros((4, 3))
        pts_np = bkd.to_numpy(points[0])
        ctrs_np = bkd.to_numpy(centers[0])
        for ii in range(4):
            for jj in range(3):
                diff = pts_np[ii] - ctrs_np[jj]
                expected_val = np.exp(-0.5 * diff**2 / lenscale**2)
                expected[ii, jj] = expected_val

        bkd.assert_allclose(vals, expected, rtol=1e-12)

    def test_evaluate_2d(self) -> None:
        """Test with 2D kernel."""
        bkd = self._bkd
        nvars = 2
        ncenters = 3
        npts = 4
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0, 1.0]),
            (0.1, 10.0),
            nvars,
            bkd,
        )
        centers = bkd.asarray(
            [
                [0.0, 1.0, -1.0],
                [0.0, 1.0, -1.0],
            ]
        )
        basis = KernelBasis(kernel, centers)

        points = bkd.asarray(
            [
                [-1.0, 0.0, 0.5, 2.0],
                [-1.0, 0.0, 0.5, 2.0],
            ]
        )
        vals = basis(points)

        bkd.assert_allclose(
            bkd.asarray([vals.shape[0]]),
            bkd.asarray([npts]),
        )
        bkd.assert_allclose(
            bkd.asarray([vals.shape[1]]),
            bkd.asarray([ncenters]),
        )

    def test_hyp_list_delegation(self) -> None:
        """Changing kernel length scale via hyp_list changes basis output."""
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.asarray([[0.0, 1.0]])
        basis = KernelBasis(kernel, centers)

        points = bkd.asarray([[0.5]])
        vals_before = bkd.copy(basis(points))

        # Change length scale via hyp_list
        basis.hyp_list().set_values(bkd.asarray([0.1]))
        vals_after = basis(points)

        # Values should differ since length scale changed
        diff = bkd.sum((vals_before - vals_after) ** 2)
        assert float(bkd.to_numpy(diff)) > 1e-6

    def test_invalid_kernel_type(self) -> None:
        with self.assertRaises(TypeError):
            KernelBasis("not_a_kernel", self._bkd.asarray([[0.0]]))

    def test_invalid_centers_ndim(self) -> None:
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            1,
            bkd,
        )
        with self.assertRaises(ValueError):
            KernelBasis(kernel, bkd.asarray([0.0, 1.0]))

    def test_invalid_centers_nvars(self) -> None:
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            1,
            bkd,
        )
        with self.assertRaises(ValueError):
            KernelBasis(kernel, bkd.asarray([[0.0, 1.0], [0.0, 1.0]]))


class TestKernelBasisNumpy(TestKernelBasis[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKernelBasisTorch(TestKernelBasis[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
