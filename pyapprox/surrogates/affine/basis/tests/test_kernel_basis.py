"""Tests for KernelBasis."""

import numpy as np
import pytest

from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)


class TestKernelBasis:

    def test_construction_and_shapes(self, bkd) -> None:
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

    def test_evaluate_shape(self, bkd) -> None:
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

    def test_evaluate_values_1d(self, bkd) -> None:
        """Verify basis values match direct kernel evaluation."""
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

    def test_evaluate_2d(self, bkd) -> None:
        """Test with 2D kernel."""
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

    def test_hyp_list_delegation(self, bkd) -> None:
        """Changing kernel length scale via hyp_list changes basis output."""
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

    def test_invalid_kernel_type(self, bkd) -> None:
        with pytest.raises(TypeError):
            KernelBasis("not_a_kernel", bkd.asarray([[0.0]]))

    def test_invalid_centers_ndim(self, bkd) -> None:
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            1,
            bkd,
        )
        with pytest.raises(ValueError):
            KernelBasis(kernel, bkd.asarray([0.0, 1.0]))

    def test_invalid_centers_nvars(self, bkd) -> None:
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            1,
            bkd,
        )
        with pytest.raises(ValueError):
            KernelBasis(kernel, bkd.asarray([[0.0, 1.0], [0.0, 1.0]]))
