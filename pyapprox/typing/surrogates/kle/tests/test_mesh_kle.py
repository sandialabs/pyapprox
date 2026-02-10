"""Tests for MeshKLE.

Ports all relevant legacy tests from pyapprox/surrogates/affine/tests/test_kle.py
and adds new tests for improved coverage.
"""
import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
    Matern52Kernel,
    Matern32Kernel,
    ExponentialKernel,
)
from pyapprox.typing.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.typing.surrogates.kle.analytical import (
    AnalyticalExponentialKLE1D,
)
from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
)
from pyapprox.typing.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)


def _gauss_legendre_quad(lb, ub, npts, bkd):
    """Gauss-Legendre quadrature on [lb, ub] for Lebesgue integration.

    Returns
    -------
    pts : Array, shape (1, npts)
        Quadrature points.
    weights : Array, shape (npts,)
        Quadrature weights (1D).
    """
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(npts)
    quad_rule = GaussQuadratureRule(poly)
    pts, wts = quad_rule(npts)
    # pts shape (1, npts), wts shape (npts, 1) - on [-1, 1]
    # The Legendre polynomial weights integrate against probability density
    # 1/2 on [-1,1] (sum to 1). For Lebesgue integration on [lb, ub],
    # multiply by (ub - lb) to get weights summing to (ub - lb).
    dom_len = ub - lb
    half_len = dom_len / 2.0
    mid = (lb + ub) / 2.0
    pts = pts * half_len + mid
    wts = wts * dom_len
    return pts, wts[:, 0]  # return weights as 1D


def _trapezoid_rule(lb, ub, npts):
    """Trapezoid rule on [lb, ub].

    Returns
    -------
    pts : ndarray, shape (npts,)
    weights : ndarray, shape (npts,)
    """
    pts = np.linspace(lb, ub, npts)
    deltax = pts[1] - pts[0]
    weights = np.ones(npts) * deltax
    weights[0] /= 2
    weights[-1] /= 2
    return pts, weights


class TestMeshKLE(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    def test_mesh_kle_1D_exponential(self) -> None:
        """Port of legacy test_mesh_kle_1D.

        Creates MeshKLE with ExponentialKernel and compares eigenvalues
        against analytical KLE1D. Also checks basis orthonormality
        with quadrature weights.
        """
        bkd = self._bkd
        level = 10
        nterms = 3
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0
        npts = 2 ** level + 1

        # Get Gauss-Legendre quadrature
        mesh_coords, quad_weights = _gauss_legendre_quad(lb, ub, npts, bkd)

        # Create ExponentialKernel (nu=0.5)
        lenscale_arr = bkd.array([len_scale])
        kernel = ExponentialKernel(
            lenscale_arr, (0.01, 100.0), 1, bkd
        )

        kle = MeshKLE(
            mesh_coords,
            kernel,
            sigma=sigma,
            nterms=nterms,
            quad_weights=quad_weights,
            bkd=bkd,
        )

        # Analytical reference
        kle_exact = AnalyticalExponentialKLE1D(
            corr_len=len_scale, sigma2=sigma, dom_len=ub - lb, nterms=nterms
        )
        mesh_1d = bkd.to_numpy(mesh_coords[0, :])
        exact_basis = kle_exact.basis_values(mesh_1d)
        exact_eig_vals = kle_exact.eigenvalues()

        # Check eigenvalues match
        bkd.assert_allclose(
            kle.eigenvalues(),
            bkd.array(exact_eig_vals),
            rtol=3e-5,
        )

        # Check analytical basis is orthonormal under quadrature weights
        exact_basis_arr = bkd.array(exact_basis)
        identity = bkd.array(np.eye(nterms))
        bkd.assert_allclose(
            exact_basis_arr.T @ (quad_weights[:, None] * exact_basis_arr),
            identity,
            atol=1e-6,
        )

        # Check KLE eigenvectors are orthonormal under quadrature weights
        eig_vecs = kle.eigenvectors()
        bkd.assert_allclose(
            eig_vecs.T @ (quad_weights[:, None] * eig_vecs),
            identity,
            atol=1e-6,
        )

    def test_mesh_kle_1D_discretization_independence(self) -> None:
        """Port of legacy test_mesh_kle_1D_discretization.

        Tests that two different mesh resolutions with trapezoid rule
        give the same eigenvalues.
        """
        bkd = self._bkd
        level1, level2 = 6, 8
        nterms = 3
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0

        # Fine mesh
        npts2 = 2 ** (level2 + 1) + 1
        pts2, wts2 = _trapezoid_rule(lb, ub, npts2)
        mesh_coords2 = bkd.array(pts2[None, :])
        quad_weights2 = bkd.array(wts2)

        lenscale_arr = bkd.array([len_scale])
        kernel2 = ExponentialKernel(
            lenscale_arr, (0.01, 100.0), 1, bkd
        )
        kle2 = MeshKLE(
            mesh_coords2, kernel2, sigma=sigma, nterms=nterms,
            quad_weights=quad_weights2, bkd=bkd,
        )

        # Coarse mesh
        npts1 = 2 ** level1 + 1
        pts1, wts1 = _trapezoid_rule(lb, ub, npts1)
        mesh_coords1 = bkd.array(pts1[None, :])
        quad_weights1 = bkd.array(wts1)

        kernel1 = ExponentialKernel(
            lenscale_arr, (0.01, 100.0), 1, bkd
        )
        kle1 = MeshKLE(
            mesh_coords1, kernel1, sigma=sigma, nterms=nterms,
            quad_weights=quad_weights1, bkd=bkd,
        )

        # Eigenvectors should be orthonormal under respective weights
        eig_vecs2 = kle2.eigenvectors()
        eig_vecs1 = kle1.eigenvectors()
        identity = bkd.array(np.eye(nterms))

        bkd.assert_allclose(
            bkd.sum(quad_weights2[:, None] * eig_vecs2 ** 2, axis=0),
            bkd.ones(nterms),
            atol=1e-6,
        )
        bkd.assert_allclose(
            bkd.sum(quad_weights1[:, None] * eig_vecs1 ** 2, axis=0),
            bkd.ones(nterms),
            atol=1e-6,
        )

        # Eigenvalues should match across resolutions
        bkd.assert_allclose(
            kle2._sqrt_eig_vals,
            kle1._sqrt_eig_vals,
            atol=3e-4,
        )

    def test_mesh_kle_multiple_kernels(self) -> None:
        """Test MeshKLE works with all supported kernel types."""
        bkd = self._bkd
        nterms = 3
        npts = 50
        mesh_coords = bkd.array(
            np.linspace(0, 1, npts)[None, :]
        )

        kernel_classes = [
            SquaredExponentialKernel,
            Matern52Kernel,
            Matern32Kernel,
            ExponentialKernel,
        ]

        for KernelClass in kernel_classes:
            with self.subTest(kernel=KernelClass.__name__):
                lenscale = bkd.array([0.5])
                kernel = KernelClass(lenscale, (0.01, 10.0), 1, bkd)
                kle = MeshKLE(
                    mesh_coords, kernel, nterms=nterms, bkd=bkd,
                )
                # Eigenvalues should be positive
                self.assertTrue(
                    bkd.all_bool(kle.eigenvalues() > 0)
                )
                # Check shapes
                self.assertEqual(
                    kle.eigenvectors().shape, (npts, nterms)
                )
                self.assertEqual(
                    kle.weighted_eigenvectors().shape, (npts, nterms)
                )
                self.assertEqual(kle.eigenvalues().shape, (nterms,))

                # Evaluate
                coef = bkd.array(np.random.randn(nterms, 5))
                result = kle(coef)
                self.assertEqual(result.shape, (npts, 5))

    def test_mesh_kle_2D(self) -> None:
        """Test MeshKLE with a 2D mesh."""
        bkd = self._bkd
        nterms = 3
        nx, ny = 5, 5
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        mesh_coords = bkd.array(
            np.vstack([xx.ravel(), yy.ravel()])
        )  # shape (2, 25)

        lenscale = bkd.array([0.5, 0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 2, bkd)
        kle = MeshKLE(mesh_coords, kernel, nterms=nterms, bkd=bkd)

        self.assertTrue(bkd.all_bool(kle.eigenvalues() > 0))
        self.assertEqual(kle.eigenvectors().shape, (25, nterms))

    def test_mesh_kle_use_log(self) -> None:
        """Test that use_log=True gives exp() of use_log=False result."""
        bkd = self._bkd
        nterms = 3
        npts = 30
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        lenscale = bkd.array([0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 1, bkd)

        kle_no_log = MeshKLE(
            mesh_coords, kernel, nterms=nterms,
            use_log=False, bkd=bkd,
        )
        kle_log = MeshKLE(
            mesh_coords, kernel, nterms=nterms,
            use_log=True, bkd=bkd,
        )

        coef = bkd.array(np.random.randn(nterms, 4))
        result_no_log = kle_no_log(coef)
        result_log = kle_log(coef)

        bkd.assert_allclose(result_log, bkd.exp(result_no_log))

    def test_mesh_kle_edge_cases(self) -> None:
        """Test edge cases: nterms=1, nterms=all, non-zero mean."""
        bkd = self._bkd
        npts = 20
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        lenscale = bkd.array([0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 1, bkd)

        # nterms=1
        kle1 = MeshKLE(mesh_coords, kernel, nterms=1, bkd=bkd)
        coef = bkd.array(np.random.randn(1, 3))
        result = kle1(coef)
        self.assertEqual(result.shape, (npts, 3))

        # nterms=all (None)
        kle_all = MeshKLE(mesh_coords, kernel, bkd=bkd)
        self.assertEqual(kle_all.nterms(), npts)

        # Non-zero mean
        mean = 5.0
        kle_mean = MeshKLE(
            mesh_coords, kernel, mean_field=mean, nterms=3, bkd=bkd,
        )
        zero_coef = bkd.zeros((3, 1))
        result = kle_mean(zero_coef)
        bkd.assert_allclose(result, bkd.full((npts, 1), mean))

    def test_mesh_kle_coef_validation(self) -> None:
        """Test that invalid coefficients raise errors."""
        bkd = self._bkd
        npts = 20
        mesh_coords = bkd.array(np.linspace(0, 1, npts)[None, :])
        lenscale = bkd.array([0.5])
        kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), 1, bkd)
        kle = MeshKLE(mesh_coords, kernel, nterms=3, bkd=bkd)

        # Wrong ndim
        with self.assertRaises(ValueError):
            kle(bkd.array(np.random.randn(3)))

        # Wrong nterms
        with self.assertRaises(ValueError):
            kle(bkd.array(np.random.randn(5, 2)))


class TestMeshKLENumpy(TestMeshKLE[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMeshKLETorch(TestMeshKLE[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
