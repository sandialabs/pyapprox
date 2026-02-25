"""Tests for KernelDensityBasis."""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.probability.density.protocols import (
    DensityBasisProtocol,
)
from pyapprox.surrogates.affine.expansions.pce_density import (
    composite_gauss_legendre,
)


class TestKernelDensityBasis(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_basis(
        self, ncenters: int = 5, lenscale: float = 0.5, y_min: float = -2.0,
        y_max: float = 2.0,
    ) -> KernelDensityBasis:
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([lenscale]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.reshape(
            bkd.linspace(y_min, y_max, ncenters), (1, -1)
        )
        kb = KernelBasis(kernel, centers)
        return KernelDensityBasis(kb)

    def test_evaluate_shape(self) -> None:
        bkd = self._bkd
        basis = self._make_basis(ncenters=5)
        y = bkd.reshape(bkd.linspace(-1.0, 1.0, 20), (1, -1))
        vals = basis.evaluate(y)
        bkd.assert_allclose(
            bkd.asarray([vals.shape[0]]),
            bkd.asarray([5]),
        )
        bkd.assert_allclose(
            bkd.asarray([vals.shape[1]]),
            bkd.asarray([20]),
        )

    def test_analytical_mass_matrix_formula(self) -> None:
        """Verify mass matrix entries match the closed-form formula."""
        bkd = self._bkd
        lenscale = 0.5
        basis = self._make_basis(ncenters=3, lenscale=lenscale,
                                  y_min=0.0, y_max=2.0)
        M = basis.mass_matrix()
        centers = basis.kernel_basis().centers()
        mu = bkd.to_numpy(centers[0])
        l = lenscale

        for ii in range(3):
            for jj in range(3):
                expected = l * math.sqrt(math.pi) * math.exp(
                    -(mu[ii] - mu[jj])**2 / (4.0 * l**2)
                )
                bkd.assert_allclose(
                    bkd.asarray([float(bkd.to_numpy(M[ii, jj]))]),
                    bkd.asarray([expected]),
                    rtol=1e-12,
                )

    def test_mass_matrix_vs_numerical(self) -> None:
        """Verify analytical mass matrix against numerical integration."""
        bkd = self._bkd
        basis = self._make_basis(ncenters=4, lenscale=0.3,
                                  y_min=-1.0, y_max=1.0)
        M = basis.mass_matrix()
        n = basis.nbasis()
        y_min, y_max = basis.domain()

        for ii in range(n):
            for jj in range(ii, n):
                _ii, _jj = ii, jj

                def integrand(y_np: np.ndarray, i=_ii, j=_jj) -> np.ndarray:
                    y_arr = bkd.reshape(bkd.asarray(y_np), (1, -1))
                    Phi = basis.evaluate(y_arr)
                    return bkd.to_numpy(Phi[i]) * bkd.to_numpy(Phi[j])

                numerical = composite_gauss_legendre(
                    integrand, y_min, y_max, n_intervals=500, n_points=5
                )
                bkd.assert_allclose(
                    bkd.asarray([float(bkd.to_numpy(M[ii, jj]))]),
                    bkd.asarray([numerical]),
                    atol=1e-10,
                    rtol=1e-4,
                )

    def test_mass_matrix_symmetry(self) -> None:
        bkd = self._bkd
        basis = self._make_basis(ncenters=7)
        M = basis.mass_matrix()
        bkd.assert_allclose(M, bkd.transpose(M, (1, 0)), rtol=1e-14)

    def test_protocol_conformance(self) -> None:
        bkd = self._bkd
        basis = self._make_basis()
        assert isinstance(basis, DensityBasisProtocol)

    def test_hyp_list_delegation(self) -> None:
        """Changing length scale via hyp_list updates mass matrix."""
        bkd = self._bkd
        basis = self._make_basis(ncenters=3, lenscale=1.0)
        M_before = bkd.copy(basis.mass_matrix())

        basis.hyp_list().set_values(bkd.log(bkd.asarray([0.3])))
        M_after = basis.mass_matrix()

        diff = bkd.sum((M_before - M_after) ** 2)
        assert float(bkd.to_numpy(diff)) > 1e-6

    def test_invalid_non_se_kernel(self) -> None:
        """Should reject non-SE kernels."""
        bkd = self._bkd
        from pyapprox.surrogates.kernels.matern import Matern32Kernel
        kernel = Matern32Kernel(
            bkd.asarray([1.0]),
            (0.1, 10.0),
            1,
            bkd,
        )
        centers = bkd.asarray([[0.0, 1.0]])
        kb = KernelBasis(kernel, centers)
        with self.assertRaises(TypeError):
            KernelDensityBasis(kb)

    def test_invalid_2d_kernel(self) -> None:
        """Should reject multi-dimensional kernels."""
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([1.0, 1.0]),
            (0.1, 10.0),
            2,
            bkd,
        )
        centers = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        kb = KernelBasis(kernel, centers)
        with self.assertRaises(ValueError):
            KernelDensityBasis(kb)


class TestKernelDensityBasisNumpy(TestKernelDensityBasis[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKernelDensityBasisTorch(TestKernelDensityBasis[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
