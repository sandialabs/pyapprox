"""
Tests for GeneralMaternKernel with arbitrary nu.
"""

from typing import Sequence

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.kernels import Matern32Kernel, Matern52Kernel
from pyapprox.surrogates.kernels.general_matern import GeneralMaternKernel
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend


def _make_kernel(
    bkd: Backend[Array], nu: float = 2.5, lenscale: Sequence[float] = (1.0,)
) -> GeneralMaternKernel[Array]:
    return GeneralMaternKernel(
        nu=nu,
        lenscale=list(lenscale),
        lenscale_bounds=(0.1, 10.0),
        nvars=len(lenscale),
        bkd=bkd,
    )


def _grid(bkd: Backend[Array], npoints: int) -> Array:
    """npoints equispaced 1D samples in [-2, 2], shape (1, npoints)."""
    return bkd.reshape(bkd.linspace(-2.0, 2.0, npoints), (1, -1))


class TestGeneralMaternKernel:
    """Tests for GeneralMaternKernel."""

    def test_kernel_shape(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd, lenscale=[1.0, 1.0])
        X1 = bkd.asarray(np.random.randn(2, 10))
        X2 = bkd.asarray(np.random.randn(2, 5))
        assert kernel(X1, X2).shape == (10, 5)

    def test_kernel_symmetry(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd)
        X = bkd.asarray(np.random.randn(1, 10))
        K = kernel(X, X)
        bkd.assert_allclose(K, K.T, rtol=1e-10)

    def test_kernel_positive_definite(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd)
        K = kernel(_grid(bkd, 20), _grid(bkd, 20))
        # Add small nugget for numerical stability
        K_stable = K + 1e-10 * bkd.eye(K.shape[0])
        eigenvalues = bkd.eigh(K_stable)[0]
        assert bkd.to_bool(bkd.all(eigenvalues > 0))

    def test_diagonal_is_one(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd, lenscale=[1.0, 1.0])
        X = bkd.asarray(np.random.randn(2, 10))
        diag = bkd.diag(kernel(X, X))
        bkd.assert_allclose(diag, bkd.ones_like(diag), rtol=1e-6)

    def test_nu_25_matches_matern52(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd, nu=2.5)
        ref_kernel = Matern52Kernel(
            lenscale=bkd.array([1.0]), lenscale_bounds=(0.1, 10.0), nvars=1, bkd=bkd
        )
        X = _grid(bkd, 20)
        bkd.assert_allclose(kernel(X, X), ref_kernel(X, X), rtol=1e-5, atol=1e-6)

    def test_nu_15_matches_matern32(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd, nu=1.5)
        ref_kernel = Matern32Kernel(
            lenscale=bkd.array([1.0]), lenscale_bounds=(0.1, 10.0), nvars=1, bkd=bkd
        )
        X = _grid(bkd, 20)
        bkd.assert_allclose(kernel(X, X), ref_kernel(X, X), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("nu", [0.5, 1.0, 2.0, 2.3, 3.0, 4.5])
    def test_arbitrary_nu(self, bkd: Backend[Array], nu: float) -> None:
        kernel = _make_kernel(bkd, nu=nu)
        X = _grid(bkd, 10)
        K = kernel(X, X)
        assert K.shape == (10, 10)
        bkd.assert_allclose(K, K.T, rtol=1e-10)
        # For general nu, asymptotic approximation may not give exactly 1
        # on diagonal due to numerical stability epsilon
        bkd.assert_allclose(bkd.diag(K), bkd.ones((10,)), rtol=1e-3)

    def test_large_nu_approximates_rbf(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd, nu=150.0)
        X = _grid(bkd, 20)
        K_rbf = bkd.exp(-0.5 * bkd.cdist(X.T, X.T) ** 2)
        bkd.assert_allclose(kernel(X, X), K_rbf, rtol=1e-3, atol=1e-4)

    def test_length_scale_effect(self, bkd: Backend[Array]) -> None:
        """Off-diagonal correlation grows with the length scale."""
        X = _grid(bkd, 20)
        K_small = _make_kernel(bkd, lenscale=[0.5])(X, X)
        K_large = _make_kernel(bkd, lenscale=[2.0])(X, X)
        assert bkd.to_float(K_large[0, 5]) > bkd.to_float(K_small[0, 5])

    def test_input_jacobian_autodiff(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Autodiff through the kernel evaluation matches finite
        differences of k(x, X2) w.r.t. x."""
        bkd = autodiff_bkd
        kernel = _make_kernel(bkd)
        X2 = _grid(bkd, 10)

        def k_flat(x_flat: Array) -> Array:
            return bkd.flatten(kernel(bkd.reshape(x_flat, (1, 1)), X2))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=X2.shape[1],
            nvars=1,
            fun=lambda samples: kernel(samples, X2).T,
            jacobian=lambda sample: bkd.jacobian(k_flat, bkd.flatten(sample)),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        # x0 lies on no node of X2, where cdist is not differentiable
        x0 = bkd.array([[0.3]])
        jac_error = checker.check_derivatives(
            x0, fd_eps=bkd.flip(bkd.logspace(-14, 0, 15)), relative=True, verbosity=0
        )[0]
        ratio = bkd.to_float(checker.error_ratio(jac_error))
        assert ratio <= 1e-6, f"Kernel jacobian error ratio {ratio:.2e} > 1e-6"

    def test_hyp_list(self, bkd: Backend[Array]) -> None:
        kernel = _make_kernel(bkd, lenscale=[1.0, 2.0])
        assert kernel.hyp_list().nparams() == 2

    def test_repr(self, bkd: Backend[Array]) -> None:
        repr_str = repr(_make_kernel(bkd))
        assert "GeneralMaternKernel" in repr_str
        assert "nu=2.5" in repr_str
