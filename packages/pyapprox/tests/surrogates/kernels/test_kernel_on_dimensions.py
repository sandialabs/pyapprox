import numpy as np
import pytest
from pyapprox.surrogates.kernels.composition import ProductKernel
from pyapprox.surrogates.kernels.kernel_on_dimensions import (
    KernelOnDimensions,
)
from pyapprox.surrogates.kernels.matern import (
    Matern52Kernel,
)


class TestKernelOnDimensions:
    def _setup_data(self, bkd):
        np.random.seed(42)
        self.total_nvars = 4
        self.dims = [1, 3]
        self.inner_nvars = len(self.dims)
        self.n1 = 5
        self.n2 = 4

        self.inner_kernel = Matern52Kernel(
            [1.0] * self.inner_nvars,
            (0.1, 10.0),
            self.inner_nvars,
            bkd,
        )
        self.wrapped = KernelOnDimensions(
            self.inner_kernel, self.dims, self.total_nvars
        )
        self.X1 = bkd.array(np.random.randn(self.total_nvars, self.n1))
        self.X2 = bkd.array(np.random.randn(self.total_nvars, self.n2))

    def test_nvars_reports_total(self, bkd) -> None:
        self._setup_data(bkd)
        assert self.wrapped.nvars() == self.total_nvars

    def test_call_matches_inner_on_subset(self, bkd) -> None:
        self._setup_data(bkd)
        K_wrapped = self.wrapped(self.X1, self.X2)
        K_inner = self.inner_kernel(
            self.X1[self.dims, :], self.X2[self.dims, :]
        )
        bkd.assert_allclose(K_wrapped, K_inner)

    def test_call_self(self, bkd) -> None:
        self._setup_data(bkd)
        K_wrapped = self.wrapped(self.X1)
        K_inner = self.inner_kernel(self.X1[self.dims, :])
        bkd.assert_allclose(K_wrapped, K_inner)

    def test_diag_matches_inner(self, bkd) -> None:
        self._setup_data(bkd)
        diag_wrapped = self.wrapped.diag(self.X1)
        diag_inner = self.inner_kernel.diag(self.X1[self.dims, :])
        bkd.assert_allclose(diag_wrapped, diag_inner)

    def test_hyp_list_delegates(self, bkd) -> None:
        self._setup_data(bkd)
        bkd.assert_allclose(
            self.wrapped.hyp_list().get_values(),
            self.inner_kernel.hyp_list().get_values(),
        )

    def test_jacobian_zero_pads_unused_dims(self, bkd) -> None:
        self._setup_data(bkd)
        jac = self.wrapped.jacobian(self.X1, self.X2)
        assert jac.shape == (self.n1, self.n2, self.total_nvars)

        inner_jac = self.inner_kernel.jacobian(
            self.X1[self.dims, :], self.X2[self.dims, :]
        )
        for ii, d in enumerate(self.dims):
            bkd.assert_allclose(jac[:, :, d], inner_jac[:, :, ii])

        non_dims = [d for d in range(self.total_nvars) if d not in self.dims]
        for d in non_dims:
            bkd.assert_allclose(
                jac[:, :, d], bkd.zeros((self.n1, self.n2))
            )

    def test_jacobian_wrt_params_delegates(self, bkd) -> None:
        self._setup_data(bkd)
        jac_wrapped = self.wrapped.jacobian_wrt_params(self.X1)
        jac_inner = self.inner_kernel.jacobian_wrt_params(
            self.X1[self.dims, :]
        )
        bkd.assert_allclose(jac_wrapped, jac_inner)

    def test_product_of_disjoint_dims(self, bkd) -> None:
        np.random.seed(42)
        total = 4
        k1 = Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        k2 = Matern52Kernel([0.5, 0.5], (0.1, 10.0), 2, bkd)
        w1 = KernelOnDimensions(k1, [0, 1], total)
        w2 = KernelOnDimensions(k2, [2, 3], total)
        product = ProductKernel(w1, w2)

        X = bkd.array(np.random.randn(total, 6))
        K_product = product(X)
        K_expected = k1(X[[0, 1], :]) * k2(X[[2, 3], :])
        bkd.assert_allclose(K_product, K_expected)

    def test_invalid_dims_raises(self, bkd) -> None:
        k = Matern52Kernel([1.0], (0.1, 10.0), 1, bkd)
        with pytest.raises(ValueError, match="must be in"):
            KernelOnDimensions(k, [5], 4)

    def test_nvars_mismatch_raises(self, bkd) -> None:
        k = Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        with pytest.raises(ValueError, match="must equal"):
            KernelOnDimensions(k, [0], 4)

    def test_too_many_dims_raises(self, bkd) -> None:
        k = Matern52Kernel([1.0, 1.0, 1.0], (0.1, 10.0), 3, bkd)
        with pytest.raises(ValueError, match="must be <="):
            KernelOnDimensions(k, [0, 1, 2], 2)

    def test_dims_accessor(self, bkd) -> None:
        self._setup_data(bkd)
        assert self.wrapped.dims() == self.dims

    def test_inner_kernel_accessor(self, bkd) -> None:
        self._setup_data(bkd)
        assert self.wrapped.inner_kernel() is self.inner_kernel

    def test_mul_and_add_operators(self, bkd) -> None:
        self._setup_data(bkd)
        k2 = Matern52Kernel(
            [1.0] * self.total_nvars,
            (0.1, 10.0),
            self.total_nvars,
            bkd,
        )
        product = self.wrapped * k2
        assert product.nvars() == self.total_nvars
        summed = self.wrapped + k2
        assert summed.nvars() == self.total_nvars
