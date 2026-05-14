import numpy as np
import pytest

from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.encoders.pca import (
    PCAFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.regressors.scalar_kernel import (
    ScalarKernelLatentRegressor,
)
from pyapprox.surrogates.kerneloperator.surrogate import (
    KernelOperatorSurrogate,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestKernelOperatorSurrogate:
    def _make_fitted_surrogate(self, bkd, ngrid=8, ncodes=8, N=10):
        """Build a surrogate with identity encoder and pre-fitted regressor."""
        np.random.seed(42)
        enc_in = IdentityFunctionEncoder(ngrid, bkd)
        enc_out = IdentityFunctionEncoder(ngrid, bkd)
        kernel = Matern52Kernel(
            [1.0] * ncodes, (0.1, 10.0), ncodes, bkd
        )
        reg = ScalarKernelLatentRegressor(
            kernel, ncodes, ncodes, bkd, nugget=1e-8
        )
        U = bkd.array(np.random.randn(ncodes, N))
        V = bkd.array(np.random.randn(ncodes, N))
        reg.fit_internal(U, V)

        surr = KernelOperatorSurrogate([enc_in], [enc_out], reg)
        return surr, U, V

    def test_predict_shapes_single_io(self, bkd) -> None:
        surr, U, V = self._make_fitted_surrogate(bkd)
        result = surr.predict([U])
        assert len(result) == 1
        assert result[0].shape == V.shape

    def test_predict_std_shapes(self, bkd) -> None:
        surr, U, V = self._make_fitted_surrogate(bkd)
        result = surr.predict_std([U])
        assert len(result) == 1
        assert result[0].shape == V.shape

    def test_predict_std_nonnegative_with_pca(self, bkd) -> None:
        """PCA decode_std via variance propagation must be non-negative."""
        np.random.seed(42)
        ngrid, N, ncodes = 20, 15, 5
        data = bkd.array(np.random.randn(ngrid, N))
        enc_in = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=ncodes)
        enc_out = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=ncodes)
        kernel = Matern52Kernel(
            [1.0] * ncodes, (0.1, 10.0), ncodes, bkd
        )
        reg = ScalarKernelLatentRegressor(
            kernel, ncodes, ncodes, bkd, nugget=1e-8
        )
        U_codes = enc_in.encode(data)
        V_codes = enc_out.encode(data)
        reg.fit_internal(U_codes, V_codes)
        surr = KernelOperatorSurrogate([enc_in], [enc_out], reg)

        std_result = surr.predict_std([data])
        assert bkd.all_bool(std_result[0] >= 0)

    def test_dimension_mismatch_raises(self, bkd) -> None:
        enc = IdentityFunctionEncoder(5, bkd)
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        reg = ScalarKernelLatentRegressor(kernel, 2, 3, bkd)
        with pytest.raises(ValueError, match="ncodes_in"):
            KernelOperatorSurrogate([enc], [enc], reg)

    def test_clone_unfitted(self, bkd) -> None:
        surr, U, V = self._make_fitted_surrogate(bkd)
        clone = surr.clone_unfitted()
        assert not clone.is_fitted()
        assert surr.is_fitted()

    def test_predict_shapes_multi_io(self, bkd) -> None:
        np.random.seed(42)
        ngrid1, ngrid2 = 5, 7
        N = 10
        ncodes_in = ngrid1 + ngrid2
        ncodes_out = ngrid1 + ngrid2

        enc_in1 = IdentityFunctionEncoder(ngrid1, bkd)
        enc_in2 = IdentityFunctionEncoder(ngrid2, bkd)
        enc_out1 = IdentityFunctionEncoder(ngrid1, bkd)
        enc_out2 = IdentityFunctionEncoder(ngrid2, bkd)

        kernel = Matern52Kernel(
            [1.0] * ncodes_in, (0.1, 10.0), ncodes_in, bkd
        )
        reg = ScalarKernelLatentRegressor(
            kernel, ncodes_in, ncodes_out, bkd, nugget=1e-8
        )
        U = bkd.array(np.random.randn(ncodes_in, N))
        V = bkd.array(np.random.randn(ncodes_out, N))
        reg.fit_internal(U, V)

        surr = KernelOperatorSurrogate(
            [enc_in1, enc_in2], [enc_out1, enc_out2], reg
        )

        u1 = bkd.array(np.random.randn(ngrid1, 3))
        u2 = bkd.array(np.random.randn(ngrid2, 3))
        result = surr.predict([u1, u2])
        assert len(result) == 2
        assert result[0].shape == (ngrid1, 3)
        assert result[1].shape == (ngrid2, 3)
