import numpy as np
import pytest

from pyapprox.surrogates.kerneloperator.protocols import (
    LatentRegressorProtocol,
)
from pyapprox.surrogates.kerneloperator.regressors.factory import (
    make_latent_regressor,
)
from pyapprox.surrogates.kerneloperator.regressors.multioutput_kernel import (
    MultiOutputKernelLatentRegressor,
)
from pyapprox.surrogates.kerneloperator.regressors.scalar_kernel import (
    ScalarKernelLatentRegressor,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.kernels.multioutput.independent import (
    IndependentMultiOutputKernel,
)


class TestScalarKernelLatentRegressor:
    def _setup(self, bkd, ncodes_in=3, ncodes_out=2, N=10):
        np.random.seed(42)
        kernel = Matern52Kernel(
            [1.0] * ncodes_in, (0.1, 10.0), ncodes_in, bkd
        )
        self.reg = ScalarKernelLatentRegressor(
            kernel, ncodes_in, ncodes_out, bkd, nugget=1e-8
        )
        self.U = bkd.array(np.random.randn(ncodes_in, N))
        self.V = bkd.array(np.random.randn(ncodes_out, N))
        self.ncodes_in = ncodes_in
        self.ncodes_out = ncodes_out
        self.N = N

    def test_fit_predict_shapes(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        N_test = 4
        U_test = bkd.array(np.random.randn(self.ncodes_in, N_test))
        pred = self.reg.predict(U_test)
        assert pred.shape == (self.ncodes_out, N_test)

    def test_predict_std_shapes(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        N_test = 4
        U_test = bkd.array(np.random.randn(self.ncodes_in, N_test))
        std = self.reg.predict_std(U_test)
        assert std.shape == (self.ncodes_out, N_test)

    def test_nll_finite(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        nll = self.reg.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(nll)))

    def test_clone_unfitted_independent(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        clone = self.reg.clone_unfitted()
        assert not clone.is_fitted()
        assert self.reg.is_fitted()

    def test_protocol_compliance(self, bkd) -> None:
        self._setup(bkd)
        assert isinstance(self.reg, LatentRegressorProtocol)

    def test_dimensions(self, bkd) -> None:
        self._setup(bkd)
        assert self.reg.ncodes_in() == self.ncodes_in
        assert self.reg.ncodes_out() == self.ncodes_out


class TestMultiOutputKernelLatentRegressor:
    def _setup(self, bkd, ncodes_in=3, ncodes_out=2, N=10):
        np.random.seed(42)
        kernels = [
            Matern52Kernel([1.0] * ncodes_in, (0.1, 10.0), ncodes_in, bkd)
            for _ in range(ncodes_out)
        ]
        mo_kernel = IndependentMultiOutputKernel(kernels)
        self.reg = MultiOutputKernelLatentRegressor(
            mo_kernel, ncodes_in, ncodes_out, bkd, nugget=1e-8
        )
        self.U = bkd.array(np.random.randn(ncodes_in, N))
        self.V = bkd.array(np.random.randn(ncodes_out, N))
        self.ncodes_in = ncodes_in
        self.ncodes_out = ncodes_out
        self.N = N

    def test_fit_predict_shapes(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        N_test = 4
        U_test = bkd.array(np.random.randn(self.ncodes_in, N_test))
        pred = self.reg.predict(U_test)
        assert pred.shape == (self.ncodes_out, N_test)

    def test_predict_std_shapes(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        N_test = 4
        U_test = bkd.array(np.random.randn(self.ncodes_in, N_test))
        std = self.reg.predict_std(U_test)
        assert std.shape == (self.ncodes_out, N_test)

    def test_noutputs_mismatch_raises(self, bkd) -> None:
        np.random.seed(42)
        kernels = [
            Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
            for _ in range(3)
        ]
        mo_kernel = IndependentMultiOutputKernel(kernels)
        with pytest.raises(ValueError, match="must equal"):
            MultiOutputKernelLatentRegressor(mo_kernel, 2, 5, bkd)

    def test_clone_unfitted(self, bkd) -> None:
        self._setup(bkd)
        self.reg.fit_internal(self.U, self.V)
        clone = self.reg.clone_unfitted()
        assert not clone.is_fitted()
        assert self.reg.is_fitted()

    def test_protocol_compliance(self, bkd) -> None:
        self._setup(bkd)
        assert isinstance(self.reg, LatentRegressorProtocol)


class TestMakeLatentRegressor:
    def test_scalar_kernel(self, bkd) -> None:
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
        reg = make_latent_regressor(kernel, 2, 3, bkd)
        assert isinstance(reg, ScalarKernelLatentRegressor)

    def test_multioutput_kernel(self, bkd) -> None:
        kernels = [
            Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)
            for _ in range(3)
        ]
        mo_kernel = IndependentMultiOutputKernel(kernels)
        reg = make_latent_regressor(mo_kernel, 2, 3, bkd)
        assert isinstance(reg, MultiOutputKernelLatentRegressor)

    def test_bad_type_raises(self, bkd) -> None:
        with pytest.raises(TypeError, match="must satisfy"):
            make_latent_regressor("not_a_kernel", 2, 3, bkd)
