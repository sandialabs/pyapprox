import numpy as np
from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.fitters.maximum_likelihood_fitter import (
    KernelOperatorMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _identity_factory(ngrid):
    def factory(data, bkd):
        return IdentityFunctionEncoder(ngrid, bkd)
    return factory


class TestKernelOperatorMaximumLikelihoodFitter:
    def _setup(self, bkd, ngrid=8, N=15):
        np.random.seed(42)
        self.ngrid = ngrid
        self.N = N
        self.u_train = [bkd.array(np.random.randn(ngrid, N))]
        self.v_train = [bkd.array(np.random.randn(ngrid, N))]

    def test_nll_finite(self, bkd) -> None:
        self._setup(bkd)
        kernel = Matern52Kernel(
            [1.0] * self.ngrid, (0.1, 10.0), self.ngrid, bkd
        )
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(self.ngrid)],
            [_identity_factory(self.ngrid)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(self.u_train, self.v_train)
        nll = result.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(nll)))

    def test_hyps_change_after_optimization(self, bkd) -> None:
        self._setup(bkd)
        kernel = Matern52Kernel(
            [1.0] * self.ngrid, (0.1, 10.0), self.ngrid, bkd
        )
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(self.ngrid)],
            [_identity_factory(self.ngrid)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(self.u_train, self.v_train)
        initial = bkd.to_numpy(result.initial_hyperparameters())
        optimized = bkd.to_numpy(result.optimized_hyperparameters())
        assert not np.allclose(initial, optimized)

    def test_fixed_hyps_no_optimization(self, bkd) -> None:
        """All inactive hyperparameters skip optimization."""
        self._setup(bkd)
        kernel = Matern52Kernel(
            [1.0] * self.ngrid, (0.1, 10.0), self.ngrid, bkd
        )
        kernel.hyp_list().set_all_inactive()
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(self.ngrid)],
            [_identity_factory(self.ngrid)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(self.u_train, self.v_train)
        assert result.optimization_result() is None
        initial = bkd.to_numpy(result.initial_hyperparameters())
        optimized = bkd.to_numpy(result.optimized_hyperparameters())
        np.testing.assert_array_equal(initial, optimized)
