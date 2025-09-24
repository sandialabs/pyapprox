import unittest

import numpy as np

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.variational.flowmatching import (
    ContinuousNormalizingFlow,
    VelocityModel,
    BasisExpansionVelocityModel,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import GaussianMarginal
from pyapprox.surrogates.affine.basisexp import (
    MonomialExpansion,
    MultiIndexBasis,
)
from pyapprox.surrogates.univariate.base import Monomial1D


class TestFlows:
    def setUp(self):
        np.random.seed(1)

    def test_independent_gaussians_sampling(self):
        bkd = self.get_backend()
        nvars = 2
        cov_diag = bkd.array([2.0, 3.0])
        mean = bkd.array([0.5, -1.0])
        target_variable = IndependentMarginalsVariable(
            [
                GaussianMarginal(mu, bkd.sqrt(var), bkd)
                for mu, var in zip(mean, cov_diag)
            ],
            backend=bkd,
        )
        source_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars)],
            backend=bkd,
        )

        class CustomVelModel(VelocityModel):
            def _value(self, state):
                if self._time == 0:
                    self._init_state = self._bkd.copy(state)
                return (
                    self._bkd.sqrt(cov_diag) * self._init_state + mean - state
                ) / (1.0 - self._time)

        vel_model = CustomVelModel(nvars, backend=bkd)

        flow = ContinuousNormalizingFlow(source_variable, vel_model, 0.3)
        print(flow)

        samples = flow._map_from_latent(bkd.zeros((nvars, 1)))
        assert bkd.allclose(samples[:, 0], mean)

        nsamples = 10000
        samples = flow.rvs(nsamples)
        # print(
        #     (bkd.mean(samples, axis=1) - target_variable.mean()[:, 0])
        #     / target_variable.mean()[:, 0]
        # )
        assert bkd.allclose(
            bkd.mean(samples, axis=1), target_variable.mean()[:, 0], rtol=3e-2
        )
        # print(bkd.cov(samples, ddof=1))
        assert bkd.allclose(
            bkd.cov(samples, ddof=1),
            target_variable.covariance(),
            rtol=3e-2,
            atol=3e-2,
        )

    def test_independent_gaussians_fit(self):
        bkd = self.get_backend()
        nvars = 2
        cov_diag = bkd.array([2.0, 3.0])
        mean = bkd.array([0.5, -1.0])
        target_variable = IndependentMarginalsVariable(
            [
                GaussianMarginal(mu, bkd.sqrt(var), bkd)
                for mu, var in zip(mean, cov_diag)
            ],
            backend=bkd,
        )
        source_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars)],
            backend=bkd,
        )

        ntrain_samples = 10000
        train_samples = target_variable.rvs(ntrain_samples)

        basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars + 1)]
        )
        basis.set_tensor_product_indices([2] * (nvars + 1))
        bexp = MonomialExpansion(basis, nqoi=nvars)
        vel_model = BasisExpansionVelocityModel(bexp)
        flow = ContinuousNormalizingFlow(source_variable, vel_model, 0.3)

        train_times = bkd.asarray(
            np.random.uniform(0.0, 1.0, (1, ntrain_samples))
        )
        source_samples = source_variable.rvs(ntrain_samples)
        print(train_samples.shape, train_times.shape)
        time_samples = (
            1.0 - train_times
        ) * source_samples + train_times * train_samples
        basis_mat = vel_model._bexp.basis()(
            bkd.vstack((train_times, time_samples))
        )
        print(basis_mat.shape)
        coef = bkd.lstsq(
            basis_mat,
            (train_samples - source_samples).T,
        )
        print(coef)
        bexp.set_coefficients(coef)
        samples = flow._map_from_latent(bkd.zeros((nvars, 1)))
        print(samples[:, 0], mean)
        assert bkd.allclose(samples[:, 0], mean)


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
