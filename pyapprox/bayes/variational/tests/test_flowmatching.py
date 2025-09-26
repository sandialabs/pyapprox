import unittest

import numpy as np

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.variational.flowmatching import (
    BasisExpansionContinuousNormalizingFlow,
    ContinuousNormalizingFlow,
    VelocityField,
    BasisExpansionVelocityField,
    FixedDataFlowMatchingObjectiveSampler,
    MonteCarloModelBasedFlowMatchingObjectiveSampler,
    TensorProductGaussQuadratureModelBasedFlowMatchingObjectiveSampler,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import GaussianMarginal
from pyapprox.surrogates.affine.basisexp import (
    MonomialExpansion,
    MultiIndexBasis,
)
from pyapprox.surrogates.affine.linearsystemsolvers import LstSqSolver
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from scipy import stats
from pyapprox.pde.collocation.timeintegration import HeunResidual, RK4


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

        class CustomVelModel(VelocityField):
            def _value(self, state):
                if self._time == 0:
                    self._init_state = self._bkd.copy(state)
                eps = 1e-8  # to avoid division by zero at time=1.0
                return (
                    self._bkd.sqrt(cov_diag) * self._init_state + mean - state
                ) / (1.0 - self._time + eps)

        vel_model = CustomVelModel(nvars, backend=bkd)

        ntrain_samples = 10
        train_samples = target_variable.rvs(ntrain_samples)
        obj_sampler = FixedDataFlowMatchingObjectiveSampler(
            source_variable, train_samples
        )

        flow = ContinuousNormalizingFlow(obj_sampler, vel_model, 0.3)
        print(flow)

        samples = flow._map_from_latent(bkd.zeros((nvars, 1)))
        assert bkd.allclose(samples[:, 0], mean)

        nsamples = 10000
        samples = flow.rvs(nsamples)
        print(
            (bkd.mean(samples, axis=1) - target_variable.mean()[:, 0])
            / target_variable.mean()[:, 0]
        )
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

        # Monomial becomes poorly conditioned with high-order quadrature
        # basis = MultiIndexBasis(
        #     [Monomial1D(backend=bkd) for ii in range(nvars + 1)]
        # )

        # Orthopolynomials are better conditioned, but I do not
        # yet have good guidance on how to choose the correct basis
        # when everything is not Gaussian
        bases_1d = [
            [
                setup_univariate_orthogonal_polynomial_from_marginal(
                    marginal, backend=bkd
                )
                for marginal in variable.marginals()
            ]
            for variable in [source_variable]
        ]
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                stats.uniform(0.0, 1.0), backend=bkd
            )
        ] + sum(bases_1d, [])
        print(bases_1d)
        basis = OrthonormalPolynomialBasis(bases_1d)

        # increaseing nterms increases accuracy of recovered mean
        nterms = 9
        basis.set_tensor_product_indices([nterms] * (nvars + 1))
        bexp = MonomialExpansion(
            basis, nqoi=nvars, solver=LstSqSolver(backend=bkd)
        )
        vel_model = BasisExpansionVelocityField(bexp)

        # ntrain_samples = 10000
        # train_samples = target_variable.rvs(ntrain_samples)
        # obj_sampler = FixedDataFlowMatchingObjectiveSampler(
        #     source_variable, train_samples
        # )
        obj_sampler = (
            TensorProductGaussQuadratureModelBasedFlowMatchingObjectiveSampler(
                source_variable,
                target_variable,
                bkd.asarray([nterms + 1] * (nvars * 2 + 1)),
            )
        )

        # the size of deltat and the type of integrator effects
        # the error recovered in mean
        flow = BasisExpansionContinuousNormalizingFlow(
            obj_sampler, vel_model, 0.01, time_residual_cls=RK4
        )
        flow.fit()
        samples = flow._map_from_latent(bkd.zeros((nvars, 1)))
        assert bkd.allclose(samples[:, 0], mean, rtol=2e-6)
        pdf_vals = flow.pdf(mean[:, None])
        # print(pdf_vals - target_variable.pdf(mean[:, None]))
        assert bkd.allclose(
            pdf_vals, target_variable.pdf(mean[:, None]), atol=1e-6
        )
        usamples = flow._map_to_latent(mean[:, None])
        assert bkd.allclose(usamples, bkd.zeros(usamples.shape), atol=1e-6)
        print(flow._map_from_latent(usamples), mean)
        assert bkd.allclose(
            flow._map_from_latent(usamples)[:, 0], mean, rtol=1e-6
        )


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
