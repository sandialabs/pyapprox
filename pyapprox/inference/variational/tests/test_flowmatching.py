import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.inference.variational.flowmatching import (
    BasisExpansionContinuousNormalizingFlow,
    ContinuousNormalizingFlow,
    VelocityField,
    BasisExpansionVelocityField,
    FixedDataFlowMatchingPathSampler,
    TensorProductGaussQuadratureModelBasedFlowMatchingPathSampler,
    # MonteCarloModelBasedFlowMatchingPathSampler,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import GaussianMarginal, UniformMarginal
from pyapprox.inference.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.surrogates.affine.linearsystemsolvers import LstSqSolver
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from scipy import stats
from pyapprox.pde.timeintegration import (
    RK4,
    HeunResidual,
    BackwardEulerResidual,
)
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.inference.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.interface.model import DenseMatrixLinearModel

# from pyapprox.util.print_wrapper import *


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

        vel_field = CustomVelModel(nvars, 0, backend=bkd)

        ntrain_samples = 10
        train_samples = target_variable.rvs(ntrain_samples)
        path_sampler = FixedDataFlowMatchingPathSampler(
            source_variable, train_samples
        )

        flow = ContinuousNormalizingFlow(path_sampler, vel_field, 0.3)
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

    def _setup_basisexpansion_flow_field(self, nvars, nlabels, variable):
        # Monomial becomes poorly conditioned with high-order quadrature
        # basis = MultiIndexBasis(
        #     [Monomial1D(backend=bkd) for ii in range(nvars + 1)]
        # )

        # Orthopolynomials are better conditioned, but I do not
        # yet have good guidance on how to choose the correct basis
        # when everything is not Gaussian
        bkd = self.get_backend()
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                stats.uniform(0.0, 1.0), backend=bkd
            )
        ] + [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=bkd
            )
            for marginal in variable.marginals()
        ]
        basis = OrthonormalPolynomialBasis(bases_1d)
        bexp = PolynomialChaosExpansion(
            basis, nqoi=nvars, solver=LstSqSolver(backend=bkd)
        )
        vel_field = BasisExpansionVelocityField(bexp, nlabels)
        return vel_field

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

        # increaseing nterms increases accuracy of recovered mean
        nterms = 9
        vel_field = self._setup_basisexpansion_flow_field(
            nvars, 0, source_variable
        )
        vel_field._bexp.basis().set_tensor_product_indices(
            [nterms] * (nvars + 1)
        )

        # ntrain_samples = 10000
        # train_samples = target_variable.rvs(ntrain_samples)
        # path_sampler = FixedDataFlowMatchingPathSampler(
        #     source_variable, train_samples
        # )
        path_sampler = (
            TensorProductGaussQuadratureModelBasedFlowMatchingPathSampler(
                source_variable,
                target_variable,
                bkd.asarray([nterms + 1] * (nvars * 2 + 1)),
            )
        )

        # the size of deltat and the type of integrator effects
        # the error recovered in mean
        flow = BasisExpansionContinuousNormalizingFlow(
            path_sampler, vel_field, 0.01, time_residual_cls=RK4
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

    def test_conditional_correlated_gaussians_fit(self):
        bkd = self.get_backend()
        nobs = 1

        # Define the prior
        nvars = 2
        prior = IndependentMarginalsVariable(
            [GaussianMarginal(0.0, 1.0, bkd) for ii in range(nvars)],
            backend=bkd,
        )

        # Define the observation model
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        obs_mat = 1.0 / bkd.norm(obs_mat, axis=1)[:, None] * obs_mat

        # Define the noise used in the likelihood
        noise_std = 0.5

        # increaseing nterms increases accuracy of recovered mean
        nterms = 5  # 9
        bexp_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0.0, 1.0, bkd) for ii in range(nvars + nobs)],
            backend=bkd,
        )
        vel_field = self._setup_basisexpansion_flow_field(
            nvars, nobs, bexp_variable
        )
        # vel_field._bexp.basis().set_tensor_product_indices(
        #    [nterms] * (bexp_variable.nvars() + 1)
        # )
        vel_field._bexp.basis().set_hyperbolic_indices(nterms, 1.0)
        print(f"{vel_field._bexp.nterms()=}")

        # ntrain_samples = 10000
        # train_samples = target_variable.rvs(ntrain_samples)
        # path_sampler = FixedDataFlowMatchingPathSampler(
        #     source_variable, train_samples
        # )
        noise_cov = bkd.diag(bkd.full((nobs,), noise_std**2))
        obs_model = DenseMatrixLinearModel(obs_mat, backend=bkd)
        source_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0.0, 1.0, bkd) for ii in range(nvars)],
            backend=bkd,
        )
        latent_data_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0.0, 1.0, bkd) for ii in range(nobs)],
            backend=bkd,
        )
        path_sampler = (
            TensorProductGaussQuadratureModelBasedFlowMatchingPathSampler(
                source_variable,
                prior,
                bkd.asarray([nterms + 1] * (nvars * 2 + nobs + 1)),
                latent_data_variable=latent_data_variable,
                loglike=ModelBasedGaussianLogLikelihood(obs_model, noise_cov),
            )
        )

        # the size of deltat and the type of integrator effects
        # the error recovered in mean
        flow = BasisExpansionContinuousNormalizingFlow(
            path_sampler, vel_field, 0.01, nobs, time_residual_cls=HeunResidual
        )
        flow.fit()

        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_mat,
            prior.mean(),
            prior.covariance(),
            bkd.eye(nobs) * noise_std**2,
            backend=bkd,
        )

        label = obs_mat @ prior.rvs(1) + bkd.asarray(
            np.random.normal(0, noise_std, (nobs, 1))
        )
        # print(label, "LABEL")

        laplace.compute(label)
        target_variable = laplace.posterior_variable()
        test_samples = target_variable.rvs(10)
        # print(
        #     flow.pdf(flow.append_labels(test_samples, label))
        #     - target_variable.pdf(test_samples)
        # )
        assert bkd.allclose(
            flow.pdf(flow.append_labels(test_samples, label)),
            target_variable.pdf(test_samples),
            atol=1e-4,
        )

    def moon_rvs(self, nsamples1, nsamples2, std, bounds=[-0.5, 0.5]):
        bkd = self.get_backend()
        moon1_samples = bkd.stack(
            [
                bkd.cos(bkd.linspace(0.0, np.pi, nsamples1)),
                bkd.sin(bkd.linspace(0.0, np.pi, nsamples1)),
            ],
            axis=0,
        )
        moon2_samples = bkd.stack(
            [
                1.0 - bkd.cos(bkd.linspace(0.0, np.pi, nsamples2)),
                1.0 - bkd.sin(bkd.linspace(0.0, np.pi, nsamples2)) - 0.5,
            ],
            axis=0,
        )
        indices1 = np.arange(nsamples1)
        np.random.shuffle(indices1)
        noise1 = bkd.asarray(np.random.normal(0.0, std, moon1_samples.shape))
        moon1_samples = moon1_samples[:, indices1] + noise1
        indices2 = np.arange(nsamples2)
        np.random.shuffle(indices2)
        noise2 = bkd.asarray(np.random.normal(0.0, std, moon2_samples.shape))
        moon2_samples = moon2_samples[:, indices2] + noise2
        moon1_samples[0] = (moon1_samples[0] + 1) / 3
        moon1_samples[1] = (moon1_samples[1] + 0.75) / 2
        moon2_samples[0] = (moon2_samples[0] + 1) / 3
        moon2_samples[1] = (moon2_samples[1] + 0.75) / 2
        # map from [0,1] to [-1,1]
        width = bounds[1] - bounds[0]
        moon1_samples = moon1_samples * width + bounds[0]
        moon2_samples = moon2_samples * width + bounds[0]
        print(bounds)
        return moon1_samples, moon2_samples

    def example(self):
        bkd = self.get_backend()
        nvars = 2
        source_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars)],
            backend=bkd,
        )

        ntrain_samples = 10000  # 000
        moon_bounds = [-2, 4]
        moon_bounds = [-1, 1]
        train_samples = bkd.hstack(
            self.moon_rvs(
                ntrain_samples // 2, ntrain_samples // 2, 0.05, moon_bounds
            )
        )
        train_samples = self.moon_rvs(
            ntrain_samples // 2, ntrain_samples // 2, 0.05, moon_bounds
        )[0]
        print(train_samples.min(), train_samples.max())

        path_sampler = FixedDataFlowMatchingPathSampler(
            source_variable, train_samples
        )
        path_samples, time_derivs = path_sampler.generate_path_samples()
        # plt.scatter(*path_samples)
        # plt.show()
        basis_marginals = [
            UniformMarginal(
                path_samples[ii].min() - 0.1,
                path_samples[ii].max() + 0.1,
                backend=bkd,
            )
            for ii in range(1, nvars + 1)
        ]
        basis_variable = IndependentMarginalsVariable(
            basis_marginals, backend=bkd
        )
        basis_variable = source_variable

        nterms = 10
        vel_field = self._setup_basisexpansion_flow_field(
            nvars, 0, basis_variable
        )
        vel_field._bexp.basis().set_tensor_product_indices(
            [nterms] * (nvars + 1)
        )

        from pyapprox.surrogates.affine.basis import (
            QRBasedRotatedOrthonormalPolynomialBasis,
        )

        basis = QRBasedRotatedOrthonormalPolynomialBasis(
            vel_field._bexp.basis()._bases_1d
        )
        basis.set_tensor_product_indices([nterms for ii in range(nvars + 1)])
        basis.set_quadrature_rule_tuple(
            bkd.vstack(
                (
                    bkd.asarray(
                        np.random.uniform(0, 1, (1, train_samples.shape[1]))
                    ),
                    train_samples,
                )
            ),
            path_sampler.get_weights(),
        )
        bexp = PolynomialChaosExpansion(
            basis, nqoi=nvars, solver=LstSqSolver(backend=bkd)
        )
        vel_field = BasisExpansionVelocityField(bexp, 0)

        # the size of deltat and the type of integrator effects
        # the error recovered in mean
        flow = BasisExpansionContinuousNormalizingFlow(
            # path_sampler, vel_field, 0.01, time_residual_cls=HeunResidual
            path_sampler,
            vel_field,
            0.01,
            time_residual_cls=HeunResidual,  # BackwardEulerResidual,
        )
        flow.fit()
        print("Training complete")
        print(
            bkd.abs(vel_field._bexp.get_coefficients()).min(),
            bkd.abs(vel_field._bexp.get_coefficients()).max(),
        )
        axs = plt.subplots(1, 2, sharey=True)[1]
        axs[0].scatter(*train_samples, alpha=0.1, color="k")
        # increase deltat to speed up plotting since high accuracy is not
        # needed when just visualizing
        label = None
        print(flow)
        flow._set_deltat(0.1)
        nsamples = 1000  # 0
        flow_samples = flow.rvs(nsamples, label)
        axs[1].scatter(*flow_samples, alpha=0.1, color="k")
        flow._set_deltat(0.1)
        flow.plot_pdf(
            axs[1],
            moon_bounds + moon_bounds,
            label=label,
            npts_1d=31,
            levels=31,
            cmap="coolwarm",
        )
        axs[0].set_xlim(moon_bounds)
        axs[0].set_ylim(moon_bounds)
        axs[1].set_xlim(moon_bounds)
        axs[1].set_ylim(moon_bounds)


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
