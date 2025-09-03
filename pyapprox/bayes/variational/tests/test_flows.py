import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.variables.gaussian import (
    IndependentMultivariateGaussian,
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    IndependentGroupsVariable,
)
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.variables.marginals import GaussianMarginal
from pyapprox.surrogates.affine.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.bayes.variational.flows import (
    Flow,
    RealNVPLayer,
    RealNVPScalingConstraint,
)
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation


class TestFlows:
    def setUp(self):
        np.random.seed(1)

    def _setup_polynomial_real_nvp(
        self, nvars, layer_nterms, layer_coefs=None, nlabels=0, tp_basis=True
    ):
        bkd = self.get_backend()
        latent_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars)],
            backend=bkd,
        )
        assert nvars % 2 == 0  # hack for convenience
        nlayer_sample_inputs = nvars // 2
        nlayer_inputs = nlayer_sample_inputs + nlabels

        layer_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nlayer_inputs)],
            backend=bkd,
        )
        nlayers = len(layer_nterms)
        # nlayer_sample_inputs * 2 because need nlayer_sample_inputs
        # for shift and nlayer_sample_inputs for scale
        nqoi = nlayer_sample_inputs * 2
        bexps = [
            setup_polynomial_chaos_expansion_from_variable(
                layer_variable, nqoi
            )
            for ii in range(nlayers)
        ]
        if layer_coefs is None:
            layer_coefs = [None for nterm in layer_nterms]
        for bexp, nterm, coef in zip(bexps, layer_nterms, layer_coefs):
            if tp_basis:
                bexp.basis().set_tensor_product_indices(
                    [nterm] * (nlayer_inputs)
                )
            else:
                bexp.basis().set_hyperbolic_indices(nterm, 1)

            if coef is None:
                # nlayer_sample_inputs * 2 because need nlayer_sample_inputs
                # for shift and nlayer_sample_inputs for scale
                coef = bkd.asarray(
                    np.random.normal(
                        0.0, 0.01, (bexp.nterms() * nlayer_sample_inputs * 2)
                    )
                )
                # before coef is flattend, coef=
                # [
                #     [
                #         shift_const_const_qoi1,
                #         shift_const_qoi2,
                #         scale_const_qoi1,
                #         scale_const_qoi2,
                #     ],
                #     [
                #         shift_linear_qoi1,
                #         shift_linear_qoi2,
                #         scale_linear_qoi1,
                #         scale_linear_qoi2,
                #     ],
                # ]
                # latent variable is multiplied by exp(scale) so specify
                # desired multiplication in log space, e.g. exp(0) = 1.
                coef[1::2] = -0.01

            # place large bounds on shift coeffients
            bounds = bkd.array([-5.0, 5.0])
            bounds = bkd.tile(bounds, (coef.shape[0],))
            # Place smaller bounds on scale coefficients because scale is
            # exponentiated
            bounds[2::4] = -1.0
            bounds[3::4] = 0.5
            bexp.set_coefficient_bounds(coef, bounds)

        mask = bkd.ones(nvars, dtype=bool)
        mask[1::2] = 0
        mask_complement = ~mask
        layers = []
        for ii, bexp in enumerate(bexps):
            if ii % 2 == 0:
                layers.append(
                    RealNVPLayer(nvars, bexp, mask=mask, nlabels=nlabels)
                ),
            else:
                layers.append(
                    RealNVPLayer(
                        nvars, bexp, mask=mask_complement, nlabels=nlabels
                    )
                )

        return Flow(latent_variable, layers)

    def test_realnvp_2d_independent_gaussians_sampling(self):
        """
        Test that RealNVP can recover independent gaussians with
        certain marginal means and variances when using a polynomial
        expansion with coefficients set to reproduce analytical solution.
        That is, this test does not check optimization
        """
        bkd = self.get_backend()
        nvars = 2
        mean = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, 1)))
        cov_diag = bkd.array([2.0, 3.0])
        ntrain_samples = 10000
        target_variable = IndependentMultivariateGaussian(
            mean, cov_diag, backend=bkd
        )
        train_samples = target_variable.rvs(ntrain_samples)

        # coef is constant for shift then constant for scaling along columns
        # rows correspond to coefficients of each polynomial for scaling
        # and columns
        cov = target_variable.covariance()
        coef1 = bkd.array(
            [[mean[1, 0], bkd.log(bkd.sqrt(cov[1, 1]))], [0.0, 0.0]],
        ).flatten()
        coef2 = bkd.array(
            [[mean[0, 0], bkd.log(bkd.sqrt(cov[0, 0]))], [0.0, 0.0]],
        ).flatten()
        flow = self._setup_polynomial_real_nvp(nvars, [2, 2], [coef1, coef2])

        usamples = flow._map_to_latent(train_samples)
        recovered_samples = flow._map_from_latent(usamples)
        assert bkd.allclose(recovered_samples, train_samples)

        nsamples = 10
        samples = target_variable.rvs(nsamples)
        assert bkd.allclose(
            flow.pdf(samples)[:, 0], target_variable.pdf(samples)[:, 0]
        )

        new_samples = flow.rvs(int(5e6))
        # print(bkd.mean(new_samples, axis=1)[:, None] - mean)
        assert bkd.allclose(
            bkd.mean(new_samples, axis=1)[:, None], mean, rtol=1e-3
        )
        # print(bkd.cov(new_samples, ddof=1) - cov)
        assert bkd.allclose(
            bkd.cov(new_samples, ddof=1), cov, rtol=1e-3, atol=3e-3
        )

        # test plots run
        axs = plt.subplots(1, 2)[1]
        target_variable.plot_pdf(
            axs[0], [-6, 6, -6, 6], levels=31, cmap="coolwarm"
        )
        flow.plot_pdf(axs[1], [-6, 6, -6, 6], levels=31, cmap="coolwarm")
        # nsamples = 1000
        # target_samples = target_variable.rvs(nsamples)
        # flow_samples = flow.rvs(nsamples)
        # axs[0].scatter(*target_samples, alpha=0.1, color="k")
        # axs[1].scatter(*flow_samples, alpha=0.1, color="k")
        # plt.show()

    def test_realnvp_2d_correlated_gaussians_sampling(self):
        """
        Test that RealNVP can recover correlated gaussians with
        certain marginal means and variances when using a polynomial
        expansion with coefficients set to reproduce analytical solution.
        That is, this test does not check optimization
        """
        bkd = self.get_backend()
        nvars = 2
        # mean = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, 1)))
        # mat = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
        # cov = mat.T @ mat
        mean = bkd.array([1.0, 2.0])[:, None]
        cov = bkd.array([[2.0, 0.5], [0.5, 4.0]])
        ntrain_samples = 10000
        target_variable = DenseCholeskyMultivariateGaussian(
            mean, cov, backend=bkd
        )
        train_samples = target_variable.rvs(ntrain_samples)

        # coef is constant for shift then constant for scaling along columns
        # rows correspond to coefficients of each polynomial for scaling
        # and columns
        # layer 1 params
        # tau1**2+delta1**2=c11
        delta1 = cov[0, 1] / bkd.sqrt(cov[0, 0])
        tau1 = bkd.sqrt(cov[1, 1] - delta1**2)
        log_tau1 = bkd.log(tau1)
        # layer 2 params
        tau2 = bkd.sqrt(cov[0, 0])
        log_tau2 = bkd.log(tau2)
        coef1 = bkd.array(
            [
                [mean[1, 0], log_tau1],
                [delta1, 0.0],
            ],
        ).flatten()
        coef2 = bkd.array(
            [[mean[0, 0], log_tau2], [0.0, 0.0]],
        ).flatten()
        flow = self._setup_polynomial_real_nvp(nvars, [2, 2], [coef1, coef2])

        usamples = flow._map_to_latent(train_samples)
        recovered_samples = flow._map_from_latent(usamples)
        assert bkd.allclose(recovered_samples, train_samples)

        nsamples = 10
        samples = target_variable.rvs(nsamples)
        assert bkd.allclose(
            flow.pdf(samples)[:, 0], target_variable.pdf(samples)[:, 0]
        )

        new_samples = flow.rvs(int(5e6))
        # print(bkd.mean(new_samples, axis=1)[:, None] - mean)
        assert bkd.allclose(
            bkd.mean(new_samples, axis=1)[:, None], mean, rtol=1e-3
        )
        assert bkd.allclose(
            bkd.cov(new_samples, ddof=1), cov, rtol=1e-3, atol=3e-3
        )

        # test plots run
        axs = plt.subplots(1, 2)[1]
        target_variable.plot_pdf(
            axs[0], [-6, 6, -6, 6], levels=31, cmap="coolwarm"
        )
        flow.plot_pdf(axs[1], [-6, 6, -6, 6], levels=31, cmap="coolwarm")
        # nsamples = 1000
        # target_samples = target_variable.rvs(nsamples)
        # flow_samples = flow.rvs(nsamples)
        # axs[0].scatter(*target_samples, alpha=0.1, color="k")
        # axs[1].scatter(*flow_samples, alpha=0.1, color="k")
        # plt.show()

    def test_realnvp_independent_gaussians_fit(self):
        """
        Test that RealNVP can recover independent gaussians with
        certain marginal means and variances when using a polynomial
        expansion with coefficients set to reproduce analytical solution.
        That is, this test does not check optimization
        """
        bkd = self.get_backend()
        nvars = 2
        mean = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, 1)))
        cov_diag = bkd.array([2.0, 3.0])
        marginals = [
            GaussianMarginal(
                mean=mean[0], stdev=bkd.sqrt(cov_diag[0]), backend=bkd
            ),
            GaussianMarginal(
                mean=mean[1], stdev=bkd.sqrt(cov_diag[1]), backend=bkd
            ),
        ]
        target_variable = IndependentMarginalsVariable(marginals, backend=bkd)

        # ntrain_samples = 10000
        # train_samples = target_variable.rvs(ntrain_samples)
        # train_samples = HaltonSequence(nvars, 1, target_variable, bkd=bkd).rvs(
        #     ntrain_samples
        # )
        # train_weights = bkd.full((ntrain_samples, 1), 1.0 / ntrain_samples)
        quad_rule = setup_tensor_product_gauss_quadrature_rule(target_variable)
        train_samples, train_weights = quad_rule([5, 5])
        # print(train_samples @ train_weights - mean, "m")

        # define exact answer
        cov = target_variable.covariance()
        exact_coef1 = bkd.stack(
            [
                bkd.array([mean[1, 0], bkd.log(bkd.sqrt(cov[1, 1]))]),
                bkd.zeros((2,)),
            ],
            axis=0,
        ).flatten()
        exact_coef2 = bkd.stack(
            [
                bkd.array([mean[0, 0], bkd.log(bkd.sqrt(cov[0, 0]))]),
                bkd.zeros((2,)),
            ],
            axis=0,
        ).flatten()
        # perturb them to use as initial guess for optimizer
        # if initial iterate values are too large optimization will fail
        coef1 = 0.0 * exact_coef1 + bkd.asarray(
            np.random.normal(0, 0.1, exact_coef1.shape)
        )
        coef2 = 0.0 * exact_coef2 + bkd.asarray(
            np.random.normal(0, 0.1, exact_coef2.shape)
        )
        flow = self._setup_polynomial_real_nvp(nvars, [2, 2], [coef1, coef2])

        flow._loss.set_samples(train_samples)
        flow._loss.set_weights(train_weights)
        iterate = flow._hyp_list.get_active_opt_params()[:, None]
        errors = flow._loss.check_apply_jacobian(iterate, disp=False)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1e-6
        flow.set_optimizer(
            flow.default_optimizer(verbosity=0, method="trust-constr")
        )
        flow.fit(train_samples, weights=train_weights)

        nsamples = 100
        samples = target_variable.rvs(nsamples)
        # print(flow.pdf(samples)[:, 0] - target_variable.pdf(samples)[:, 0])
        assert bkd.allclose(
            flow.pdf(samples)[:, 0],
            target_variable.pdf(samples)[:, 0],
            rtol=1e-10,
        )

    def test_realnvp_2d_conditional_correlated_gaussians_fit(self):
        """
        Test that RealNVP can recover the posterior from a Gaussian prior,
        likelihood and linear observation model
        """
        bkd = self.get_backend()

        nobs = 1
        # Define the prior
        nvars = 2
        mean = bkd.array([-1.0, -2.0])[:, None]
        cov = bkd.array([[1.0, 0.0], [0.0, 1.0]])
        prior = DenseCholeskyMultivariateGaussian(mean, cov, backend=bkd)

        # Define the observation model
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        obs_mat = 1.0 / bkd.norm(obs_mat, axis=1)[:, None] * obs_mat

        # Define the likelihood
        noise_std = 0.5

        latent_prior_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, backend=bkd) for ii in range(nvars)],
            backend=bkd,
        )
        latent_data_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, backend=bkd) for ii in range(nobs)],
            backend=bkd,
        )
        latent_joint_prior_data_variable = IndependentGroupsVariable(
            [latent_prior_variable, latent_data_variable]
        )

        # quad_type = "MC"
        # quad_type = "Halton"
        quad_type = "Gauss"
        if quad_type == "Gauss":
            # Generate the training data with Gaussian Quadrature
            # Gauss quadrature with Gaussian latent space can
            # generate samples to far into the tails it causes inf
            quad_rule = setup_tensor_product_gauss_quadrature_rule(
                latent_joint_prior_data_variable
            )
            quad_samples, train_weights = quad_rule([3, 3, 3])
        elif quad_type == "Halton":
            # Generate the training data with Halton Sequences
            ntrain_samples = 10000
            quad_samples = HaltonSequence(
                nvars + nobs, 1, latent_joint_prior_data_variable, bkd=bkd
            ).rvs(ntrain_samples)
            train_weights = bkd.full((ntrain_samples, 1), 1.0 / ntrain_samples)
        else:
            # Generate the training data with MC
            ntrain_samples = 10000
            quad_samples = latent_joint_prior_data_variable.rvs(ntrain_samples)
            train_weights = bkd.full((ntrain_samples, 1), 1.0 / ntrain_samples)

        ntrain_samples = train_weights.shape[0]
        latent_prior_samples = quad_samples[:nvars]
        prior_samples = (
            prior._cov_sqrt.apply(latent_prior_samples) + prior._mean
        )
        latent_noise_samples = quad_samples[nvars:]
        noise = latent_noise_samples * noise_std
        # quad_mean = prior_samples @ train_weights
        # print(noise @ (train_weights * noise.T), noise_std**2)
        # print(
        #     (prior_samples - quad_mean)
        #     @ (train_weights * (prior_samples - quad_mean).T)
        # )
        # print(prior.covariance())
        # print(quad_mean)
        # print(prior.mean())

        noiseless_obs = obs_mat @ prior_samples
        obs = noiseless_obs + noise
        train_samples = bkd.vstack((prior_samples, obs))

        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_mat,
            prior.mean(),
            prior.covariance(),
            bkd.eye(nobs) * noise_std**2,
            backend=bkd,
        )

        # Setup the flow model
        flow = self._setup_polynomial_real_nvp(
            nvars, [2, 2], None, nlabels=nobs, tp_basis=True
        )

        flow._loss.set_samples(train_samples)
        flow._loss.set_weights(train_weights)
        iterate = flow._hyp_list.get_active_opt_params()[:, None]
        errors = flow._loss.check_apply_jacobian(
            iterate, disp=True, fd_eps=bkd.flip(bkd.logspace(-13, -1, 13))
        )
        assert errors.min() / errors.max() < 1e-6

        flow.set_optimizer(
            flow.default_multistart_optimizer(
                maxiter=1000,
                verbosity=3,
                exit_hard=False,
                method="trust-constr",  # , ncandidates=10
            )
        )

        constraint = RealNVPScalingConstraint(
            ntrain_samples, backend=bkd, keep_feasible=True
        )
        constraint.set_bounds(bkd.array([-np.inf, 5.0])[None, :])
        constraint.set_flow(flow)
        # print(constraint)
        # print(constraint(iterate))
        if quad_type == "Gauss":
            # only use contraint with Gauss quadrature rule
            # because current contraint implementation jacobian in slow
            # and this test is only to test it works. The constraint is
            # not active for this tests problem. But it is useful for
            # avoiding infs caused by large exp(scale) values which
            # arrise in more ill coniditioned problems. Also changing bounds
            # on scale parameters can help, but it is difficult to
            # set good bounds a priori as I do in setup_flow
            flow._optimizer.set_constraints([constraint])
        # assert False

        flow.fit(train_samples, iterate=iterate, weights=train_weights)
        axs = plt.subplots(1, 2, sharey=True)[1]
        np.random.seed(2)
        label = obs_mat @ prior.rvs(1) + bkd.asarray(
            np.random.normal(0, noise_std, (nobs, 1))
        )
        print(label, "LABEL")
        flow.plot_pdf(
            axs[1],
            [-6, 6, -6, 6],
            label=label,
            levels=31,
            cmap="coolwarm",
        )
        # nsamples = 10
        # flow_samples = flow.rvs(nsamples, label)
        # axs[1].scatter(*flow_samples, alpha=0.1, color="k")

        laplace.compute(label)
        target_variable = laplace.posterior_variable()
        target_variable.plot_pdf(
            axs[0], [-6, 6, -6, 6], levels=31, cmap="coolwarm"
        )
        test_samples = target_variable.rvs(10)
        print(flow.pdf(flow.append_labels(test_samples, label)))
        print(target_variable.pdf(test_samples))
        print(
            flow.pdf(flow.append_labels(test_samples, label))
            - target_variable.pdf(test_samples)
        )
        # axs[0].scatter(*test_samples, alpha=0.1, color="k")

        plt.show()
        axs, im = flow._loss.plot_cross_sections(npts_1d=10)
        # plt.colorbar(im, ax=ax)
        plt.show()


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
