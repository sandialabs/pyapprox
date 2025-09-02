import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.variables.gaussian import (
    IndependentMultivariateGaussian,
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import GaussianMarginal
from pyapprox.surrogates.affine.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.bayes.variational.flows import Flow, RealNVPLayer
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)


class TestFlows:
    def setUp(self):
        np.random.seed(1)

    def _setup_2_layer_polynomial_real_nvp(
        self, nvars, layer_degrees, layer_coefs=None
    ):
        bkd = self.get_backend()
        latent_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars)],
            backend=bkd,
        )
        # The number of QoI must be 2 * the input dimension
        # which we assume is nvars // 2, so nqoi = nvars
        assert nvars % 2 == 0
        layer_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars // 2)],
            backend=bkd,
        )
        nlayers = 2
        bexps = [
            setup_polynomial_chaos_expansion_from_variable(
                layer_variable, nvars
            )
            for ii in range(nlayers)
        ]
        if layer_coefs is None:
            layer_coefs = [None for degree in layer_degrees]
        for bexp, degree, coef in zip(bexps, layer_degrees, layer_coefs):
            bexp.basis().set_tensor_product_indices([degree] * (nvars // 2))
            if coef is None:
                # nvars because need nvars//2 for shift and nvars//2 for scale
                coef = bkd.ones(bexp.nterms() * nvars)
            bexp.set_coefficient_bounds(coef, [-10.0, 10.0])

        mask = bkd.ones(nvars, dtype=bool)
        mask[1::2] = 0
        mask_complement = ~mask
        layers = []
        for ii, bexp in enumerate(bexps):
            if ii % 2 == 0:
                layers.append(RealNVPLayer(nvars, bexp, mask=mask)),
            else:
                layers.append(RealNVPLayer(nvars, bexp, mask=mask_complement))

        return Flow(latent_variable, layers)

    def test_realnvp_independent_gaussians_sampling(self):
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
        flow = self._setup_2_layer_polynomial_real_nvp(
            nvars, [2, 2], [coef1, coef2]
        )

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

    def test_realnvp_correlated_gaussians_sampling(self):
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
        flow = self._setup_2_layer_polynomial_real_nvp(
            nvars, [2, 2], [coef1, coef2]
        )

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
        plt.show()

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
        flow = self._setup_2_layer_polynomial_real_nvp(
            nvars, [2, 2], [coef1, coef2]
        )

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


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
