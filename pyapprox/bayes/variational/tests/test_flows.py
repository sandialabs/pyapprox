import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.variables.gaussian import (
    DenseCholeskyMultivariateGaussian,
    IndependentMultivariateGaussian,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import GaussianMarginal
from pyapprox.surrogates.affine.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.bayes.variational.flows import Flow, RealNVPLayer


class TestFlows:
    def setUp(self):
        np.random.seed(1)

    def test_realnvp(self):
        bkd = self.get_backend()
        nvars = 2
        mean = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))
        mat = bkd.asarray(np.random.uniform(0, 1, (nvars, nvars)))
        cov = mat.T @ mat
        cov = bkd.diag(bkd.asarray((np.random.uniform(0, 1, (nvars)))))
        mean = bkd.zeros((nvars, 1))
        cov = bkd.eye(nvars)
        ntrain_samples = 10000
        target_variable = DenseCholeskyMultivariateGaussian(
            mean, cov, backend=bkd
        )
        train_samples = target_variable.rvs(ntrain_samples)

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
        bexp = setup_polynomial_chaos_expansion_from_variable(
            layer_variable, nvars
        )
        bexp.basis().set_tensor_product_indices([1])
        bexp.set_coefficient_bounds(
            # bkd.ones((bexp.nterms(), bexp.nqoi())).flatten(),
            bkd.zeros((bexp.nterms(), bexp.nqoi())).flatten(),
            # bkd.array([-3, 3]),
            None,
        )
        print(bexp._hyp_list)
        mask = bkd.ones(nvars, dtype=bool)
        mask[::2] = 0
        mask_complement = ~mask
        layers = [
            RealNVPLayer(bexp, mask=mask),
            RealNVPLayer(bexp, mask=mask_complement),
            # RealNVPLayer(bexp, mask=mask),
            # RealNVPLayer(bexp, mask=mask_complement),
        ]

        flow = Flow(latent_variable, layers)
        flow._loss.set_samples(train_samples)

        usamples = flow._map_to_latent(train_samples)
        recovered_samples = flow._map_from_latent(usamples)
        assert bkd.allclose(recovered_samples, train_samples)

        iterate = flow._hyp_list.get_active_opt_params()[:, None]
        errors = flow._loss.check_apply_jacobian(iterate, disp=True)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 4e-6

        import matplotlib.pyplot as plt

        # axs = plt.subplots(1, 2)[1]
        # nsamples = 1000
        # target_samples = target_variable.rvs(nsamples)
        # flow_samples = flow.rvs(nsamples)
        # target_variable.plot_pdf(
        #     axs[0], [-3, 3, -3, 3], levels=31, cmap="coolwarm"
        # )
        # # axs[0].scatter(*target_samples, alpha=0.1, color="k")
        # flow.plot_pdf(axs[1], [-3, 3, -3, 3], levels=31, cmap="coolwarm")
        # # axs[1].scatter(*flow_samples, alpha=0.1, color="k")
        # plt.show()

        flow.set_optimizer(
            flow.default_optimizer(verbosity=3, method="trust-constr")
        )
        flow.fit(train_samples)

        nsamples = 1000
        target_samples = target_variable.rvs(nsamples)
        flow_samples = flow.rvs(nsamples)

        print(flow_samples.mean(axis=1))
        print(target_variable.mean()[:, 0])
        print(target_variable.covariance())
        print(bkd.cov(flow_samples, ddof=1))

        axs = plt.subplots(1, 2)[1]
        target_variable.plot_pdf(
            axs[0], [-3, 3, -3, 3], levels=31, cmap="coolwarm"
        )
        # axs[0].scatter(*target_samples, alpha=0.1)
        flow.plot_pdf(axs[1], [-3, 3, -3, 3], levels=31, cmap="coolwarm")
        # axs[1].scatter(*flow_samples, alpha=0.1, color="k")
        plt.show()


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
