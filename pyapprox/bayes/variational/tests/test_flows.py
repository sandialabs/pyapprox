import unittest

import numpy as np

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
        ntrain_samples = 100
        target_variable = DenseCholeskyMultivariateGaussian(
            mean, cov, backend=bkd
        )
        train_samples = target_variable.rvs(ntrain_samples)

        latent_variable = IndependentMarginalsVariable(
            [GaussianMarginal(0, 1, bkd) for ii in range(nvars)], backend=bkd
        )
        bexp = setup_polynomial_chaos_expansion_from_variable(
            latent_variable, nvars * 2
        )
        bexp.basis().set_tensor_product_indices([2, 2])
        bexp.set_coefficient_bounds(
            bkd.ones((bexp.nterms(), bexp.nqoi())).flatten(), None
        )
        print(bexp._hyp_list)
        layers = [RealNVPLayer(bexp, mask=bkd.ones(nvars, dtype=bool))]
        flow = Flow(latent_variable, layers)
        flow.fit(train_samples)
        print(flow._opt_result)


class TestTorchFlows(TestFlows, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
