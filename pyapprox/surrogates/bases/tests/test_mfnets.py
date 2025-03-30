import unittest

import numpy as np
import networkx as nx
from scipy import stats

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.mfnets import (
    MultiplicativeAndAdditiveDiscrepancyModel,
    MFNetModel,
    LeafMFNetNode,
    RootMFNetNode,
    MFNetEdge,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.basisexp import (
    HyperbolicMonomialExpansion,
    Monomial1D,
)
from pyapprox.interface.model import ModelFromVectorizedCallable


class TestMFNets:
    def setUp(self):
        np.random.seed(1)

    def _setup_multiplicative_additive_discrepancy(
        self, nvars, ndelta_qoi, nscaled_qoi, delta_level, scaling_level
    ):
        bkd = self.get_backend()
        delta_bases_1d = [Monomial1D(backend=bkd) for ii in range(nvars)]
        delta = HyperbolicMonomialExpansion(
            delta_bases_1d, delta_level, nqoi=ndelta_qoi
        )
        delta.set_coefficients(bkd.ones((delta.nterms(), delta.nqoi())))
        scaling_bases_1d = [Monomial1D(backend=bkd) for ii in range(nvars)]
        scalings = [
            HyperbolicMonomialExpansion(
                scaling_bases_1d, scaling_level, nqoi=nscaled_qoi
            )
            for ii in range(ndelta_qoi)
        ]
        for ii in range(ndelta_qoi):
            scalings[ii].set_coefficients(
                bkd.ones((scalings[ii].nterms(), scalings[ii].nqoi()))
            )
        return MultiplicativeAndAdditiveDiscrepancyModel(
            scalings, delta, nscaled_qoi
        )

    def test_multiplicative_additive_discrepancy_model_evaluation(self):
        bkd = self.get_backend()
        nvars = 1
        nscaled_qoi = 2
        model0_level = 4
        delta_level, scaling_level = 3, 1
        ndelta_qoi = 3
        node_models = [
            HyperbolicMonomialExpansion(
                [Monomial1D(backend=bkd) for ii in range(nvars)],
                model0_level,
                nqoi=nscaled_qoi,
            ),
            self._setup_multiplicative_additive_discrepancy(
                nvars, ndelta_qoi, nscaled_qoi, delta_level, scaling_level
            ),
        ]

        marginals = [stats.uniform(0, 1)] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=bkd)

        nsamples = 10
        test_samples = variable.rvs(nsamples)
        inputs = bkd.vstack((test_samples, node_models[0](test_samples).T))
        test_values = node_models[1](inputs)
        # models[1].check_apply_jacobian(inputs[:, :1])
        # assert False

        mfnet = MFNetModel(nvars, nx.DiGraph(), backend=bkd)

        nodes = [
            LeafMFNetNode(0, node_models[0], 1.0),
            RootMFNetNode(1, node_models[1], 1.0),
        ]

        mfnet.add_node(nodes[0])  # lf
        mfnet.add_node(nodes[1])  # hf
        mfnet.add_edge(MFNetEdge(nodes[0], nodes[1]))  # edge lf-> hf
        mfnet.validate()

        assert bkd.allclose(mfnet(test_samples), test_values)

    def _setup_multilevel_model_ensemble(self, degree, nmodels):
        bkd = self.get_backend()
        rho = bkd.full(((nmodels - 1) * (degree + 1),), 0.9)

        def scale(x, rho, kk):
            if degree == 0:
                return rho[kk]
            return rho[2 * kk] + x.T * rho[2 * kk + 1]

        def f1(x):
            y = x[0:1].T
            return ((y * 6 - 2) ** 2) / 5

        def f2(x):
            y = x[0:1].T
            return scale(x, rho, 0) * f1(x) + y * 5 / 10

        def f3(x):
            y = x[0:1].T
            return scale(x, rho, -1) * f2(x) + ((y - 0.5) * 1.0 - 5) / 5

        model1 = ModelFromVectorizedCallable(1, 1, f1, backend=bkd)
        model2 = ModelFromVectorizedCallable(1, 1, f2, backend=bkd)
        model3 = ModelFromVectorizedCallable(1, 1, f3, backend=bkd)
        if nmodels == 3:
            return rho, (model1, model2, model3)
        return rho, (model1, model2)

    def test_multiplicative_additive_discrepancy_model_fit(self):
        bkd = self.get_backend()
        nvars = 1
        nscaled_qoi = 1
        model0_level = 5
        delta_level, scaling_level = 5, 0
        ndelta_qoi = 1
        node_models = [
            HyperbolicMonomialExpansion(
                [Monomial1D(backend=bkd) for ii in range(nvars)],
                model0_level,
                nqoi=nscaled_qoi,
            ),
            self._setup_multiplicative_additive_discrepancy(
                nvars, ndelta_qoi, nscaled_qoi, delta_level, scaling_level
            ),
        ]

        marginals = [stats.uniform(0, 1)] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=bkd)

        mfnet = MFNetModel(nvars, nx.DiGraph(), backend=bkd)

        nodes = [
            LeafMFNetNode(0, node_models[0], 1.0e-8),
            RootMFNetNode(1, node_models[1], 1.0e-8),
        ]

        mfnet.add_node(nodes[0])  # lf
        mfnet.add_node(nodes[1])  # hf
        mfnet.add_edge(MFNetEdge(nodes[0], nodes[1]))  # edge lf-> hf
        mfnet.validate()

        nsamples = 10
        test_samples = variable.rvs(nsamples)

        degree, nmodels = scaling_level, 2
        rho, models = self._setup_multilevel_model_ensemble(degree, nmodels)

        ntrain_samples_per_model = [9, 7, 5][:nmodels]
        train_samples_per_model = [
            bkd.linspace(0, 1, nsamples)[None, :]
            for nsamples in ntrain_samples_per_model
        ]
        train_values_per_model = [
            model(samples)
            for model, samples in zip(models, train_samples_per_model)
        ]
        print(node_models[1]._scalings[0].get_coefficients())
        mfnet.fit(train_samples_per_model, train_values_per_model)

        test_values_per_model = [model(test_samples) for model in models]
        # print(mfnet._subgraph_values(test_samples, 0))
        # print(test_values_per_model[0])
        # print(mfnet(test_samples))
        # print(test_values_per_model[-1])
        # print(mfnet(test_samples) - test_values_per_model[-1])
        print(node_models[1]._delta.get_coefficients())
        print(node_models[1]._scalings[0].get_coefficients())

        assert bkd.allclose(
            mfnet._subgraph_values(test_samples, 0),
            test_values_per_model[0],
            atol=3e-5,
        )
        assert bkd.allclose(
            mfnet(test_samples), test_values_per_model[-1], atol=3e-5
        )
        #  TODO test different numbers of models and graphs

    def test_alternating_least_squares(self):
        # also test co-regionalization like graphs with latent kernels with no
        # data and mulitple root nodes.
        # Use genetic algorithms to get initial guess for other types of node models when I implement them
        raise NotImplementedError(
            "TODO implement alternating least squares optimization for additive multiplicative node models"
        )


# class TestNumpyMFNets(TestMFNets, unittest.TestCase):
#     def get_backend(self):
#         return NumpyLinAlgMixin


class TestTorchMFNets(TestMFNets, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


# class TestJaxMFNets(TestMFNets, unittest.TestCase):
#     def setUp(self):
#         if not package_available("jax"):
#             self.skipTest("jax not available")
#         TestMFNets.setUp(self)

#     def get_backend(self):
#         return JaxLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
