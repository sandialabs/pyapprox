"""Tests for DGPMaximumLikelihoodFitter (Phase 9)."""

import numpy as np
import pytest

import networkx as nx

from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
    DGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPOptimizedFitResult,
)
from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import ZeroMean
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _make_layer(bkd, nvars=1, num_inducing=5, noise_std=0.1,
                with_likelihood=True, fixed=False, seed=42):
    rng = np.random.RandomState(seed)
    kernel = Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
        fixed=fixed,
    )
    mean = ZeroMean(bkd)
    locs = bkd.array(rng.randn(nvars, num_inducing))
    ip = InducingPoints(
        nvars=nvars,
        num_inducing=num_inducing,
        bkd=bkd,
        inducing_locations=locs,
        inducing_bounds=(-5.0, 5.0),
    )
    vd = GaussianVariationalDistribution(num_inducing, bkd)
    lik = None
    if with_likelihood:
        lik = GaussianLikelihood(noise_std, (1e-6, 1.0), bkd)
    return DGPLayer(kernel, mean, ip, vd, bkd, likelihood=lik)


class TestDGPFitterNoActiveParams:
    def test_no_optimization_when_all_fixed(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=True, seed=10)
        layer.inducing_points().hyp_list().set_all_inactive()
        layer.variational_dist().hyp_list().set_all_inactive()
        if layer.likelihood() is not None:
            layer.likelihood().hyp_list().set_all_inactive()

        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))
        data = {0: (X, y)}

        fitter = DGPMaximumLikelihoodFitter(bkd, n_propagation=1)
        result = fitter.fit(dgp, data)

        assert isinstance(result, GPOptimizedFitResult)
        assert result.optimization_result() is None


class TestDGPFitterSingleLayer:
    def test_neg_elbo_decreases(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 10))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=200, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        assert isinstance(result, GPOptimizedFitResult)
        initial_nll = float(
            bkd.to_numpy(result.neg_log_marginal_likelihood())[0, 0]
        )
        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        initial_loss = DGPELBOLoss(dgp, data, n_propagation=1)
        initial_val = float(
            bkd.to_numpy(
                initial_loss(dgp.hyp_list().get_active_values())
            )[0, 0]
        )
        assert initial_nll < initial_val

    def test_fitted_dgp_can_predict(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 10))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        fitted_dgp = result.surrogate()
        assert fitted_dgp.is_fitted()

        X_test = bkd.array(rng.randn(1, 5))
        mean = fitted_dgp.predict(X_test)
        std = fitted_dgp.predict_std(X_test)
        assert mean.shape == (1, 5)
        assert std.shape == (1, 5)
        assert np.all(np.isfinite(bkd.to_numpy(mean)))
        assert np.all(np.isfinite(bkd.to_numpy(std)))

    def test_result_predict_delegates(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 6))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=50, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        X_test = bkd.array(rng.randn(1, 3))
        bkd.assert_allclose(result.predict(X_test),
                            result.surrogate().predict(X_test),
                            rtol=1e-12)
        bkd.assert_allclose(result(X_test),
                            result.surrogate().predict(X_test),
                            rtol=1e-12)

    def test_clone_independence(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 6))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        old_vals = bkd.to_numpy(dgp.hyp_list().get_values()).copy()

        optimizer = AdamOptimizer(lr=1e-2, maxiter=50, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        fitter.fit(dgp, data)

        bkd.assert_allclose(
            dgp.hyp_list().get_values(),
            bkd.array(old_vals),
            rtol=1e-12,
        )


class TestDGPFitterChain:
    def test_chain_neg_elbo_decreases(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3,
                             with_likelihood=False, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(
            dag, {0: layer0, 1: layer1}, prop, bkd,
        )

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))
        data = {1: (X, y)}

        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        initial_loss = DGPELBOLoss(dgp, data, n_propagation=1)
        initial_val = float(
            bkd.to_numpy(
                initial_loss(dgp.hyp_list().get_active_values())
            )[0, 0]
        )

        optimizer = AdamOptimizer(lr=1e-2, maxiter=200, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        final_nll = float(
            bkd.to_numpy(result.neg_log_marginal_likelihood())[0, 0]
        )
        assert final_nll < initial_val
