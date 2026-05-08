"""Tests for DeepGaussianProcess (Phase 6)."""

import numpy as np
import pytest

import networkx as nx

from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
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
                with_likelihood=True, seed=42):
    rng = np.random.RandomState(seed)
    kernel = Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
        fixed=True,
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
    ip.hyp_list().set_all_inactive()
    vd.hyp_list().set_all_inactive()
    if lik is not None:
        lik.hyp_list().set_all_inactive()
    return DGPLayer(kernel, mean, ip, vd, bkd, likelihood=lik)


class TestDeepGPValidation:
    def test_cycle_raises(self, bkd):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 0)])
        layer0 = _make_layer(bkd, seed=10)
        layer1 = _make_layer(bkd, seed=20)
        prop = LayerPropagator(bkd)
        with pytest.raises(ValueError, match="directed acyclic graph"):
            DeepGaussianProcess(dag, {0: layer0, 1: layer1}, prop, bkd)

    def test_missing_layer_raises(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, seed=10)
        prop = LayerPropagator(bkd)
        with pytest.raises(ValueError, match="layers keys must match"):
            DeepGaussianProcess(dag, {0: layer0}, prop, bkd)

    def test_extra_layer_raises(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer0 = _make_layer(bkd, seed=10)
        layer1 = _make_layer(bkd, seed=20)
        prop = LayerPropagator(bkd)
        with pytest.raises(ValueError, match="layers keys must match"):
            DeepGaussianProcess(dag, {0: layer0, 1: layer1}, prop, bkd)


class TestDeepGPSingleLayer:
    """Single-layer DGP must match standalone layer exactly."""

    def test_predict_matches_layer(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 8))

        dgp_mean = dgp.predict(X)
        layer_mean, _ = layer.predict_marginal(X)
        bkd.assert_allclose(dgp_mean, layer_mean, rtol=1e-12)

    def test_predict_std_matches_layer(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 8))

        dgp_std = dgp.predict_std(X)
        _, layer_var = layer.predict_marginal(X)
        layer_std = bkd.sqrt(layer_var)
        bkd.assert_allclose(dgp_std, layer_std, rtol=1e-12)


class TestDeepGPChain:
    def test_predict_shape(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=3, num_inducing=4, seed=20)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer0, 1: layer1}, prop, bkd)

        X = bkd.array(np.random.RandomState(0).randn(2, 6))
        mean = dgp.predict(X, n_propagation=5)
        std = dgp.predict_std(X, n_propagation=5)
        assert mean.shape == (1, 6)
        assert std.shape == (1, 6)

    def test_predictive_samples_shape(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer0, 1: layer1}, prop, bkd)

        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        samples = dgp.predictive_samples(X, n_samples=7)
        assert samples.shape == (7, 1, 5)


class TestDeepGPHyperparameters:
    def test_hyp_list_aggregates_all_layers(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        layers = {0: layer0, 1: layer1}
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, layers, prop, bkd)

        expected = (
            layer0.hyp_list().nparams() + layer1.hyp_list().nparams()
        )
        assert dgp.hyp_list().nparams() == expected

    def test_kl_total_is_sum(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        layers = {0: layer0, 1: layer1}
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, layers, prop, bkd)

        kl_expected = layer0.kl_to_prior() + layer1.kl_to_prior()
        bkd.assert_allclose(
            bkd.asarray([dgp.kl_total()]),
            bkd.asarray([kl_expected]),
            rtol=1e-12,
        )

    def test_kl_total_zero_at_prior(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        bkd.assert_allclose(
            bkd.asarray([dgp.kl_total()]),
            bkd.zeros((1,)),
            atol=1e-12,
        )


class TestDeepGPLifecycle:
    def test_clone_unfitted_independence(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        clone = dgp._clone_unfitted()
        old_vals = bkd.to_numpy(dgp.hyp_list().get_values()).copy()
        clone.hyp_list().set_values(
            clone.hyp_list().get_values() + bkd.ones(
                (clone.hyp_list().nparams(),)
            ) * 0.5
        )
        bkd.assert_allclose(
            dgp.hyp_list().get_values(),
            bkd.array(old_vals),
            rtol=1e-12,
        )

    def test_leaf_nodes(self, bkd):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 2), (1, 2)])
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        layer1 = _make_layer(bkd, nvars=1, num_inducing=3, seed=20)
        layer2 = _make_layer(bkd, nvars=3, num_inducing=3, seed=30)
        layers = {0: layer0, 1: layer1, 2: layer2}
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, layers, prop, bkd)

        assert dgp.leaf_nodes() == [2]

    def test_multiple_leaves_requires_target(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        layer2 = _make_layer(bkd, nvars=2, num_inducing=3, seed=30)
        layers = {0: layer0, 1: layer1, 2: layer2}
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, layers, prop, bkd)

        X = bkd.array(np.random.RandomState(0).randn(1, 4))
        with pytest.raises(ValueError, match="target must be specified"):
            dgp.predict(X)

        mean = dgp.predict(X, target=1, n_propagation=3)
        assert mean.shape == (1, 4)

    def test_set_propagator(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop1 = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop1, bkd)

        prop2 = LayerPropagator(bkd)
        dgp.set_propagator(prop2)
        assert dgp.propagator() is prop2

    def test_repr(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer0, 1: layer1}, prop, bkd)
        r = repr(dgp)
        assert "nodes=2" in r
        assert "edges=1" in r
