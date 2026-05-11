"""Tests for DGPFitterChain and DGPChainedFitResult."""

import networkx as nx
import numpy as np
import pytest
from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.fitters.composition import (
    DGPChainedFitResult,
    DGPFitterChain,
)
from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
    DGPMaximumLikelihoodFitter,
    MFDGPSequentialFitter,
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


def _nonlinear_a_lo(x):
    return np.sin(8.0 * np.pi * x)


def _nonlinear_a_hi(x):
    return (x - np.sqrt(2.0)) * _nonlinear_a_lo(x) ** 2


def _make_two_layer_dgp(bkd, num_inducing=8, seed=10):
    rng = np.random.RandomState(seed)
    dag = nx.DiGraph()
    dag.add_edge(0, 1)
    locs0 = bkd.array(rng.rand(1, num_inducing))
    ip0 = InducingPoints(
        nvars=1, num_inducing=num_inducing, bkd=bkd,
        inducing_locations=locs0, inducing_bounds=(0.0, 1.0),
    )
    kernel0 = Matern52Kernel(
        lenscale=[0.3], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    vd0 = GaussianVariationalDistribution(num_inducing, bkd)
    lik0 = GaussianLikelihood(0.1, (1e-6, 1.0), bkd)
    layer0 = DGPLayer(kernel0, ZeroMean(bkd), ip0, vd0, bkd, likelihood=lik0)

    locs1 = bkd.array(rng.rand(2, num_inducing))
    ip1 = InducingPoints(
        nvars=2, num_inducing=num_inducing, bkd=bkd,
        inducing_locations=locs1, inducing_bounds=(0.0, 1.0),
    )
    kernel1 = Matern52Kernel(
        lenscale=[0.3, 0.3], lenscale_bounds=(0.01, 5.0), nvars=2, bkd=bkd,
    )
    vd1 = GaussianVariationalDistribution(num_inducing, bkd)
    lik1 = GaussianLikelihood(0.1, (1e-6, 1.0), bkd)
    layer1 = DGPLayer(kernel1, ZeroMean(bkd), ip1, vd1, bkd, likelihood=lik1)

    prop = LayerPropagator(bkd)
    return DeepGaussianProcess(dag, {0: layer0, 1: layer1}, prop, bkd)


def _make_two_level_data(bkd, n_lo=30, n_hi=10):
    X_lo_np = np.linspace(0.0, 1.0, n_lo).reshape(1, -1)
    X_hi_np = X_lo_np[:, ::3][:, :n_hi]
    y_lo_np = _nonlinear_a_lo(X_lo_np)
    y_hi_np = _nonlinear_a_hi(X_hi_np)
    return {
        0: (bkd.array(X_lo_np), bkd.array(y_lo_np)),
        1: (bkd.array(X_hi_np), bkd.array(y_hi_np)),
    }


class TestDGPFitterChainValidation:
    def test_empty_chain_raises(self, torch_bkd):
        with pytest.raises(ValueError, match="at least one fitter"):
            DGPFitterChain([])


class TestDGPFitterChainThreading:
    def test_chain_threads_model(self, torch_bkd):
        """Second fitter's initial hyperparameters must match first fitter's
        final optimized hyperparameters."""
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd)
        data = _make_two_level_data(bkd)

        initial_vals = bkd.to_numpy(
            dgp.hyp_list().get_values()
        ).copy()

        optimizer = AdamOptimizer(lr=1e-2, maxiter=200, verbosity=0)
        seq_fitter = MFDGPSequentialFitter(bkd, optimizer=optimizer)
        seq_result = seq_fitter.fit(dgp, data)
        seq_final_vals = bkd.to_numpy(
            seq_result.surrogate().hyp_list().get_values()
        ).copy()

        assert not np.allclose(initial_vals, seq_final_vals, rtol=1e-6), (
            "Sequential fitter did not change any parameters"
        )

        chain = DGPFitterChain([
            MFDGPSequentialFitter(bkd, optimizer=optimizer),
            DGPMaximumLikelihoodFitter(
                bkd, optimizer=optimizer, n_propagation=1,
            ),
        ])
        chain_result = chain.fit(dgp, data)

        seq_stage_vals = bkd.to_numpy(
            chain_result.intermediate_results()[0].surrogate().hyp_list().get_values()
        )
        bkd.assert_allclose(
            bkd.array(seq_stage_vals),
            bkd.array(seq_final_vals),
            rtol=1e-10,
        )

    def test_chain_intermediate_results(self, torch_bkd):
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd)
        data = _make_two_level_data(bkd)

        optimizer = AdamOptimizer(lr=1e-2, maxiter=50, verbosity=0)
        chain = DGPFitterChain([
            MFDGPSequentialFitter(bkd, optimizer=optimizer),
            DGPMaximumLikelihoodFitter(
                bkd, optimizer=optimizer, n_propagation=1,
            ),
        ])
        result = chain.fit(dgp, data)

        assert isinstance(result, DGPChainedFitResult)
        assert len(result.intermediate_results()) == 2

    def test_chain_runs_in_order(self, torch_bkd):
        """Verify each fitter receives the previous fitter's output model,
        not the original."""
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd)
        data = _make_two_level_data(bkd)

        initial_vals = bkd.to_numpy(dgp.hyp_list().get_values()).copy()

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        chain = DGPFitterChain([
            MFDGPSequentialFitter(bkd, optimizer=optimizer),
            DGPMaximumLikelihoodFitter(
                bkd, optimizer=optimizer, n_propagation=1,
            ),
        ])
        result = chain.fit(dgp, data)

        intermediates = result.intermediate_results()
        stage0_vals = bkd.to_numpy(
            intermediates[0].surrogate().hyp_list().get_values()
        )
        stage1_vals = bkd.to_numpy(
            intermediates[1].surrogate().hyp_list().get_values()
        )
        assert not np.allclose(initial_vals, stage0_vals, rtol=1e-6), (
            "Stage 0 did not change params from initial"
        )
        assert not np.allclose(stage0_vals, stage1_vals, rtol=1e-6), (
            "Stage 1 did not change params from stage 0"
        )


class TestDGPFitterChainIntegration:
    @pytest.mark.slow_on("*")
    def test_chain_with_dgp_fitters(self, torch_bkd):
        """Full integration: DGPFitterChain([Sequential, Joint]) on 2-level
        data. Verify: (a) produces fitted model, (b) second fitter started
        from first fitter's optimized state, (c) final RMSE is reasonable."""
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd)
        data = _make_two_level_data(bkd)

        initial_vals = bkd.to_numpy(
            dgp.hyp_list().get_values()
        ).copy()

        seq_opt = AdamOptimizer(lr=1e-2, maxiter=300, verbosity=0)
        joint_opt = AdamOptimizer(lr=1e-3, maxiter=200, verbosity=0)

        chain = DGPFitterChain([
            MFDGPSequentialFitter(bkd, optimizer=seq_opt),
            DGPMaximumLikelihoodFitter(
                bkd, optimizer=joint_opt, n_propagation=3,
            ),
        ])
        result = chain.fit(dgp, data)

        fitted_dgp = result.surrogate()
        assert fitted_dgp.is_fitted()

        final_vals = bkd.to_numpy(
            fitted_dgp.hyp_list().get_values()
        )
        assert not np.allclose(initial_vals, final_vals, rtol=1e-6)

        intermediates = result.intermediate_results()
        seq_vals = bkd.to_numpy(
            intermediates[0].surrogate().hyp_list().get_values()
        )
        assert not np.allclose(initial_vals, seq_vals, rtol=1e-6), (
            "Sequential stage did not change params"
        )

        X_test_np = np.linspace(0.0, 1.0, 50).reshape(1, -1)
        X_test = bkd.array(X_test_np)
        y_test_np = _nonlinear_a_hi(X_test_np)
        pred = bkd.to_numpy(
            fitted_dgp.predict(X_test, target=1, n_propagation=10)
        )
        rmse = float(np.sqrt(np.mean((pred - y_test_np) ** 2)))
        assert rmse < 0.5, f"Chain RMSE too high: {rmse}"
