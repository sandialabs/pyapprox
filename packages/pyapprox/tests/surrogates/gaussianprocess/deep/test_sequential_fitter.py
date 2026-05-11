"""Tests for SingleLayerELBOLoss and MFDGPSequentialFitter."""

import networkx as nx
import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.deep.single_layer_loss import (
    SingleLayerELBOLoss,
    TorchSingleLayerELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
    DGPELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
    MFDGPSequentialFitter,
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


def _nonlinear_a_lo(x):
    return np.sin(8.0 * np.pi * x)


def _nonlinear_a_hi(x):
    return (x - np.sqrt(2.0)) * _nonlinear_a_lo(x) ** 2


def _make_two_layer_dgp(bkd, num_inducing=8, seed=10):
    """Build a 2-level chain DGP: 0 -> 1 on [0,1] with nested inputs."""
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
    """Nonlinear-A benchmark data on [0,1] with nested high-fidelity inputs."""
    X_lo_np = np.linspace(0.0, 1.0, n_lo).reshape(1, -1)
    X_hi_np = X_lo_np[:, ::3][:, :n_hi]
    y_lo_np = _nonlinear_a_lo(X_lo_np)
    y_hi_np = _nonlinear_a_hi(X_hi_np)
    return {
        0: (bkd.array(X_lo_np), bkd.array(y_lo_np)),
        1: (bkd.array(X_hi_np), bkd.array(y_hi_np)),
    }


class TestSingleLayerELBOLossMatchesDGP:
    def test_single_layer_loss_matches_dgp_loss(self, bkd):
        """1-node DGP: SingleLayerELBOLoss must match DGPELBOLoss(n_prop=1)
        to numerical precision."""
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))

        dgp_loss = DGPELBOLoss(dgp, {0: (X, y)}, n_propagation=1)
        dgp_val = dgp_loss(dgp.hyp_list().get_active_values())

        sl_loss = SingleLayerELBOLoss(layer, X, y)
        sl_val = sl_loss(layer.hyp_list().get_active_values())

        bkd.assert_allclose(sl_val, dgp_val, rtol=1e-10, atol=1e-12)


class TestSingleLayerELBOLossBasic:
    def test_shape_and_finite(self, bkd):
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))

        loss = SingleLayerELBOLoss(layer, X, y)
        params = layer.hyp_list().get_active_values()
        result = loss(params)

        assert result.shape == (1, 1)
        assert np.isfinite(float(bkd.to_numpy(result).ravel()[0]))

    def test_no_likelihood_raises(self, bkd):
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            with_likelihood=False, seed=10)
        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        y = bkd.array(np.random.RandomState(1).randn(1, 5))
        with pytest.raises(ValueError, match="likelihood"):
            SingleLayerELBOLoss(layer, X, y)

    def test_nvars_and_nqoi(self, bkd):
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        y = bkd.array(np.random.RandomState(1).randn(1, 5))
        loss = SingleLayerELBOLoss(layer, X, y)
        assert loss.nvars() == layer.hyp_list().nactive_params()
        assert loss.nqoi() == 1


class TestTorchSingleLayerELBOLoss:
    def test_numpy_backend_raises(self, numpy_bkd):
        layer = _make_layer(numpy_bkd, nvars=1, num_inducing=3, seed=10)
        X = numpy_bkd.array(np.random.RandomState(0).randn(1, 5))
        y = numpy_bkd.array(np.random.RandomState(1).randn(1, 5))
        with pytest.raises(TypeError, match="TorchBkd"):
            TorchSingleLayerELBOLoss(layer, X, y)

    def test_jacobian_shape(self, torch_bkd):
        bkd = torch_bkd
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=False, seed=10)
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        y = bkd.array(np.sin(rng.randn(1, 5)))
        loss = TorchSingleLayerELBOLoss(layer, X, y)
        params = layer.hyp_list().get_active_values()
        jac = loss.jacobian(params)
        assert jac.shape == (1, loss.nvars())

    def test_autograd_jacobian_matches_fd(self, torch_bkd):
        bkd = torch_bkd
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=False, seed=10)
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        y = bkd.array(np.sin(rng.randn(1, 5)))
        loss = TorchSingleLayerELBOLoss(layer, X, y)
        n_active = loss.nvars()

        def value_fn(sample):
            return loss(sample[:, 0])

        def jac_fn(sample):
            return loss.jacobian(sample[:, 0])

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=n_active, fun=value_fn,
            jacobian=jac_fn, bkd=bkd,
        )

        checker = DerivativeChecker(wrapper)
        x0 = bkd.reshape(
            layer.hyp_list().get_active_values(), (n_active, 1),
        )
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0,
        )
        jac_error = errors[0]
        assert bkd.all_bool(bkd.isfinite(jac_error))
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < 1e-6, f"Jacobian error ratio: {jac_ratio}"


class TestMFDGPSequentialFitterBasic:
    def test_sequential_elbo_decreases(self, torch_bkd):
        """Verify: (a) joint DGPELBOLoss at fitted params < initial,
        (b) predictive RMSE on held-out test data < 0.5 (absolute).
        Uses 15 inducing points to resolve the 4-oscillation lo-fi function.
        """
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd, num_inducing=15)
        data = _make_two_level_data(bkd)

        initial_joint_loss = DGPELBOLoss(dgp, data, n_propagation=1)
        initial_joint_val = float(bkd.to_numpy(
            initial_joint_loss(dgp.hyp_list().get_active_values())
        )[0, 0])

        X_test_np = np.linspace(0.0, 1.0, 50).reshape(1, -1)
        X_test = bkd.array(X_test_np)
        y_test_np = _nonlinear_a_hi(X_test_np)

        optimizer = AdamOptimizer(lr=1e-2, maxiter=1000, verbosity=0)
        fitter = MFDGPSequentialFitter(bkd, optimizer=optimizer)
        result = fitter.fit(dgp, data)

        seq_nll = float(bkd.to_numpy(
            result.neg_log_marginal_likelihood()
        ).ravel()[0])
        assert np.isfinite(seq_nll)

        fitted_dgp = result.surrogate()
        fitted_joint_loss = DGPELBOLoss(fitted_dgp, data, n_propagation=1)
        fitted_joint_val = float(bkd.to_numpy(
            fitted_joint_loss(fitted_dgp.hyp_list().get_active_values())
        )[0, 0])
        assert fitted_joint_val < initial_joint_val, (
            f"Joint ELBO did not improve: {fitted_joint_val} >= "
            f"{initial_joint_val}"
        )

        fitted_pred = fitted_dgp.predict(X_test, target=1, n_propagation=20)
        fitted_rmse = float(np.sqrt(np.mean(
            (bkd.to_numpy(fitted_pred) - y_test_np) ** 2
        )))
        assert fitted_rmse < 0.5, (
            f"Fitted RMSE too high: {fitted_rmse:.4f}"
        )

    def test_parent_posteriors_propagated(self, torch_bkd):
        """After sequential fit, layer 0 should track low-fidelity data
        with training RMSE < 0.15 (noise is 0.1, target is sin(8*pi*x)).
        Uses more inducing points to resolve the 4-oscillation lo-fi function.
        """
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd, num_inducing=15)
        data = _make_two_level_data(bkd)

        optimizer = AdamOptimizer(lr=1e-2, maxiter=1000, verbosity=0)
        fitter = MFDGPSequentialFitter(bkd, optimizer=optimizer)
        result = fitter.fit(dgp, data)

        fitted_dgp = result.surrogate()
        layer0 = fitted_dgp.layers()[0]

        X_lo, y_lo = data[0]
        mean0, _ = layer0.predict_marginal(X_lo)
        residual = bkd.to_numpy(mean0) - bkd.to_numpy(y_lo)
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        assert rmse < 0.15, f"Layer 0 training RMSE too high: {rmse:.4f}"

    def test_clone_independence(self, torch_bkd):
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd)
        data = _make_two_level_data(bkd)

        old_vals = bkd.to_numpy(dgp.hyp_list().get_values()).copy()

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        fitter = MFDGPSequentialFitter(bkd, optimizer=optimizer)
        fitter.fit(dgp, data)

        bkd.assert_allclose(
            dgp.hyp_list().get_values(),
            bkd.array(old_vals),
            rtol=1e-12,
        )

    def test_no_data_nodes_skipped(self, torch_bkd):
        """Node without data stays at prior (params unchanged)."""
        bkd = torch_bkd
        dgp = _make_two_layer_dgp(bkd)
        data = {1: _make_two_level_data(bkd)[1]}

        layer0_vals_before = bkd.to_numpy(
            dgp.layers()[0].hyp_list().get_values()
        ).copy()

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        fitter = MFDGPSequentialFitter(bkd, optimizer=optimizer)
        result = fitter.fit(dgp, data)

        fitted_layer0_vals = bkd.to_numpy(
            result.surrogate().layers()[0].hyp_list().get_values()
        )
        np.testing.assert_array_equal(
            layer0_vals_before, fitted_layer0_vals,
            err_msg="Layer 0 params changed despite no data",
        )

    def test_no_active_params_skipped(self, torch_bkd):
        """All params fixed -> returns without optimization."""
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

        fitter = MFDGPSequentialFitter(bkd)
        result = fitter.fit(dgp, data)
        assert isinstance(result, GPOptimizedFitResult)
        assert result.optimization_result() is None
