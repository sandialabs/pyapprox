"""Tests for Deep GP builder functions (Phase 11)."""

import numpy as np
import pytest

from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer
from pyapprox.surrogates.gaussianprocess.deep.builders import (
    build_single_fidelity_dgp,
    build_multilevel_dgp,
)
from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
    DGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _matern_factory(nvars, bkd):
    return Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
    )


class TestBuildSingleFidelityDGP:
    def test_single_layer_structure(self, bkd):
        dgp = build_single_fidelity_dgp(1, nvars=2, num_inducing=5,
                               kernel_factory=_matern_factory, bkd=bkd)
        assert dgp.dag().number_of_nodes() == 1
        assert dgp.dag().number_of_edges() == 0
        assert len(dgp.leaf_nodes()) == 1
        assert dgp.layers()[0].likelihood() is not None

    def test_two_layer_structure(self, bkd):
        dgp = build_single_fidelity_dgp(2, nvars=3, num_inducing=4,
                               kernel_factory=_matern_factory, bkd=bkd)
        assert dgp.dag().number_of_nodes() == 2
        assert dgp.dag().number_of_edges() == 1
        assert dgp.layers()[0].likelihood() is None
        assert dgp.layers()[1].likelihood() is not None

    def test_three_layer_chain_topology(self, bkd):
        dgp = build_single_fidelity_dgp(3, nvars=1, num_inducing=3,
                               kernel_factory=_matern_factory, bkd=bkd)
        dag = dgp.dag()
        assert dag.number_of_nodes() == 3
        assert list(dag.edges()) == [(0, 1), (1, 2)]
        assert dgp.layers()[0].likelihood() is None
        assert dgp.layers()[1].likelihood() is None
        assert dgp.layers()[2].likelihood() is not None

    def test_hidden_layer_input_dim_includes_parent(self, bkd):
        nvars = 2
        dgp = build_single_fidelity_dgp(2, nvars=nvars, num_inducing=4,
                               kernel_factory=_matern_factory, bkd=bkd)
        layer0_ip = dgp.layers()[0].inducing_points()
        layer1_ip = dgp.layers()[1].inducing_points()
        assert layer0_ip.nvars() == nvars
        assert layer1_ip.nvars() == nvars + 1

    def test_invalid_n_layers_raises(self, numpy_bkd):
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            build_single_fidelity_dgp(0, nvars=1, num_inducing=3,
                            kernel_factory=_matern_factory, bkd=numpy_bkd)

    def test_can_predict(self, bkd):
        dgp = build_single_fidelity_dgp(2, nvars=1, num_inducing=4,
                               kernel_factory=_matern_factory, bkd=bkd,
                               n_propagation=2)
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        mean = dgp.predict(X)
        std = dgp.predict_std(X)
        assert mean.shape == (1, 5)
        assert std.shape == (1, 5)
        assert np.all(np.isfinite(bkd.to_numpy(mean)))
        assert np.all(np.isfinite(bkd.to_numpy(std)))


class TestBuildMultilevelDGP:
    def test_two_level_structure(self, bkd):
        dgp = build_multilevel_dgp(
            level_nvars=[2, 2], num_inducing=4,
            kernel_factory=_matern_factory, bkd=bkd,
        )
        assert dgp.dag().number_of_nodes() == 2
        assert dgp.dag().number_of_edges() == 1
        assert dgp.layers()[0].likelihood() is not None
        assert dgp.layers()[1].likelihood() is not None

    def test_three_level_chain(self, bkd):
        dgp = build_multilevel_dgp(
            level_nvars=[1, 1, 1], num_inducing=3,
            kernel_factory=_matern_factory, bkd=bkd,
        )
        assert dgp.dag().number_of_nodes() == 3
        assert list(dgp.dag().edges()) == [(0, 1), (1, 2)]
        for i in range(3):
            assert dgp.layers()[i].likelihood() is not None

    def test_child_layer_nvars_includes_parent(self, bkd):
        dgp = build_multilevel_dgp(
            level_nvars=[2, 2], num_inducing=4,
            kernel_factory=_matern_factory, bkd=bkd,
        )
        assert dgp.layers()[0].inducing_points().nvars() == 2
        assert dgp.layers()[1].inducing_points().nvars() == 3

    def test_invalid_empty_raises(self, numpy_bkd):
        with pytest.raises(ValueError, match="at least 1"):
            build_multilevel_dgp(
                level_nvars=[], num_inducing=3,
                kernel_factory=_matern_factory, bkd=numpy_bkd,
            )

    def test_missing_data_raises(self, bkd):
        """Multilevel DGP requires data at every level."""
        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        dgp = build_multilevel_dgp(
            level_nvars=[1, 1], num_inducing=3,
            kernel_factory=_matern_factory, bkd=bkd,
        )
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        y = bkd.array(rng.randn(1, 5))
        with pytest.raises(ValueError, match="no training data"):
            DGPELBOLoss(dgp, {1: (X, y)}, n_propagation=1)

    def test_can_predict(self, bkd):
        dgp = build_multilevel_dgp(
            level_nvars=[1, 1], num_inducing=4,
            kernel_factory=_matern_factory, bkd=bkd,
            n_propagation=2,
        )
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        mean = dgp.predict(X)
        assert mean.shape == (1, 5)
        assert np.all(np.isfinite(bkd.to_numpy(mean)))


class TestBuilderFitAndPredict:
    def test_chain_dgp_fit_and_predict(self, torch_bkd):
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        dgp = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=5,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.1, n_propagation=1,
        )

        X_train = bkd.array(rng.randn(1, 10))
        y_train = bkd.array(np.sin(rng.randn(1, 10)))
        data = {1: (X_train, y_train)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        fitted = result.surrogate()
        assert fitted.is_fitted()

        X_test = bkd.array(rng.randn(1, 5))
        mean = fitted.predict(X_test)
        std = fitted.predict_std(X_test)
        assert mean.shape == (1, 5)
        assert std.shape == (1, 5)
        assert np.all(np.isfinite(bkd.to_numpy(mean)))
        assert np.all(np.isfinite(bkd.to_numpy(std)))

    def test_multilevel_dgp_fit_and_predict(self, torch_bkd):
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        dgp = build_multilevel_dgp(
            level_nvars=[1, 1], num_inducing=5,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.1, n_propagation=1,
        )

        X_lo = bkd.array(rng.randn(1, 15))
        y_lo = bkd.array(np.sin(rng.randn(1, 15)))
        X_hi = bkd.array(rng.randn(1, 8))
        y_hi = bkd.array(np.sin(rng.randn(1, 8)))
        data = {0: (X_lo, y_lo), 1: (X_hi, y_hi)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        fitted = result.surrogate()
        assert fitted.is_fitted()
        X_test = bkd.array(rng.randn(1, 3))
        mean = fitted.predict(X_test)
        assert mean.shape == (1, 3)
        assert np.all(np.isfinite(bkd.to_numpy(mean)))
