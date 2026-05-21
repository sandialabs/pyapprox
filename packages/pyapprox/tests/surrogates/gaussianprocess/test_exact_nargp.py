"""Tests for ExactNARGPFitter and ExactNARGPModel."""

import networkx as nx
import numpy as np
import pytest
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.exact_nargp import (
    ExactNARGPModel,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    ParentPassthroughMean,
    ZeroMean,
)
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel

from pyapprox.surrogates.gaussianprocess.fitters import (
    ExactNARGPFitResult,
    ExactNARGPFitter,
    GPMaximumLikelihoodFitter,
)


def _se_factory(nvars, bkd):
    return SquaredExponentialKernel(
        lenscale=[0.3] * nvars,
        lenscale_bounds=(0.01, 10.0),
        nvars=nvars,
        bkd=bkd,
    )


def _make_2level_data(bkd):
    x = np.linspace(0, 1, 30)
    y_low = np.sin(8 * np.pi * x)
    x_high = x[::3]
    y_high = (x_high - np.sqrt(2.0)) * np.sin(8 * np.pi * x_high) ** 2
    X_low = bkd.array(x.reshape(1, -1))
    Y_low = bkd.array(y_low.reshape(1, -1))
    X_high = bkd.array(x_high.reshape(1, -1))
    Y_high = bkd.array(y_high.reshape(1, -1))
    return X_low, Y_low, X_high, Y_high


def _make_2level_dag():
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1])
    dag.add_edge(0, 1)
    return dag


class TestExactNARGPFitter:
    def test_fit_produces_model(self, bkd):
        X_low, Y_low, X_high, Y_high = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low), 1: (X_high, Y_high)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        result = fitter.fit(dag, data)

        assert isinstance(result, ExactNARGPFitResult)
        model = result.surrogate()
        assert isinstance(model, ExactNARGPModel)
        assert len(model.gps()) == 2
        assert 0 in model.gps()
        assert 1 in model.gps()

    def test_root_layer_fits_data(self, bkd):
        X_low, Y_low, X_high, Y_high = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low), 1: (X_high, Y_high)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        result = fitter.fit(dag, data)
        model = result.surrogate()

        pred = model.gps()[0].predict(X_low)
        residual = bkd.to_numpy(pred) - bkd.to_numpy(Y_low)
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        assert rmse < 0.15

    def test_predict_shape(self, bkd):
        X_low, Y_low, X_high, Y_high = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low), 1: (X_high, Y_high)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        model = fitter.fit(dag, data).surrogate()

        n_test = 15
        X_test = bkd.array(np.linspace(0, 1, n_test).reshape(1, -1))
        mean = model.predict(X_test)
        std = model.predict_std(X_test)

        assert mean.shape == (1, n_test)
        assert std.shape == (1, n_test)

    def test_improves_over_baseline(self, bkd):
        X_low, Y_low, X_high, Y_high = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low), 1: (X_high, Y_high)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        model = fitter.fit(dag, data).surrogate()

        x_test = np.linspace(0, 1, 100)
        X_test = bkd.array(x_test.reshape(1, -1))
        y_truth = (x_test - np.sqrt(2.0)) * np.sin(8 * np.pi * x_test) ** 2

        nargp_mean = bkd.to_numpy(model.predict(X_test)).ravel()
        nargp_rmse = float(np.sqrt(np.mean((nargp_mean - y_truth) ** 2)))

        baseline_gp = ExactGaussianProcess(
            _se_factory(1, bkd), 1, bkd, nugget=1e-6,
        )
        result = GPMaximumLikelihoodFitter(bkd).fit(
            baseline_gp, X_high, Y_high
        )
        baseline_gp = result.surrogate()
        baseline_mean = bkd.to_numpy(baseline_gp.predict(X_test)).ravel()
        baseline_rmse = float(np.sqrt(np.mean((baseline_mean - y_truth) ** 2)))

        assert nargp_rmse < baseline_rmse

    def test_skips_nodes_without_data(self, bkd):
        X_low, Y_low, _, _ = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        result = fitter.fit(dag, data)
        model = result.surrogate()

        assert 0 in model.gps()
        assert 1 not in model.gps()

    def test_neg_log_marginal_likelihood(self, bkd):
        X_low, Y_low, X_high, Y_high = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low), 1: (X_high, Y_high)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        result = fitter.fit(dag, data)

        nll = result.neg_log_marginal_likelihood()
        assert nll.shape == (1, 1)
        assert np.isfinite(float(bkd.to_numpy(nll).ravel()[0]))

        per_layer = result.per_layer_nll()
        assert len(per_layer) == 2

    def test_mean_functions(self, bkd):
        X_low, Y_low, X_high, Y_high = _make_2level_data(bkd)
        dag = _make_2level_dag()
        data = {0: (X_low, Y_low), 1: (X_high, Y_high)}

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        model = fitter.fit(dag, data).surrogate()

        assert isinstance(model.gps()[0].mean(), ZeroMean)
        assert isinstance(model.gps()[1].mean(), ParentPassthroughMean)


class TestExactNARGPModel3Level:
    def test_3level_chain(self, bkd):
        x = np.linspace(0, 1, 40)
        y0 = np.sin(4 * np.pi * x)
        x1 = x[::2]
        y1 = np.sin(4 * np.pi * x1) ** 2
        x2 = x1[::2]
        y2 = (x2 - 0.5) * np.sin(4 * np.pi * x2) ** 2

        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 2)])

        data = {
            0: (bkd.array(x.reshape(1, -1)),
                bkd.array(y0.reshape(1, -1))),
            1: (bkd.array(x1.reshape(1, -1)),
                bkd.array(y1.reshape(1, -1))),
            2: (bkd.array(x2.reshape(1, -1)),
                bkd.array(y2.reshape(1, -1))),
        }

        fitter = ExactNARGPFitter(bkd, _se_factory, nvars=1)
        model = fitter.fit(dag, data).surrogate()

        assert len(model.gps()) == 3
        X_test = bkd.array(np.linspace(0, 1, 20).reshape(1, -1))
        mean = model.predict(X_test)
        std = model.predict_std(X_test)
        assert mean.shape == (1, 20)
        assert std.shape == (1, 20)
