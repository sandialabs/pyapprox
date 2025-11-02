import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.optimization.sampleaverage import (
    SampleAverageMean,
    SampleAverageVariance,
    SampleAverageStdev,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
    SampleAverageSmoothedAverageValueAtRisk,
    SampleAverageSmoothedAverageValueAtRiskDeviation,
    SampleAverageConstraint,
)
from pyapprox.optimization.risk import (
    GaussianAnalyticalRiskMeasures,
    AverageValueAtRisk,
)
from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.benchmarks import CantileverBeamUncertainOptimizationBenchmark


class TestSampleAverage:
    def setUp(self):
        np.random.seed(1)

    def _check_sample_average_stat(self, stat, attr_name, attr_args):
        bkd = self.get_backend()
        mu, sigma = 0.5, 1.0
        risks = [
            GaussianAnalyticalRiskMeasures(mu, sigma),
            GaussianAnalyticalRiskMeasures(mu, 2 * sigma),
        ]
        exact_values = bkd.asarray(
            [getattr(risk, attr_name)(*attr_args) for risk in risks]
        )
        rvs = [stats.norm(mu, sigma), stats.norm(mu, 2 * sigma)]
        nsamples = int(1e6)
        values = bkd.stack(
            [bkd.asarray(rv.rvs(nsamples)) for rv in rvs], axis=1
        )
        weights = bkd.full((nsamples, 1), 1.0 / nsamples)
        estimate = stat(values, weights)
        # print(estimate, exact_values)
        assert bkd.allclose(estimate, exact_values, rtol=1e-2)

        # test jacobians
        nqoi, nvars = 2, 2

        # create a parameterized model that scales a set of samples from
        # the two gaussian distributions above
        def param_fun(p):
            return bkd.flatten(p * values.T)[None, :]

        def param_jac(p):
            jac = bkd.zeros((nqoi * nsamples, nvars))
            jac[:nsamples, 0] = values[:, 0]
            jac[nsamples:, 1] = values[:, 1]
            return jac

        model = ModelFromSingleSampleCallable(
            2 * nsamples, 2, param_fun, param_jac, backend=bkd
        )
        # check its jacobian is correct
        param = bkd.array([2.0, 3.0])[:, None]
        errors = model.check_apply_jacobian(param)
        assert errors.min() / errors.max() < 1e-6

        # create a model that estimates the statistic
        def fun(p):
            return stat(bkd.reshape(model(p), (nsamples, nqoi)), weights)

        def jac(p):
            vals = bkd.reshape(model(p), (nsamples, nqoi))
            jac_vals = model.jacobian(p)
            return stat.jacobian(
                vals, bkd.reshape(jac_vals, (nsamples, nqoi, nvars)), weights
            )

        stat_model = ModelFromSingleSampleCallable(
            nqoi, nvars, fun, jac, backend=bkd
        )
        errors = stat_model.check_apply_jacobian(param)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 3e-6

    def test_sample_average_stats(self):
        bkd = self.get_backend()
        for test_case in [
            [SampleAverageMean(backend=bkd), "mean", []],
            [SampleAverageVariance(backend=bkd), "variance", []],
            [SampleAverageStdev(backend=bkd), "stdev", []],
            [
                SampleAverageMeanPlusStdev(2, backend=bkd),
                "mean_plus_stddev",
                [2.0],
            ],
            [
                SampleAverageEntropicRisk(0.5, backend=bkd),
                "entropic",
                [0.5],
            ],
            [
                SampleAverageSmoothedAverageValueAtRisk(0.5, bkd, 100000),
                "AVaR",
                [0.5],
            ],
        ]:
            self._check_sample_average_stat(*test_case)

    def test_sample_average_constraints(self):
        bkd = self.get_backend()
        benchmark = CantileverBeamUncertainOptimizationBenchmark(backend=bkd)
        constraint_model = benchmark.constraints()[0]._model

        # test jacobian and hessian
        nsamples = 1000
        samples = benchmark.prior().rvs(nsamples)
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        for stat in [
            SampleAverageMean(backend=bkd),
            SampleAverageVariance(backend=bkd),
            SampleAverageStdev(backend=bkd),
            SampleAverageMeanPlusStdev(2, backend=bkd),
            SampleAverageEntropicRisk(0.5, backend=bkd),
        ]:
            constraint_bounds = bkd.hstack(
                [bkd.zeros((2, 1)), bkd.full((2, 1), np.inf)]
            )
            constraint = SampleAverageConstraint(
                constraint_model,
                samples,
                weights,
                stat,
                constraint_bounds,
                benchmark.prior().nvars()
                + benchmark.design_variable().nvars(),
                benchmark.design_var_indices(),
                backend=bkd,
            )
            design_sample = bkd.array([3.0, 3.0])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

            if not stat.hessian_implemented():
                continue
            # assert False
            errors = constraint.check_apply_hessian(
                design_sample, weights=bkd.ones((constraint.nqoi(), 1))
            )
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

    def test_smoothed_average_value_at_risk(self):
        bkd = self.get_backend()
        mu, sigma, beta = 0, 1, 0.5
        risks = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = bkd.asarray(risks.AVaR(beta))
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.norm(mu, sigma)
        nsamples = int(1e5)
        samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        AVaR.set_samples(samples)
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        smooth_avar = SampleAverageSmoothedAverageValueAtRisk(
            alpha=beta, delta=100000, backend=bkd
        )
        nsamples = int(1e6)
        # samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        # a more accurate ansswer can be obtained by using 1D
        # halton like sequence, i.e. equidistant points
        samples = bkd.asarray(rv.ppf(bkd.linspace(1e-6, 1 - 1e-6, nsamples)))[
            None, :
        ]
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        # print(smooth_avar(samples.T, weights) - exact_avar, exact_avar)
        assert bkd.allclose(
            smooth_avar(samples.T, weights), exact_avar, rtol=2e-5
        )

        # Test AVAR with multiple qoi. Use positive homogeneity R(tZ) = tR(Z)
        # to compute true answer
        samples = bkd.vstack([samples, 2 * samples])
        assert bkd.allclose(
            smooth_avar(samples.T, weights),
            bkd.asarray([exact_avar, 2 * exact_avar]),
            rtol=2e-5,
        )

        # Test avar with importance sampling
        smooth_avar = SampleAverageSmoothedAverageValueAtRisk(
            alpha=beta, delta=100000, backend=bkd
        )
        nsamples = int(1e6)
        # samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        # a more accurate ansswer can be obtained by using 1D
        # halton like sequence, i.e. equidistant points
        dominating_rv = stats.norm(mu, sigma * 2)
        samples = bkd.asarray(
            dominating_rv.ppf(bkd.linspace(1e-6, 1 - 1e-6, nsamples))
        )[None, :]
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        weights *= bkd.asarray(
            rv.pdf(samples.T) / dominating_rv.pdf(samples.T)
        )
        # print(smooth_avar(samples.T, weights) - exact_avar, exact_avar)
        assert bkd.allclose(
            smooth_avar(samples.T, weights), exact_avar, rtol=2e-5
        )

    def test_smoothed_average_value_at_risk_deviation(self):
        # test smooth cvar deviation using importance sampling
        bkd = self.get_backend()
        mu, sigma, beta = 0.5, 1, 0.5
        risks = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avardev = bkd.array(risks.AVaR(beta) - mu)
        rv = stats.norm(mu, sigma)
        smooth_avardev = SampleAverageSmoothedAverageValueAtRiskDeviation(
            alpha=beta, delta=100000, backend=bkd
        )
        nsamples = int(1e6)
        # samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        # a more accurate ansswer can be obtained by using 1D
        # halton like sequence, i.e. equidistant points
        dominating_rv = stats.norm(mu, sigma * 2)
        np_samples = (
            dominating_rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        )[None, :]
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        weights *= bkd.asarray(
            rv.pdf(np_samples.T) / dominating_rv.pdf(np_samples.T)
        )
        smooth_avardev.set_mean(mu)
        # print(
        #     smooth_avardev(bkd.asarray(np_samples).T, weights), exact_avardev
        # )
        # print(
        #     smooth_avardev(bkd.asarray(np_samples).T, weights) - exact_avardev
        # )
        assert bkd.allclose(
            smooth_avardev(bkd.asarray(np_samples).T, weights),
            exact_avardev,
            rtol=2e-5,
        )


class TestNumpySampleAverage(TestSampleAverage, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchSampleAverage(TestSampleAverage, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
