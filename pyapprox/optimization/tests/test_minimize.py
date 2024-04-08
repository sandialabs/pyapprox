import unittest

import numpy as np

from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer, Bounds, Constraint,
    SampleAverageMean, SampleAverageVariance, SampleAverageStdev,
    SampleAverageMeanPlusStdev, SampleAverageEntropicRisk,
    SampleAverageConstraint)
from pyapprox.benchmarks import setup_benchmark
from pyapprox.interface.model import ModelFromCallable


class TestMinimize(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_constrained_scipy(self):
        # check that no bounds is handled correctly
        nvars = 2
        benchmark = setup_benchmark("rosenbrock", nvars=nvars)
        optimizer = ScipyConstrainedOptimizer(benchmark.fun)
        result = optimizer.minimize(benchmark.variable.get_statistics("mean"))
        assert np.allclose(result.x, np.full(nvars, 1))

        # check that constraints are handled correctly
        nvars = 2
        bounds = Bounds(np.full((nvars,), -2), np.full((nvars,), 2))
        benchmark = setup_benchmark("rosenbrock", nvars=nvars)
        optimizer = ScipyConstrainedOptimizer(benchmark.fun, bounds=bounds)
        result = optimizer.minimize(benchmark.variable.get_statistics("mean"))
        assert np.allclose(result.x, np.full(nvars, 1))

        # check apply_jacobian and apply_hessian with 1D samples
        objective = ModelFromCallable(
            lambda x: ((x[0] - 1)**2 + (x[1] - 2.5)**2),
            jacobian=lambda x: np.array([2*(x[0] - 1), 2*(x[1] - 2.5)]),
            apply_jacobian=lambda x, v: 2*(x[0] - 1)*v[0]+2*(x[1] - 2.5)*v[1],
            apply_hessian=lambda x, v: np.array(np.diag([2, 2])) @ v,
            sample_ndim=1, values_ndim=0)

        constraint_model = ModelFromCallable(
            lambda x:  np.array(
                [x[0]-2*x[1]+2, -x[0]-2*x[1]+6, -x[0]+2*x[1]+2]),
            lambda x:  np.array(
                [[1., -2.], [-1., -2], [-1, 2.]]),
            sample_ndim=1, values_ndim=1)
        sample = np.array([2, 0])[:, None]
        errors = constraint_model.check_apply_jacobian(sample, disp=True)
        # jacobian is constant so check first finite difference is exact
        assert errors[0] < 1e-15

        constraint_bounds = np.hstack(
            [np.full((3, 1), 0), np.full((3, 1), np.inf)])
        print(constraint_bounds.shape)
        constraint = Constraint(constraint_model, constraint_bounds)

        bounds = Bounds(np.full((nvars,), 0), np.full((nvars,), np.inf))
        optimizer = ScipyConstrainedOptimizer(
            objective, bounds=bounds, constraints=[constraint])
        result = optimizer.minimize(np.array([2, 0])[:, None])
        assert np.allclose(result.x, np.array([1.4, 1.7]))

    def test_sample_average_constraints(self):
        benchmark = setup_benchmark('cantilever_beam')
        constraint_model = benchmark.funs[1]

        # test jacobian
        nsamples = 10000
        samples = benchmark.variable.rvs(nsamples)
        weights = np.full((nsamples, 1), 1/nsamples)
        for stat in [SampleAverageMean(), SampleAverageVariance(),
                     SampleAverageStdev(), SampleAverageMeanPlusStdev(2),
                     SampleAverageEntropicRisk()]:
            constraint_bounds = np.hstack(
                [np.zeros((2, 1)), np.full((2, 1), np.inf)])
            constraint = SampleAverageConstraint(
                constraint_model, samples, weights, stat, constraint_bounds,
                benchmark.variable.num_vars() +
                benchmark.design_variable.num_vars(),
                benchmark.design_var_indices)
            design_sample = np.array([3, 3])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min()/errors.max() < 1.3e-6

        # test apply_jacobian
        constraint_model._apply_jacobian_implemented = True
        constraint_model._apply_jacobian = (
            lambda x, v: constraint_model.jacobian(x) @ v)

        nsamples = 1000
        samples = benchmark.variable.rvs(nsamples)
        weights = np.full((nsamples, 1), 1/nsamples)
        for stat in [SampleAverageMean(), SampleAverageVariance(),
                     SampleAverageStdev(), SampleAverageMeanPlusStdev(2),
                     SampleAverageEntropicRisk()]:
            constraint_bounds = np.hstack(
                [np.zeros((2, 1)), np.full((2, 1), np.inf)])
            constraint = SampleAverageConstraint(
                constraint_model, samples, weights, stat, constraint_bounds,
                benchmark.variable.num_vars() +
                benchmark.design_variable.num_vars(),
                benchmark.design_var_indices)
            design_sample = np.array([3, 3])[:, None]
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max(), stat)
            assert errors.min()/errors.max() < 1.3e-6


if __name__ == '__main__':
    minimize_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMinimize)
    unittest.TextTestRunner(verbosity=2).run(minimize_test_suite)
