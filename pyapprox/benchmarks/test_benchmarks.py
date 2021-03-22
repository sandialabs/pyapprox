#!/usr/bin/env python
import sys
from functools import partial
import unittest, pytest

import numpy as np
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    pytestmark = pytest.mark.skip("Skipping test on Windows")
else:
    from pyapprox.benchmarks.benchmarks import *
    import pyapprox as pya


class TestBenchmarks(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_ishigami_function_gradient_and_hessian(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        init_guess = benchmark.variable.get_statistics('mean') +\
            benchmark.variable.get_statistics('std')
        errors = pya.check_gradients(
            benchmark.fun, benchmark.jac, init_guess, disp=False)
        # print(errors.min())
        assert errors.min() < 2e-6
        def hess_matvec(x, v): return np.dot(benchmark.hess(x), v)
        errors = pya.check_hessian(
            benchmark.jac, hess_matvec, init_guess, disp=False)
        assert errors.min() < 2e-7

    def test_rosenbrock_function_gradient_and_hessian_prod(self):
        benchmark = setup_benchmark("rosenbrock", nvars=2)
        init_guess = benchmark.variable.get_statistics('mean') +\
            benchmark.variable.get_statistics('std')
        errors = pya.check_gradients(
            benchmark.fun, benchmark.jac, init_guess, disp=False)
        assert errors.min() < 1e-5
        errors = pya.check_hessian(
            benchmark.jac, benchmark.hessp, init_guess, disp=False)
        assert errors.min() < 1e-5

    def test_incorrect_benchmark_name(self):
        self.assertRaises(Exception, setup_benchmark, "missing", a=7, b=0.1)
        benchmark = Benchmark(
            {'fun': rosenbrock_function, 'jac': rosenbrock_function_jacobian,
             'hessp': rosenbrock_function_hessian_prod})

    def test_cantilever_beam_gradients(self):
        benchmark = setup_benchmark('cantilever_beam')
        from pyapprox.models.wrappers import ActiveSetVariableModel
        fun = ActiveSetVariableModel(
            benchmark.fun,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            benchmark.variable.get_statistics('mean'),
            benchmark.design_var_indices)
        jac = ActiveSetVariableModel(
            benchmark.jac,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            benchmark.variable.get_statistics('mean'),
            benchmark.design_var_indices)
        init_guess = 2*np.ones((2, 1))
        errors = pya.check_gradients(
            fun, jac, init_guess, disp=True)
        assert errors.min() < 4e-7

        constraint_fun = ActiveSetVariableModel(
            benchmark.constraint_fun,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            benchmark.variable.get_statistics('mean'),
            benchmark.design_var_indices)
        constraint_jac = ActiveSetVariableModel(
            benchmark.constraint_jac,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            benchmark.variable.get_statistics('mean'),
            benchmark.design_var_indices)
        init_guess = 2*np.ones((2, 1))
        errors = pya.check_gradients(
            constraint_fun, constraint_jac, init_guess, disp=True)
        assert errors.min() < 4e-7

        nsamples = 10
        samples = pya.generate_independent_random_samples(
            benchmark.variable, nsamples)
        constraint_fun = ActiveSetVariableModel(
            benchmark.constraint_fun,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            samples, benchmark.design_var_indices)
        constraint_jac = ActiveSetVariableModel(
            benchmark.constraint_jac,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            samples, benchmark.design_var_indices)
        init_guess = 2*np.ones((2, 1))
        errors = pya.check_gradients(
            lambda x: constraint_fun(x).flatten(order='F'), constraint_jac,
            init_guess, disp=True)
        assert errors.min() < 4e-7


if __name__ == "__main__":
    benchmarks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestBenchmarks)
    unittest.TextTestRunner(verbosity=2).run(benchmarks_test_suite)
