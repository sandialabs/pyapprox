import unittest
import numpy as np
import pickle
import tempfile
import os

from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.benchmarks.surrogate_benchmarks import (
    wing_weight_function, wing_weight_gradient,
    define_wing_weight_random_variables
)
from pyapprox.util.utilities import check_gradients, check_hessian


class TestBenchmarks(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_ishigami_function_gradient_and_hessian(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        init_guess = benchmark.variable.get_statistics('mean') +\
            benchmark.variable.get_statistics('std')
        errors = check_gradients(
            benchmark.fun, benchmark.jac, init_guess, disp=False)
        # print(errors.min())
        assert errors.min() < 7e-6
        def hess_matvec(x, v): return np.dot(benchmark.hess(x), v)
        errors = check_hessian(
            benchmark.jac, hess_matvec, init_guess, disp=False)
        assert errors.min() < 2e-7

    def test_rosenbrock_function_gradient_and_hessian_prod(self):
        benchmark = setup_benchmark("rosenbrock", nvars=2)
        init_guess = benchmark.variable.get_statistics('mean') +\
            benchmark.variable.get_statistics('std')
        errors = check_gradients(
            benchmark.fun, benchmark.jac, init_guess, disp=False)
        assert errors.min() < 1e-5
        errors = check_hessian(
            benchmark.jac, benchmark.hessp, init_guess, disp=False)
        assert errors.min() < 1e-5

    def test_incorrect_benchmark_name(self):
        self.assertRaises(Exception, setup_benchmark, "missing", a=7, b=0.1)

    def test_cantilever_beam_gradients(self):
        benchmark = setup_benchmark('cantilever_beam')
        from pyapprox.interface.wrappers import ActiveSetVariableModel
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
        errors = check_gradients(
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
        errors = check_gradients(
            constraint_fun, constraint_jac, init_guess, disp=True)
        assert errors.min() < 4e-7

        nsamples = 10
        samples = benchmark.variable.rvs(nsamples)
        constraint_fun = ActiveSetVariableModel(
            benchmark.constraint_fun,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            samples, benchmark.design_var_indices)
        constraint_jac = ActiveSetVariableModel(
            benchmark.constraint_jac,
            benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            samples, benchmark.design_var_indices)
        init_guess = 2*np.ones((2, 1))
        print(constraint_jac(init_guess))
        from pyapprox.util.utilities import approx_jacobian
        print(approx_jacobian(lambda x: constraint_fun(x).flatten(order="F"),
                              init_guess)-constraint_jac(init_guess))
        errors = check_gradients(
            lambda x: constraint_fun(x).flatten(order='F'), constraint_jac,
            init_guess, disp=True)
        assert errors.min() < 4e-7

    def test_wing_weight_gradient(self):
        variable = define_wing_weight_random_variables()
        fun = wing_weight_function
        grad = wing_weight_gradient
        sample = variable.rvs(1)
        errors = check_gradients(fun, grad, sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 7e-7

    def test_random_oscillator_analytical_solution(self):
        benchmark = setup_benchmark("random_oscillator")
        time = benchmark.fun.t
        sample = benchmark.variable.get_statistics("mean")
        asol = benchmark.fun.analytical_solution(sample, time)
        nsol = benchmark.fun.numerical_solution(sample.squeeze())
        assert np.allclose(asol, nsol[:, 0])

    def test_piston_gradient(self):
        benchmark = setup_benchmark("piston")
        sample = benchmark.variable.rvs(1)
        print(benchmark.jac(sample))
        errors = check_gradients(benchmark.fun, benchmark. jac, sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 6e-7

    def test_genz_pickle(self):
        tmp_dir = tempfile.TemporaryDirectory()
        g = setup_benchmark("genz", nvars=2, test_name="oscillatory")
        print(tmp_dir.name)
        with open(os.path.join(tmp_dir.name, 'function.pkl'), 'wb') as f:
            pickle.dump(g, f)
        with open(os.path.join(tmp_dir.name, 'function.pkl'), 'rb') as f:
            g1 = pickle.load(f)
        tmp_dir.cleanup()


if __name__ == "__main__":
    benchmarks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestBenchmarks)
    unittest.TextTestRunner(verbosity=2).run(benchmarks_test_suite)
