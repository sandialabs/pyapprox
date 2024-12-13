import unittest
import numpy as np
import pickle
import tempfile
import os

from pyapprox.benchmarks import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
    RosenbrockBenchmark,
    CantileverBeamDeterminsticOptimizationBenchmark,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class TestBenchmarks(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyLinAlgMixin

    def test_ishigami(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(a=7, b=0.1, bkd=bkd)
        init_guess = benchmark.variable().get_statistics('mean') +\
            benchmark.variable().get_statistics('std')
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.model().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        samples = benchmark.variable().rvs(1e5)
        values = benchmark.model()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)
        assert bkd.allclose(
            bkd.var(values, ddof=1), benchmark.variance(), rtol=1e-2
        )
        # check statistics run
        benchmark.main_effects()
        benchmark.total_effects()
        benchmark.sobol_indices()

    def test_oakley(self):
        bkd = self.get_backend()
        benchmark = OakleyBenchmark()
        samples = benchmark.variable().rvs(1e5)
        values = benchmark.model()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)
        assert bkd.allclose(
            bkd.var(values, ddof=1), benchmark.variance(), rtol=1e-2
        )
        # check statistics run
        benchmark.main_effects()

    def test_sobolg(self):
        bkd = self.get_backend()
        benchmark = SobolGBenchmark(backend=bkd)
        samples = benchmark.variable().rvs(1e5)
        values = benchmark.model()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)
        assert bkd.allclose(
            bkd.var(values, ddof=1), benchmark.variance(), rtol=1e-2
        )
        # check statistics run
        benchmark.main_effects()
        benchmark.total_effects()
        benchmark.sobol_indices()

    def test_rosenbrock(self):
        bkd = self.get_backend()
        benchmark = RosenbrockBenchmark(nvars=2, backend=bkd)
        init_guess = benchmark.variable().get_statistics('mean') +\
            benchmark.variable().get_statistics('std')
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.model().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        samples = benchmark.variable().rvs(1e5)
        values = benchmark.model()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)

    def test_cantileverbeam_determistic_optimiaztion_benchmark(self):
        bkd = self.get_backend()
        benchmark = CantileverBeamDeterminsticOptimizationBenchmark(bkd)
        objective = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        errors = objective.check_apply_jacobian(init_guess)
        assert errors.min()/errors.max() < 1e-6
        errors = objective.check_apply_hessian(init_guess)
        assert errors.min() < 2e-7

        constraint = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        errors = constraint.check_apply_jacobian(init_guess)
        assert errors.min()/errors.max() < 1e-6

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
