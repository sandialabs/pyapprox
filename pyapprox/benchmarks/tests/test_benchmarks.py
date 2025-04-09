import unittest
import numpy as np
import itertools

from pyapprox.benchmarks import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
    RosenbrockUnconstrainedOptimizationBenchmark,
    CantileverBeamDeterminsticOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
    PistonBenchmark,
    WingWeightBenchmark,
    GenzBenchmark,
    ChemicalReactionBenchmark,
    LotkaVolterraBenchmark,
    CoupledSpringsBenchmark,
    HastingsEcologyBenchmark,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.newton import NewtonSolver
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual,
    ForwardEulerResidual,
    CrankNicholsonResidual,
    HeunResidual,
)
from pyapprox.expdesign.sequences import SobolSequence


class TestBenchmarks:

    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyMixin

    def test_ishigami(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(a=7, b=0.1, backend=bkd)
        init_guess = benchmark.variable().mean() + benchmark.variable().std()
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.model().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        samples = benchmark.variable().rvs(1e5)
        values = benchmark.model()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)
        print(values.shape)
        assert bkd.allclose(
            bkd.var(values, ddof=1), benchmark.variance(), rtol=1e-2
        )
        # check statistics run
        benchmark.main_effects()
        benchmark.total_effects()
        benchmark.sobol_indices()

    def test_oakley(self):
        bkd = self.get_backend()
        benchmark = OakleyBenchmark(backend=bkd)
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
        print(benchmark.mean(), bkd.mean(values))
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
        benchmark = RosenbrockUnconstrainedOptimizationBenchmark(
            nvars=2, backend=bkd
        )
        init_guess = benchmark.variable().mean() + benchmark.variable().std()
        errors = benchmark.objective().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.objective().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        samples = benchmark.variable().rvs(1e5)
        values = benchmark.objective()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)

    def test_cantileverbeam_optimization_benchmarks(self):
        bkd = self.get_backend()
        benchmark = CantileverBeamDeterminsticOptimizationBenchmark(bkd)
        objective = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        errors = objective.check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6
        errors = objective.check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        constraint = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        errors = constraint.check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6

        benchmark = CantileverBeamUncertainOptimizationBenchmark(bkd)
        objective = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        errors = objective.check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6
        errors = objective.check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        constraint = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        errors = constraint.check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6

    def test_piston(self):
        bkd = self.get_backend()
        benchmark = PistonBenchmark(bkd)
        init_guess = benchmark.variable().mean()
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6

    def test_wing_weight(self):
        bkd = self.get_backend()
        benchmark = WingWeightBenchmark(bkd)
        init_guess = benchmark.variable().mean()
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6

    def _check_genz(self, name, nvars, decay):
        bkd = self.get_backend()
        cfactor, wfactor = 1, 0.5
        benchmark = GenzBenchmark(
            name, nvars, decay, cfactor, wfactor, backend=bkd
        )
        integral = benchmark.integral()

        nsamples = int(1e4)
        seq = SobolSequence(nvars, 0, benchmark.variable(), bkd)
        samples = seq.rvs(nsamples)
        weights = bkd.full((nsamples, 1), 1.0 / nsamples)
        vals = benchmark.model()(samples)
        qmc_integral = vals.T @ weights
        # print(integral, qmc_integral)
        # print((qmc_integral-integral)/integral)
        assert np.allclose(qmc_integral, integral, rtol=7e-4)

        if benchmark.model().jacobian_implemented():
            sample = benchmark.variable().mean() * 0.25
            errors = benchmark.model().check_apply_jacobian(sample)
            assert errors.min() / errors.max() < 1e-6

    def test_genz(self):
        names = [
            "oscillatory",
            "product_peak",
            "corner_peak",
            "gaussian",
            "c0continuous",
            "discontinuous",
        ]
        nvars = np.arange(2, 7)
        decays = ["none", "quadratic", "quartic", "exp", "sqexp"]
        test_scenarios = itertools.product(*[names, nvars, decays])
        for test_scenario in test_scenarios:
            np.random.seed(1)
            # print(test_scenario)
            self._check_genz(*test_scenario)

    def test_chemical_reaction(self):
        bkd = self.get_backend()
        benchmark = ChemicalReactionBenchmark(bkd)
        init_guess = benchmark.variable().mean()

        # some variable ranges are small so restrict fd sizes else
        # solve will not converge
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        errors = benchmark.model().check_apply_jacobian(init_guess, fd_eps)
        assert errors.min() / errors.max() < 1e-6

    def _check_lotka_volterra(self, time_residual_cls):
        bkd = self.get_backend()
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-12)
        benchmark = LotkaVolterraBenchmark(
            time_residual_cls, newton_solver, bkd
        )
        # The error in check apply jacobian depend on newton tolerance
        # because finite difference is only accurate to that tolerance
        sample = bkd.array(
            np.random.uniform(0.3, 0.7, benchmark.model().nvars())
        )[:, None]
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = benchmark.model().check_apply_jacobian(sample, fd_eps)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1.4e-6

    def test_lotka_volterra(self):
        test_cases = [
            [BackwardEulerResidual],
            [CrankNicholsonResidual],
            [ForwardEulerResidual],
            [HeunResidual],
        ]
        # useful to test all types of timestepping that supports
        # timestepping as it picked up some bugs during development
        for test_case in test_cases:
            self._check_lotka_volterra(*test_case)

    def test_coupled_springs(self):
        bkd = self.get_backend()
        benchmark = CoupledSpringsBenchmark(bkd)
        sample = benchmark.variable().mean()
        errors = benchmark.model().check_apply_jacobian(sample)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 5e-6

    def test_hastings_ecology(self):
        bkd = self.get_backend()
        benchmark = HastingsEcologyBenchmark(bkd)
        # The error in check apply jacobian depend on newton tolerance
        # because finite difference is only accurate to that tolerance
        sample = benchmark.variable().mean()
        errors = benchmark.model().check_apply_jacobian(sample)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1.4e-6


class TestNumpyBenchmarks(TestBenchmarks, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchBenchmarks(TestBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
