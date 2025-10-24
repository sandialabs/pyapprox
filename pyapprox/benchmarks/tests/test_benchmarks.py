import unittest
import numpy as np
import itertools

from pyapprox.benchmarks import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
    RosenbrockUnconstrainedOptimizationBenchmark,
    RosenbrockConstrainedOptimizationBenchmark,
    CantileverBeamDeterminsticOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
    PistonBenchmark,
    WingWeightBenchmark,
    GenzBenchmark,
    ChemicalReactionBenchmark,
    LotkaVolterraBenchmark,
    LotkaVolterraOEDBenchmark,
    CoupledSpringsBenchmark,
    HastingsEcologyBenchmark,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.newton import NewtonSolver
from pyapprox.pde.timeintegration import (
    BackwardEulerResidual,
    ForwardEulerResidual,
    CrankNicholsonResidual,
    HeunResidual,
)
from pyapprox.expdesign.sequences import SobolSequence


class TestBenchmarks:

    def setUp(self):
        np.random.seed(1)

    def test_ishigami(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(a=7, b=0.1, backend=bkd)
        init_guess = benchmark.prior().mean() + benchmark.prior().std()
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.model().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        samples = benchmark.prior().rvs(1e5)
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
        benchmark = OakleyBenchmark(backend=bkd)
        samples = benchmark.prior().rvs(1e5)
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
        samples = benchmark.prior().rvs(1e5)
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
        benchmark = RosenbrockUnconstrainedOptimizationBenchmark(
            nvars=2, backend=bkd
        )
        init_guess = benchmark.prior().mean() + benchmark.prior().std()
        errors = benchmark.objective().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.objective().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        samples = benchmark.prior().rvs(1e5)
        values = benchmark.objective()(samples)
        assert bkd.allclose(bkd.mean(values), benchmark.mean(), rtol=1e-2)

    def test_constrained_rosenbrock(self):
        bkd = self.get_backend()
        benchmark = RosenbrockConstrainedOptimizationBenchmark(backend=bkd)
        init_guess = benchmark.prior().mean() + benchmark.prior().std()
        errors = benchmark.objective().check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.objective().check_apply_hessian(init_guess)
        assert errors.min() < 2e-7
        errors = benchmark.constraints()[0].check_apply_jacobian(init_guess)
        assert errors.min() < 7e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_guess, weights=bkd.ones((2, 1))
        )
        assert errors.min() < 2e-7

    def test_cantileverbeam_optimization_benchmarks(self):
        bkd = self.get_backend()
        # benchmark = CantileverBeamDeterminsticOptimizationBenchmark(bkd)
        # objective = benchmark.objective()
        # init_guess = bkd.ones((2, 1))
        # errors = objective.check_apply_jacobian(init_guess)
        # assert errors.min() / errors.max() < 1e-6
        # errors = objective.check_apply_hessian(init_guess)
        # assert errors.min() < 2e-7
        # constraint = benchmark.objective()
        # init_guess = bkd.ones((2, 1))
        # errors = constraint.check_apply_jacobian(init_guess)
        # assert errors.min() / errors.max() < 1e-6

        benchmark = CantileverBeamUncertainOptimizationBenchmark(bkd)
        objective = benchmark.objective()
        init_guess = bkd.ones((2, 1))
        print(objective)
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
        init_guess = benchmark.prior().mean()
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6

    def test_wing_weight(self):
        bkd = self.get_backend()
        benchmark = WingWeightBenchmark(bkd)
        init_guess = benchmark.prior().mean()
        errors = benchmark.model().check_apply_jacobian(init_guess)
        assert errors.min() / errors.max() < 1e-6

    def _check_genz(self, name, nvars, decay):
        bkd = self.get_backend()
        cfactor, wfactor = 1.0, 0.5
        benchmark = GenzBenchmark(
            name, nvars, decay, cfactor, wfactor, backend=bkd
        )
        integral = benchmark.integral()

        nsamples = int(1e4)
        seq = SobolSequence(nvars, 0, benchmark.prior(), bkd)
        samples = seq.rvs(nsamples)
        weights = bkd.full((nsamples, 1), 1.0 / nsamples)
        vals = benchmark.model()(samples)
        qmc_integral = vals.T @ weights
        # print(integral, qmc_integral)
        # print((qmc_integral-integral)/integral)
        assert np.allclose(qmc_integral, integral, rtol=7e-4)

        if benchmark.model().jacobian_implemented():
            sample = benchmark.prior().mean() * 0.25
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
        init_guess = benchmark.prior().mean()

        # some variable ranges are small so restrict fd sizes else
        # solve will not converge
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        errors = benchmark.model().check_apply_jacobian(init_guess, fd_eps)
        assert errors.min() / errors.max() < 1e-6

    def _check_lotka_volterra(self, time_residual_cls):
        bkd = self.get_backend()
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-12)
        benchmark = LotkaVolterraBenchmark(
            bkd, time_residual_cls, newton_solver
        )
        # stack the standard deviation of the MSE functional to a prior sample
        sample = bkd.vstack([bkd.ones((1, 1)), benchmark.prior().rvs(1)])

        # The error in check apply jacobian depend on newton tolerance
        # because finite difference is only accurate to that tolerance
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

    def test_lotka_volterra_oed(self):
        bkd = self.get_backend()
        benchmark = LotkaVolterraOEDBenchmark(backend=bkd)
        sample = benchmark.prior().rvs(1)
        obs = benchmark.observation_model()(sample)
        assert obs.shape == (1, benchmark.nobservations())
        preds = benchmark.prediction_model()(sample)
        assert preds.shape == (1, benchmark.npredictions())
        assert benchmark.nvars() == 12

        # regression test
        print(np.array2string(bkd.to_numpy(obs), separator=", "))
        # fmt: off
        ref_obs = bkd.array([
            [0.3       , 0.49916859, 0.67410872, 0.81292888, 0.92120916, 1.00752906,
             1.07860341, 1.13881465, 1.19085892, 1.23640198, 1.27652427, 1.31198331,
             1.34335727, 1.37111982, 1.39567793, 1.41739065, 1.43657858, 1.45352911,
             1.46849971, 1.48172048, 1.49339654, 1.50371012, 1.51282277, 1.52087737,
             1.52800006, 1.53430202, 0.3       , 0.34642637, 0.3535113 , 0.33661577,
             0.30871531, 0.27741579, 0.24648662, 0.2175804 , 0.1913011 , 0.1677619 ,
             0.14685686, 0.12838954, 0.11213315, 0.09785867, 0.08534768, 0.07439798,
             0.06482557, 0.05646484, 0.04916784, 0.04280314, 0.03725446, 0.03241922,
             0.02820719, 0.02453916, 0.02134568, 0.01856596
             ]
        ])
        assert bkd.allclose(obs, ref_obs)

        print(np.array2string(bkd.to_numpy(preds), separator=", "))
        ref_preds = bkd.array([
            [0.4       , 0.88978974, 1.06583232, 1.09380394, 1.08017557, 1.05804101,
             1.03634514, 1.01730436, 1.00124297, 0.98793399, 0.97700324, 0.96806452,
             0.96076678]]
        )
        # fmt: on
        assert bkd.allclose(preds, ref_preds)

    def test_coupled_springs(self):
        bkd = self.get_backend()
        benchmark = CoupledSpringsBenchmark(bkd)
        sample = benchmark.prior().mean()
        errors = benchmark.model().check_apply_jacobian(sample)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 5e-6

    def test_hastings_ecology(self):
        bkd = self.get_backend()
        benchmark = HastingsEcologyBenchmark(bkd)
        # The error in check apply jacobian depend on newton tolerance
        # because finite difference is only accurate to that tolerance
        sample = benchmark.prior().mean()
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
