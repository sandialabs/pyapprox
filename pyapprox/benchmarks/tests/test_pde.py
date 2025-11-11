import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
    TransientViscousBurgers1DOperatorBenchmark,
    SteadyDarcy2DOperatorBenchmark,
    NonlinearSystemOfEquationsBenchmark,
    ObstructedAdvectionDiffusionOEDBenchmark,
)
from pyapprox.pde.collocation.functions import ScalarFunction


class TestPDEBenchmarks:
    def setUp(self):
        np.random.seed(1)

    def test_nonlinear_system_of_equations(self):
        bkd = self.get_backend()
        benchmark = NonlinearSystemOfEquationsBenchmark(bkd)
        # The error in check apply jacobian depend on newton tolerance
        # because finite difference is only accurate to that tolerance
        sample = benchmark.prior().mean()
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = benchmark.model().check_apply_jacobian(sample, fd_eps=fd_eps)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1.4e-6

    def test_pyapprox_paper_inversion_benchmark(self):
        # This test picked up cross platform differences in KLE
        # caused by differences in scipy.linalg.eigh. That function
        # was replaced but keep test as is here to make sure that issue
        # does not resurface
        bkd = self.get_backend()
        np.random.seed(1)
        benchmark = PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
            backend=bkd
        )
        sample = benchmark.prior().rvs(1)
        sol = benchmark.obs_model().forward_solve(sample)

        # regression test
        if bkd.bkd_equal(bkd, NumpyMixin):
            # Numerical drfit causes torch and numpy solution to differ slightly
            # import torch
            # torch.set_printoptions(precision=16)
            # print(bkd.max(sol))
            # print(bkd.norm(sol))
            assert bkd.allclose(bkd.max(sol), bkd.asarray(-7.218008911052107))
            assert bkd.allclose(bkd.norm(sol), bkd.asarray(156.320730706938))

        # test plots run
        ax = plt.subplots(1)[1]
        benchmark.obs_model().physics().solution_from_array(sol).plot(
            ax, npts_1d=100, levels=20, cmap="coolwarm"
        )

        # test convergence of QoI model solution at last timestep
        model_config = [
            [10, 10, 0.2 / 10],
            [20, 20, 0.2 / 10],
            [30, 30, 0.2 / 10],
        ]
        qoi_models = benchmark._qoi_models(model_config)
        sols = []
        for model in qoi_models:
            sol_array = model.forward_solve(sample)[0]
            sols.append(model.physics().solution_from_array(sol_array[:, -1]))

        hf_sol = sols[-1]
        diff = ScalarFunction(hf_sol.basis())
        sol_errors = []
        ndofs = []
        for sol in sols[:-1]:
            diff.set_values(
                (hf_sol.get_values() - sol(hf_sol.basis().mesh().mesh_pts()))
                ** 2
            )
            sol_errors.append(diff.integrate())
            ndofs.append(sol.basis().mesh().nmesh_pts())
        sol_errors = bkd.asarray(sol_errors)
        ndofs = bkd.asarray(ndofs)
        convergence_rate = bkd.log(sol_errors[0] / sol_errors[-1]) / bkd.log(
            ndofs[0] / ndofs[-1]
        )
        # print(sol_errors)
        print(convergence_rate)
        assert convergence_rate < -7.0

        if not bkd.jacobian_implemented():
            return
        # test gradient and hessian of loglike
        x0 = benchmark.observation_generating_parameters() + 1e-2
        errors = benchmark.loglike().check_apply_jacobian(x0)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 2e-7
        errors = benchmark.loglike().check_apply_hessian(x0)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 3e-6

    def test_transient_viscous_burgers_1d_benchmark(self):
        bkd = self.get_backend()
        benchmark = TransientViscousBurgers1DOperatorBenchmark(backend=bkd)
        sample = benchmark.prior().rvs(1)
        sol, times = benchmark.model().forward_solve(sample)
        # regression test
        assert np.allclose(bkd.max(sol[:, -1]), 0.0080807)
        assert np.allclose(bkd.norm(sol), 36.9231765)

    def test_steady_darcy_2d_kle_benchmark(self):
        bkd = self.get_backend()
        benchmark = SteadyDarcy2DOperatorBenchmark(backend=bkd)
        sample = benchmark.prior().rvs(1)
        sol = benchmark.model().forward_solve(sample)
        # regression test
        # print(bkd.max(sol.get_values()), bkd.norm(sol.get_values()))
        # difference between torch and numpy when computing kle eig
        # decomposition mean that we can only achieve consistenty to 4 digits
        assert np.allclose(bkd.max(sol.get_values()), 0.043, atol=1e-3)
        assert np.allclose(bkd.norm(sol.get_values()), 0.406, atol=2e-3)

    def test_obstructed_advection_diffusion_oed_benchmark(self):
        bkd = self.get_backend()
        benchmark = ObstructedAdvectionDiffusionOEDBenchmark(bkd)
        # set number of observations smaller so reference arrays
        # for regression tests are small
        benchmark._nobs = 10
        benchmark._set_obs_model()
        obs_model = benchmark.observation_model()
        sample = benchmark.prior().rvs(1)

        # regression test
        # print(np.array2string(bkd.to_numpy(obs), separator=", "))
        # fmt: off
        obs = obs_model(sample)
        ref_obs = bkd.array(
            [[0.50404555, 0.86627557, 1.12947464, 0.49326366, 0.90510361, 1.22957678,
              0.50331619, 0.83547626, 1.07224894, 0.47150367, 0.81459434, 1.07525192,
              0.44476742, 0.81367648, 1.10492837, 0.34040704, 0.65978668, 0.92981599,
              0.19935085, 0.37056546, 0.5157221 , 0.4697795 , 0.84763621, 1.14741092,
              0.42749568, 0.77486119, 1.0534409 , 0.44745031, 0.81736022, 1.10931204]]
        )
        assert bkd.allclose(obs, ref_obs)

        # print(
        #     np.array2string(benchmark.observation_locations(), separator=", ")
        # )
        ref_locs = bkd.array(
            [[0.28571429, 0.41071429, 0.14285714, 0.21428571, 0.33928571, 0.60714286,
              0.89285714, 0.46428571, 0.60714286, 0.33928571],
             [0.34375   , 0.1875    , 0.28125   , 0.875     , 0.59375   , 0.28125   ,
              0.375     , 1.        , 0.96875   , 0.625     ]
             ]
        )
        # fmt: on
        assert bkd.allclose(benchmark.observation_locations(), ref_locs)

        pred_model = benchmark.prediction_model()
        qoi = pred_model(sample)
        # print(np.array2string(bkd.to_numpy(qoi), separator=", "))
        ref_qoi = bkd.asarray([[0.15857682]])
        assert bkd.allclose(qoi, ref_qoi)

        # ax = plt.figure().gca()
        # ax.plot(*benchmark.observation_locations(), "o")
        # plt.show()


class TestTorchPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


class TestNumpyPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
