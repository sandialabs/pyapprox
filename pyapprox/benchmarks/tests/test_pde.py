import unittest
import copy

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
    TransientViscousBurgers1DOperatorBenchmark,
    SteadyDarcy2DOperatorBenchmark,
)
from pyapprox.pde.collocation.functions import ScalarFunction


class TestPDEBenchmarks:
    def setUp(self):
        np.random.seed(1)

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
        # import torch

        # torch.set_printoptions(precision=16)
        # print(bkd.max(sol))
        # print(bkd.norm(sol))
        assert bkd.allclose(bkd.max(sol), bkd.asarray(-7.2928795756009910))
        assert bkd.allclose(bkd.norm(sol), bkd.asarray(156.3263428926894960))
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
            print(model.basis())
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
        # print(convergence_rate)
        assert convergence_rate < -10.0

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
        sample = benchmark.variable().rvs(1)
        sol, times = benchmark.model().forward_solve(sample)
        # regression test
        assert np.allclose(bkd.max(sol[:, -1]), 0.0080807)
        assert np.allclose(bkd.norm(sol), 36.9231765)

    def test_steady_darcy_2d_kle_benchmark(self):
        bkd = self.get_backend()
        benchmark = SteadyDarcy2DOperatorBenchmark(backend=bkd)
        sample = benchmark.variable().rvs(1)
        sol = benchmark.model().forward_solve(sample)
        # regression test
        # print(bkd.max(sol.get_values()), bkd.norm(sol.get_values()))
        # difference between torch and numpy when computing kle eig
        # decomposition mean that we can only achieve consistenty to 4 digits
        assert np.allclose(bkd.max(sol.get_values()), 0.043, atol=1e-3)
        assert np.allclose(bkd.norm(sol.get_values()), 0.406, atol=2e-3)


class TestTorchPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


class TestNumpyPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
