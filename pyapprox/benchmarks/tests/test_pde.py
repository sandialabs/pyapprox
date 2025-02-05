import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
    TransientViscousBurgers1DOperatorBenchmark,
    SteadyDarcy2DOperatorBenchmark,
)


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
        sample = benchmark.variable().rvs(1)
        sol = benchmark.model().forward_solve(sample)
        # regression test
        # import torch
        # torch.set_printoptions(precision=8)
        assert np.allclose(bkd.max(sol), -2.82280765)
        assert np.allclose(bkd.norm(sol), 167.32975628280943)
        # test plots run
        ax = plt.subplots(1)[1]
        benchmark.model().physics().solution_from_array(sol).plot(
            ax, npts_1d=100, levels=20, cmap="coolwarm"
        )

        if not bkd.jacobian_implemented():
            return
        # test gradient and hessian
        x0 = benchmark.true_params() + 1e-2
        errors = benchmark.model().check_apply_jacobian(x0)
        assert errors.min() / errors.max() > 1e-7
        errors = benchmark.model().check_apply_hessian(x0)
        assert errors.min() / errors.max() > 1e-7

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
        print(bkd.max(sol.get_values()), bkd.norm(sol.get_values()))
        # difference between torch and numpy when computing kle eig
        # decomposition mean that we can only achieve consistenty to 4 digits
        assert np.allclose(bkd.max(sol.get_values()), 0.0436, atol=1e-4)
        assert np.allclose(bkd.norm(sol.get_values()), 0.2935, atol=1e-4)


class TestTorchPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


class TestNumpyPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
