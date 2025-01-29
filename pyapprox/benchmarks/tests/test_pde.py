import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
    TransientViscousBurgers1DOperatorBenchmark,
)
from pyapprox.pde.collocation.functions import ScalarFunction


class TestPDEBenchmarks:
    def setUp(self):
        np.random.seed(1)

    def test_pyapprox_paper_inversion_benchmark(self):
        bkd = self.get_backend()
        benchmark = PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
            backend=bkd
        )
        x0 = benchmark.true_params() + 1e-3
        errors = benchmark.model().check_apply_jacobian(x0)
        assert errors.min() / errors.max() > 1e-7
        errors = benchmark.model().check_apply_hessian(x0)
        assert errors.min() / errors.max() > 1e-7

    def test_transient_viscous_burgers_1d_benchmark(self):
        bkd = self.get_backend()
        benchmark = TransientViscousBurgers1DOperatorBenchmark(backend=bkd)

        # test benchmark model and plots run
        axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
        for ii in range(3):
            sol, times = benchmark.model().forward_solve(
                benchmark.variable().rvs(1)
            )
            init_sol, final_sol = sol[:, [0, -1]].T
            init_sol = ScalarFunction(benchmark.model().basis(), init_sol)
            final_sol = ScalarFunction(benchmark.model().basis(), final_sol)
            init_sol.plot(axs[0], npts_1d=201)
            final_sol.plot(axs[1], npts_1d=201)
        plt.show()


class TestTorchPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
