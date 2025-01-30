import unittest

import numpy as np
import matplotlib.pyplot as plt

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
        bkd = self.get_backend()
        np.random.seed(1)
        benchmark = PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
            backend=bkd
        )
        sample = benchmark.variable().rvs(1)
        sol = benchmark.model().forward_solve(sample)
        # regression test
        assert np.allclose(bkd.max(sol), -3.2342921)
        assert np.allclose(bkd.norm(sol), 166.6196595)
        # test plots run
        ax = plt.subplots(1)[1]
        benchmark.model().physics().solution_from_array(sol).plot(
            ax, npts_1d=100, levels=20, cmap="coolwarm"
        )
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
        assert np.allclose(bkd.max(sol.get_values()), 0.0502538)
        assert np.allclose(bkd.norm(sol.get_values()), 0.3568277)


class TestTorchPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
