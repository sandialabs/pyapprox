import unittest

import numpy as np

from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark
)


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


class TestTorchPDEBenchmarks(TestPDEBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
