import unittest
import math

import numpy as np

from pyapprox.benchmarks.multifidelity_benchmarks import (
    PolynomialModelEnsembleBenchmark,
    TunableModelEnsembleBenchmark,
    MultiOutputModelEnsembleBenchmark,
    ShortColumnModelEnsembleBenchmark,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestMFBenchmarks:

    def setUp(self):
        np.random.seed(1)

    def _check_benchmark(self, bench):
        # check quadrature and exact method if implemented match
        # when no exact method is given then it just checks quadrature
        # with itself
        bkd = self.get_backend()
        assert bkd.allclose(super(type(bench), bench)._mean(), bench.mean())

        assert bkd.allclose(
            super(type(bench), bench)._covariance(), bench.covariance()
        )

        print(bench.costs())

        if hasattr(bench, "get_kurtosis"):
            kurtosis = bench.get_kurtosis()
            quad_x, quad_w = bench._quadx, bench._quadw
            ref_kurtosis = bkd.hstack(
                [
                    (((m(quad_x) - mu) ** 4).T @ quad_w)
                    for m, mu in zip(bench.models(), bench.mean())
                ]
            )
            assert bkd.allclose(ref_kurtosis, kurtosis)

        nsamples = int(1e7)
        samples = bench.prior().rvs(nsamples)
        values = bkd.hstack([m(samples) for m in bench.models()])

        assert bkd.allclose(
            bench.mean(),
            bkd.mean(values, axis=0).reshape(bench.nmodels(), bench.nqoi()),
            rtol=4e-2,
            atol=1e-3,
        )

        assert bkd.allclose(
            bench.covariance(),
            bkd.cov(values, rowvar=False, ddof=1),
            rtol=4e-2,
        )

    def test_acv_benchmarks(self):
        bkd = self.get_backend()
        cases = [
            PolynomialModelEnsembleBenchmark(bkd, 5),
            TunableModelEnsembleBenchmark(math.pi / 4, bkd, shifts=[1, 2]),
            MultiOutputModelEnsembleBenchmark(bkd),
            ShortColumnModelEnsembleBenchmark(bkd),
        ]
        for case in cases:
            print(case)
            self._check_benchmark(case)


class TestTorchMFBenchmarks(TestMFBenchmarks, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


class TestNumpyMFBenchmarks(TestMFBenchmarks, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
