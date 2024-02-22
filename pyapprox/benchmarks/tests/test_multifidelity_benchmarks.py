import unittest

import numpy as np

from pyapprox.benchmarks.benchmarks import setup_benchmark


class TestMFBenchmarks(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def _check_benchmark(self, name, args, kwargs):
        bench = setup_benchmark(name, *args, **kwargs)
        quad_x, quad_w = bench.fun._quadrature_rule


        # check quadrature and exact method if implemented match
        # when no exact method is given then it just checks quadrature
        # with itself
        print(np.allclose(
            super(type(bench.fun), bench.fun)._get_means(),
            bench.mean))
        assert np.allclose(
            super(type(bench.fun), bench.fun)._get_means(),
            bench.mean)

        assert np.allclose(
            super(type(bench.fun), bench.fun)._get_covariance_matrix(),
            bench.covariance)

        if hasattr(bench.fun, "get_kurtosis"):
            kurtosis = bench.fun.get_kurtosis()
            quad_x, quad_w = bench.fun._quadrature_rule
            ref_kurtosis = np.hstack([
                (((m(quad_x)-mu)**4).T.dot(quad_w)[:, 0])
                for m, mu in zip(bench.funs, bench.mean)])
            assert np.allclose(ref_kurtosis, kurtosis)

        nsamples = int(1e6)
        samples = bench.variable.rvs(nsamples)
        values = np.hstack([m(samples) for m in bench.funs])

        assert np.allclose(
            bench.mean,
            np.mean(values, axis=0).reshape(
                bench.fun.nmodels, bench.fun.nqoi), rtol=4e-2, atol=1e-3)

        assert np.allclose(
            bench.covariance,
            np.cov(values, rowvar=False, ddof=1), rtol=4e-2)

    def test_acv_benchmarks(self):
        cases = [
            ["polynomial_ensemble", [], {}],
            ["tunable_model_ensemble", [], {"shifts": [1, 2]}],
            ["multioutput_model_ensemble", [], {}],
            ["short_column_ensemble", [], {}]
        ]
        for case in cases:
            print(case)
            self._check_benchmark(*case)


if __name__ == "__main__":
    mf_benchmarks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMFBenchmarks)
    unittest.TextTestRunner(verbosity=2).run(mf_benchmarks_test_suite)
