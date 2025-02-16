import unittest

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.analysis.parameter_sweeps import (
    BoundedParameterSweeper,
    GaussianParameterSweeper,
)

from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation,
)


class TestParameterSweeps(unittest.TestCase):
    def test_get_hypercube_parameter_sweeps_samples(self):
        nvars = 2
        trans = define_iid_random_variable_transformation(
            stats.uniform(0, 1), nvars
        )
        sweeper = BoundedParameterSweeper(nvars, trans, 50)
        nsweeps = 3
        sweep_samples = sweeper.rvs(nsweeps)
        for ii in range(sweep_samples.shape[0]):
            assert np.all(
                (sweep_samples[ii, :] >= 0) & (sweep_samples[ii, :] <= 1)
            )

        # test plotting runs
        def fun(samples):
            return np.stack(
                (np.sum(samples**2, axis=0), np.sum(samples, axis=0)), axis=1
            )

        ax = plt.subplots(1, 1)[1]
        sweep_vals = fun(sweep_samples)
        # plot 1st qoi of 1st sweep
        sweeper.plot_single_qoi_sweep(sweep_vals[:, 0], 0, ax)
        # plot 2nd qoi of  3rd sweep
        sweeper.plot_single_qoi_sweep(sweep_vals[:, 1], 2, ax)

    def test_get_gaussian_parameter_sweep_samples(self):
        nvars = 2
        sweep_radius = 2
        nsamples_per_sweep = 50
        nsweeps = 2
        mean = np.ones((nvars, 1))
        covariance = np.asarray([[1, 0.7], [0.7, 1.0]])

        cov_sqrt = np.linalg.cholesky(covariance)
        sweeper = GaussianParameterSweeper(
            mean, lambda x: cov_sqrt @ x, sweep_radius, nsamples_per_sweep
        )
        samples = sweeper.rvs(nsweeps)

        for ii in range(nsweeps):
            # true bounds of sweep in standard normal space
            stdnormal_sweep_bounds_samples = np.zeros((nvars, 2))
            stdnormal_sweep_bounds_samples[ii, 0] = -sweep_radius
            stdnormal_sweep_bounds_samples[ii, 1] = sweep_radius
            # true bounds of sweep in correlated Gaussian space
            correlated_sweep_bounds_samples = mean + np.dot(
                cov_sqrt, stdnormal_sweep_bounds_samples
            )
            # pdf values of true bounds in correlated gaussian space
            # Note: multivariate_normal.pdf() takes samples.T
            # because it stores variables in each column, unlike in this
            # package, which stores each variable in a different row
            true_sweep_bounds_pdf_vals = stats.multivariate_normal.pdf(
                correlated_sweep_bounds_samples.T,
                mean=mean[:, 0],
                cov=covariance,
            )

            # pdf values of computed sweep bounds
            lb = nsamples_per_sweep * ii
            ub = nsamples_per_sweep * (ii + 1) - 1
            sweep_bounds_samples = samples[:, [lb, ub]]
            sweep_bounds_pdf_vals = stats.multivariate_normal.pdf(
                sweep_bounds_samples.T, mean=mean[:, 0], cov=covariance
            )
            assert np.allclose(
                sweep_bounds_pdf_vals, true_sweep_bounds_pdf_vals
            )


if __name__ == "__main__":
    parameter_sweeps_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestParameterSweeps
    )
    unittest.TextTestRunner(verbosity=2).run(parameter_sweeps_test_suite)
