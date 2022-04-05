import unittest
import numpy as np
from scipy.stats import multivariate_normal

from pyapprox.analysis.parameter_sweeps import (
    get_hypercube_parameter_sweeps_samples,
    get_gaussian_parameter_sweeps_samples
)


class TestParameterSweeps(unittest.TestCase):
    def test_get_hypercube_parameter_sweeps_samples(self):
        ranges = np.asarray([0., 1., 0., 1])
        samples, active_samples, W = get_hypercube_parameter_sweeps_samples(
            ranges, num_samples_per_sweep=50, num_sweeps=1)

        for ii in range(samples.shape[0]):
            assert np.all(
                (samples[ii, :] >= ranges[2*ii]) &
                (samples[ii, :] <= ranges[2*ii+1]))

    def test_get_gaussian_parameter_sweep_samples(self):
        num_vars = 2
        sweep_radius = 2
        num_samples_per_sweep = 50
        num_sweeps = 2
        mean = np.ones(num_vars)
        covariance = np.asarray([[1, 0.7], [0.7, 1.]])

        samples, active_samples, W = get_gaussian_parameter_sweeps_samples(
            mean, covariance, sweep_radius=sweep_radius,
            num_samples_per_sweep=num_samples_per_sweep, num_sweeps=num_sweeps)

        covariance_chol_factor = np.linalg.cholesky(covariance)
        for ii in range(num_sweeps):
            # true bounds of sweep in standard normal space
            stdnormal_sweep_bounds_samples = np.zeros((num_vars, 2))
            stdnormal_sweep_bounds_samples[ii, 0] = -sweep_radius
            stdnormal_sweep_bounds_samples[ii, 1] = sweep_radius
            # true bounds of sweep in correlated Gaussian space
            correlated_sweep_bounds_samples = mean+np.dot(
                covariance_chol_factor, stdnormal_sweep_bounds_samples)
            # pdf values of true bounds in correlated gaussian space
            # Note: multivariate_normal.pdf() takes samples.T
            # because it stores variables in each column, unlike in this
            # package, which stores each variable in a different row
            true_sweep_bounds_pdf_vals = multivariate_normal.pdf(
                correlated_sweep_bounds_samples.T, mean=mean, cov=covariance)

            # pdf values of computed sweep bounds
            lb = num_samples_per_sweep*ii
            ub = num_samples_per_sweep*(ii+1)-1
            sweep_bounds_samples = samples[:, [lb, ub]]
            sweep_bounds_pdf_vals = multivariate_normal.pdf(
                sweep_bounds_samples.T, mean=mean, cov=covariance)
            assert np.allclose(sweep_bounds_pdf_vals,
                               true_sweep_bounds_pdf_vals)


if __name__ == "__main__":
    parameter_sweeps_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestParameterSweeps)
    unittest.TextTestRunner(verbosity=2).run(parameter_sweeps_test_suite)
