import unittest
from scipy import stats
import numpy as np

from pyapprox.variables.transforms import (
    map_hypercube_samples, AffineTransform,
    RosenblattTransform,
    NatafTransform, define_iid_random_variable_transformation,
    ComposeTransforms, UniformMarginalTransformation
)
from pyapprox.variables.marginals import float_rv_discrete
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.tests.test_rosenblatt_transformation import (
    rosenblatt_example_2d
)
from pyapprox.variables.nataf import (
    gaussian_copula_compute_x_correlation_from_z_correlation,
    generate_x_samples_using_gaussian_copula, correlation_to_covariance
)
from pyapprox.variables.sampling import generate_independent_random_samples


class TestVariableTransformations(unittest.TestCase):

    def test_map_hypercube_samples(self):
        num_vars = 3
        num_samples = 4
        current_samples = np.random.uniform(0., 1., (num_vars, num_samples))
        current_ranges = np.ones(2*num_vars)
        current_ranges[::2] = 0.
        new_ranges = np.ones(2*num_vars)
        new_ranges[::2] = -1.
        samples = map_hypercube_samples(
            current_samples, current_ranges, new_ranges)
        true_samples = 2*current_samples-1.
        assert np.allclose(true_samples, samples)

        num_vars = 3
        num_samples = 6
        current_samples = np.random.uniform(0., 1., (num_vars, num_samples))
        current_samples[:, [2, 4]] = 1
        current_ranges = np.ones(2*num_vars)
        current_ranges[::2] = 0.
        new_ranges = np.ones(2*num_vars)
        new_ranges[::2] = -1.
        true_samples = 2*current_samples-1.
        true_samples[1, :] = current_samples[1, :]
        # perturb some samples with numerical noise to make sure
        # that these samples are clipped correctly
        current_samples[:, [2, 4]] += np.finfo(float).eps
        samples = map_hypercube_samples(
            current_samples, current_ranges, new_ranges,
            active_vars=[0, 2])
        assert np.allclose(true_samples, samples)

    def test_define_mixed_tensor_product_random_variable(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable
        type the distribution parameters ARE NOT the same
        """
        univariate_variables = [
            stats.uniform(-1, 2), stats.beta(1, 1, -1, 2),
            stats.norm(-1, np.sqrt(4)), stats.uniform(),
            stats.uniform(-1, 2), stats.beta(2, 1, -2, 3)]
        var_trans = AffineTransform(univariate_variables)

        # from pyapprox.variables.sampling import print_statistics
        # print_statistics(IndependentMarginalsVariable(
        #     univariate_variables).rvs(3))
        # print_statistics(IndependentMarginalsVariable(
        #     univariate_variables).rvs(3), np.ones((3, 1)))

        # first sample is on left boundary of all bounded variables
        # and one standard deviation to left of mean for gaussian variable
        # second sample is on right boundary of all bounded variables
        # and one standard deviation to right of mean for gaussian variable
        true_user_samples = np.asarray(
            [[-1, -1, -3, 0, -1, -2], [1, 1, 1, 1, 1, 1]]).T

        canonical_samples = var_trans.map_to_canonical(true_user_samples)
        true_canonical_samples = np.ones_like(true_user_samples)
        true_canonical_samples[:, 0] = -1
        assert np.allclose(true_canonical_samples, canonical_samples)

        user_samples = var_trans.map_from_canonical(canonical_samples)
        assert np.allclose(user_samples, true_user_samples)

    def test_define_mixed_tensor_product_random_variable_contin_discrete(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable
        type the distribution parameters ARE NOT the same
        """
        # parameters of binomial distribution
        num_trials = 10
        prob_success = 0.5
        univariate_variables = [
            stats.uniform(), stats.norm(-1, np.sqrt(4)),
            stats.norm(-1, np.sqrt(4)),
            stats.binom(num_trials, prob_success),
            stats.norm(-1, np.sqrt(4)), stats.uniform(0, 1),
            stats.uniform(0, 1), stats.binom(num_trials, prob_success)]
        var_trans = AffineTransform(univariate_variables)

        # first sample is on left boundary of all bounded variables
        # and one standard deviation to left of mean for gaussian variables
        # second sample is on right boundary of all bounded variables
        # and one standard deviation to right of mean for gaussian variable
        true_user_samples = np.asarray(
            [[0, -3, -3, 0, -3, 0, 0, 0],
             [1, 1, 1, num_trials, 1, 1, 1, 10]]).T

        canonical_samples = var_trans.map_to_canonical(true_user_samples)
        true_canonical_samples = np.ones_like(true_user_samples)
        true_canonical_samples[:, 0] = -1
        true_canonical_samples[5, 0] = -1
        true_canonical_samples[3, :] = [-1, 1]
        true_canonical_samples[7, :] = [-1, 1]
        assert np.allclose(true_canonical_samples, canonical_samples)

        user_samples = var_trans.map_from_canonical(canonical_samples)
        assert np.allclose(user_samples, true_user_samples)

    def test_rosenblatt_transformation(self):

        true_samples, true_canonical_samples, joint_density, limits = \
            rosenblatt_example_2d(num_samples=10)

        num_vars = 2
        opts = {'limits': limits, 'num_quad_samples_1d': 100}
        var_trans = RosenblattTransform(joint_density, num_vars, opts)

        samples = var_trans.map_from_canonical(
            true_canonical_samples)
        assert np.allclose(true_samples, samples)

        canonical_samples = var_trans.map_to_canonical(samples)
        assert np.allclose(true_canonical_samples, canonical_samples)

    def test_nataf_transformation(self):
        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        bisection_opts = {'tol': 1e-10, 'max_iterations': 100}

        def beta_cdf(x): return stats.beta.cdf(x, a=alpha_stat, b=beta_stat)
        def beta_icdf(x): return stats.beta.ppf(x, a=alpha_stat, b=beta_stat)
        x_marginal_cdfs = [beta_cdf]*num_vars
        x_marginal_inv_cdfs = [beta_icdf]*num_vars
        x_marginal_means = np.asarray(
            [stats.beta.mean(a=alpha_stat, b=beta_stat)]*num_vars)
        x_marginal_stdevs = np.asarray(
            [stats.beta.std(a=alpha_stat, b=beta_stat)]*num_vars)

        def beta_pdf(x): return stats.beta.pdf(x, a=alpha_stat, b=beta_stat)
        x_marginal_pdfs = [beta_pdf]*num_vars

        z_correlation = np.array([[1, 0.7], [0.7, 1]])

        x_correlation = \
            gaussian_copula_compute_x_correlation_from_z_correlation(
                x_marginal_inv_cdfs, x_marginal_means, x_marginal_stdevs,
                z_correlation)

        x_covariance = correlation_to_covariance(
            x_correlation, x_marginal_stdevs)

        var_trans = NatafTransform(
            x_marginal_cdfs, x_marginal_inv_cdfs, x_marginal_pdfs,
            x_covariance, x_marginal_means, bisection_opts)

        assert np.allclose(var_trans.z_correlation, z_correlation)

        num_samples = 1000
        true_samples, true_canonical_samples = \
            generate_x_samples_using_gaussian_copula(
                num_vars, z_correlation, x_marginal_inv_cdfs, num_samples)

        canonical_samples = var_trans.map_to_canonical(true_samples)
        assert np.allclose(true_canonical_samples, canonical_samples)

        samples = var_trans.map_from_canonical(
            true_canonical_samples)
        assert np.allclose(true_samples, samples)

    def test_transformation_composition_I(self):

        np.random.seed(2)
        true_samples, true_canonical_samples, joint_density, limits = \
            rosenblatt_example_2d(num_samples=10)

        #  rosenblatt_example_2d is defined on [0,1] remap to [-1,1]
        true_canonical_samples = true_canonical_samples*2-1

        num_vars = 2
        opts = {'limits': limits, 'num_quad_samples_1d': 100}
        var_trans_1 = RosenblattTransform(joint_density, num_vars, opts)
        var_trans_2 = define_iid_random_variable_transformation(
            stats.uniform(0, 1), num_vars)
        var_trans = ComposeTransforms([var_trans_1, var_trans_2])

        samples = var_trans.map_from_canonical(
            true_canonical_samples)
        assert np.allclose(true_samples, samples)

        canonical_samples = var_trans.map_to_canonical(samples)
        assert np.allclose(true_canonical_samples, canonical_samples)

    def test_transformation_composition_II(self):
        num_vars = 2
        alpha_stat = 5
        beta_stat = 2
        def beta_cdf(x): return stats.beta.cdf(x, a=alpha_stat, b=beta_stat)
        def beta_icdf(x): return stats.beta.ppf(x, a=alpha_stat, b=beta_stat)
        x_marginal_cdfs = [beta_cdf]*num_vars
        x_marginal_inv_cdfs = [beta_icdf]*num_vars
        x_marginal_means = np.asarray(
            [stats.beta.mean(a=alpha_stat, b=beta_stat)]*num_vars)
        x_marginal_stdevs = np.asarray(
            [stats.beta.std(a=alpha_stat, b=beta_stat)]*num_vars)

        def beta_pdf(x): return stats.beta.pdf(x, a=alpha_stat, b=beta_stat)
        x_marginal_pdfs = [beta_pdf]*num_vars

        z_correlation = -0.9*np.ones((num_vars, num_vars))
        for ii in range(num_vars):
            z_correlation[ii, ii] = 1.

        x_correlation = \
            gaussian_copula_compute_x_correlation_from_z_correlation(
                x_marginal_inv_cdfs, x_marginal_means, x_marginal_stdevs,
                z_correlation)
        x_covariance = correlation_to_covariance(
            x_correlation, x_marginal_stdevs)

        var_trans_1 = NatafTransform(
            x_marginal_cdfs, x_marginal_inv_cdfs, x_marginal_pdfs,
            x_covariance, x_marginal_means)

        # rosenblatt maps to [0,1] but polynomials of bounded variables
        # are in [-1,1] so add second transformation for this second mapping
        def normal_cdf(x): return stats.norm.cdf(x)
        def normal_icdf(x): return stats.norm.ppf(x)
        std_normal_marginal_cdfs = [normal_cdf]*num_vars
        std_normal_marginal_inv_cdfs = [normal_icdf]*num_vars
        var_trans_2 = UniformMarginalTransformation(
            std_normal_marginal_cdfs, std_normal_marginal_inv_cdfs)
        var_trans = ComposeTransforms([var_trans_1, var_trans_2])

        num_samples = 1000
        true_samples, true_canonical_samples = \
            generate_x_samples_using_gaussian_copula(
                num_vars, z_correlation, x_marginal_inv_cdfs, num_samples)
        true_canonical_samples = stats.norm.cdf(true_canonical_samples)

        samples = var_trans.map_from_canonical(
            true_canonical_samples)
        assert np.allclose(true_samples, samples)

        canonical_samples = var_trans.map_to_canonical(samples)
        assert np.allclose(true_canonical_samples, canonical_samples)

    def test_pickle_rosenblatt_transformation(self):
        import pickle
        import os
        true_samples, true_canonical_samples, joint_density, limits = \
            rosenblatt_example_2d(num_samples=10)

        num_vars = 2
        opts = {'limits': limits, 'num_quad_samples_1d': 100}
        var_trans = RosenblattTransform(joint_density, num_vars, opts)

        filename = 'rv_trans.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(var_trans, f)

        with open(filename, 'rb') as f:
            pickle.load(f)

        os.remove(filename)

    def test_pickle_affine_random_variable_transformation(self):
        import pickle
        import os

        num_vars = 2
        alpha_stat = 2
        beta_stat = 10
        var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat, 0, 1), num_vars)

        filename = 'rv_trans.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(var_trans, f)

        with open(filename, 'rb') as f:
            pickle.load(f)

        os.remove(filename)

    def test_map_rv_discrete(self):
        nvars = 2

        mass_locs = np.arange(5, 501, step=50)
        nmasses = mass_locs.shape[0]
        mass_probs = np.ones(nmasses, dtype=float)/float(nmasses)
        univariate_variables = [
            float_rv_discrete(name='float_rv_discrete',
                              values=(mass_locs, mass_probs))()]*nvars

        variables = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variables)

        samples = np.vstack(
            [mass_locs[np.newaxis, :], mass_locs[0]*np.ones((1, nmasses))])
        canonical_samples = var_trans.map_to_canonical(samples)

        assert(canonical_samples[0].min() == -1)
        assert(canonical_samples[0].max() == 1)

        recovered_samples = var_trans.map_from_canonical(
            canonical_samples)
        assert np.allclose(recovered_samples, samples)

    def test_identity_map_subset(self):
        num_vars = 3
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(0, 1), num_vars)
        var_trans.set_identity_maps([1])

        samples = np.random.uniform(0, 1, (num_vars, 4))
        canonical_samples = var_trans.map_to_canonical(samples)
        assert np.allclose(canonical_samples[1, :], samples[1, :])

        assert np.allclose(
            var_trans.map_from_canonical(canonical_samples), samples)

        univariate_variables = [
            stats.uniform(-1, 2), stats.beta(1, 1, -1, 2),
            stats.norm(-1, np.sqrt(4)), stats.uniform(),
            stats.uniform(-1, 2), stats.beta(2, 1, -2, 3)]
        var_trans = AffineTransform(univariate_variables)
        var_trans.set_identity_maps([4, 2])

        samples = generate_independent_random_samples(var_trans.variable, 10)
        canonical_samples = var_trans.map_to_canonical(samples)
        assert np.allclose(canonical_samples[[2, 4], :], samples[[2, 4], :])

        assert np.allclose(
            var_trans.map_from_canonical(canonical_samples), samples)

    def test_map_derivatives(self):
        nvars = 2
        nsamples = 10
        x = np.random.uniform(0, 1, (nvars, nsamples))
        # vals = np.sum(x**2, axis=0)[:, None]
        grad = np.vstack([2*x[ii:ii+1, :] for ii in range(nvars)])
        var_trans = AffineTransform(
            [stats.uniform(0, 1), stats.uniform(2, 2)])
        canonical_derivs = var_trans.map_derivatives_to_canonical_space(grad)
        for ii in range(nvars):
            lb, ub = var_trans.variable.marginals()[ii].interval(1)
            assert np.allclose(canonical_derivs[ii, :], (ub-lb)*grad[ii, :]/2)
        recovered_derivs = var_trans.map_derivatives_from_canonical_space(
            canonical_derivs)
        assert np.allclose(recovered_derivs, grad)


if __name__ == "__main__":
    variable_transformations_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestVariableTransformations)
    unittest.TextTestRunner(verbosity=2).run(
        variable_transformations_test_suite)
