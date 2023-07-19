import unittest
import numpy as np
from scipy import stats

from pyapprox.util.utilities import lists_of_arrays_equal
from pyapprox.variables.joint import (
    IndependentMarginalsVariable, GaussCopulaVariable,
    define_iid_random_variable
)
from pyapprox.variables.nataf import correlation_to_covariance


class TestJoint(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_define_mixed_tensor_product_random_variable_I(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable type
        the distribution parameters ARE the same
        """
        univariate_variables = [
            stats.uniform(-1, 2), stats.beta(1, 1, -1, 2), stats.norm(0, 1),
            stats.uniform(-1, 2), stats.uniform(-1, 2),
            stats.beta(1, 1, -1, 2)]
        variable = IndependentMarginalsVariable(univariate_variables)

        assert len(variable.unique_variables) == 3
        assert lists_of_arrays_equal(variable.unique_variable_indices,
                                     [[0, 3, 4], [1, 5], [2]])

    def test_define_mixed_tensor_product_random_variable_II(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable
        type the distribution parameters ARE NOT the same
        """
        univariate_variables = [
            stats.uniform(-1, 2), stats.beta(1, 1, -1, 2),
            stats.norm(-1, 2), stats.uniform(), stats.uniform(-1, 2),
            stats.beta(2, 1, -2, 3)]
        variable = IndependentMarginalsVariable(univariate_variables)

        assert len(variable.unique_variables) == 5
        assert lists_of_arrays_equal(variable.unique_variable_indices,
                                     [[0, 4], [1], [2], [3], [5]])

    def test_get_statistics(self):
        univariate_variables = [
            stats.uniform(2, 4), stats.beta(1, 1, -1, 2), stats.norm(0, 1)]
        variable = IndependentMarginalsVariable(univariate_variables)
        mean = variable.get_statistics('mean')
        assert np.allclose(mean.squeeze(), [4, 0, 0])

        intervals = variable.get_statistics('interval', 1)
        assert np.allclose(intervals, np.array(
            [[2, 6], [-1, 1], [-np.inf, np.inf]]))

    def test_define_iid_random_variable(self):
        """
        Construct a independent and identiically distributed (iid)
        multivariate random variable from the tensor-product of
        the same one-dimensional variable.
        """
        var = stats.norm(loc=2, scale=3)
        num_vars = 2
        iid_variable = define_iid_random_variable(var, num_vars)

        assert len(iid_variable.unique_variables) == 1
        assert np.allclose(
            iid_variable.unique_variable_indices, np.arange(num_vars))

    def test_gauss_copula_variable(self):
        num_samples = 10000
        num_vars, alpha_stat, beta_stat = 2, 2, 5
        marginals = [stats.beta(a=alpha_stat, b=beta_stat)]*num_vars

        x_correlation = np.array([[1, 0.7], [0.7, 1]])
        variable = GaussCopulaVariable(marginals, x_correlation)
        x_samples, true_u_samples = variable.rvs(num_samples, True)
        x_sample_covariance = np.cov(x_samples)

        true_x_covariance = correlation_to_covariance(
            x_correlation, [m.std() for m in marginals])
        assert np.allclose(true_x_covariance, x_sample_covariance, atol=1e-2)


if __name__ == "__main__":
    joint_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestJoint)
    unittest.TextTestRunner(verbosity=2).run(joint_test_suite)
