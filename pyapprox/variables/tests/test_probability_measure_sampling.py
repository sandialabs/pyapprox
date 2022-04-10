import unittest
import numpy as np
from scipy import stats

from pyapprox.variables.transforms import AffineTransform
from pyapprox.variables.marginals import float_rv_discrete
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.sampling import generate_independent_random_samples


class TestProbabilitySampling(unittest.TestCase):
    def setUp(self):
        self.continuous_variables = [
            stats.uniform(-1, 2), stats.beta(1, 1, -1, 2), stats.norm(-1, 2),
            stats.uniform(), stats.uniform(-1, 2), stats.beta(2, 1, -2, 3)]

        self.continuous_mean = np.array(
            [0., 0., -1, 0.5, 0., stats.beta.mean(a=2, b=1, loc=-2, scale=3)])

        nmasses1 = 10
        mass_locations1 = np.geomspace(1.0, 32.0, num=nmasses1)
        masses1 = np.ones(nmasses1, dtype=float)/nmasses1

        nmasses2 = 10
        mass_locations2 = np.arange(0, nmasses2)
        masses2 = np.geomspace(1.0, 32.0, num=nmasses2)
        masses2 /= masses2.sum()
        # second () is to freeze variable which creates var.dist member
        # variable
        var1 = float_rv_discrete(
            name='var1', values=(mass_locations1, masses1))()
        var2 = float_rv_discrete(
            name='var2', values=(mass_locations2, masses2))()
        self.discrete_variables = [var1, var2]
        self.discrete_mean = np.empty(len(self.discrete_variables))
        for ii, var in enumerate(self.discrete_variables):
            self.discrete_mean[ii] = var.moment(1)

    def test_independent_continuous_samples(self):

        variable = IndependentMarginalsVariable(
            self.continuous_variables)

        var_trans = AffineTransform(variable)

        num_samples = int(1e6)
        samples = generate_independent_random_samples(
            var_trans.variable, num_samples)

        mean = samples.mean(axis=1)
        assert np.allclose(mean, self.continuous_mean, atol=1e-2)

    def test_independent_discrete_samples(self):

        variable = IndependentMarginalsVariable(
            self.discrete_variables)
        var_trans = AffineTransform(variable)

        num_samples = int(1e6)
        samples = generate_independent_random_samples(
            var_trans.variable, num_samples)
        mean = samples.mean(axis=1)
        assert np.allclose(mean, self.discrete_mean, rtol=1e-2)

    def test_independent_mixed_continuous_discrete_samples(self):

        univariate_variables = (
            self.continuous_variables+self.discrete_variables)
        II = np.random.permutation(len(univariate_variables))
        univariate_variables = [univariate_variables[ii] for ii in II]

        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)

        num_samples = int(5e6)
        samples = generate_independent_random_samples(
            var_trans.variable, num_samples)
        mean = samples.mean(axis=1)

        true_mean = np.concatenate(
            [self.continuous_mean, self.discrete_mean])[II]
        assert np.allclose(mean, true_mean, atol=1e-2)


if __name__ == '__main__':
    probability_sampling_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestProbabilitySampling)
    unittest.TextTestRunner(verbosity=2).run(probability_sampling_test_suite)
