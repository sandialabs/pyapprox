import unittest
import numpy as np
from functools import partial
from scipy import stats

from pyapprox.surrogates.polychaos.adaptive_polynomial_chaos import (
    variance_pce_refinement_indicator, AdaptiveLejaPCE,
    AdaptiveInducedPCE
)
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    max_level_admissibility_function
)
from pyapprox.surrogates.orthopoly.quadrature import clenshaw_curtis_rule_growth


class TestAdaptivePCE(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def helper(self, function, var_trans, pce, max_level, error_tol):
        max_level_1d = [max_level]*(pce.num_vars)
        max_num_samples = 100

        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_samples, error_tol)
        refinement_indicator = variance_pce_refinement_indicator

        pce.set_function(function, var_trans)
        pce.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        pce.build()
        validation_samples = generate_independent_random_samples(
            var_trans.variable, int(1e3))
        validation_vals = function(validation_samples)
        pce_vals = pce(validation_samples)
        error = np.linalg.norm(pce_vals-validation_vals)/np.sqrt(
            validation_samples.shape[1])
        return error, pce

    def test_adaptive_leja_sampling_I(self):
        """
        If function is isotropic small changes in priority can cause
        different subspace to be refined when using slow vs fast method. This
        will lead to different leja sequence due to the fact that LU
        factorization is dependent on order of the columns, i.e. the order
        in which basis functions are added. The numerical difference is
        caused by applying preconditioning directly to basis matrix using slow
        approach and re-weighting LU_factor matrix with ratio of
        precond_weights/precond_weights_prev using fast approach.
        """
        num_vars = 2
        alph = 5
        bet = 5.

        # randomize coefficients of random variables to create anisotropy
        a = np.random.uniform(0, 1, (num_vars, 1))
        def function(x): return np.sum(a*x**2, axis=0)[:, np.newaxis]

        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.beta(alph, bet, 0, 1)], [np.arange(num_vars)]))

        candidate_samples = -np.cos(
            np.random.uniform(0, np.pi, (num_vars, int(1e4))))
        pce = AdaptiveLejaPCE(
            num_vars, candidate_samples, factorization_type='fast')
        error, pce_slow = self.helper(function, var_trans, pce, 2, 0.)
        print('leja sampling error', error)
        assert error < 1e-14

        pce = AdaptiveLejaPCE(
            num_vars, candidate_samples, factorization_type='slow')
        error, pce_fast = self.helper(function, var_trans, pce, 2, 0.)
        print('leja sampling error', error)
        assert error < 1e-14

        assert np.allclose(pce_slow.samples, pce_fast.samples)

    def test_adaptive_leja_sampling_II(self):
        """
        Using variance refinement indicator on additive function can
        lead to some polynomial terms with more than one active variable.
        This is because errors on these terms will be small but not
        necessarily near machine precision. Consequently test that index
        set is additive except for terms corresponding to subspace [1,1]
        with only moderate accuracy 1e-6

        """
        num_vars = 2
        alph = 5
        bet = 5.
        error_tol = 1e-7

        # randomize coefficients of random variables to create anisotropy
        a = np.random.uniform(0, 1, (num_vars, 1))

        def function(x):
            vals = [np.cos(np.pi*a[ii]*x[ii, :]) for ii in range(x.shape[0])]
            vals = np.array(vals).sum(axis=0)[:, np.newaxis]
            return vals
        # function = lambda x: np.sum(a*x**2,axis=0)[:,np.newaxis]

        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.beta(alph, bet, 0, 1)], [np.arange(num_vars)]))

        candidate_samples = -np.cos(
            np.random.uniform(0, np.pi, (num_vars, int(1e4))))
        pce = AdaptiveLejaPCE(
            num_vars, candidate_samples, factorization_type='fast')
        error, pce_slow = self.helper(
            function, var_trans, pce, np.inf, error_tol)
        print('leja sampling error', error)
        assert error < 10*error_tol

        # assert index is additive except for [1,1] subspace terms
        subspace_num_active_vars = np.count_nonzero(
            pce.subspace_indices, axis=0)
        assert np.where(subspace_num_active_vars > 1)[0].shape[0] == 1

        # from pyapprox.surrogates.interp.sparse_grid import plot_sparse_grid_2d
        # import matplotlib.pyplot as plt
        # plot_sparse_grid_2d(
        #    pce.samples,np.ones(pce.samples.shape[1]),
        #    pce.pce.indices, pce.subspace_indices)
        # plt.show()

    def test_adaptive_least_squares_induced_sampling(self):
        num_vars = 2
        alph = 5
        bet = 5.

        def function(x):
            vals = [np.cos(np.pi*x[ii, :]) for ii in range(x.shape[0])]
            vals = np.array(vals).sum(axis=0)[:, np.newaxis]
            return vals
        # function = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.beta(alph, bet, 0, 1)], [np.arange(num_vars)]))

        pce = AdaptiveInducedPCE(num_vars, cond_tol=1e2)
        error, pce = self.helper(function, var_trans, pce, 4, 0.)

        print('induced sampling error', error)
        assert error < 1e-7

    def test_adaptive_least_squares_probability_measure_sampling(self):
        num_vars = 2
        alph = 5
        bet = 5.

        def function(x):
            vals = [np.cos(np.pi*x[ii, :]) for ii in range(x.shape[0])]
            vals = np.array(vals).sum(axis=0)[:, np.newaxis]
            return vals
        # function = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.beta(alph, bet, 0, 1)], [np.arange(num_vars)]))

        pce = AdaptiveInducedPCE(
            num_vars, cond_tol=1e2, induced_sampling=False)
        pce.sample_ratio = 2
        error, pce = self.helper(function, var_trans, pce, 4, 0.)

        print('probability sampling error', error)
        assert error < 5e-8


if __name__ == '__main__':
    adaptive_pce_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestAdaptivePCE)
    unittest.TextTestRunner(verbosity=2).run(adaptive_pce_test_suite)
