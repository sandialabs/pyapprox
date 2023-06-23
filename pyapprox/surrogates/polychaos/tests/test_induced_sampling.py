import unittest
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.surrogates.orthopoly.leja_sequences import christoffel_function
from pyapprox.surrogates.polychaos.induced_sampling import (
    continuous_induced_measure_cdf,
    continuous_induced_measure_ppf, generate_induced_samples,
    discrete_induced_sampling, basis_matrix_generator_1d,
    random_induced_measure_sampling, idistinv_jacobi,
    inverse_transform_sampling_1d,
    generate_induced_samples_migliorati_tolerance,
    compute_preconditioned_basis_matrix_condition_number,
    increment_induced_samples_migliorati
)
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, compute_hyperbolic_level_indices
)
from pyapprox.variables.marginals import (
    float_rv_discrete, get_probability_masses
)
from pyapprox.variables.transforms import AffineTransform
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation
)
from pyapprox.variables.density import tensor_product_pdf
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence
)
from pyapprox.surrogates.orthopoly.recursion_factory import (
    get_recursion_coefficients_from_variable
)
from pyapprox.util.utilities import cartesian_product, outer_product


class TestInducedSampling(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_continous_induced_measure_ppf(self):
        degree = 2
        alpha_stat, beta_stat = 3, 3
        ab = jacobi_recurrence(
            degree+1, alpha=beta_stat-1, beta=alpha_stat-1, probability=True)

        tol = 1e-15
        var = stats.beta(alpha_stat, beta_stat, -5, 10)
        can_lb, can_ub = -1, 1
        lb, ub = var.support()
        print(lb, ub)
        cx = np.linspace(can_lb, can_ub, 51)

        def can_pdf(xx):
            loc, scale = lb+(ub-lb)/2, (ub-lb)/2
            return var.pdf(xx*scale+loc)*scale

        cdf_vals = continuous_induced_measure_cdf(
            can_pdf, ab, degree, can_lb, can_ub, tol, cx)
        assert np.all(cdf_vals <= 1.0)
        ppf_vals = continuous_induced_measure_ppf(
            var, ab, degree, cdf_vals, 1e-10, 1e-8)
        assert np.allclose(cx, ppf_vals)

        try:
            var = stats.loguniform(1.e-5, 1.e-3)
        except:
            var = stats.reciprocal(1.e-5, 1.e-3)
        ab = get_recursion_coefficients_from_variable(var, degree+5, {})
        can_lb, can_ub = -1, 1
        cx = np.linspace(can_lb, can_ub, 51)
        lb, ub = var.support()

        def can_pdf(xx):
            loc, scale = lb+(ub-lb)/2, (ub-lb)/2
            return var.pdf(xx*scale+loc)*scale
        cdf_vals = continuous_induced_measure_cdf(
            can_pdf, ab, degree, can_lb, can_ub, tol, cx)
        # differences caused by root finding optimization tolerance
        assert np.all(cdf_vals <= 1.0)
        ppf_vals = continuous_induced_measure_ppf(
            var, ab, degree, cdf_vals, 1e-10, 1e-8)
        # import matplotlib.pyplot as plt
        # plt.plot(cx, cdf_vals)
        # plt.plot(ppf_vals, cdf_vals, 'r*', ms=2)
        # plt.show()
        assert np.allclose(cx, ppf_vals)

    def help_discrete_induced_sampling(self, var1, var2, envelope_factor):
        degree = 3

        var_trans = AffineTransform([var1, var2])
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(pce.num_vars(), degree, 1.0)
        pce.set_indices(indices)

        num_samples = int(3e4)
        np.random.seed(1)
        canonical_samples = generate_induced_samples(pce, num_samples)
        samples = var_trans.map_from_canonical(canonical_samples)

        np.random.seed(1)
        # canonical_xk = [2*get_distribution_info(var1)[2]['xk']-1,
        #                2*get_distribution_info(var2)[2]['xk']-1]
        xk = np.array(
            [get_probability_masses(var)[0]
             for var in var_trans.variable.marginals()])
        pk = np.array(
            [get_probability_masses(var)[1]
             for var in var_trans.variable.marginals()])
        canonical_xk = var_trans.map_to_canonical(xk)
        basis_matrix_generator = partial(
            basis_matrix_generator_1d, pce, degree)
        canonical_samples1 = discrete_induced_sampling(
            basis_matrix_generator, pce.indices, canonical_xk,
            pk, num_samples)
        samples1 = var_trans.map_from_canonical(canonical_samples1)

        def univariate_pdf(var, x):
            if hasattr(var, 'pdf'):
                return var.pdf(x)
            elif hasattr(var, "pmf"):
                return var.pmf(x)
            else:
                print(var.__dict__.keys())
                raise RuntimeError()
            # xk, pk = get_probability_masses(var)
                # x = np.atleast_1d(x)
                # vals = np.zeros(x.shape[0])
                # for jj in range(x.shape[0]):
                #     for ii in range(xk.shape[0]):
                #         if xk[ii] == x[jj]:
                #             vals[jj] = pk[ii]
                #             break
                # return vals

        def density(x):
            # some issue with native scipy.pmf
            # assert np.allclose(var1.pdf(x[0, :]),var1.pmf(x[0, :]))
            return univariate_pdf(var1, x[0, :])*univariate_pdf(var2, x[1, :])

        def generate_proposal_samples(n):
            samples = np.vstack([var1.rvs(n), var2.rvs(n)])
            return samples
        proposal_density = density

        # unlike fekete and leja sampling can and should use
        # pce.basis_matrix here. If use canonical_basis_matrix then
        # densities must be mapped to this space also which can be difficult
        samples2 = random_induced_measure_sampling(
            num_samples, pce.num_vars(), pce.basis_matrix, density,
            proposal_density, generate_proposal_samples, envelope_factor)

        def induced_density(x):
            vals = density(x)*christoffel_function(
                x, pce.basis_matrix, True)
            return vals

        quad_samples = cartesian_product([xk[0], xk[1]])
        quad_weights = outer_product([pk[0], pk[1]])

        # print(canonical_samples.min(axis=1),canonical_samples.max(axis=1))
        # print(samples.min(axis=1),samples.max(axis=1))
        # print(canonical_samples1.min(axis=1),canonical_samples1.max(axis=1))
        # print(samples1.min(axis=1),samples1.max(axis=1))
        # import matplotlib.pyplot as plt
        # plt.plot(quad_samples[0,:],quad_samples[1,:],'s')
        # plt.plot(samples[0,:],samples[1,:],'o')
        # plt.plot(samples1[0,:],samples1[1,:],'*')
        # plt.show()

        rtol = 1e-2
        assert np.allclose(quad_weights, density(quad_samples))
        assert np.allclose(density(quad_samples).sum(), 1)
        assert np.allclose(
            christoffel_function(quad_samples, pce.basis_matrix, True).dot(
                quad_weights), 1.0)
        true_induced_mean = quad_samples.dot(induced_density(quad_samples))
        # print(true_induced_mean)
        # print(samples.mean(axis=1))
        # print(samples1.mean(axis=1))
        # print(samples2.mean(axis=1))
        # print(samples1.mean(axis=1)-true_induced_mean,
        #       true_induced_mean*rtol)
        # print(samples2.mean(axis=1))
        assert np.allclose(samples.mean(axis=1), true_induced_mean, rtol=rtol)
        assert np.allclose(samples1.mean(axis=1), true_induced_mean, rtol=rtol)
        assert np.allclose(samples2.mean(axis=1), true_induced_mean, rtol=rtol)

    def test_discrete_induced_sampling(self):
        nmasses1 = 10
        mass_locations1 = np.geomspace(1.0, 512.0, num=nmasses1)
        # mass_locations1 = np.arange(0,nmasses1)
        masses1 = np.ones(nmasses1, dtype=float)/nmasses1
        var1 = float_rv_discrete(
            name='float_rv_discrete', values=(mass_locations1, masses1))()
        nmasses2 = 10
        mass_locations2 = np.arange(0, nmasses2)
        # if increase from 16 unmodififed becomes ill conditioned
        masses2 = np.geomspace(1.0, 16.0, num=nmasses2)
        # masses2  = np.ones(nmasses2,dtype=float)/nmasses2
        masses2 /= masses2.sum()
        var2 = float_rv_discrete(
            name='float_rv_discrete', values=(mass_locations2, masses2))()
        self.help_discrete_induced_sampling(var1, var2, 30)

        num_type1, num_type2, num_trials = [10, 10, 9]
        var1 = stats.hypergeom(num_type1+num_type2, num_type1, num_trials)
        var2 = var1
        self.help_discrete_induced_sampling(var1, var2, 300)

        num_type1, num_type2, num_trials = [10, 10, 9]
        var1 = stats.binom(10, 0.5)
        var2 = var1
        self.help_discrete_induced_sampling(var1, var2, 300)

        N = 10
        xk, pk = np.arange(N), np.ones(N)/N
        var1 = float_rv_discrete(name='discrete_chebyshev', values=(xk, pk))()
        var2 = var1
        self.help_discrete_induced_sampling(var1, var2, 30)

    def test_multivariate_sampling_jacobi(self):

        num_vars = 2
        degree = 2
        alph = 1
        bet = 1.
        univ_inv = partial(idistinv_jacobi, alph=alph, bet=bet)
        num_samples = 10
        indices = np.ones((2, num_samples), dtype=int)*degree
        indices[1, :] = degree-1
        xx = np.tile(
            np.linspace(0.01, 0.99, (num_samples))[np.newaxis, :],
            (num_vars, 1))
        samples = univ_inv(xx, indices)

        var_trans = AffineTransform(
            [stats.beta(bet+1, alph+1, -1, 2),
             stats.beta(bet+1, alph+1, -1, 2)])
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        pce.set_indices(indices)

        reference_samples = inverse_transform_sampling_1d(
            pce.var_trans.variable.unique_variables[0],
            pce.recursion_coeffs[0], degree, xx[0, :])
        # differences are just caused by different tolerances in optimizes
        # used to find roots of CDF
        assert np.allclose(reference_samples, samples[0, :], atol=1e-7)
        reference_samples = inverse_transform_sampling_1d(
            pce.var_trans.variable.unique_variables[0],
            pce.recursion_coeffs[0], degree-1, xx[0, :])
        assert np.allclose(reference_samples, samples[1, :], atol=1e-7)

        # num_samples = 30
        # samples = generate_induced_samples(pce,num_samples)
        # plt.plot(samples[0,:],samples[1,:],'o'); plt.show()

    def test_multivariate_migliorati_sampling_jacobi(self):

        num_vars = 1
        degree = 20
        alph = 5
        bet = 5.
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.beta(alph, bet, -1, 2)], [np.arange(num_vars)]))
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        pce.set_indices(indices)

        cond_tol = 1e1
        samples = generate_induced_samples_migliorati_tolerance(pce, cond_tol)
        cond = compute_preconditioned_basis_matrix_condition_number(
            pce.canonical_basis_matrix, samples)
        assert cond < cond_tol
        # plt.plot(samples[0,:],samples[1,:],'o'); plt.show()

    def test_adaptive_multivariate_sampling_jacobi(self):

        num_vars = 2
        degree = 6
        alph = 5
        bet = 5.

        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.beta(alph, bet, -1, 3)], [np.arange(num_vars)]))
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, 1, 1.0)
        pce.set_indices(indices)
        cond_tol = 1e2
        samples = generate_induced_samples_migliorati_tolerance(pce, cond_tol)

        for dd in range(2, degree):
            # num_prev_samples = samples.shape[1]
            new_indices = compute_hyperbolic_level_indices(num_vars, dd, 1.)
            samples = increment_induced_samples_migliorati(
                pce, cond_tol, samples, indices, new_indices)
            indices = np.hstack((indices, new_indices))
            pce.set_indices(indices)
            # new_samples = samples[:, num_prev_samples:]
            # prev_samples = samples[:, :num_prev_samples]
            # fig,axs = plt.subplots(1,2,figsize=(2*8,6))
            # from pyapprox.util.visualization import plot_2d_indices
            # axs[0].plot(prev_samples[0,:],prev_samples[1,:],'ko');
            # axs[0].plot(new_samples[0,:],new_samples[1,:],'ro');
            # plot_2d_indices(indices,other_indices=new_indices,ax=axs[1]);
            # plt.show()

        samples = var_trans.map_from_canonical(samples)
        cond = compute_preconditioned_basis_matrix_condition_number(
            pce.basis_matrix, samples)
        assert cond < cond_tol

    def test_random_christoffel_sampling(self):
        num_vars = 2
        degree = 10

        alpha_poly = 1
        beta_poly = 1

        alpha_stat = beta_poly+1
        beta_stat = alpha_poly+1

        num_samples = int(1e4)
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat), num_vars)
        opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        univariate_pdf = partial(stats.beta.pdf, a=alpha_stat, b=beta_stat)
        probability_density = partial(
            tensor_product_pdf, univariate_pdfs=univariate_pdf)

        envelope_factor = 10
        def generate_proposal_samples(n): return np.random.uniform(
            0., 1., size=(num_vars, n))

        def proposal_density(x): return np.ones(x.shape[1])

        # unlike fekete and leja sampling can and should use
        # pce.basis_matrix here. If use canonical_basis_matrix then
        # densities must be mapped to this space also which can be difficult
        samples = random_induced_measure_sampling(
            num_samples, num_vars, poly.basis_matrix, probability_density,
            proposal_density, generate_proposal_samples, envelope_factor)

        def univariate_quadrature_rule(x):
            x, w = gauss_jacobi_pts_wts_1D(x, alpha_poly, beta_poly)
            x = (x+1)/2
            return x, w
        x, w = get_tensor_product_quadrature_rule(
            degree*2+1, num_vars, univariate_quadrature_rule)
        # print(samples.mean(axis=1),x.dot(w))
        assert np.allclose(
            christoffel_function(x, poly.basis_matrix, True).dot(w), 1.0)
        assert np.allclose(x.dot(w), samples.mean(axis=1), atol=1e-2)

        # from pyapprox.util.visualization import get_meshgrid_function_data
        # def induced_density(x):
        #     vals=christoffel_function(x,poly.basis_matrix)*probability_density(
        #         x)/np.sqrt(poly.indices.shape[1])
        #     return vals
        # X,Y,Z = get_meshgrid_function_data(
        #     induced_density,[0,1,0,1], 50)
        # levels = np.linspace(0,Z.max(),20)
        # cset = plt.contourf(X, Y, Z, levels=levels)
        # plt.plot(samples[0,:],samples[1,:],'o')
        # plt.plot(x[0,:],x[1,:],'s')
        # plt.show()


if __name__ == "__main__":
    induced_sampling_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestInducedSampling)
    unittest.TextTestRunner(verbosity=2).run(induced_sampling_test_suite)
