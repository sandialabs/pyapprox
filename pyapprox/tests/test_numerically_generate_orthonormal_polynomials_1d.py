import unittest

import numpy as np

from scipy.stats import binom, hypergeom
from scipy import stats
from scipy.special import factorial

from functools import partial

from pyapprox.numerically_generate_orthonormal_polynomials_1d import *
from pyapprox.orthonormal_polynomials_1d import *
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D, \
    gauss_hermite_pts_wts_1D
from pyapprox.variables import float_rv_discrete


class TestNumericallyGenerateOrthonormalPolynomials1D(unittest.TestCase):
    def test_krawtchouk(self):
        num_coef = 6
        ntrials = 10
        p = 0.5

        xk = np.array(range(ntrials+1), dtype='float')
        pk = binom.pmf(xk, ntrials, p)

        ab_lanczos = lanczos(xk, pk, num_coef)
        ab_stieltjes = stieltjes(xk, pk, num_coef)

        ab_exact = krawtchouk_recurrence(num_coef, ntrials, p)

        # ab_lanczos[-1, 0] is a dummy entry so set to exact so
        # comparison will pass if all other entries are correct
        ab_lanczos[-1, 0] = ab_exact[-1, 0]
        
        assert np.allclose(ab_lanczos, ab_exact)
        assert np.allclose(ab_stieltjes, ab_exact)

        from pyapprox.univariate_quadrature import gauss_quadrature
        x, w = gauss_quadrature(ab_lanczos, num_coef)
        moments = np.array([(x**ii).dot(w) for ii in range(num_coef)])
        true_moments = np.array([(xk**ii).dot(pk)for ii in range(num_coef)])
        assert np.allclose(moments, true_moments)
        p = evaluate_orthonormal_polynomial_1d(x, num_coef-1, ab_lanczos)
        assert np.allclose((p.T*w).dot(p), np.eye(num_coef))
        p = evaluate_orthonormal_polynomial_1d(xk, num_coef-1, ab_lanczos)
        assert np.allclose((p.T*pk).dot(p), np.eye(num_coef))

    def test_discrete_chebyshev(self):
        num_coef = 5
        nmasses = 10

        xk = np.array(range(nmasses), dtype='float')
        pk = np.ones(nmasses)/nmasses

        ab_lanczos = lanczos(xk, pk, num_coef)
        ab_stieltjes = stieltjes(xk, pk, num_coef)

        ab_exact = discrete_chebyshev_recurrence(num_coef, nmasses)

        # ab_lanczos[-1, 0] is a dummy entry so set to exact so
        # comparison will pass if all other entries are correct
        ab_lanczos[-1, 0] = ab_exact[-1, 0]
        
        assert np.allclose(ab_lanczos, ab_exact)
        assert np.allclose(ab_stieltjes, ab_exact)

        from pyapprox.univariate_quadrature import gauss_quadrature
        x, w = gauss_quadrature(ab_lanczos, num_coef)
        moments = np.array([(x**ii).dot(w) for ii in range(num_coef)])
        true_moments = np.array([(xk**ii).dot(pk)for ii in range(num_coef)])
        assert np.allclose(moments, true_moments)
        p = evaluate_orthonormal_polynomial_1d(x, num_coef-1, ab_lanczos)
        assert np.allclose((p.T*w).dot(p), np.eye(num_coef))
        p = evaluate_orthonormal_polynomial_1d(xk, num_coef-1, ab_lanczos)
        assert np.allclose((p.T*pk).dot(p), np.eye(num_coef))

    def test_float_rv_discrete(self):
        num_coef, nmasses = 5, 10
        # works for both lanczos and chebyshev algorithms
        #xk   = np.geomspace(1,512,num=nmasses)
        #pk   = np.ones(nmasses)/nmasses

        # works only for chebyshev algorithms
        pk = np.geomspace(1, 512, num=nmasses)
        pk /= pk.sum()
        xk = np.arange(0, nmasses)

        #ab  = lanczos(xk,pk,num_coef)
        ab = modified_chebyshev_orthonormal(
            num_coef, [xk, pk], probability=True)

        from pyapprox.univariate_quadrature import gauss_quadrature
        x, w = gauss_quadrature(ab, num_coef)
        moments = np.array([(x**ii).dot(w) for ii in range(num_coef)])
        true_moments = np.array([(xk**ii).dot(pk)for ii in range(num_coef)])
        assert np.allclose(moments, true_moments), (moments, true_moments)
        p = evaluate_orthonormal_polynomial_1d(x, num_coef-1, ab)
        assert np.allclose((p.T*w).dot(p), np.eye(num_coef))
        p = evaluate_orthonormal_polynomial_1d(xk, num_coef-1, ab)
        assert np.allclose((p.T*pk).dot(p), np.eye(num_coef))

    def test_modified_chebyshev(self):
        nterms = 10
        alpha_stat, beta_stat = 2, 2
        probability_measure = True
        # using scipy to compute moments is extermely slow
        # moments = [stats.beta.moment(n,alpha_stat,beta_stat,loc=-1,scale=2)
        #           for n in range(2*nterms)]
        quad_x, quad_w = gauss_jacobi_pts_wts_1D(
            4*nterms, beta_stat-1, alpha_stat-1)

        true_ab = jacobi_recurrence(
            nterms, alpha=beta_stat-1, beta=alpha_stat-1,
            probability=probability_measure)

        ab = modified_chebyshev_orthonormal(
            nterms, [quad_x, quad_w], get_input_coefs=None, probability=True)
        assert np.allclose(true_ab, ab)

        get_input_coefs = partial(
            jacobi_recurrence, alpha=beta_stat-2, beta=alpha_stat-2)
        ab = modified_chebyshev_orthonormal(
            nterms, [quad_x, quad_w], get_input_coefs=get_input_coefs,
            probability=True)
        assert np.allclose(true_ab, ab)

    def test_rv_discrete_large_moments(self):
        """
        When Modified_chebyshev_orthonormal is used when the moments of discrete
        variable are very large it will fail. To avoid this rescale the 
        variables to [-1,1] like is done for continuous random variables
        """
        N, degree = 100, 5
        xk, pk = np.arange(N), np.ones(N)/N
        rv = float_rv_discrete(name='float_rv_discrete', values=(xk, pk))
        xk_canonical = xk/(N-1)*2-1
        ab = modified_chebyshev_orthonormal(
            degree+1, [xk_canonical, pk])
        p = evaluate_orthonormal_polynomial_1d(xk_canonical, degree, ab)
        w = rv.pmf(xk)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

        ab = predictor_corrector(
            degree+1, (xk_canonical, pk), xk_canonical.min(),
            xk_canonical.max(),
            interval_size=xk_canonical.max()-xk_canonical.min())
        p = evaluate_orthonormal_polynomial_1d(xk_canonical, degree, ab)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_predictor_corrector_known_scipy_pdf(self):
        nterms = 5
        quad_options = {'nquad_samples': 10, 'atol': 1e-8, 'rtol': 1e-8,
                        'max_steps': 10000, 'verbose': 1}

        rv = stats.beta(1, 1, -1, 2)
        ab = predictor_corrector_known_scipy_pdf(nterms, rv, quad_options)
        true_ab = jacobi_recurrence(nterms, 0, 0)
        assert np.allclose(ab, true_ab)

        rv = stats.norm()
        ab = predictor_corrector_known_scipy_pdf(nterms, rv, quad_options)
        true_ab = hermite_recurrence(nterms)
        assert np.allclose(ab, true_ab)

        # lognormal is a very hard test
        rv = stats.lognorm(1)

        # mean, std = 1e4, 7.5e3
        # beta = std*np.sqrt(6)/np.pi
        # mu = mean - beta*np.euler_gamma
        # rv = stats.gumbel_r(loc=mu, scale=beta)

        ab = predictor_corrector_known_scipy_pdf(nterms, rv, quad_options)

        def integrand(x):
            p = evaluate_orthonormal_polynomial_1d(x, nterms-1, ab)
            G = np.empty((x.shape[0], nterms**2))
            kk = 0
            for ii in range(nterms):
                for jj in range(nterms):
                    G[:, kk] = p[:, ii]*p[:, jj]
                    kk += 1
            return G*rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        xx, __ = gauss_quadrature(ab, nterms)
        interval_size = xx.max()-xx.min()
        quad_opts = quad_options.copy()
        del quad_opts['nquad_samples']
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, quad_options['nquad_samples'],
            interval_size=interval_size, **quad_opts)
        res = np.reshape(res, (nterms, nterms), order='C')
        print(np.absolute(res-np.eye(nterms)).max())
        assert np.absolute(res-np.eye(nterms)).max() < 2e-4

    def test_predictor_corrector_function_of_independent_variables(self):
        """
        Test 1: Sum of Gaussians is a Gaussian

        Test 2: Product of uniforms on [0,1]
        """
        nvars, nterms = 2, 5
        variables = [stats.norm(0, 1)]*nvars

        nquad_samples_1d = 50
        quad_rules = [gauss_hermite_pts_wts_1D(nquad_samples_1d)]*nvars

        def fun(x):
            return x.sum(axis=0)

        ab = predictor_corrector_function_of_independent_variables(
            nterms, quad_rules, fun)

        rv = stats.norm(0, np.sqrt(nvars))
        measures = rv.pdf
        lb, ub = rv.interval(1)
        interval_size = rv.interval(0.99)[1] - rv.interval(0.99)[0]
        ab_full = predictor_corrector(nterms, rv.pdf, lb, ub, interval_size)
        assert np.allclose(ab_full, ab)

        nvars = 2

        def measure(x):
            return (-1)**(nvars-1)*np.log(x)**(nvars-1)/factorial(nvars-1)

        def fun(x):
            return x.prod(axis=0)

        quad_opts = {'verbose': 0, 'atol': 1e-6, 'rtol': 1e-6}
        ab_full = predictor_corrector(nterms, measure, 0, 1, 1, quad_opts)
        xx, ww = gauss_jacobi_pts_wts_1D(nquad_samples_1d, 0, 0)
        xx = (xx+1)/2
        quad_rules = [(xx, ww)]*nvars
        ab = predictor_corrector_function_of_independent_variables(
            nterms, quad_rules, fun)
        assert np.allclose(ab_full, ab)

    def test_predictor_corrector_product_of_functions_of_independent_variables(
            self):
        nvars, nterms = 3, 4

        def measure(x):
            return (-1)**(nvars-1)*np.log(x)**(nvars-1)/factorial(nvars-1)

        def fun(x):
            return x.prod(axis=0)

        nquad_samples_1d = 20
        xx, ww = gauss_jacobi_pts_wts_1D(nquad_samples_1d, 0, 0)
        xx = (xx+1)/2
        quad_rules = [(xx, ww)]*nvars
        funs = [lambda x: x]*nvars
        ab = predictor_corrector_product_of_functions_of_independent_variables(
            nterms, quad_rules, funs)

        quad_opts = {'verbose': 3, 'atol': 1e-6, 'rtol': 1e-6}
        ab_full = predictor_corrector(nterms, measure, 0, 1, 1, quad_opts)
        print(ab-ab_full)
        assert np.allclose(ab, ab_full, atol=1e-5, rtol=1e-5)

    def test_arbitraty_polynomial_chaos(self):
        nterms = 5
        alpha_stat, beta_stat = 1, 1
        
        true_ab = jacobi_recurrence(
            nterms, alpha=beta_stat-1, beta=alpha_stat-1,
            probability=True)

        rv = stats.uniform(-1, 2)
        moments = [rv.moment(n) for n in range(2*nterms+1)]
        ab = arbitrary_polynomial_chaos_recursion_coefficients(moments, nterms)

        assert np.allclose(true_ab, ab)


if __name__ == "__main__":
    num_gen_orthonormal_poly_1d_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestNumericallyGenerateOrthonormalPolynomials1D)
    unittest.TextTestRunner(verbosity=2).run(
        num_gen_orthonormal_poly_1d_test_suite)

"""
print("----------------------------")
print("Lanczos test (deprecated)")
print("----------------------------")
A       = np.zeros((ntrials+2,ntrials+2));
A[0,0]  = 1;
A[0,1:] = np.sqrt(pmfVals);
A[1:,0] = np.sqrt(pmfVals);
for i in range(1,ntrials+2):
    A[i,i] = x[i-1]

e1 = np.zeros(ntrials+2); e1[0] = 1;
abAN = lanczos_deprecated(A,e1)[:N]

print(np.allclose(abWG,abAN))
"""
