import unittest
from functools import partial

import numpy as np
from scipy import stats
from scipy.special import factorial

from pyapprox.surrogates.orthopoly.numeric_orthonormal_recursions import (
    lanczos, stieltjes, modified_chebyshev_orthonormal, predictor_corrector,
    predictor_corrector_function_of_independent_variables,
    arbitrary_polynomial_chaos_recursion_coefficients,
    predictor_corrector_product_of_functions_of_independent_variables,
    ortho_polynomial_grammian_bounded_continuous_variable,
    native_recursion_integrate_fun)
from pyapprox.surrogates.orthopoly.recursion_factory import (
    predictor_corrector_known_pdf)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d, gauss_quadrature)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    krawtchouk_recurrence, jacobi_recurrence, discrete_chebyshev_recurrence,
    hermite_recurrence)
from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D, gauss_hermite_pts_wts_1D)
from pyapprox.variables.marginals import (
    float_rv_discrete, transform_scale_parameters)


class TestNumericallyGenerateOrthonormalPolynomials1D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_krawtchouk(self):
        num_coef = 6
        ntrials = 10
        p = 0.5

        xk = np.array(range(ntrials+1), dtype='float')
        pk = stats.binom.pmf(xk, ntrials, p)

        ab_lanczos = lanczos(xk, pk, num_coef)
        ab_stieltjes = stieltjes(xk, pk, num_coef)

        ab_exact = krawtchouk_recurrence(num_coef, ntrials, p)

        # ab_lanczos[-1, 0] is a dummy entry so set to exact so
        # comparison will pass if all other entries are correct
        ab_lanczos[-1, 0] = ab_exact[-1, 0]

        assert np.allclose(ab_lanczos, ab_exact)
        assert np.allclose(ab_stieltjes, ab_exact)

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
        # xk   = np.geomspace(1,512,num=nmasses)
        # pk   = np.ones(nmasses)/nmasses

        # works only for chebyshev algorithms
        pk = np.geomspace(1, 512, num=nmasses)
        pk /= pk.sum()
        xk = np.arange(0, nmasses)

        # ab  = lanczos(xk,pk,num_coef)
        ab = modified_chebyshev_orthonormal(
            num_coef, [xk, pk], probability=True)

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

    def test_continuous_rv_sample(self):
        N, degree = int(1e6), 5
        xk, pk = np.random.normal(0, 1, N), np.ones(N)/N
        ab = modified_chebyshev_orthonormal(degree+1, [xk, pk])
        hermite_ab = hermite_recurrence(
            degree+1, 0, True)
        x, w = gauss_quadrature(hermite_ab, degree+1)
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        gaussian_moments = np.zeros(degree+1)
        gaussian_moments[0] = 1
        assert np.allclose(p.T.dot(w), gaussian_moments, atol=1e-2)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1), atol=7e-2)

    def test_rv_discrete_large_moments(self):
        """
        When Modified_chebyshev_orthonormal is used when the moments of
        discrete variable are very large it will fail. To avoid this
        rescale the variables to [-1,1] like is done for continuous
        random variables
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
            xk_canonical.max())
        p = evaluate_orthonormal_polynomial_1d(xk_canonical, degree, ab)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_predictor_corrector_known_pdf(self):
        nterms = 12
        tol = 1e-12
        quad_options = {'epsrel': tol, 'epsabs': tol, "limlst": 10,
                        "limit": 1000}

        rv = stats.beta(1, 1, -1, 2)
        ab = predictor_corrector_known_pdf(
            nterms, -1, 1, rv.pdf, quad_options)
        true_ab = jacobi_recurrence(nterms, 0, 0)
        assert np.allclose(ab, true_ab)

        rv = stats.beta(3, 3, -1, 2)
        ab = predictor_corrector_known_pdf(
            nterms, -1, 1, rv.pdf, quad_options)
        true_ab = jacobi_recurrence(nterms, 2, 2)

        rv = stats.norm(0, 2)
        loc, scale = transform_scale_parameters(rv)
        ab = predictor_corrector_known_pdf(
            nterms, -np.inf, np.inf,
            lambda x: rv.pdf(x*scale+loc)*scale, quad_options)
        true_ab = hermite_recurrence(nterms)
        assert np.allclose(ab, true_ab)

        # lognormal is a very hard test
        # rv = stats.lognorm(1)
        # custom_integrate_fun = native_recursion_integrate_fun
        # interval_size = abs(np.diff(rv.interval(0.99)))
        # integrate_fun = partial(custom_integrate_fun, interval_size)
        # quad_opts = {"integrate_fun": integrate_fun}
        # # quad_opts = {}
        # opts = {"numeric": True, "quad_options": quad_opts}

        # loc, scale = transform_scale_parameters(rv)
        # ab = predictor_corrector_known_pdf(
        #     nterms, 0, np.inf, lambda x: rv.pdf(x*scale+loc)*scale, opts)
        # for ii in range(1, nterms):
        #     assert np.all(gauss_quadrature(ab, ii)[0] > 0)
        # gram_mat = ortho_polynomial_grammian_bounded_continuous_variable(
        #     rv, ab, nterms-1, tol=tol, integrate_fun=integrate_fun)
        # # print(gram_mat-np.eye(gram_mat.shape[0]))
        # # print(np.absolute(gram_mat-np.eye(gram_mat.shape[0])).max())
        # assert np.absolute(gram_mat-np.eye(gram_mat.shape[0])).max() < 5e-10

        nterms = 2
        mean, std = 1e4, 7.5e3
        beta = std*np.sqrt(6)/np.pi
        mu = mean - beta*np.euler_gamma
        # mu, beta = 1, 1
        rv = stats.gumbel_r(loc=mu, scale=beta)
        custom_integrate_fun = native_recursion_integrate_fun
        tabulated_quad_rules = {}
        from numpy.polynomial.legendre import leggauss
        for nquad_samples in [100, 200, 400]:
            tabulated_quad_rules[nquad_samples] = leggauss(nquad_samples)
        # interval_size must be in canonical domain
        interval_size = abs(np.diff(rv.interval(0.99)))/beta
        integrate_fun = partial(
            custom_integrate_fun, interval_size,
            tabulated_quad_rules=tabulated_quad_rules, verbose=3)
        quad_opts = {"integrate_fun": integrate_fun}
        # quad_opts = {}
        opts = {"numeric": True, "quad_options": quad_opts}

        loc, scale = transform_scale_parameters(rv)
        ab = predictor_corrector_known_pdf(
            nterms, -np.inf, np.inf, lambda x: rv.pdf(x*scale+loc)*scale, opts)
        gram_mat = ortho_polynomial_grammian_bounded_continuous_variable(
            rv, ab, nterms-1, tol=tol, integrate_fun=integrate_fun)
        # print(gram_mat-np.eye(gram_mat.shape[0]))
        print(np.absolute(gram_mat-np.eye(gram_mat.shape[0])).max())
        assert np.absolute(gram_mat-np.eye(gram_mat.shape[0])).max() < 5e-10

    def test_predictor_corrector_function_of_independent_variables(self):
        """
        Test 1: Sum of Gaussians is a Gaussian

        Test 2: Product of uniforms on [0,1]
        """
        nvars, nterms = 2, 5

        nquad_samples_1d = 50
        quad_rules = [gauss_hermite_pts_wts_1D(nquad_samples_1d)]*nvars

        def fun(x):
            return x.sum(axis=0)

        ab = predictor_corrector_function_of_independent_variables(
            nterms, quad_rules, fun)

        rv = stats.norm(0, np.sqrt(nvars))
        lb, ub = rv.interval(1)
        ab_full = predictor_corrector(nterms, rv.pdf, lb, ub)
        assert np.allclose(ab_full, ab)

        nvars = 2

        def measure(x):
            return (-1)**(nvars-1)*np.log(x)**(nvars-1)/factorial(nvars-1)

        def fun(x):
            return x.prod(axis=0)

        quad_opts = {}
        ab_full = predictor_corrector(nterms, measure, 0, 1, quad_opts)
        xx, ww = gauss_jacobi_pts_wts_1D(nquad_samples_1d, 0, 0)
        xx = (xx+1)/2
        quad_rules = [(xx, ww)]*nvars
        ab = predictor_corrector_function_of_independent_variables(
            nterms, quad_rules, fun)
        print(ab_full, ab)
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

        quad_opts = {}
        ab_full = predictor_corrector(nterms, measure, 0, 1, quad_opts)
        # print(ab-ab_full)
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
