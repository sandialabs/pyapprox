import unittest
import numpy as np
from scipy.stats import binom, hypergeom

from pyapprox.numerically_generate_orthonormal_polynomials_1d import *
from pyapprox.orthonormal_polynomials_1d import *
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from scipy import stats
from functools import partial
from pyapprox.variables import float_rv_discrete
class TestNumericallyGenerateOrthonormalPolynomials1D(unittest.TestCase):
    def test_krawtchouk(self):
        num_coef=6
        ntrials  = 10
        p        = 0.5

        xk = np.array(range(ntrials+1),dtype='float')
        pk = binom.pmf(xk, ntrials, p)
        
        ab_lanzcos   = lanczos(xk,pk,num_coef)
        ab_stieltjes = stieltjes(xk,pk,num_coef)

        ab_exact = krawtchouk_recurrence(num_coef, ntrials, p)
        assert np.allclose(ab_lanzcos,ab_exact)
        assert np.allclose(ab_stieltjes,ab_exact)

        from pyapprox.univariate_quadrature import gauss_quadrature
        x,w = gauss_quadrature(ab_lanzcos,num_coef)
        moments = np.array([(x**ii).dot(w) for ii in range(num_coef)])
        true_moments = np.array([(xk**ii).dot(pk)for ii in range(num_coef)])
        assert np.allclose(moments,true_moments)
        p = evaluate_orthonormal_polynomial_1d(x, num_coef-1, ab_lanzcos)
        assert np.allclose((p.T*w).dot(p),np.eye(num_coef))
        p = evaluate_orthonormal_polynomial_1d(xk, num_coef-1, ab_lanzcos)
        assert np.allclose((p.T*pk).dot(p),np.eye(num_coef))

        
    def test_discrete_chebyshev(self):
        num_coef=5
        nmasses  = 10

        xk = np.array(range(nmasses),dtype='float')
        pk = np.ones(nmasses)/nmasses
        
        ab_lanzcos   = lanczos(xk,pk,num_coef)
        ab_stieltjes = stieltjes(xk,pk,num_coef)

        ab_exact = discrete_chebyshev_recurrence(num_coef,nmasses)
        assert np.allclose(ab_lanzcos,ab_exact)
        assert np.allclose(ab_stieltjes,ab_exact)

        from pyapprox.univariate_quadrature import gauss_quadrature
        x,w = gauss_quadrature(ab_lanzcos,num_coef)
        moments = np.array([(x**ii).dot(w) for ii in range(num_coef)])
        true_moments = np.array([(xk**ii).dot(pk)for ii in range(num_coef)])
        assert np.allclose(moments,true_moments)
        p = evaluate_orthonormal_polynomial_1d(x, num_coef-1, ab_lanzcos)
        assert np.allclose((p.T*w).dot(p),np.eye(num_coef))
        p = evaluate_orthonormal_polynomial_1d(xk, num_coef-1, ab_lanzcos)
        assert np.allclose((p.T*pk).dot(p),np.eye(num_coef))

    def test_float_rv_discrete(self):
        num_coef,nmasses = 5,10
        #works for both lanczos and chebyshev algorithms
        #xk   = np.geomspace(1,512,num=nmasses)
        #pk   = np.ones(nmasses)/nmasses
        
        #works only for chebyshev algorithms
        pk  = np.geomspace(1,512,num=nmasses)
        pk /= pk.sum()
        xk  = np.arange(0,nmasses)
        
        #ab  = lanczos(xk,pk,num_coef)
        ab  = modified_chebyshev_orthonormal(
            num_coef,[xk,pk],probability=True)
        
        from pyapprox.univariate_quadrature import gauss_quadrature
        x,w = gauss_quadrature(ab,num_coef)
        moments = np.array([(x**ii).dot(w) for ii in range(num_coef)])
        true_moments = np.array([(xk**ii).dot(pk)for ii in range(num_coef)])
        assert np.allclose(moments,true_moments),(moments,true_moments)
        p = evaluate_orthonormal_polynomial_1d(x, num_coef-1, ab)
        assert np.allclose((p.T*w).dot(p),np.eye(num_coef))
        p = evaluate_orthonormal_polynomial_1d(xk, num_coef-1, ab)
        assert np.allclose((p.T*pk).dot(p),np.eye(num_coef))

    def test_modified_chebyshev(self):
        nterms=10
        alpha_stat,beta_stat=2,2
        probability_measure=True
        # using scipy to compute moments is extermely slow
        #moments = [stats.beta.moment(n,alpha_stat,beta_stat,loc=-1,scale=2)
        #           for n in range(2*nterms)]
        quad_x,quad_w = gauss_jacobi_pts_wts_1D(
            4*nterms,beta_stat-1,alpha_stat-1)

        true_ab = jacobi_recurrence(
            nterms,alpha=beta_stat-1,beta=alpha_stat-1,
            probability=probability_measure)

        ab = modified_chebyshev_orthonormal(
            nterms,[quad_x,quad_w],get_input_coefs=None,probability=True)
        assert np.allclose(true_ab,ab)

        get_input_coefs = partial(
            jacobi_recurrence,alpha=beta_stat-2,beta=alpha_stat-2)
        ab = modified_chebyshev_orthonormal(
            nterms,[quad_x,quad_w],get_input_coefs=get_input_coefs,
            probability=True)
        assert np.allclose(true_ab,ab)

    def test_rv_discrete_large_moments(self):
        """
        When Modified_chebyshev_orthonormal is used when the moments of discrete
        variable are very large it will fail. To avoid this rescale the 
        variables to [-1,1] like is done for continuous random variables
        """
        N,degree=100,5
        xk,pk = np.arange(N),np.ones(N)/N
        rv = float_rv_discrete(name='float_rv_discrete',values=(xk,pk))
        xk_canonical = xk/(N-1)*2-1
        ab  = modified_chebyshev_orthonormal(
            degree+1,[xk_canonical,pk])
        p = evaluate_orthonormal_polynomial_1d(xk_canonical, degree, ab)
        w = rv.pmf(xk)
        assert np.allclose(np.dot(p.T*w,p),np.eye(degree+1))

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
        assert np.absolute(res-np.eye(nterms)).max() < 5e-6

        


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
