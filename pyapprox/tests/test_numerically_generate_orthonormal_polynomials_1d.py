import unittest
import numpy as np
from scipy.stats import binom, hypergeom

from pyapprox.numerically_generate_orthonormal_polynomials_1d import *
from pyapprox.orthonormal_polynomials_1d import *
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from scipy.stats import beta as beta_rv
from functools import partial
class TestNumericallyGenerateOrthonormalPolynomials1D(unittest.TestCase):
    def test_krawtchouk(self):
        num_coef=6
        ntrials  = 10
        p        = 0.5

        xk = np.array(range(ntrials+1),dtype='float')
        pk = binom.pmf(xk, ntrials, p)
        
        ab_lanzcos   = lanczos(xk,pk,num_coef)
        ab_stieltjes = stieltjes(xk,pk,num_coef)

        ab_exact = krawtchouk_recurrence(num_coef, ntrials, p, probability=True)
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

        ab_exact = discrete_chebyshev_recurrence(
            num_coef,nmasses,probability=True)
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
        #moments = [beta_rv.moment(n,alpha_stat,beta_stat,loc=-1,scale=2)
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
