from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy import special as sp

def charlier_recurrence(N, a, probability=False):
    """These polynomials are orthogonal with respect to the Poisson distribution
    """

    if N < 1:
        return np.ones((0,2))

    ab = np.zeros((N+1,2))
    ab[0,0] = a
    ab[0,1] = 1
    for i in range(1,N+1):
        ab[i,0] = a + i
        ab[i,1] = a * i

    # orthonormal
    ab[:,1] = np.sqrt(ab[:,1])

    return ab

def discrete_chebyshev_recurrence(N, Ntrials, probability=False):
    """
    These polynomials are orthogonal with respect to the weight function
    w(x) = \delta_i(x) where \delta_i(x) is the dirac delta function which
    is one when x==i, i==1, Ntrials and zero otherwise
    """
    assert(N<=Ntrials)

    if N < 1:
        return np.ones((0,2))

    ab = np.zeros((N,2))
    ab[:,0] = 0.5 * Ntrials * (1. - 1./Ntrials)
    ab[0,1] = Ntrials
    for i in range(1,N):
        ab[i,1] = 0.25 * Ntrials**2 * (1-(i * 1./Ntrials)**2)/(4-1./i**2)

    ab[:,1] = np.sqrt(ab[:,1])

    ab[0,1] = 1.0

    return ab

def hahn_recurrence(N, Ntrials, alphaPoly, betaPoly, probability=False):
    """
    These polynomials are orthogonal with respect to the weight function 

    (K choose k)(N-K choose n-k)/( N choose n).

    This corresponds to the hypergeometric probability mass function, which
    describes the probability of k successes in n draws, without 
    replacement, from a finite population of size N that contains exactly 
    K successes.
    """
    assert(N<=Ntrials)

    if N < 1:
        return np.ones((0,2))

    An = np.zeros(N)
    Cn = np.zeros(N)
    for n in range(N):
        numA = (alphaPoly+n+1) * (Ntrials-n) * (n+alphaPoly+betaPoly+1)
        numC = n * (betaPoly+n) * (Ntrials+alphaPoly+betaPoly+n+1)
        denA = (alphaPoly+betaPoly+2*n+1) * (alphaPoly+betaPoly+2*n+2)
        denC = (alphaPoly+betaPoly+2*n+1) * (alphaPoly+betaPoly+2*n  )
        An[n] = numA / denA
        Cn[n] = numC / denC

    if N==1:
        return np.array([[An[0]+Cn[0],1]])

    ab = np.array(
        [[An[0]+Cn[0],1]]+[[An[n]+Cn[n],An[n-1]*Cn[n]] for n in range(1,N)])

    ab[:,1] = np.sqrt(ab[:,1])

    ab[0,1] = 1.0

    return ab

def krawtchouk_recurrence(N, Ntrials, p, probability=False):
    """
    These polynomials
    are orthogonal with respect to the weight function 
    (N choose k)*p^k*(1-p)^(n-k).
    This corresponds to the binomial probability mass function,  
    which is the probability of k successes from N trials.
    """
    
    assert(N<=Ntrials)
    assert p>0 and p<1

    if N < 1:
        return np.ones((0,2))

    ab = np.array(
        [[p*(Ntrials-n)+n*(1-p), p*(1-p)*n*(Ntrials-n+1)] for n in range(N)])

    ab[:,1] = np.sqrt(ab[:,1])

    ab[0,1] = 1.0

    # the probability flag does not apply here 
    # (beta0 comes out 0 in the three term recurrence), instead we set it
    # to 1, the norm of the p0 polynomial

    return ab

def jacobi_recurrence(N, alpha=0., beta=0., probability=False):
    # Returns the first N recurrence coefficient pairs for the Jacobi
    # polynomial family.

    if N < 1:
        return np.ones((0,2))

    ab = np.ones((N,2)) * np.array([beta**2.- alpha**2., 1.])

    # Special cases
    ab[0,0] = (beta - alpha) / (alpha + beta + 2.)
    ab[0,1] = np.exp( (alpha + beta + 1.) * np.log(2.) +
                      sp.gammaln(alpha + 1.) + sp.gammaln(beta + 1.) -
                      sp.gammaln(alpha + beta + 2.)
                    )

    if N > 1:
        ab[1,0] /= (2. + alpha + beta) * (4. + alpha + beta)
        ab[1,1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.) )

    inds = np.arange(2.,N)
    ab[2:,0] /= (2. * inds + alpha + beta) * (2 * inds + alpha + beta + 2.)
    ab[2:,1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
    ab[2:,1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)

    ab[:,1] = np.sqrt(ab[:,1])

    if probability:
        ab[0,1] = 1.

    return ab

def hermite_recurrence(N, rho=0., probability=False):
    r""" 
    Returns the first N recurrence coefficient pairs for the Hermite
    polynomial family.

    x^{2\rho}exp(-x^2)

    and for special case of rho=0 and probability=True

    exp(-x^2/2) return the probablists hermite polynomial
    """
    
    if N < 1:
        return np.ones((0,2))

    ab = np.zeros((N,2))
    ab[0,1] = sp.gamma(rho+0.5)# = np.sqrt(np.pi) for rho=0


    if rho==0 and probability:
        ab[1:,1] = np.arange(1., N)
    else:
        ab[1:,1] = 0.5*np.arange(1., N)

        
    ab[np.arange(N) % 2 == 1,1] += rho

    ab[:,1] = np.sqrt(ab[:,1])

    if probability:
        ab[0,1] = 1.

    return ab

def evaluate_monic_polynomial_1d(x,nmax,ab):
    p = np.zeros((x.shape[0],nmax+1),dtype=float)

    p[:,0] = 1/ab[0,1]

    if nmax > 0:
        p[:,1] =(x - ab[0,0])*p[:,0]

    for jj in range(2, nmax+1):
        p[:,jj] = (x-ab[jj-1,0])*p[:,jj-1]-ab[jj-1,1]*p[:,jj-2]

    return p
    

def evaluate_orthonormal_polynomial_1d(x, nmax, ab):
    r""" 
    Evaluate univariate orthonormal polynomials using their
    three-term recurrence coefficients.

    The the degree-n orthonormal polynomial p_n(x) is associated with
    the recurrence coefficients a, b (with positive leading coefficient)
    satisfy the recurrences

      b_{n+1} p_{n+1} = (x - a_n) p_n - sqrt(b_n) p_{n-1}

    This assumes that the orthonormal recursion coefficients satisfy
    
      b_{n+1} = sqrt(\hat{b}_{n+1})

    where \hat{b}_{n+1} are the orthogonal recursion coefficients.

    Parameters
    ----------
    x : np.ndarray (num_samples)
       The samples at which to evaluate the polynomials

    nmax : integer
       The maximum degree of the polynomials to be evaluated

    ab : np.ndarray (num_recusion_coeffs,2)
       The recursion coefficients. num_recusion_coeffs>degree

    Return
    ------
    p : np.ndarray (num_samples, nmax+1)
       The values of the polynomials
    """
    assert ab.shape[1]==2
    assert nmax < ab.shape[0]

    try:
        # necessary when discrete variables are define on integers
        x = np.asarray(x,dtype=float)
        from pyapprox.cython.orthonormal_polynomials_1d import \
            evaluate_orthonormal_polynomial_1d_pyx
        return evaluate_orthonormal_polynomial_1d_pyx(x, nmax, ab)
        # from pyapprox.weave import c_evaluate_orthonormal_polynomial
        # return c_evaluate_orthonormal_polynomial_1d(x, nmax, ab)
    except Exception as e:
        print ('evalaute_orthornormal_polynomial_1d extension failed')

    p = np.zeros((x.shape[0],nmax+1),dtype=float)

    p[:,0] = 1/ab[0,1]

    if nmax > 0:
        p[:,1] = 1/ab[1,1] * ( (x - ab[0,0])*p[:,0] )

    for jj in range(2, nmax+1):
        #p[:,jj] = ((x-ab[jj-1,0])*p[:,jj-1]-ab[jj-1,1]*p[:,jj-2])
        p[:,jj] = 1.0/ab[jj,1]*((x-ab[jj-1,0])*p[:,jj-1]-ab[jj-1,1]*p[:,jj-2])

    return p


def evaluate_orthonormal_polynomial_deriv_1d(x, nmax, ab, deriv_order):
    r""" 
    Evaluate the univariate orthonormal polynomials and its s-derivatives 
    (s=1,...,num_derivs) using a three-term recurrence coefficients.

    The the degree-n orthonormal polynomial p_n(x) is associated with
    the recurrence coefficients a, b (with positive leading coefficient)
    satisfy the recurrences

      b_{n+1} p_{n+1} = (x - a_n) p_n - sqrt(b_n) p_{n-1}

    This assumes that the orthonormal recursion coefficients satisfy
    
      b_{n+1} = sqrt(\hat{b}_{n+1})

    where \hat{b}_{n+1} are the orthogonal recursion coefficients.

    Parameters
    ----------
    x : np.ndarray (num_samples)
       The samples at which to evaluate the polynomials

    nmax : integer
       The maximum degree of the polynomials to be evaluated

    ab : np.ndarray (num_recursion_coeffs,2)
       The recursion coefficients

    deriv_order : integer
       The maximum order of the derivatives to evaluate.

    Return
    ------
    p : np.ndarray (num_samples, num_indices)
       The values of the s-th derivative of the polynomials
    """
    try:
        # necessary when discrete variables are define on integers
        x = np.asarray(x,dtype=float)
        from pyapprox.cython.orthonormal_polynomials_1d import \
            evaluate_orthonormal_polynomial_deriv_1d_pyx
        return evaluate_orthonormal_polynomial_deriv_1d_pyx(
            x, nmax, ab, deriv_order)
    except:
        print ('evaluate_orthonormal_polynomial_deriv_1d_pyx extension failed')

    num_samples = x.shape[0]
    num_indices = nmax+1
    a = ab[:,0]; b = ab[:,1]
    result = np.empty((num_samples,num_indices*(deriv_order+1)))
    p = evaluate_orthonormal_polynomial_1d(x, nmax, ab)
    result[:,:num_indices] = p

    for deriv_num in range(1,deriv_order+1):
        pd = np.zeros((num_samples,num_indices),dtype=float)
        for jj in range(deriv_num,num_indices):

            if (jj == deriv_num):
                # use following expression to avoid overflow issues when
                # computing oveflow
                pd[:,jj] = np.exp(
                    sp.gammaln(deriv_num+1)-0.5*np.sum(np.log(b[:jj+1]**2)))
            else:
                
                pd[:,jj]=\
                  (x-a[jj-1])*pd[:,jj-1]-b[jj-1]*pd[:,jj-2]+deriv_num*p[:,jj-1]
                pd[:,jj] *= 1.0/b[jj]
        p = pd
        result[:,deriv_num*num_indices:(deriv_num+1)*num_indices] = p
    return result

from scipy.sparse import diags as sparse_diags
def gauss_quadrature(recursion_coeffs,N):
    """Computes Gauss quadrature from recurrence coefficients

    x,w = gauss_quadrature(recursion+coeffs,N)

    Computes N Gauss quadrature nodes (x) and weights (w) from 
    standard orthonormal recurrence coefficients. 
    """
    assert N > 0
    assert N<=recursion_coeffs.shape[0]
    assert recursion_coeffs.shape[1]==2

    a = recursion_coeffs[:,0]; b = recursion_coeffs[:,1];

    # Form Jacobi matrix
    J = np.diag(a[:N],0)+np.diag(b[1:N],1)+np.diag(b[1:N],-1)
    x, __ = np.linalg.eigh(J)

    w = evaluate_orthonormal_polynomial_1d(x, N-1, recursion_coeffs)
    w = 1./np.sum(w**2,axis=1)
    w[~np.isfinite(w)]= 0.
    return x, w

def convert_orthonormal_polynomials_to_monomials_1d(ab,nmax):
    """
    Get the monomial expansion of each orthonormal basis up to a given
    degree.

    Returns
    -------
    monomial_coefs : np.ndarray (nmax+1,nmax+1)
        The coefficients of x**i i=0,...,nmax for each orthonormal basis p_j
        Each row is the coefficients of a single basis p_j.
    """
    assert nmax < ab.shape[0]
    
    monomial_coefs = np.zeros((nmax+1,nmax+1))

    monomial_coefs[0,0] = 1/ab[0,1]

    if nmax > 0:
        monomial_coefs[1,:2] = np.array([-ab[0,0],1])*monomial_coefs[0,0]/ab[1,1]
    
    for jj in range(2,nmax+1):
        monomial_coefs[jj,:jj]+=(
            -ab[jj-1,0]*monomial_coefs[jj-1,:jj]
            -ab[jj-1,1]*monomial_coefs[jj-2,:jj])/ab[jj,1]
        monomial_coefs[jj,1:jj+1]+=monomial_coefs[jj-1,:jj]/ab[jj,1]
    return monomial_coefs

    

def evaluate_three_term_recurrence_polynomial_1d(abc,nmax,x):
    assert nmax < abc.shape[0]
    
    p = np.zeros((x.shape[0],nmax+1),dtype=float)

    p[:,0] = abc[0,0]

    if nmax > 0:
        p[:,1] = (abc[1,0]*x - abc[1,1])*p[:,0]

    for jj in range(2, nmax+1):
        p[:,jj] = (abc[jj,0]*x-abc[jj,1])*p[:,jj-1]-abc[jj,2]*p[:,jj-2]

    return p
    

def convert_orthonormal_recurence_to_three_term_recurence(recursion_coefs):
    num_terms = recursion_coefs.shape[0]
    abc = np.zeros((num_terms,3))
    abc[:,0] = 1./recursion_coefs[:,1]
    abc[1:,1] = recursion_coefs[:-1,0]/recursion_coefs[1:,1]
    abc[1:,2] = recursion_coefs[:-1,1]/recursion_coefs[1:,1]
    return abc

from pyapprox.manipulate_polynomials import shift_momomial_expansion
def convert_orthonormal_expansion_to_monomial_expansion_1d(ortho_coef,ab,
                                                           shift,scale):
    assert ortho_coef.ndim==1
    # get monomial expansion of each orthonormal basis
    basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
        ab,ortho_coef.shape[0]-1)
    # scale each monomial coefficient by the corresponding orthonormal expansion
    # coefficients and collect terms
    mono_coefs = np.sum(basis_mono_coefs.T*ortho_coef,axis=1)
    # the orthonormal basis is defined on canonical domain so
    # shift to desired domain
    mono_coefs =  shift_momomial_expansion(mono_coefs,shift,scale)
    return mono_coefs
