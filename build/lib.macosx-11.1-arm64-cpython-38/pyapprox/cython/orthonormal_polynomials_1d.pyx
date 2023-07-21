import numpy as np
import cython
from scipy.special.cython_special cimport gammaln
from libc.math cimport exp, log

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef evaluate_orthonormal_polynomial_1d_pyx(double [:] x, int nmax, double [:,:] ab):
    """ 
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
    cdef int ii, jj
    cdef int npoints = x.shape[0]

    p = np.zeros((x.shape[0],nmax+1),dtype=np.double)
    cdef double [:,:] p_view = p

    p_view[:,0] = 1/ab[0,1]

    if nmax > 0:
        for ii in range(npoints):
            p_view[ii,1] = 1/ab[1,1] * ( (x[ii] - ab[0,0])*p_view[ii,0] )

    for jj in range(2, nmax+1):
        for ii in range(npoints):
            p_view[ii,jj] = 1.0/ab[jj,1]*((x[ii]-ab[jj-1,0])*p_view[ii,jj-1]-ab[jj-1,1]*p_view[ii,jj-2])

    return p


@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef evaluate_orthonormal_polynomial_deriv_1d_pyx(double [:] x, int nmax, double [:,:] ab, int deriv_order):
    """ 
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

    cdef int ii,jj,deriv_num
    cdef int num_samples = x.shape[0]
    cdef int num_indices = nmax+1
    result = np.empty((num_samples,num_indices*(deriv_order+1)))
    cdef double [:,:] result_view = result
    cdef double [:,:] pd = np.zeros((num_samples,num_indices),dtype=float)
    cdef double bsum=0

    cdef double [:,:] p = evaluate_orthonormal_polynomial_1d_pyx(x, nmax, ab)
    result_view[:,:num_indices] = p

    for deriv_num in range(1,deriv_order+1):
        pd[:,:] = 0
        for jj in range(deriv_num,num_indices):
            if (jj == deriv_num):
                bsum = 0
                for ii in range(jj+1):
                    bsum += log(ab[ii,1]**2)
                # use following expression to avoid overflow issues when
                # computing oveflow
                pd[:,jj] = exp(gammaln(deriv_num+1)-0.5*bsum)
            else:
                for ii in range(num_samples):
                    pd[ii,jj]=\
			(x[ii]-ab[jj-1,0])*pd[ii,jj-1]-\
			ab[jj-1,1]*pd[ii,jj-2]+deriv_num*p[ii,jj-1]
                    pd[ii,jj] *= 1.0/ab[jj,1]
        # p = pd # does not work
        p[:,:] = pd[:,:]
        result_view[:,deriv_num*num_indices:(deriv_num+1)*num_indices] = p
    return result

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef induced_measure_pyx(double x, int ii, double [:,:] ab, pdf):
    cdef double [:] xx = np.atleast_1d(x)
    cdef double [:,:] val = evaluate_orthonormal_polynomial_1d_pyx(xx,ii,ab)
    cdef double pdf_val = pdf(xx[0])
    return pdf_val*val[0,ii]*val[0,ii]

from scipy import integrate
@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef continuous_induced_measure_cdf_pyx(pdf, double [:,:] ab, int ii, double lb, double tol, double x):
    cdef double integral, err
    integral,err = integrate.quad(
    	induced_measure_pyx,lb,x,args=(ii,ab,pdf),epsrel=tol,epsabs=tol,
	limit=100)
    return integral

cpdef vector_continuous_induced_measure_cdf_pyx(pdf, double [:,:] ab, int ii, double lb, double tol, double [:] x):
    cdef int jj
    vals = np.zeros((x.shape[0]),dtype=np.double)
    cdef double [:] vals_view = vals
    for jj in range(x.shape[0]):
        vals_view[jj]=continuous_induced_measure_cdf_pyx(pdf,ab,ii,lb,tol,x[jj])
        if vals_view[jj]>1 and vals_view[jj]-1<tol:
            vals_view[jj]=1.
    return vals
      