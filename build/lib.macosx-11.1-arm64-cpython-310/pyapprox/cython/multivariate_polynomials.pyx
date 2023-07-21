import numpy as np
import cython

from pyapprox.cython.orthonormal_polynomials_1d import \
  evaluate_orthonormal_polynomial_deriv_1d_pyx

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def evaluate_multivariate_orthonormal_polynomial_pyx(
        double [:,:] samples, cython.numeric [:,:] indices,
        double [:,:] recursion_coeffs,
        int deriv_order):

    """
    Evaluate a multivaiate orthonormal polynomial and its s-derivatives 
    (s=1,...,num_derivs) using a three-term recurrence coefficients.

    Parameters
    ----------

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the polynomial

    indices : np.ndarray (num_vars, num_indices)
        The exponents of each polynomial term

    recursion_coeffs : np.ndarray (2, num_indices)
        The coefficients of each monomial term

    deriv_order : integer in [0,1]
       The maximum order of the derivatives to evaluate.

    Return
    ------
    values : np.ndarray (1+deriv_order*num_samples,num_indices)
        The values of the polynomials at the samples
    """
    cdef int ii,jj,dd,kk
    cdef int num_vars = indices.shape[0], num_indices = indices.shape[1]
    cdef int num_samples = samples.shape[1]
    
    # precompute 1D basis functions for faster evaluation of multivariate terms
    max_level_1d = np.max(indices,axis=1)
    basis_vals_1d = []
    for dd in range(num_vars):
        basis_vals_1d_dd = evaluate_orthonormal_polynomial_deriv_1d_pyx(
            samples[dd,:],max_level_1d[dd],recursion_coeffs,
            deriv_order)
        basis_vals_1d.append(basis_vals_1d_dd)

    values = np.zeros(((1+deriv_order*num_vars)*num_samples,num_indices))
    cdef double [:,:] values_view = values
    cdef double [:] basis_vals

    for ii in range(num_indices):
        index = indices[:,ii]
        for jj in range(num_samples):
            values_view[jj,ii]=basis_vals_1d[0][jj,index[0]]
            for dd in range(1,num_vars):
                values_view[jj,ii]*=basis_vals_1d[dd][jj,index[dd]]

    if deriv_order==0:
        return values

    for ii in range(num_indices):
        index = indices[:,ii]
        for jj in range(num_vars):
            # derivative in jj direction
            basis_vals=\
              basis_vals_1d[jj][:,(max_level_1d[jj]+1)+index[jj]].copy()
            # basis values in other directions
            for dd in range(num_vars):
                if dd!=jj:
                    basis_vals*=basis_vals_1d[dd][:,index[dd]]

            for kk in range(num_samples):
                values_view[(jj+1)*num_samples+kk,ii] = basis_vals[kk]
        
    return values
