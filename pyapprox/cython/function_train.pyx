import numpy as np
from pyapprox.cython.orthonormal_polynomials_1d import \
evaluate_orthonormal_polynomial_1d

cimport cython
@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef evaluate_core_pyx(double [:] sample, double [:] core_params, 
                        core_params_map, 
                        int [:] ranks,
                        double [:,:] recursion_coeffs):
    """
    Evaluate a core of the function train at a sample

    Parameters
    ----------
    sample : float
        The sample at which to evaluate the function train

    univariate_params : [ np.ndarray (num_coeffs_i) ] (ranks[0]*ranks[2])
        The coeffs of each univariate function. May be of different size
        i.e. num_coeffs_i can be different for i=0,...,ranks[0]*ranks[1]

    ranks : np.ndarray (2)
        The ranks of the core [r_{k-1},r_k]

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    Returns
    -------
    core_values : np.ndarray (ranks[0],ranks[1])
        The values of each univariate function evaluated at the sample

    Notes
    -----
    If we assume each univariate function for variable ii is fixed
    we only need to compute basis matrix once. This is also true
    if we compute basis matrix for max degree of the univariate functions
    of the ii variable. If degree of a given univariate function is
    smaller we can just use subset of matrix. This comes at the cost of
    more storage but less computations than if vandermonde was computed
    for each different degree. We build max_degree vandermonde here.
    """

    cdef int ii,jj,nn
    core_values = np.empty((ranks[0],ranks[1]),dtype=float)
    cdef double [:,:] core_values_view = core_values    
    cdef int max_degree = recursion_coeffs.shape[0]-1
    cdef int num_core_params = core_params.shape[0]
    cdef int num_univariate_functions=core_params_map[0]-1, nparams=0
    cdef int lb=0,ub=0, univariate_function_num=0

    cdef double [:,:] basis_matrix = evaluate_orthonormal_polynomial_1d(
        np.asarray([sample]), max_degree, recursion_coeffs)
    for jj in range(ranks[1]):
        for ii in range(ranks[0]):
            univariate_function_num=jj*ranks[0]+ii
            if (univariate_function_num==num_univariate_functions):
                ub = num_core_params
            else:
                ub = (core_params_map[univariate_function_num+1])
            lb = core_params_map[univariate_function_num]
            nparams = ub-lb
            core_values_view[ii,jj]=0.0
            for nn in range(nparams):
               core_values_view[ii,jj] += basis_matrix[0,nn]*core_params[lb+nn]
    return core_values