cimport cython
import numpy as np

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef multiply_multivariate_polynomials_pyx(cython.integral [:,:] indicesI, double [:] coeffsI, cython.integral [:,:] indicesII, double [:] coeffsII):
    """
    Parameters
    ----------
    Warning indices returned can be duplicate, but if using to compute 
    moments this does not matter unless cost of computing moments is comparable
    to this function

    index : multidimensional index
        multidimensional index specifying the polynomial degree in each
        dimension

    Returns
    -------
    """
    cdef int num_vars = indicesI.shape[0]
    cdef int num_indicesI = indicesI.shape[1]
    cdef int num_indicesII = indicesII.shape[1]

    if cython.integral is int:
      int_dtype=np.int32 
    elif cython.integral is long:
      int_dtype=np.int64 
    else:
      # short
      int_dtype=np.short
    
    indices = np.empty((num_vars,num_indicesI*num_indicesII),
                       dtype=int_dtype)
    cdef cython.integral [:,:] indices_view = indices

    coeffs = np.empty((num_indicesI*num_indicesII),dtype=np.float)
    cdef double [:] coeffs_view = coeffs

    cdef cython.integral [:] index1 = np.empty((num_vars),dtype=int_dtype)
    cdef int kk=0,ii,jj,dd

    for ii in range(num_indicesI):
        index1 = indicesI[:,ii]
        for jj in range(num_indicesII):
            for dd in range(num_vars):
                indices_view[dd,kk] = index1[dd]+indicesII[dd,jj]
            coeffs_view[kk] = coeffsI[ii]*coeffsII[jj]
            kk+=1

    return indices, coeffs