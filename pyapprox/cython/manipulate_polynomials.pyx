cimport cython
import numpy as np
cimport numpy as np

ctypedef np.int64_t int64_t

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef multiply_multivariate_polynomials_pyx(int64_t[:,:] indicesI, double[:] coeffsI, int64_t[:,:] indicesII, double[:] coeffsII):
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
    cdef:
        Py_ssize_t num_vars = indicesI.shape[0]
        Py_ssize_t num_indicesI = indicesI.shape[1]
        Py_ssize_t num_indicesII = indicesII.shape[1]
    
    indices = np.empty((num_vars,num_indicesI*num_indicesII),
                       dtype=np.int64)
    coeffs = np.empty((num_indicesI*num_indicesII), dtype=np.float64)

    cdef:
        int64_t[:,:] indices_view = indices
        double [:] coeffs_view = coeffs
        int64_t[:] index1 = np.empty((num_vars), dtype=np.int64)
        Py_ssize_t kk=0, ii, jj, dd

    for ii in range(num_indicesI):
        index1 = indicesI[:,ii]
        for jj in range(num_indicesII):
            for dd in range(num_vars):
                indices_view[dd,kk] = index1[dd]+indicesII[dd,jj]
            coeffs_view[kk] = coeffsI[ii]*coeffsII[jj]
            kk+=1

    return indices, coeffs