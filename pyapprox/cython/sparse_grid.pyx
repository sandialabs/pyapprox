import numpy as np
cimport numpy as cnp
import cython
cimport cython
fused_type = cython.fused_type(cython.numeric, cnp.float64_t)

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_smolyak_coefficients_without_sorting_pyx(cython.integral [:,:] subspace_indices):
    """
    Given an arbitrary set of downward close indices determine the  
    smolyak coefficients.
    """
    cdef int ii,jj,kk
    cdef int num_vars = subspace_indices.shape[0]
    cdef int num_subspace_indices = subspace_indices.shape[1]
    cdef double diff, diff_sum
    smolyak_coeffs = np.zeros((num_subspace_indices),dtype=float)
    cdef double [:] smolyak_coeffs_view = smolyak_coeffs
    cdef int add
    
    for ii in range(num_subspace_indices):
        for jj in range(num_subspace_indices):
            diff_sum = 0
            add = 1
            for kk in range(num_vars):
                diff = subspace_indices[kk,jj]-subspace_indices[kk,ii]
                if diff>1 or diff<0:
                    add = 0
                    break
                diff_sum += diff
            if add:
                smolyak_coeffs_view[ii]+=(-1.)**diff_sum
    return smolyak_coeffs

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_smolyak_coefficients_pyx(cython.integral [:,:] subspace_indices, long [:] levels, long [:] level_change_indices):
    """
    Given an arbitrary set of downward close indices determine the  
    smolyak coefficients.
    """
    cdef int ii,jj,kk,idx
    cdef int num_vars = subspace_indices.shape[0]
    cdef int num_subspace_indices = subspace_indices.shape[1]
    cdef int num_levels = levels.shape[0]
    cdef double diff, diff_sum
    smolyak_coeffs = np.zeros((num_subspace_indices),dtype=float)
    cdef double [:] smolyak_coeffs_view = smolyak_coeffs
    cdef int add

    idx=0
    for ii in range(num_subspace_indices):
        if idx<num_levels and subspace_indices[0,ii]>levels[idx]:
            idx += 1
        for jj in range(ii,level_change_indices[idx]):
            diff_sum=0
            add=1
            for kk in range(num_vars):
                diff = subspace_indices[kk,jj]-subspace_indices[kk,ii]
                if diff>1 or diff<0:
                    add = 0
                    break
                diff_sum += diff
            if add:
                smolyak_coeffs_view[ii]+=(-1.)**diff_sum
    return smolyak_coeffs

