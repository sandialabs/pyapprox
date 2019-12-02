import numpy as np
import cython

ctypedef fused mixed_type:
    int
    long

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef update_smolyak_coefficients_pyx(mixed_type [:] new_index, mixed_type [:,:] subspace_indices, smolyak_coeffs):

    if mixed_type is int:
       dtype = np.int32
    if mixed_type is long:
       dtype = np.int64

    cdef double [:] smolyak_coeffs_view = smolyak_coeffs

    cdef int ii,jj
    cdef int num_vars = subspace_indices.shape[0]
    cdef int num_subspace_indices = subspace_indices.shape[1]
    cdef mixed_type [:] diff = np.empty((num_vars),dtype=dtype)
    cdef int diff_sum = 0
    update = True

    for ii in range(num_subspace_indices):
        diff_sum = 0
        update = True
        for jj in range(num_vars):
            diff[jj] = new_index[jj]-subspace_indices[jj,ii]
            diff_sum += diff[jj]
            if (diff[jj]<0) or (diff[jj]>1):
                update = False
                break
        if update:
           smolyak_coeffs_view[ii]+=(-1.)**diff_sum
    return smolyak_coeffs