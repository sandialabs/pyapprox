import cython
import numpy as np

from libc.math cimport cos, fabs

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef clenshaw_curtis_pts_wts_1D_pyx(int level):

    cdef int jj,kk
    cdef int num_samples = 1
    if (level>0):
        num_samples = int(2**level+1)
  
    cdef double wt_factor = 1./2.

    x = np.empty((num_samples),dtype=np.double)
    w = np.empty((num_samples),dtype=np.double)
    cdef double [:] x_view = x
    cdef double [:] w_view = w

    cdef double pi = np.pi
    cdef double mach_eps = np.finfo(float).eps
    cdef double mysum

    if ( level == 0 ):
        x_view[0]=0.; w_view[0]=1.;
    else:
        for jj in range(num_samples):
            if ( jj == 0 ):
                x_view[jj] = -1.;
                w_view[jj] = wt_factor / float(num_samples*(num_samples -2.))
            elif ( jj == num_samples-1 ):
                x_view[jj] = 1.;
                w_view[jj] = wt_factor / float(num_samples*(num_samples-2.))
            else:
                x_view[jj] = -cos(pi*jj/(num_samples-1))
                mysum = 0.0
                for kk in range(1,(num_samples-3)//2+1):
                    mysum += (1. / (4.*kk*kk-1.)*
                      cos(2.*pi*kk*jj/(num_samples-1.)))
                w_view[jj] = 2./(num_samples-1.)*(1.-cos(pi*float(jj) ) /
                          float(num_samples*(num_samples -2.))-2.*(mysum))
                w_view[jj] *= wt_factor;
            if ( fabs( x_view[jj] ) < 2.*mach_eps): x_view[jj] = 0.
    return x, w
