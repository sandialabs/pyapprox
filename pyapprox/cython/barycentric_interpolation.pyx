import numpy as np
cimport cython

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef compute_barycentric_weights_1d_pyx(double [:] samples, double C_inv):
    cdef int jj, kk
    cdef int num_samples = samples.shape[0]

    weights = np.empty((num_samples, num_samples),dtype=float)

    cdef double [:,:] weights_view = weights
    cdef double [:]   samples_view = samples
    
    weights_view[0,0] = 1.
    for jj in range(1,num_samples):
        for kk in range(jj):
            weights_view[jj,kk] = C_inv*(
	        samples_view[kk]-samples_view[jj])*weights_view[jj-1,kk]
        weights_view[jj,jj] = 1
        for kk in range(jj):
            weights_view[jj,jj] *= C_inv*(samples_view[jj]-samples_view[kk])
            weights_view[jj-1,kk] = 1./weights_view[jj-1,kk]

    for kk in range(num_samples):
        weights_view[num_samples-1,kk] = 1./weights_view[num_samples-1,kk]

    return weights

@cython.cdivision(True)     # Deactivate division by zero checking
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef multivariate_hierarchical_barycentric_lagrange_interpolation_pyx(
    double [:,:] x, double [:,:] fn_vals, long [:] active_dims,
    long [:,:] active_abscissa_indices_1d, long [:] num_abscissa_1d,
    long [:] num_active_abscissa_1d, long [:] shifts,
    double [:,:] abscissa_and_weights):
    
    cdef int num_act_dims_pt,ii,jj,kk,mm,cnt,dim,num_abscissa,act_dim_idx,fn_val_index,dd
    cdef bint has_inactive_abscissa, is_active_dim, done
    cdef double x_dim_k, denom, denom_d, basis

    cdef double mach_eps = np.finfo(float).eps
    cdef int num_pts = x.shape[1]
    cdef int num_act_dims = active_dims.shape[0]

    cdef int max_num_abscissa_1d = abscissa_and_weights.shape[0]//2
    cdef long [:] multi_index  = np.empty((num_act_dims),dtype=int)
  
    cdef int num_qoi = fn_vals.shape[1]

    result = np.empty((num_pts,num_qoi),dtype=float)
    cdef double [:,:] result_view = result
    
    # Allocate persistent memory. Each point will fill in a varying amount
    # of entries. We use a view of this memory to stop reallocation for each 
    # data point
    cdef long [:] act_dims_pt_persistent = np.empty((num_act_dims),dtype=int)
    cdef long [:] act_dim_indices_pt_persistent = np.empty(
        (num_act_dims),dtype=int)
    cdef double [:,:] c_persistent=np.empty((num_qoi,num_act_dims),dtype=float)
    cdef double [:,:] bases = np.empty(
        (max_num_abscissa_1d, num_act_dims),dtype=float)
    for kk in range(num_pts):
        # compute the active dimension of the kth point in x and the 
        # set multi_index accordingly
        multi_index[:] = 0
        num_act_dims_pt = 0
        has_inactive_abscissa = False
        for act_dim_idx in range(num_act_dims):
            cnt = 0
            is_active_dim = True
            dim = active_dims[act_dim_idx]
            num_abscissa = num_abscissa_1d[act_dim_idx]
            x_dim_k = x[dim,kk]
            for ii in range(num_abscissa):
                if ( ( cnt < num_active_abscissa_1d[act_dim_idx] ) and 
                    ( ii == active_abscissa_indices_1d[act_dim_idx][cnt] ) ):
                    cnt+=1
                if ( abs( x_dim_k - abscissa_and_weights[2*ii,act_dim_idx] ) < mach_eps ):
                    is_active_dim = False
                    if ( ( cnt > 0 ) and 
                          (active_abscissa_indices_1d[act_dim_idx][cnt-1]==ii)):
                        multi_index[act_dim_idx] = cnt-1
                    else:
                        has_inactive_abscissa = True
                    break

            if ( is_active_dim ):
                act_dims_pt_persistent[num_act_dims_pt] = dim
                act_dim_indices_pt_persistent[num_act_dims_pt] = act_dim_idx
                num_act_dims_pt+=1
        # end for act_dim_idx in range(num_act_dims):
        
        if ( has_inactive_abscissa ):
            result_view[kk,:] = 0.
        else:
            # compute barycentric basis functions
            denom = 1.
            for dd in range(num_act_dims_pt):
                dim = act_dims_pt_persistent[dd]
                act_dim_idx = act_dim_indices_pt_persistent[dd]
                num_abscissa = num_abscissa_1d[act_dim_idx]
                x_dim_k = x[dim,kk]
                bases[0,dd] = abscissa_and_weights[1,act_dim_idx] /\
                  (x_dim_k - abscissa_and_weights[0,act_dim_idx])	      
                denom_d = bases[0,dd]
                for ii in range(1,num_abscissa):
                    basis = abscissa_and_weights[2*ii+1,act_dim_idx] /\
                      (x_dim_k - abscissa_and_weights[2*ii,act_dim_idx])
                    bases[ii,dd] = basis 
                    denom_d += basis

                if ( abs(denom_d) < mach_eps ):
                    raise Exception, "interpolation absacissa are not unique" 
                denom *= denom_d

            # end for dd in range(num_act_dims_pt):
                
            if ( num_act_dims_pt == 0 ):
                # if point is an abscissa return the fn value at that point
                # fn_val_index = np.sum(multi_index*shifts)
                fn_val_index = 0
                for dd in range(num_act_dims):
                    fn_val_index += multi_index[dd]*shifts[dd]
                result_view[kk,:] = fn_vals[fn_val_index,:]
            else:
                # compute interpolant
                c_persistent[:,:] = 0.
                done = True
                if ( num_act_dims_pt > 1 ): done = False
                fn_val_index = 0
                for dd in range(num_act_dims):
                    fn_val_index += multi_index[dd]*shifts[dd]
		#fn_val_index = np.sum(multi_index*shifts)
                while (True):
                    act_dim_idx = act_dim_indices_pt_persistent[0]
                    for ii in range(num_active_abscissa_1d[act_dim_idx]):
                        fn_val_index+=shifts[act_dim_idx]*(ii-multi_index[act_dim_idx])
                        multi_index[act_dim_idx] = ii
                        basis=bases[active_abscissa_indices_1d[act_dim_idx][ii],0]
                        #c_persistent[:,0]+= basis * fn_vals[fn_val_index,:]
                        for mm in range(num_qoi):
                            c_persistent[mm,0] += basis * fn_vals[fn_val_index,mm]
			
                        
                    for dd in range(1,num_act_dims_pt):
                        act_dim_idx = act_dim_indices_pt_persistent[dd]
                        basis = bases[active_abscissa_indices_1d[act_dim_idx][multi_index[act_dim_idx]],dd]
                        for mm in range(num_qoi):
                            c_persistent[mm,dd] += basis * c_persistent[mm,dd-1]
                            c_persistent[mm,dd-1] = 0.
                        #c_persistent[:,dd] += basis * c_persistent[:,dd-1]
                        #c_persistent[:,dd-1] = 0.
                        if (multi_index[act_dim_idx]<num_active_abscissa_1d[act_dim_idx]-1):
                            fn_val_index += shifts[act_dim_idx]
                            multi_index[act_dim_idx] += 1
                            break
                        elif ( dd < num_act_dims_pt - 1 ):
                            fn_val_index-=shifts[act_dim_idx]*multi_index[act_dim_idx]
                            multi_index[act_dim_idx] = 0
                        else:
                            done = True
                    if ( done ):
                        break
                for mm in range(num_qoi):
                    result_view[kk,mm]=c_persistent[mm,num_act_dims_pt-1]/denom
                #result[kk,:] = c_persistent[:,num_act_dims_pt-1] / denom
                #if np.any(np.isnan(result[kk,:])):
                #    #print (c_persistent [:,num_act_dims_pt-1])
                #    #print (denom)
                #    raise Exception, 'Error values not finite'
    return result