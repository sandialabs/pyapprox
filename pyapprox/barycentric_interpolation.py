from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from scipy.special import comb as nchoosek
from pyapprox.univariate_quadrature import clenshaw_curtis_pts_wts_1D
from scipy.special import factorial

def compute_barycentric_weights_1d(samples, interval_length=None,
                                   return_sequence=False,normalize_weights=False):
    """
    Return barycentric weights for a sequence of samples. e.g. of sequence
    x0,x1,x2 where order represents the order in which the samples are added
    to the interpolant.

    Parameters
    ----------
    return_sequence : boolean
        True  - return [1],[1/(x0-x1),1/(x1-x0)],
                       [1/((x0-x2)(x0-x1)),1/((x1-x2)(x1-x0)),1/((x2-x1)(x2-x0))]
        False - return [1/((x0-x2)(x0-x1)),1/((x1-x2)(x1-x0)),1/((x2-x1)(x2-x0))]

    Note
    ----
    If length of interval [a,b]=4C then weights will grow or decay
    exponentially at C^{-n} where n is number of points causing overflow
    or underflow.

    To minimize this effect multiply each x_j-x_k by C^{-1}. This has effect
    of rescaling all weights by C^n. In rare situations where n is so large
    randomize or use Leja ordering of the samples before computing weights.
    See Barycentric Lagrange Interpolation by
    Jean-Paul Berrut and Lloyd N. Trefethen 2004
    """
    if interval_length is None:
        scaling_factor = 1.
    else:
        scaling_factor = interval_length/4.
        
    C_inv = 1/scaling_factor
    num_samples = samples.shape[0]

    try:
        from pyapprox.cython.barycentric_interpolation import \
            compute_barycentric_weights_1d_pyx
        weights = compute_barycentric_weights_1d_pyx(samples,C_inv)
    except:
        print ('compute_barycentric_weights_1d extension failed')

        #X=np.tile(samples[:,np.newaxis],[1,samples.shape[0]])
        #result=1./np.prod(X-X.T+np.eye(samples.shape[0]),axis=0)
        #return result
    
        weights = np.empty((num_samples, num_samples),dtype=float)
        weights[0,0] = 1.
        for jj in range(1,num_samples):
            weights[jj,:jj] = C_inv*(samples[:jj]-samples[jj])*weights[jj-1,:jj]
            weights[jj,jj]  = np.prod(C_inv*(samples[jj]-samples[:jj]))
            weights[jj-1,:jj] = 1./weights[jj-1,:jj]

        weights[num_samples-1,:num_samples]=\
            1./weights[num_samples-1,:num_samples]

    
    if not return_sequence:
        result = weights[num_samples-1,:]
        # make sure magintude of weights is approximately O(1)
        # useful to sample sets like leja for gaussian variables
        # where interval [a,b] is not very useful
        #print('max_weights',result.min(),result.max())
        if normalize_weights:
            raise Exception('I do not think I want to support this option')
            result /= np.absolute(result).max()
            #result[I]=result

    else:
        result = weights

    assert np.all(np.isfinite(result)),(num_samples)
    return result


def barycentric_lagrange_interpolation_precompute(
        num_act_dims,abscissa_1d,barycentric_weights_1d,
        active_abscissa_indices_1d_list):
    num_abscissa_1d = np.empty((num_act_dims),dtype=int )
    num_active_abscissa_1d = np.empty((num_act_dims),dtype=int )
    shifts = np.empty((num_act_dims),dtype=int)

    
    shifts[0] = 1
    num_abscissa_1d[0] = abscissa_1d[0].shape[0]
    num_active_abscissa_1d[0] = active_abscissa_indices_1d_list[0].shape[0]
    max_num_abscissa_1d = num_abscissa_1d[0]
    for act_dim_idx in range(1,num_act_dims):
        num_abscissa_1d[act_dim_idx] = abscissa_1d[act_dim_idx].shape[0]
        num_active_abscissa_1d[act_dim_idx] = \
            active_abscissa_indices_1d_list[act_dim_idx].shape[0]
        # multi-index needs only be defined over active_abscissa_1d 
        shifts[act_dim_idx] = \
            shifts[act_dim_idx-1]*num_active_abscissa_1d[act_dim_idx-1]
        max_num_abscissa_1d=max(
            max_num_abscissa_1d,num_abscissa_1d[act_dim_idx])

    max_num_active_abscissa_1d = num_active_abscissa_1d.max()
    active_abscissa_indices_1d = np.empty(
        (num_act_dims,max_num_active_abscissa_1d),dtype=int)
    for dd in range(num_act_dims):
        active_abscissa_indices_1d[dd,:num_active_abscissa_1d[dd]] = \
          active_abscissa_indices_1d_list[dd]
    
    # Create locality of data for increased preformance
    abscissa_and_weights = np.empty(
        (2*max_num_abscissa_1d, num_act_dims),dtype=float)
    for dd in range(num_act_dims):
        for ii in range(num_abscissa_1d[dd]):
            abscissa_and_weights[2*ii,dd] = abscissa_1d[dd][ii]
            abscissa_and_weights[2*ii+1,dd] = barycentric_weights_1d[dd][ii]
        
    return num_abscissa_1d,num_active_abscissa_1d,shifts,abscissa_and_weights,\
      active_abscissa_indices_1d

def multivariate_hierarchical_barycentric_lagrange_interpolation( 
			x,
			abscissa_1d, 
			barycentric_weights_1d,
			fn_vals,
			active_dims,
			active_abscissa_indices_1d):
    """
    Parameters
    ----------
    x : np.ndarray (num_vars, num_samples)
        The samples at which to evaluate the interpolant

    abscissa_1d : [np.ndarray]
        List of interpolation nodes in each active dimension. Each array
        has ndim==1

    barycentric_weights_1d : [np.ndarray]
        List of barycentric weights in each active dimension, corresponding to 
        each of the interpolation nodes. Each array has ndim==1

    fn_vals : np.ndarray (num_samples, num_qoi)
        The function values at each of the interpolation nodes
        Each column is a flattened array that assumes the nodes
        were created with the same ordering as generated by
        the function cartesian_product.

        if active_abscissa_1d is not None the fn_vals must be same size as
        the tensor product of the active_abscissa_1d.

        Warning: Python code takes fn_vals as num_samples x num_qoi
        but c++ code takes num_qoi x num_samples. Todo change c++ code
        also look at c++ code to compute barycentric weights. min() on line 154
        seems to have no effect.

    active_dims : np.ndarray (num_active_dims)
        The dimensions which have more than one interpolation node. TODO
        check if this can be simply extracted in this function by looking 
        at abscissa_1d.

    active_abscissa_indices_1d : [np.ndarray]
        The list (over each dimension) of indices for which we will compute
        barycentric basis functions. This is useful when used with
        heirarchical interpolation where the function values will be zero
        at some nodes and thus there is no need to compute associated basis
        functions

    Returns
    -------
    result : np.ndarray (num_samples,num_qoi)
        The values of the interpolant at the samples x
    """
    eps = 2*np.finfo(float).eps
    num_pts = x.shape[1]
    num_act_dims = active_dims.shape[0]

    """
    # this can be true for univariate quadrature rules that are not closed
    # i.e on bounded domain and with samples on both boundaries
    # need to make this check better
    x_min, x_max = x.min(axis=1), x.max(axis=1)
    for ii in range(num_act_dims):
        if (x_min<abscissa_1d[active_dims[ii]].min() or
            x_max>abscissa_1d[active_dims[ii]].max()):
            print ('warning extrapolating outside abscissa')
            print(x_min,x_max,abscissa_1d[active_dims[ii]].min(),
                  abscissa_1d[active_dims[ii]].max())
    """


    num_abscissa_1d, num_active_abscissa_1d, shifts, abscissa_and_weights, \
      active_abscissa_indices_1d = \
        barycentric_lagrange_interpolation_precompute(
            num_act_dims,abscissa_1d,barycentric_weights_1d,
            active_abscissa_indices_1d)

    try:
        from pyapprox.cython.barycentric_interpolation import \
            multivariate_hierarchical_barycentric_lagrange_interpolation_pyx
        return multivariate_hierarchical_barycentric_lagrange_interpolation_pyx(
            x, fn_vals, active_dims, active_abscissa_indices_1d, 
            num_abscissa_1d, num_active_abscissa_1d, shifts, 
            abscissa_and_weights)
    
        # from pyapprox.weave.barycentric_interpolation import \
        #     c_multivariate_hierarchical_barycentric_lagrange_interpolation
        # return c_multivariate_hierarchical_barycentric_lagrange_interpolation(
        #     x,abscissa_1d,barycentric_weights_1d,fn_vals,active_dims,
        #     active_abscissa_indices_1d)
    except:
        msg =  'multivariate_hierarchical_barycentric_lagrange_interpolation '
        msg += 'extension failed'
        print (msg)

    max_num_abscissa_1d = abscissa_and_weights.shape[0]//2
    multi_index = np.empty((num_act_dims),dtype=int)
  
    num_qoi = fn_vals.shape[1]
    result = np.empty((num_pts,num_qoi),dtype=float )
    # Allocate persistent memory. Each point will fill in a varying amount
    # of entries. We use a view of this memory to stop reallocation for each 
    # data point
    act_dims_pt_persistent = np.empty((num_act_dims),dtype=int)
    act_dim_indices_pt_persistent = np.empty((num_act_dims),dtype=int)
    c_persistent = np.empty((num_qoi,num_act_dims),dtype=float)
    bases = np.empty((max_num_abscissa_1d, num_act_dims),dtype=float)
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
                if ( abs( x_dim_k - abscissa_1d[act_dim_idx][ii] ) < eps ):
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
            result[kk,:] = 0.
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
                
                if ( abs(denom_d) < eps ):
                    # print ('#')
                    print('abscissa',abscissa_and_weights[::2,act_dim_idx])
                    print('weights',abscissa_and_weights[1::2,act_dim_idx])
                    print(denom_d)
                    # print(bases)
                    # print(dim)
                    # print(num_abscissa)
                    # print(num_act_dims_pt)
                    # print(x.shape)
                    raise Exception("interpolation absacissa are not unique")
                denom *= denom_d

            # end for dd in range(num_act_dims_pt):
                
            if ( num_act_dims_pt == 0 ):
                # if point is an abscissa return the fn value at that point
                #fn_val_index = 0
                #for act_dim_idx in range(num_act_dims):
                #    fn_val_index+=multi_index[act_dim_idx]*shifts[act_dim_idx]
                fn_val_index = np.sum(multi_index*shifts)
                result[kk,:] = fn_vals[fn_val_index,:]
            else:
                # compute interpolant
                c_persistent[:,:] = 0.
                done = True
                if ( num_act_dims_pt > 1 ): done = False
                #fn_val_index = 0
                #for act_dim_idx in range(num_act_dims):
                #    fn_val_index += multi_index[act_dim_idx]*shifts[act_dim_idx]
                fn_val_index = np.sum(multi_index*shifts)
                while (True):
                    act_dim_idx = act_dim_indices_pt_persistent[0]
                    for ii in range(num_active_abscissa_1d[act_dim_idx]):
                        fn_val_index+=shifts[act_dim_idx]*(ii-multi_index[act_dim_idx])
                        multi_index[act_dim_idx] = ii
                        basis=bases[active_abscissa_indices_1d[act_dim_idx][ii],0]
                        c_persistent[:,0]+= basis * fn_vals[fn_val_index,:]
                        
                    for dd in range(1,num_act_dims_pt):
                        act_dim_idx = act_dim_indices_pt_persistent[dd]
                        basis = bases[active_abscissa_indices_1d[act_dim_idx][multi_index[act_dim_idx]],dd]
                        c_persistent[:,dd] += basis * c_persistent[:,dd-1]
                        c_persistent[:,dd-1] = 0.
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
                result[kk,:] = c_persistent[:,num_act_dims_pt-1] / denom
                if np.any(np.isnan(result[kk,:])):
                    #print (c_persistent [:,num_act_dims_pt-1])
                    #print (denom)
                    raise Exception('Error values not finite')
    return result

def multivariate_barycentric_lagrange_interpolation(
        x, abscissa_1d, barycentric_weights_1d,fn_vals,active_dims):
    
    num_active_dims = active_dims.shape[0]
    active_abscissa_indices_1d = []
    for active_index in range(num_active_dims):
        active_abscissa_indices_1d.append(np.arange(
            abscissa_1d[active_index].shape[0]))

    return multivariate_hierarchical_barycentric_lagrange_interpolation(
        x,abscissa_1d,barycentric_weights_1d,fn_vals,active_dims,
        active_abscissa_indices_1d)
    

def clenshaw_curtis_barycentric_weights( level ):
    if ( level == 0 ):
        return np.array( [0.5], np.float )
    else:
        mi = 2**(level) + 1
        w = np.ones( mi, np.double )
        w[0] = 0.5; w[mi-1] = 0.5;
        w[1::2] = -1.
        return w

def equidistant_barycentric_weights( n ):
    w = np.zeros( n, np.double )
    for i in range( 0, n - n%2, 2 ):
        w[i] = 1. * nchoosek( n-1, i )
        w[i+1] = -1. * nchoosek( n-1, i+1 )
    if ( n%2 == 1 ): 
        w[n-1] = 1.
    return w
