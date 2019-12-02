from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

def s_sparse_projection(signal,sparsity):
    assert signal.ndim==1
    projected_signal = signal.copy()
    projected_signal[np.argsort(np.absolute(signal))[:-sparsity]]=0.
    return projected_signal

def iterative_hard_thresholding(approx_eval, apply_approx_adjoint_jacobian,
                                project, obs, initial_guess, tol, maxiter,
                                max_linesearch_iter=10, backtracking_tol=0.1,
                                resid_tol=1e-10, verbosity=0):
    """
    """
    assert obs.ndim==1
    obsnorm = np.linalg.norm(obs)
    assert initial_guess.ndim==1

    guess = initial_guess.copy()
    # ensure approx_eval has ndim==1 using squeeze
    resid = obs-approx_eval(guess).squeeze()
    residnorm = np.linalg.norm(resid)
    
    for ii in range(maxiter):
        grad = apply_approx_adjoint_jacobian(guess,resid)
        descent_dir = -grad
        step_size = 1.0
        # backtracking line search
        tau = 0.5
        obj = 0.5*np.dot(resid.T,resid)
        init_slope = np.dot(grad,descent_dir)
        it = 0
        while True:
            new_guess = guess + step_size*descent_dir
            new_resid = obs-approx_eval(new_guess).squeeze()
            new_obj = np.dot(new_resid.T,new_resid)
            if (new_obj<=obj+backtracking_tol*step_size*init_slope):
                if verbosity>1:
                    print ('lineserach achieved sufficient decrease')
                break
            step_size *= tau
            it += 1
            if it==max_linesearch_iter:
                if verbosity>1:
                    print ('maximum linsearch iterations reached')
                break
        if verbosity>1:
            print(("step_size", step_size))
        new_guess = project(new_guess)
        
        guess = new_guess.copy()
        resid = obs-approx_eval(guess).squeeze()
        residnorm_prev = residnorm
        residnorm = np.linalg.norm(resid)

        if verbosity>0:
            if ( ii%10 == 0 ):
                print(('iteration %d the residual norm is %1.2e' %(
                    ii,residnorm/obsnorm)))
        
        if residnorm/obsnorm < tol:
            if verbosity>0:
                print ('Terminating: relative norm or residual is below tolerance')
            break

        if abs(residnorm_prev-residnorm)<resid_tol:
            msg = 'Terminating: decrease in residual norm did not exceed '
            msg += 'tolerance'
            if verbosity>0:
                print (msg)
            break
        
    return guess, residnorm

def orthogonal_matching_pursuit(approx_eval, apply_approx_adjoint_jacobian,
                                least_squares_regression,
                                obs, active_indices, num_indices, tol,
                                max_sparsity, verbosity=0):

    if active_indices is not None:
        assert active_indices.ndim==1
        
    inactive_indices_mask = np.asarray([True]*num_indices)    
    obsnorm = np.linalg.norm(obs)

    max_sparsity=min(max_sparsity,num_indices)

    if active_indices is not None:
        inactive_indices_mask[active_indices]=False
        assert active_indices.shape[0]>0
        sparse_guess = least_squares_regression(
            np.asarray(active_indices),None)
        guess = np.zeros((num_indices),dtype=float)
        guess[active_indices] = sparse_guess
        resid = obs-approx_eval(guess).squeeze()
        assert resid.ndim==1
    else:
        active_indices = np.empty((0),dtype=int)
        resid = obs.copy()
        guess = np.zeros(num_indices)
    if verbosity>0:
        print(('sparsity'.center(8),'index'.center(5),'||r||'.center(9)))
    while True:
        residnorm = np.linalg.norm(resid)
        if verbosity>0:
            if active_indices.shape[0]>0:
                print((repr(active_indices.shape[0]).center(8),repr(active_indices[-1]).center(5),format(residnorm,'1.3e').center(9)))
        if residnorm/obsnorm < tol:
            if verbosity>0:
                print ('Terminating: relative residual norm is below tolerance')
            break

        if len(active_indices)>=num_indices:
            if verbosity>0:
                print ('Terminating: all basis functions have been added')
            break

        if len(active_indices)>=max_sparsity:
            if verbosity>0:
                print ('Terminating: maximum number of basis functions added')
            break
        
        grad = apply_approx_adjoint_jacobian(guess,resid)
        
        num_inactive_indices = num_indices-active_indices.shape[0]
        inactive_indices = np.arange(num_indices)[inactive_indices_mask]
        best_inactive_index = np.argmax(np.absolute(grad[inactive_indices]))
        best_index = inactive_indices[best_inactive_index]
        active_indices = np.append(active_indices,best_index)
        inactive_indices_mask[best_index]=False
        initial_lstsq_guess = guess[active_indices]
        initial_lstsq_guess[-1] = 1e-2
        sparse_guess = least_squares_regression(
            active_indices,initial_lstsq_guess)
        guess = np.zeros((num_indices),dtype=float)
        guess[active_indices] = sparse_guess
        resid = obs-approx_eval(guess).squeeze()
        assert resid.ndim==1

    return guess, residnorm

        
        
        
    
    
