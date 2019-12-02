from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from pyapprox.utilities import truncated_pivoted_lu_factorization
from scipy.linalg import qr as qr_factorization
from scipy.linalg import solve_triangular
from pyapprox.probability_measure_sampling import rejection_sampling
from pyapprox.orthogonal_least_interpolation import LeastInterpolationSolver
from pyapprox.indexing import get_total_degree, compute_hyperbolic_indices, \
    compute_hyperbolic_level_indices
def christoffel_function(samples,basis_matrix_generator):
    """
    Evaluate the christoffel function K(x) at a set of samples x. 

    Useful for preconditioning linear systems generated using 
    orthonormal polynomials
    """
    basis_matrix = basis_matrix_generator(samples)
    return 1./christoffel_weights(basis_matrix)

def christoffel_weights(basis_matrix):
    """
    Evaluate the 1/K(x),from a basis matrix, where K(x) is the 
    Christoffel function.
    """
    return 1./np.sum(basis_matrix**2,axis=1)

def christoffel_preconditioner(basis_matrix,samples):
    return christoffel_weights(basis_matrix)


def get_fekete_samples(generate_basis_matrix,generate_candidate_samples,
                       num_candidate_samples,preconditioning_function=None,
                       precond_opts=dict()):
    """
    Generate Fekete samples using QR factorization. 

    The number of samples is determined by the number of basis functions.

    Parameters
    ----------
    generate_basis_matrix : callable
        basis_matrix = generate_basis_matrix(candidate_samples)
        Function to evaluate a basis at a set of samples

    generate_candidate_samples : callable
        candidate_samples = generate_candidate_samples(num_candidate_samples)
        Function to generate candidate samples. This can siginficantly effect
        the fekete samples generated

    num_candidate_samples : integer
        The number of candidate_samples

    preconditioning_function : callable
        basis_matrix = preconditioning_function(basis_matrix,samples)
        precondition a basis matrix to improve stability.
        samples are the samples used to build the basis matrix. They must
        be in the same order as they were used to create the rows of the basis 
        matrix. Note if using probability density function for 
        preconditioning make sure density has been mapped to canonical domain

    TODO unfortunately some preconditioing_functions need only basis matrix
    or samples, but cant think of a better way to generically pass in function
    here other than to require functions that use both arguments

    Returns
    -------
    fekete_samples : np.ndarray (num_vars, num_indices)
        The Fekete samples

    data_structures : tuple
        (Q,R,p) the QR factors and pivots. This can be useful for
        quickly building an interpolant from the samples
    """
    candidate_samples = generate_candidate_samples(num_candidate_samples)
    basis_matrix = generate_basis_matrix(candidate_samples)
    if preconditioning_function is not None:
        weights = np.sqrt(
            preconditioning_function(basis_matrix,candidate_samples))
        basis_matrix = np.dot(np.diag(weights),basis_matrix)
    else:
        weights = None
    Q,R,p = qr_factorization(basis_matrix.T,pivoting=True)
    p = p[:basis_matrix.shape[1]]
    fekete_samples = candidate_samples[:,p]
    data_structures=(Q,R[:,:basis_matrix.shape[1]],p,weights[p])
    return fekete_samples, data_structures

def get_lu_leja_samples(generate_basis_matrix,generate_candidate_samples,
                        num_candidate_samples,num_leja_samples,
                        preconditioning_function=None,initial_samples=None):
    """
    Generate Leja samples using LU factorization. 

    Parameters
    ----------
    generate_basis_matrix : callable
        basis_matrix = generate_basis_matrix(candidate_samples)
        Function to evaluate a basis at a set of samples

    generate_candidate_samples : callable
        candidate_samples = generate_candidate_samples(num_candidate_samples)
        Function to generate candidate samples. This can siginficantly effect
        the fekete samples generated

    num_candidate_samples : integer
        The number of candidate_samples

    preconditioning_function : callable
        basis_matrix = preconditioning_function(basis_matrix)
        precondition a basis matrix to improve stability
        samples are the samples used to build the basis matrix. They must
        be in the same order as they were used to create the rows of the basis 
        matrix.

    TODO unfortunately some preconditioing_functions need only basis matrix
    or samples, but cant think of a better way to generically pass in function
    here other than to require functions that use both arguments

    num_leja_samples : integer
        The number of desired leja samples. Must be <= num_indices

    initial_samples : np.ndarray (num_vars,num_initial_samples)
       Enforce that the initial samples are chosen (in the order specified)
       before any other candidate sampels are chosen. This can lead to
       ill conditioning and leja sequence terminating early

    Returns
    -------
    laja_samples : np.ndarray (num_vars, num_indices)
        The samples of the Leja sequence

    data_structures : tuple
        (Q,R,p) the QR factors and pivots. This can be useful for
        quickly building an interpolant from the samples
    """
    candidate_samples = generate_candidate_samples(num_candidate_samples)
    if initial_samples is not None:
        candidate_samples = np.hstack((initial_samples,candidate_samples))
        num_initial_rows = initial_samples.shape[1]
    else:
        num_initial_rows=0
        
    basis_matrix = generate_basis_matrix(candidate_samples)

    assert num_leja_samples <= basis_matrix.shape[1]
    if preconditioning_function is not None:
        weights = np.sqrt(
            preconditioning_function(basis_matrix,candidate_samples))
        basis_matrix = (basis_matrix.T*weights).T
    else:
        weights = None
    L,U,p = truncated_pivoted_lu_factorization(
        basis_matrix,num_leja_samples,num_initial_rows)
    assert p.shape[0]==num_leja_samples, (p.shape, num_leja_samples)
    p = p[:num_leja_samples]
    leja_samples = candidate_samples[:,p]
    # Ignore basis functions (columns) that were not considered during the
    # incomplete LU factorization
    L = L[:,:num_leja_samples]
    U = U[:num_leja_samples,:num_leja_samples]
    data_structures=[L,U,p,weights[p]]
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        print(('N:', basis_matrix.shape[1]))
        plt.plot(leja_samples[0,0],leja_samples[1,0],'*')
        plt.plot(leja_samples[0,:],leja_samples[1,:],'ro',zorder=10)
        plt.scatter(candidate_samples[0,:],candidate_samples[1,:],s=weights*100,color='b')
        #plt.xlim(-1,1)
        #plt.ylim(-1,1)
        #plt.title('Leja sequence and candidates')
        #print (weights[p])
        plt.show()
    return leja_samples, data_structures

def total_degree_basis_generator(num_vars,degree):
    """
    Generate all indices i such that ||i||_1=degree.
    This function is useful when computing oli_leja sequences
    """
    return (degree+1,compute_hyperbolic_level_indices(num_vars,degree,1.0))

def get_oli_leja_samples(pce, generate_candidate_samples, num_candidate_samples,
                         num_leja_samples, preconditioning_function=None,
                         basis_generator=total_degree_basis_generator,
                         initial_samples=None):
    """
    Generate Leja samples using orthogonal least interpolation. 

    The number of samples is determined by the number of basis functions.

    Parameters
    ----------
    generate_basis_matrix : callable
        basis_matrix = generate_basis_matrix(candidate_samples)
        Function to evaluate a basis at a set of samples

    generate_candidate_samples : callable
        candidate_samples = generate_candidate_samples(num_candidate_samples)
        Function to generate candidate samples. This can siginficantly effect
        the fekete samples generated

    num_candidate_samples : integer
        The number of candidate_samples

    preconditioning_function : callable
        basis_matrix = preconditioning_function(basis_matrix)
        precondition a basis matrix to improve stability

    num_leja_samples : integer
        The number of desired leja samples. Must be <= num_indices

    Returns
    -------
    laja_samples : np.ndarray (num_vars, num_indices)
        The samples of the Leja sequence

    data_structures : tuple
        (oli_solver,) the final state of the othogonal least interpolation 
        solver. This is useful for quickly building an interpolant
    """
    oli_opts = dict()
    oli_solver = LeastInterpolationSolver()
    oli_solver.configure(oli_opts)
    oli_solver.set_pce(pce)
    
    if preconditioning_function is not None:
        oli_solver.set_preconditioning_function(preconditioning_function)
        
    oli_solver.set_basis_generator(basis_generator)

    num_vars = pce.num_vars()
    max_degree = get_total_degree(num_vars,num_leja_samples)
    indices = compute_hyperbolic_indices(num_vars, max_degree, 1.)
    # warning this assumes basis generator is always compute_hyperbolic_indices
    # with p=1
    assert indices.shape[1]>=num_leja_samples
    pce.set_indices(indices)

    assert num_leja_samples <= num_candidate_samples

    candidate_samples = generate_candidate_samples(num_candidate_samples)

    oli_solver.factorize(
        candidate_samples, initial_samples, num_selected_pts=num_leja_samples)

    leja_samples = oli_solver.get_current_points()

    data_structures = (oli_solver,)
    
    return leja_samples, data_structures

def interpolate_fekete_samples(fekete_samples,values,data_structures):
    """
    Assumes ordering of values and rows of L and U are consistent.
    Typically this is done by computing leja samples then evaluating function
    at these samples.
    """
    Q,R = data_structures[0],data_structures[1]
    precond_weights = data_structures[3]
    # QR is a decomposition of V.T, V=basis_matrix(samples)
    # and we want to solve V*coeff = values
    temp = solve_triangular(R.T,(values.T*precond_weights).T,lower=True) 
    coef = np.dot(Q,temp)
    return coef


def interpolate_lu_leja_samples(leja_samples,values,data_structures):
    """
    Assumes ordering of values and rows of L and U are consistent.
    Typically this is done by computing leja samples then evaluating function
    at these samples.
    """
    L,U = data_structures[0],data_structures[1]
    weights = data_structures[3]
    temp = solve_triangular(L,(values.T*weights).T,lower=True)
    coef = solve_triangular(U,temp,lower=False)
    return coef

def random_induced_measure_sampling(num_samples,num_vars,
                                    basis_matrix_generator,
                                    probability_density,
                                    proposal_density, 
                                    generate_proposal_samples,
                                    envelope_factor):

    target_density = lambda x: (probability_density(x)*
                                christoffel_function(x,basis_matrix_generator))
        
    samples = rejection_sampling(
        target_density, proposal_density, generate_proposal_samples,
        envelope_factor, num_vars, num_samples, verbose=False)

    return samples

def get_quadrature_weights_from_fekete_samples(fekete_samples,data_structures):
    Q,R = data_structures[0],data_structures[1]
    precond_weights = data_structures[3]
    # QR is a decomposition of V.T, V=basis_matrix(samples)
    # and we want to compute inverse of V=(QR).T
    basis_matrix_inv = np.linalg.inv(np.dot(Q,R).T)
    quad_weights = basis_matrix_inv[0,:]
    if precond_weights is not None:
        # Since we preconditioned, we need to "un-precondition" to get
        # the right weights. Sqrt of weights has already been applied
        # so do not do it again here
        quad_weights *= precond_weights
    return quad_weights

def get_quadrature_weights_from_lu_leja_samples(leja_samples,data_structures):
    L,U = data_structures[0],data_structures[1]
    precond_weights = data_structures[3]
    basis_matrix_inv = np.linalg.inv(np.dot(L,U))
    quad_weights = basis_matrix_inv[0,:]
    if precond_weights is not None:
        # Since we preconditioned, we need to "un-precondition" to get
        # the right weights. Sqrt of weights has already been applied
        # so do not do it again here
        quad_weights *= precond_weights
    return quad_weights
    

    
    
