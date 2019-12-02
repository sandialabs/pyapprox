from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
     evaluate_multivariate_orthonormal_polynomial
from matplotlib import pyplot as plt
from pyapprox.utilities import cartesian_product, outer_product,\
    get_tensor_product_quadrature_rule
from pyapprox.indexing import set_difference
from pyapprox.monomial import monomial_basis_matrix

class APC(PolynomialChaosExpansion):
    def __init__(self,compute_moment_matrix_function=None,moments=None,
                 compute_grammian_function=None):
        super(APC,self).__init__()
        self.compute_moment_matrix_function = compute_moment_matrix_function
        self.compute_grammian_function = compute_grammian_function
        self.moment_matrix_cond = None
        self.moments=moments
        self.R_inv = None

    def compute_rotation(self):
        if self.moments is not None:
            if type(self.moments)!=np.ndarray:
                raise Exception('moments was given but was not a np.ndarray')
            assert self.compute_moment_matrix_function is None
            self.R_inv = compute_rotation_from_moments_linear_system(self.moments)
        elif self.compute_moment_matrix_function is not None:
            if not callable(self.compute_moment_matrix_function):
                msg =  'compute_moment_matrix_function was given but was '
                msg += 'not a callable function'
                raise Exception(msg)
            assert self.moments is None
            assert self.compute_grammian_function is None
            
            moment_matrix = self.compute_moment_matrix_function(
                self.unrotated_basis_matrix)
            self.R_inv = compute_rotation_qr(moment_matrix)
            self.moment_matrix_cond = np.linalg.cond(moment_matrix)
        elif self.compute_grammian_function is not None:
            if not callable(self.compute_grammian_function):
                msg =  'compute_grammian_function was given but was '
                msg += 'not a callable function'
                raise Exception(msg)
            assert self.moments is None
            assert self.compute_moment_matrix_function is None

            grammian = self.compute_grammian_function(self.unrotated_basis_matrix,
                                                      self.indices)
            # cholesky requires moment_matrix function to return grammian
            # A'*A not basis matrix A
            assert grammian.shape[0]==grammian.shape[1]
            self.R_inv = compute_rotation_cholesky(grammian)
            self.moment_matrix_cond = np.linalg.cond(grammian)
        else:
            raise Exception

    def unrotated_canonical_basis_matrix(self,canonical_samples):
        """
        Cannot just call super(APCE,self).canonical_basis because I was 
        running into inheritance problems.
        """
        deriv_order = 0
        if self.recursion_coeffs is not None:
            unrotated_basis_matrix = \
                evaluate_multivariate_orthonormal_polynomial(
                    canonical_samples,self.indices,self.recursion_coeffs,
                    deriv_order,self.basis_type_index_map)
        else:
            unrotated_basis_matrix = monomial_basis_matrix(
                self.indices,canonical_samples)
        return unrotated_basis_matrix

    def unrotated_basis_matrix(self,samples):
        assert samples.ndim==2
        assert samples.shape[0]==self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical_space(
            samples)
        matrix = self.unrotated_canonical_basis_matrix(canonical_samples)
        return matrix
    
    def canonical_basis_matrix(self,canonical_samples,opts=dict()):
        deriv_order = opts.get('deriv_order',0)
        assert deriv_order==0
        unrotated_basis_mat = self.unrotated_canonical_basis_matrix(
            canonical_samples)

        if self.R_inv is not None:
            #basis_matrix = np.dot(unrotated_basis_mat,self.R_inv)
            basis_matrix = unrotated_basis_mat.dot(
                self.R_inv[:self.num_terms(),:self.num_terms()])
        else:
            basis_matrix = unrotated_basis_mat
        return basis_matrix

    def basis_matrix(self,samples,opts=dict()):
        assert samples.ndim==2
        assert samples.shape[0]==self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical_space(
            samples)
        return self.canonical_basis_matrix(canonical_samples,opts)

    # def basis_matrix(self,samples):
    #     if self.compute_moment_matrix_function is not None:
    #         return np.dot(self.unrotated_basis_matrix(samples),self.R_inv)
    #     else:
    #         return self.unrotated_basis_matrix(samples)

    # def unrotated_basis_matrix(self,samples):
    #     return super(APC,self).basis_matrix(samples)

    def set_indices(self,indices):
        # need to perform check before base class updated self.indices
        # if (self.compute_moment_matrix_function is not None and
        #         (self.indices is None or
        #              self.indices.shape[1]!=indices.shape[1] or
        #         not np.allclose(indices,self.indices))):
        #     update_rotation=True
        # else:
        #     update_rotation=False

        # TODO eventually use following. need to distinguish between
        # indices for pce and indices used for computing basis rotation


        update_indices=False
        if self.indices is None:
            update_indices=True
        else:
            # check indices is subset of self.indices
            if indices.shape[1]>self.indices.shape[1]:
                update_indices=True
            else:
                # check indices is subset of self.indices
                update_indices=set_difference(
                    self.indices,indices).shape[1]!=0


        if update_indices:
            super(APC,self).set_indices(indices)
            if (self.compute_moment_matrix_function is not None or 
                self.moments is not None or
                self.compute_grammian_function is not None):
                self.compute_rotation()

    #TODO: __call__ take advantage of fact that
    # basis_matrix.dot(R_inv.dot(coeff))
    # (m x n)*((n x n)*(n*1)) = n**2+m*n
    # is faster than
    # (basis_matrix.dot(R_inv)).dot(coeff)
    # ((m x n)*(n x n))*(n*1) = m*n**2+m*n
    # faster way can also leverage subsitution solve
    # instead of directly inverting R to get R_inv



def compute_moment_matrix_from_samples(basis_matrix_func,samples):
    return basis_matrix_func(samples)/np.sqrt(samples.shape[1])

def compute_moment_matrix_using_tensor_product_quadrature(
        basis_matrix_func,num_samples,num_vars,
        univariate_quadrature_rule,density_function=None):
    """
    Parameters
    ----------
    num_samples : integer
       The number of samples in the 1D quadrature rule

    univariate_quadrature_rule : tuple (x,w)
       x : np.ndarray (num_samples) the quadrature points in the user space
       w : np.ndarray (num_samples) the quadrature weights

    density_function : callable
       v = density_function(x)
       A probability density function. If not None then quadrature rule
       should be for lebesque measure and weights will be multiplied by the 
       value of the density at the quarature points
    """
    samples,weights = get_tensor_product_quadrature_rule(
        num_samples,num_vars,univariate_quadrature_rule,None,density_function)
    basis_matrix = basis_matrix_func(samples)
    moment_matrix = np.dot(np.diag(np.sqrt(weights)),basis_matrix)
    return moment_matrix

def compute_coefficients_of_unrotated_basis(coefficients,R_inv):
    r"""
    Given pce coefficients a such that p(z)=\Phi(Z)*a
    where phi is the multivariate orthgonal Gram-Schmidt basis compute
    coefficients of tensor-product basis \psi where \Phi(Z)=\Psi(Z)*R_inv
    and \Phi(Z),\Psi(Z) are vandermonde matrices evaluate at the samples Z.
    """
    num_terms = coefficients.shape[0]
    unrotated_basis_coefficients = np.zeros_like(coefficients)
    for ii in range(num_terms):
        unrotated_basis_coefficients[ii,:]=\
            np.sum(coefficients[ii:,:]*(R_inv[ii,ii:])[:,np.newaxis],
                   axis=0)
            #for kk in range(pce.coefficients.shape[1]):
                #for jj in range(ii,num_terms):
                #    unrotated_basis_coefficients[ii,kk]+=\
                #        pce.coefficients[jj,kk]*pce.R_inv[ii,jj]
    return unrotated_basis_coefficients

class FPC(APC):
    def __init__(self,compute_moment_matrix_function):
        super(FPC,self).__init__(compute_moment_matrix_function)

    def configure(self, opts):
        super(FPC,self).configure(opts)
        self.truncation_tol=opts['truncation_tol']

    def compute_rotation(self):
        moment_matrix = self.compute_moment_matrix_function(
                self.unrotated_basis_matrix)
        assert moment_matrix.shape[0]>=moment_matrix.shape[1]
        grammian = np.dot(moment_matrix.T,moment_matrix)
        U_factor, S_factor, V_factor = np.linalg.svd(grammian)
        I = np.where(S_factor>self.truncation_tol)[0]
        if (I.shape[0]<grammian.shape[0]):
            raise Exception('frame approximation produced low-rank basis')
        truncated_singular_vectors = (U_factor[:,I]/np.sqrt(S_factor[I]))
        num_truncated_bases = truncated_singular_vectors.shape[1]
        self.R_inv = truncated_singular_vectors
        self.moment_matrix_cond = S_factor[I].max()/S_factor[I].min()

    def num_terms(self):
        # truncated svd creates basis with num_terms <= num_indices
        return self.R_inv.shape[1]

def compute_rotation_from_moments_linear_system(poly_moments):
    num_terms = poly_moments.shape[0]
    R_inv = np.zeros((num_terms,num_terms),dtype=float)
    R_inv[0,0]=1.
    for kk in range(1,num_terms):
        moment_matrix = np.zeros((kk+1,kk+1),dtype=float)
        for ii in range(kk):
            for jj in range(kk+1):
                moment_matrix[ii,jj] = poly_moments[ii,jj]
        moment_matrix[-1,-1]=1.

        rhs = np.zeros(kk+1); rhs[-1]=1.
        rotated_basis_coefs = np.linalg.solve(moment_matrix,rhs)
        R_inv[:kk+1,kk] = rotated_basis_coefs

        # orthonormalize
        l2_norm = 0.
        for ii in range(kk+1):
            for jj in range(kk+1):
                l2_norm += R_inv[ii,kk]*R_inv[jj,kk]*poly_moments[ii,jj]
        R_inv[:kk+1,kk]/=np.sqrt(l2_norm)
        
    return R_inv

def compute_rotation_from_moments_gram_schmidt(poly_moments):
    num_terms = poly_moments.shape[0]
    R_inv = np.zeros((num_terms,num_terms),dtype=float)
    R_inv[0,0]=1.
    for kk in range(1,num_terms):
        R_inv[kk,kk]=1.
        for ii in range(kk):
            # compute <e_k,phi_i>
            numerator=0.
            for jj in range(ii+1):
                numerator+=poly_moments[kk,jj]*R_inv[jj,ii]
            # basis_norms are one because of Step ***
            ratio = numerator#/basis_norms[ii] 
            # compute contribution to r_ik
            for jj in range(ii+1):
                R_inv[jj,kk]+=-ratio*R_inv[jj,ii]
                
        # compute <phi_k,phi_k>
        basis_norm = 0
        for ii in range(kk+1):
            for jj in range(kk+1):
                basis_norm += R_inv[ii,kk]*R_inv[jj,kk]*poly_moments[ii,jj]
        R_inv[:kk+1,kk]/=np.sqrt(basis_norm) # Step ***
        
    return R_inv

def compute_rotation_qr(moment_matrix):
    assert moment_matrix.shape[0]>=moment_matrix.shape[1]
    Q_factor, R_factor = np.linalg.qr(moment_matrix)
    for ii in range(R_factor.shape[0]):
        if R_factor[ii,ii] < 0.:
            R_factor[ii,:] *=-1.
            Q_factor[:,ii] *=-1.
    R_inv = np.linalg.inv(R_factor)
    return R_inv


def compute_rotation_cholesky(grammian_matrix):
    assert grammian_matrix.shape[0]==grammian_matrix.shape[1]
    L_factor = np.linalg.cholesky(grammian_matrix)
    R_inv = np.linalg.inv(L_factor.T)
    return R_inv

def compute_polynomial_moments_using_tensor_product_quadrature(
        basis_matrix_function, num_samples, num_vars,
        univariate_quadrature_rule, density_function=None):
    """
    Compute the moments of a polynomial basis using tensor product quadrature

    Parameters
    ----------
    num_samples : integer
       The number of samples in the 1D quadrature rule

    univariate_quadrature_rule : tuple (x,w)
       x : np.ndarray (num_samples) the quadrature points in the user space
       w : np.ndarray (num_samples) the quadrature weights

    density_function : callable
       v = density_function(x)
       A probability density function. If not None then quadrature rule
       should be for lebesque measure and weights will be multiplied by the 
       value of the density at the quarature points

    Returns
    -------
    poly_moments : np.ndarray (num_terms, num_terms)
       The symmetric matrix containing the inner product of each polynomial
       basis with every polynomial basis (including itself - diagonal entries)
    """
    samples,weights = get_tensor_product_quadrature_rule(
        num_samples,num_vars,univariate_quadrature_rule,None,density_function)
    basis_matrix = basis_matrix_function(samples)
    
    poly_moments = np.empty((basis_matrix.shape[1],basis_matrix.shape[1]),
                            dtype=float)
    for ii in range(basis_matrix.shape[1]):
        for jj in range(ii,basis_matrix.shape[1]):
            poly_moments[ii,jj] = np.dot(
                basis_matrix[:,ii]*basis_matrix[:,jj],weights)
            poly_moments[jj,ii]=poly_moments[ii,jj]
    return poly_moments

    
