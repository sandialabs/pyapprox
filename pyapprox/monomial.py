from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from pyapprox.indexing import hash_array
from pyapprox.manipulate_polynomials import multiply_multivariate_polynomials

def monomial_mean_uniform_variables(indices,coeffs):
    """
    Integrate a multivaiate monomial with respect to the uniform probability
    measure on [-1,1].

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    coeffs : np.ndarray (num_indices)
        The coefficients of each monomial term

    Return
    ------
    integral : float
        The integral of the monomial
    """
    num_vars, num_indices = indices.shape
    assert coeffs.shape[0]==num_indices
    vals = np.prod(((-1.0)**indices+1)/(2.0*(indices+1.0)),axis=0)
    integral = np.sum(vals*coeffs)
    return integral

def monomial_variance_uniform_variables(indices,coeffs):
    mean = monomial_mean_uniform_variables(indices,coeffs)
    squared_indices,squared_coeffs =multiply_multivariate_polynomials(
            indices,coeffs,indices,coeffs)
    variance = monomial_mean_uniform_variables(
        squared_indices,squared_coeffs)-mean**2
    return variance    

def evaluate_monomial(indices,coeffs,samples):
    """
    Evaluate a multivariate monomial at a set of samples.

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    coeffs : np.ndarray (num_indices,num_qoi)
        The coefficients of each monomial term

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the monomial

    Return
    ------
    integral : float
        The values of the monomial at the samples
    """
    if coeffs.ndim==1:
        coeffs = coeffs[:,np.newaxis]
    assert coeffs.ndim==2
    assert coeffs.shape[0]==indices.shape[1]

    basis_matrix = monomial_basis_matrix(indices,samples)
    values = np.dot(basis_matrix,coeffs)
    return values

def univariate_monomial_basis_matrix(max_level,samples):
    assert samples.ndim==1
    num_samples = samples.shape[0]
    
    #basis_matrix = np.ones((num_samples,max_level+1),dtype=float)
    #for ii in range(1,max_level+1):
    #    # use horners rule to compute monomial.
    #    # This speeds up this function dramatically
    #    basis_matrix[:,ii] = basis_matrix[:,ii-1]*samples
    basis_matrix = samples[:,np.newaxis]**np.arange(max_level+1)[np.newaxis,:]
    return basis_matrix
    

def monomial_basis_matrix(indices,samples,deriv_order=0):
    """
    Evaluate a multivariate monomial basis at a set of samples.

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the monomial

    deriv_order : integer in [0,1]
       The maximum order of the derivatives to evaluate.

    Return
    ------
    basis_matrix : np.ndarray (num_samples,num_indices)
        The values of the monomial basis at the samples
    """
    # weave code is slower than python version when only computing values of
    # basis. I am Not sure of timings when computing derivatives
    # return c_monomial_basis_matrix(indices,samples)
    
    num_vars, num_indices = indices.shape
    assert samples.shape[0]==num_vars
    num_samples = samples.shape[1]

    basis_matrix = np.empty(((1+deriv_order*num_vars)*num_samples,num_indices))
    basis_vals_1d = [univariate_monomial_basis_matrix(
        indices[0,:].max(),samples[0,:])]
    basis_matrix[:num_samples,:]=basis_vals_1d[0][:,indices[0,:]]
    for dd in range(1,num_vars):
        basis_vals_1d.append(univariate_monomial_basis_matrix(
            indices[dd,:].max(),samples[dd,:]))
        basis_matrix[:num_samples,:]*=basis_vals_1d[dd][:,indices[dd,:]]

    if deriv_order>0:
        for ii in range(num_indices):
            index = indices[:,ii]
            for jj in range(num_vars):
                # derivative in jj direction
                basis_vals=basis_vals_1d[jj][:,max(0,index[jj]-1)]*index[jj]
                # basis values in other directions
                for dd in range(num_vars):
                    if dd!=jj:
                        basis_vals*=basis_vals_1d[dd][:,index[dd]]
                basis_matrix[(jj+1)*num_samples:(jj+2)*num_samples,ii]=\
                    basis_vals

    return basis_matrix

def c_monomial_basis_matrix(indices,samples):
    fast_code=r"""
int num_vars=Nsamples[0], num_samples=Nsamples[1], num_indices=Nindices[1];
double ** basis_vals_1d = new double*[num_indices];
for (int jj=0; jj<num_indices; ++jj){
    basis_vals_1d[jj] = new double[num_samples];
}

for (int kk=0; kk<num_vars; ++kk){
    for (int ii=0; ii<num_samples; ++ii)
         basis_vals_1d[0][ii]=1.0;
    for (int jj=1; jj<max_degree_1d[kk]+1; ++jj){
        for (int ii=0; ii<num_samples; ++ii)
            basis_vals_1d[jj][ii] = basis_vals_1d[jj-1][ii]*SAMPLES2(kk,ii);
    }
    for (int jj=0; jj<num_indices; ++jj){
        for (int ii=0; ii<num_samples; ++ii)
            VALS2(ii,jj)*=basis_vals_1d[INDICES2(kk,jj)][ii];
    }
}
for (int jj=0; jj<num_indices; ++jj)
   delete [] basis_vals_1d[jj];
delete [] basis_vals_1d;
"""
    slow_code = r"""
int num_vars=Nsamples[0], num_samples=Nsamples[1], num_indices=Nindices[1];
for (int ii=0; ii<num_samples; ++ii){
    for (int jj=0; jj<num_indices; ++jj){
        VALS2(ii,jj)=std::pow(SAMPLES2(0,ii),INDICES2(0,jj)); 
        for (int kk=1; kk<num_vars; ++kk){
            VALS2(ii,jj)*=std::pow(SAMPLES2(kk,ii),INDICES2(kk,jj));
        }
    }
}
"""
    import weave
    num_samples=samples.shape[1]
    num_indices=indices.shape[1]
    max_degree_1d = np.max(indices,axis=1)
    # c code needs vals initialized to 1
    vals = np.ones((num_samples,num_indices),dtype=float)
    weave.inline(fast_code,[ 'indices','max_degree_1d','samples','vals'],verbose=2)
    return vals
