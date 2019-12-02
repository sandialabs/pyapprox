import numpy as np
from scipy.special import factorial
from pyapprox.indexing import hash_array
from pyapprox.indexing import compute_hyperbolic_level_indices
def multiply_multivariate_polynomials(indices1,coeffs1,indices2,coeffs2):
    """
    TODO: instead of using dictionary to colect terms consider using

    unique_indices,repeated_idx=np.unique(
        indices[active_idx,:],axis=1,return_inverse=True)

    as is done in multivariate_polynomials.conditional_moments_of_polynomial_chaos_expansion. Choose which one is faster


    Parameters
    ----------
    index : multidimensional index
        multidimensional index specifying the polynomial degree in each
        dimension

    Returns
    -------
    """
    num_vars = indices1.shape[0]
    num_indices1 = indices1.shape[1]
    num_indices2 = indices2.shape[1]
    assert num_indices1==coeffs1.shape[0]
    assert num_indices2==coeffs2.shape[0]
    assert num_vars==indices2.shape[0]
    
    indices_dict = dict()
    max_num_indices = num_indices1*num_indices2
    indices = np.empty((num_vars,max_num_indices),int)
    coeffs = np.empty((max_num_indices),float)
    kk = 0
    for ii in range(num_indices1):
        index1 = indices1[:,ii]
        coeff1 = coeffs1[ii]
        for jj in range(num_indices2):
            index= index1+indices2[:,jj]
            key = hash_array(index)
            coeff = coeff1*coeffs2[jj]
            if key in indices_dict:
                coeffs[indices_dict[key]]+=coeff
            else:
                indices_dict[key]=kk
                indices[:,kk]=index
                coeffs[kk]=coeff
                kk+=1
    indices = indices[:,:kk]
    coeffs = coeffs[:kk]
    return indices, coeffs

def coeffs_of_power_of_nd_linear_polynomial(num_vars, degree, linear_coeffs):
    """
    Compute the polynomial (coefficients and indices) obtained by raising
    a linear multivariate polynomial (no constant term) to some power.

    Parameters
    ----------
    num_vars : integer
        The number of variables

    degree : integer
        The power of the linear polynomial

    linear_coeffs: np.ndarray (num_vars)
        The coefficients of the linear polynomial

    Returns
    -------
    coeffs: np.ndarray (num_terms)
        The coefficients of the new polynomial

    indices : np.ndarray (num_vars, num_terms)
        The set of multivariate indices that define the new polynomial
    """
    assert len(linear_coeffs)==num_vars
    coeffs, indices=multinomial_coeffs_of_power_of_nd_linear_polynomial(
        num_vars, degree)
    for ii in range(indices.shape[1]):
        index = indices[:,ii]
        for dd in range(num_vars):
            degree = index[dd]
            coeffs[ii] *= linear_coeffs[dd]**degree
    return coeffs, indices

def substitute_polynomial_for_variables_in_polynomial(
        indices_in,coeffs_in,indices,coeffs,var_idx):
    num_vars, num_terms = indices.shape
    new_indices = []
    new_coeffs = []
    for ii in range(num_terms):
        index = indices[:,ii]
        pows = index[var_idx]
        ind,cf = substitute_polynomial_for_variables_in_single_basis_term(
            indices_in,coeffs_in,index,coeffs[ii],var_idx,pows)
        new_indices.append(ind)
        new_coeffs.append(cf)
    new_indices = np.hstack(new_indices)
    new_coeffs = np.vstack(new_coeffs)
    return new_indices, new_coeffs

def substitute_polynomial_for_variables_in_single_basis_term(
        indices_in,coeffs_in,basis_index,basis_coeff,var_idx,global_var_idx,
        num_global_vars):
    """
    var_idx : np.ndarray (nsub_vars)
        The dimensions in basis_index which will be substituted

    global_var_idx : [ np.ndarray(nvars[ii]) for ii in num_inputs]
        The index of the active variables for each input
    """
    num_inputs = var_idx.shape[0]
    assert num_inputs==len(indices_in)
    assert num_inputs==len(coeffs_in)
    assert basis_coeff.shape[0]==1
    assert var_idx.max()<basis_index.shape[0]
    assert basis_index.shape[1]==1
    assert len(global_var_idx)==num_inputs

    # store input indices in global_var_idx
    temp = []
    for ii in range(num_inputs):
        ind = np.zeros((num_global_vars,indices_in[ii].shape[1]))
        ind[global_var_idx,:] = indices_in[ii]
        temp.append(ind)
    indices_in = temp
    
    jj=0
    degree = basis_index[var_idx[jj]]
    c1,ind1 = coeffs_of_power_of_polynomial(
        indices_in,coeffs_in[:,jj:jj+1],degree)
    for jj in range(1,var_idx.shape[0]):
        degree = basis_index[var_idx[jj]]
        c2,ind2 = coeffs_of_power_of_polynomial(
            indices_in,coeffs_in[:,jj:jj+1],degree)
        ind1,c1 = multiply_multivariate_polynomials(ind1,c1,ind2,c2)

    # this mask may be wrong. I might be confusing global and var idx
    mask = np.ones(basis_index.shape[0],dtype=bool); mask[var_idx]=False
    print(ind1.shape,mask.shape)
    ind1[mask,:] += basis_index[mask]
    c1*=basis_coeff
    return ind1, c1 

def composition_of_polynomials(indices_list,coeffs_list):
    npolys = len(indices_list)
    assert npolys==len(coeffs_list)
    for ii in range(1,npolys):
        new_poly = 2
    return new_poly

def coeffs_of_power_of_polynomial(indices, coeffs, degree):
    """
    Compute the polynomial (coefficients and indices) obtained by raising
    a multivariate polynomial to some power.

    TODO: Deprecate coeffs_of_power_of_nd_linear_polynomial as that function
    can be obtained as a special case of this function

    Parameters
    ----------
    indices : np.ndarray (num_vars,num_terms)
        The indices of the multivariate polynomial

    coeffs: np.ndarray (num_vars)
        The coefficients of the polynomial

    Returns
    -------
    coeffs: np.ndarray (num_terms)
        The coefficients of the new polynomial

    indices : np.ndarray (num_vars, num_terms)
        The set of multivariate indices that define the new polynomial
    """
    num_vars, num_terms = indices.shape
    assert indices.shape[1]==coeffs.shape[0]
    multinomial_coeffs, multinomial_indices = \
        multinomial_coeffs_of_power_of_nd_linear_polynomial(num_terms, degree)
    new_indices = np.zeros((num_vars,multinomial_indices.shape[1]))
    new_coeffs = np.tile(multinomial_coeffs[:,np.newaxis],coeffs.shape[1])
    for ii in range(multinomial_indices.shape[1]):
        multinomial_index = multinomial_indices[:,ii]
        for dd in range(num_terms):
            deg = multinomial_index[dd]
            new_coeffs[ii] *= coeffs[dd]**deg
            new_indices[:,ii] += indices[:,dd]*deg
    return new_coeffs, new_indices


def group_like_terms(coeffs, indices):
    if coeffs.ndim==1:
        coeffs = coeffs[:,np.newaxis]
    
    num_vars,num_indices = indices.shape
    indices_dict = {}
    for ii in range(num_indices):
        key = hash_array(indices[:,ii])
        if not key in indices_dict:
            indices_dict[key] = [coeffs[ii],ii]
        else:
            indices_dict[key] = [indices_dict[key][0]+coeffs[ii],ii]

    new_coeffs = np.empty((len(indices_dict),coeffs.shape[1]))
    new_indices = np.empty((num_vars,len(indices_dict)),dtype=int)
    ii=0
    for key, item in indices_dict.items():
        new_indices[:,ii] = indices[:,item[1]]
        new_coeffs[ii] = item[0]
        ii+=1
    return new_coeffs, new_indices

def multinomial_coefficient(index):
    """Compute the multinomial coefficient of an index [i1,i2,...,id].

    Parameters
    ----------
    index : multidimensional index
        multidimensional index specifying the polynomial degree in each
        dimension

    Returns
    -------
    coeff : double
        the multinomial coefficient
    """
    level = index.sum()
    denom = np.prod(factorial(index))
    coeff = factorial(level)/denom
    return coeff

def multinomial_coefficients(indices):
    coeffs = np.empty((indices.shape[1]),float)
    for i in range(indices.shape[1]):
        coeffs[i] = multinomial_coefficient(indices[:,i])
    return coeffs

def multinomial_coeffs_of_power_of_nd_linear_polynomial(num_vars,degree):
    """ Compute the multinomial coefficients of the individual terms
    obtained  when taking the power of a linear polynomial
    (without constant term).

    Given a linear multivariate polynomial e.g.
    e.g. (x1+x2+x3)**2 = x1**2+2*x1*x2+2*x1*x3+2*x2**2+x2*x3+x3**2
    return the coefficients of each quadratic term, i.e.
    [1,2,2,1,2,1]

    Parameters
    ----------
    num_vars : integer
        the dimension of the multivariate polynomial
    degree : integer
        the power of the linear polynomial

    Returns
    -------
    coeffs: np.ndarray (num_terms)
        the multinomial coefficients of the polynomial obtained when
        raising the linear multivariate polynomial to the power=degree

    indices: np.ndarray (num_terms)
        the indices of the polynomial obtained when
        raising the linear multivariate polynomial to the power=degree
    """
    indices = compute_hyperbolic_level_indices(num_vars,degree,1.0)
    coeffs = multinomial_coefficients(indices)
    return coeffs, indices

def add_polynomials(indices_list, coeffs_list):
    """ 
    Add many polynomials together.

    Example:
        p1 = x1**2+x2+x3, p2 = x2**2+2*x3
        p3 = p1+p2 

       return the degrees of each term in the the polynomial 
       
       p3 = x1**2+x2+3*x3+x2**2
      
       [2, 1, 1, 2]

       and the coefficients of each of these terms

       [1., 1., 3., 1.]
       

    Parameters
    ----------
    indices_list : list [np.ndarray (num_vars,num_indices_i)]
        List of polynomial indices. indices_i may be different for each 
        polynomial

    coeffs_list : list [np.ndarray (num_indices_i,num_qoi)]
        List of polynomial coefficients. indices_i may be different for each 
        polynomial. num_qoi must be the same for each list element.


    Returns
    -------
    indices: np.ndarray (num_vars,num_terms)
        the polynomial indices of the polynomial obtained from
        summing the polynomials. This will be the union of the indices
        of the input polynomials

    coeffs: np.ndarray (num_terms,num_qoi)
        the polynomial coefficients of the polynomial obtained from
        summing the polynomials
    """
    
    num_polynomials = len(indices_list)
    assert num_polynomials==len(coeffs_list)
    indices_dict = dict()

    indices = []
    coeff = []
    ii=0; kk=0
    for jj in range(indices_list[ii].shape[1]):
        assert coeffs_list[ii].ndim==2
        assert coeffs_list[ii].shape[0]==indices_list[ii].shape[1]
        index=indices_list[ii][:,jj]
        indices_dict[hash_array(index)]=kk
        indices.append(index)
        coeff.append(coeffs_list[ii][jj,:].copy())
        kk+=1
        
    for ii in range(1,num_polynomials):
        #print indices_list[ii].T,num_polynomials
        assert coeffs_list[ii].ndim==2
        assert coeffs_list[ii].shape[0]==indices_list[ii].shape[1]
        for jj in range(indices_list[ii].shape[1]):
            index=indices_list[ii][:,jj]
            key = hash_array(index)
            if key in indices_dict:
                nn = indices_dict[key]
                coeff[nn]+=coeffs_list[ii][jj,:]
            else:
                indices_dict[key]=kk
                indices.append(index)
                coeff.append(coeffs_list[ii][jj,:].copy())
                kk+=1

    indices = np.asarray(indices).T
    coeff = np.asarray(coeff)

    return indices, coeff
    
def get_indices_double_set(indices):
    """
    Given muultivariate indices 

        [i1,i2,...,]
 
    Compute its double set by
        [i1*i1,i1*i2,...,i2*i2,i2*i3...]

    The double set will only contain unique indices

    Parameters
    ----------
    indices : np.ndarray (num_vars,num_indices)
        The initial indices

    Returns
    -------
    double_set_indices : np.ndarray (num_vars,num_indices)
        The double set of indices
    """
    dummy_coeffs = np.zeros(indices.shape[1])
    double_set_indices = multiply_multivariate_polynomials(
        indices,dummy_coeffs,indices,dummy_coeffs)[0]
    return double_set_indices

def shift_momomial_expansion(coef,shift,scale):
    assert coef.ndim==1
    shifted_coef = np.zeros_like(coef)
    shifted_coef[0]=coef[0]
    nterms = coef.shape[0]
    for ii in range(1,nterms):
        temp = np.polynomial.polynomial.polypow([1,-shift],ii)
        shifted_coef[:ii+1] += coef[ii]*temp[::-1]/scale**ii
    return shifted_coef
    

#Some of these functions can be replaced by numpy functions described at
#https://docs.scipy.org/doc/numpy/reference/routines.polynomials.polynomial.html
