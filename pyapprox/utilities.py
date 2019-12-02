from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy.special import beta as beta_fn
from functools import partial

def sub2ind(sizes, multi_index):
    r"""
    Map a d-dimensional index to the scalar index of the equivalent flat
    1D array

    Example:
    \f[
    \begin{bmatrix}
    1,1 & 1,2 & 1,3\\
    2,1 & 2,2 & 2,3\\
    3,1 & 3,2 & 3,3
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
    1 & 4 & 7\\
    2 & 5 & 8\\
    3 & 6 & 9
    \end{bmatrix}
    \f]

    Parameters
    ----------
    sizes : integer 
        The number of elems in each dimension. For a 2D index
        sizes = [numRows, numCols]

    multi_index : np.ndarray (len(sizes))
       The d-dimensional index

    Returns
    ------
    scalar_index : integer 
        The scalar index
    """
    num_sets = len(sizes)
    scalar_index = 0; shift = 1
    for ii in range(num_sets):
        scalar_index += shift * multi_index[ii]
        shift *= sizes[ii]
    return scalar_index

def ind2sub(sizes,scalar_index,num_elems):
    r"""
    Map a scalar index of a flat 1D array to the equivalent d-dimensional index

    Example:
    \f[
    \begin{bmatrix}
    1 & 4 & 7\\
    2 & 5 & 8\\
    3 & 6 & 9
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
    1,1 & 1,2 & 1,3\\
    2,1 & 2,2 & 2,3\\
    3,1 & 3,2 & 3,3
    \end{bmatrix}
    \f]
    
    Parameters
    ----------
    sizes : integer 
        The number of elems in each dimension. For a 2D index
        sizes = [numRows, numCols]

    scalar_index : integer 
        The scalar index
    
    num_elems : integer
        The total number of elements in the d-dimensional matrix

    Returns
    ------
    multi_index : np.ndarray (len(sizes))
       The d-dimensional index
    """
    denom = num_elems
    num_sets = len(sizes)
    multi_index = np.empty((num_sets),dtype=int)
    for ii in range(num_sets-1,-1,-1):
        denom /= sizes[ii]
        multi_index[ii] = scalar_index / denom;
        scalar_index = scalar_index % denom;
    return multi_index

def cartesian_product(input_sets, elem_size=1):
    r"""
    Compute the cartesian product of an arbitray number of sets.

    The sets can consist of numbers or themselves be lists or vectors. All 
    the lists or vectors of a given set must have the same number of entries
    (elem_size). However each set can have a different number of sclars, lists,
    or vectors.

    Parameters
    ----------
    input_sets 
        The sets to be used in the cartesian product.

    elem_size : integer 
        The size of the vectors within each set.

    Returns
    ------
    result : np.ndarray (num_sets*elem_size, num_elems)
        The cartesian product. num_elems = np.prod(sizes)/elem_size,
        where sizes[ii] = len(input_sets[ii]), ii=0,..,num_sets-1.
        result.dtype will be set to the first entry of the first input_set
    """
    try:
        from pyapprox.cython.utilities import cartesian_product_pyx
        # fused type does not work for np.in32, np.float32, np.int64
        # so envoke cython cast
        for ii in range(len(input_sets)-1):
            if type(input_sets[ii][0])!=type(input_sets[ii+1][0]):
                raise Exception('not all 1d arrays are the same type')
        if np.issubdtype(input_sets[0][0],np.signedinteger):
            return cartesian_product_pyx(input_sets,1,elem_size)
        if np.issubdtype(input_sets[0][0],np.floating):
            return cartesian_product_pyx(input_sets,1.,elem_size)
        else:
            return cartesian_product_pyx(
                input_sets,input_sets[0][0],elem_size)
    except:
        print ('cartesian_product extension failed')

    num_elems = 1;
    num_sets = len(input_sets)
    sizes = np.empty((num_sets),dtype=int)
    for ii in range(num_sets):
        sizes[ii] = input_sets[ii].shape[0]/elem_size
        num_elems *= sizes[ii]
    #try:
    #    from pyapprox.weave import c_cartesian_product
    #    # note c_cartesian_produc takes_num_elems as last arg and cython
    #    # takes elem_size
    #    return c_cartesian_product(input_sets, elem_size, sizes, num_elems)
    #except:
    #    print ('cartesian_product extension failed')

    result = np.empty(
        (num_sets*elem_size, num_elems), dtype=type(input_sets[0][0]))
    for ii in range(num_elems):
        multi_index = ind2sub( sizes, ii, num_elems)
        for jj in range(num_sets):
            for kk in range(elem_size):
                result[jj*elem_size+kk,ii]=\
                  input_sets[jj][multi_index[jj]*elem_size+kk];
    return result


def outer_product(input_sets):
    r"""
    Construct the outer product of an arbitary number of sets.
 
    Example:
    \f[ \{1,2\}\times\{3,4\}=\{1\times3, 2\times3, 1\times4, 2\times4\} =
    \{3, 6, 4, 8\} \f]

    Parameters
    ----------
    input_sets  
        The sets to be used in the outer product

    Returns
    ------
    result : np.ndarray(np.prod(sizes))
       The outer product of the sets.
       result.dtype will be set to the first entry of the first input_set
    """
    try:
        from pyapprox.cython.utilities import outer_product_pyx
        # fused type does not work for np.in32, np.float32, np.int64
        # so envoke cython cast
        if np.issubdtype(input_sets[0][0],np.signedinteger):
            return outer_product_pyx(input_sets,1)
        if np.issubdtype(input_sets[0][0],np.floating):
            return outer_product_pyx(input_sets,1.)
        else:
            return outer_product_pyx(input_sets,input_sets[0][0])        
    except:
        print ('outer_product extension failed')

    num_elems = 1
    num_sets = len(input_sets)
    sizes = np.empty((num_sets),dtype=int)
    for ii in range(num_sets):
        sizes[ii] = len(input_sets[ii])
        num_elems *= sizes[ii];

    # try:
    #     from pyapprox.weave import c_outer_product
    #     return c_outer_product(input_sets)
    # except:
    #     print ('outer_product extension failed')



    result = np.empty((num_elems), dtype=type(input_sets[0][0]))
    for ii in range(num_elems):
        result[ii] = 1.0
        multi_index = ind2sub(sizes, ii, num_elems);
        for jj in range(num_sets):
            result[ii] *= input_sets[jj][multi_index[jj]];
          
    return result

def hash_array(array):
    """
    Hash an array for dictionary or set based lookup

    Parameters
    ----------
    array : np.ndarray
       The integer array to hash

    Returns
    ------
    key : integer
       The hash value of the array
    """
    #assert array.ndim==1
    #array = np.ascontiguousarray(array)
    #array.flags.writeable = False
    #return hash(array.data)
    return hash(array.tostring())

def unique_matrix_rows(matrix):
    unique_rows = []
    unique_rows_set = set()
    for ii in range(matrix.shape[0]):
        key = hash_array(matrix[ii,:])
        if key not in unique_rows_set:
            unique_rows_set.add(key)
            unique_rows.append(matrix[ii,:])
    return np.asarray(unique_rows)

def remove_common_rows(matrices):
    num_cols = matrices[0].shape[1]
    unique_rows_dict = dict()
    for ii in range(len(matrices)):
        matrix = matrices[ii]
        assert matrix.shape[1]==num_cols
        for jj in range(matrix.shape[0]):
            key = hash_array(matrix[jj,:])
            if key not in unique_rows_dict:
                unique_rows_dict[key] = (ii,jj)
            elif unique_rows_dict[key][0]!=ii:
                del unique_rows_dict[key]
            #else:
            # entry is a duplicate entry in the current. Allow this to
            # occur but only add one of the duplicates to the unique rows dict

    unique_rows = []
    for key in list(unique_rows_dict.keys()):
        ii,jj = unique_rows_dict[key]
        unique_rows.append(matrices[ii][jj,:])

    return np.asarray(unique_rows)

def allclose_unsorted_matrix_rows(matrix1,matrix2):
    if matrix1.shape!=matrix2.shape:
        return False

    matrix1_dict = dict()
    for ii in range(matrix1.shape[0]):
        key = hash_array(matrix1[ii,:])
        # allow duplicates of rows
        if key not in matrix1_dict:
            matrix1_dict[key] = 0
        else:
            matrix1_dict[key] += 1

    matrix2_dict = dict()
    for ii in range(matrix2.shape[0]):
        key = hash_array(matrix2[ii,:])
        # allow duplicates of rows
        if key not in matrix2_dict:
            matrix2_dict[key] = 0
        else:
            matrix2_dict[key] += 1

    if len(list(matrix1_dict.keys()))!=len(list(matrix2_dict.keys())):
        return False

    for key in list(matrix1_dict.keys()):
        if key not in matrix2_dict:
            return False
        if matrix2_dict[key]!=matrix1_dict[key]:
            return False
        
    return True

def get_2d_cartesian_grid(num_pts_1d, ranges):
    """
    Get a 2d tensor grid with equidistant points.

    Parameters
    ----------
    num_pts_1d : integer
        The number of points in each dimension

    ranges : np.ndarray (4)
        The lower and upper bound of each dimension [lb_1,ub_1,lb_2,ub_2]

    Returns
    -------
    grid : np.ndarray (2,num_pts_1d**2)
        The points in the tensor product grid.
        [x1,x2,...x1,x2...]
        [y1,y1,...y2,y2...]
    """
    #from math_tools_cpp import cartesian_product_double as cartesian_product
    from PyDakota.math_tools import cartesian_product
    x1 = np.linspace( ranges[0], ranges[1], num_pts_1d )
    x2 = np.linspace( ranges[2], ranges[3], num_pts_1d )
    abscissa_1d = []
    abscissa_1d.append( x1 )
    abscissa_1d.append( x2 )
    grid = cartesian_product( abscissa_1d, 1 )
    return grid

def invert_permutation_vector( p , dtype=int):
    """
    Returns the "inverse" of a permutation vector. I.e., returns the
    permutation vector that performs the inverse of the original
    permutation operation.

    Parameters
    ----------
    p: np.ndarray
        Permutation vector
    dtype: type
        Data type passed to np.ndarray constructor

    Returns
    -------
    pt: np.ndarray
        Permutation vector that accomplishes the inverse of the
        permutation p.
    """

    N = np.max(p) + 1
    pt = np.zeros(p.size,dtype=dtype)
    pt[p] = np.arange(N,dtype=dtype)
    return pt


def nchoosek(nn,kk):
    try:  # SciPy >= 0.19
        from scipy.special import comb
    except:
        from scipy.misc import comb
    result = np.asarray(np.round(comb(nn, kk)),dtype=int)
    if np.isscalar(result):
        result=np.asscalar(result)
    return result

def total_degree_space_dimension(dimension, degree):
    """
    Return the number of basis functions in a total degree polynomial space,
    i.e. the space of all polynomials with degree at most degree.

    Parameters
    ----------
    num_vars : integer
        The number of variables of the polynomials

    degree : 
        The degree of the total-degree space
    
    Returns
    -------
    num_terms : integer
        The number of basis functions in the total degree space
    """
    #from scipy.special import gammaln
    #subspace_dimension = lambda k: int(np.round(np.exp( gammaln(k+d+1) - gammaln(k+1) - gammaln(d+1) )))
    return nchoosek(dimension+degree,degree)

def total_degree_encompassing_N(dimension, N):
    """
    Returns the smallest integer k such that the dimension of the total
    degree-k space is greater than N.
    """

    k = 0
    while total_degree_subspace_dimension(dimension, k) < N:
        k += 1
    return k

def total_degree_barrier_indices(dimension, max_degree):
    """
    Returns linear indices that bound total degree spaces

    Parameters
    ----------
    dimension: int
        Parametric dimension
    max_degree: int
        Maximum polynomial degree

    Returns
    -------
    degree_barrier_indices: list
        List of degree barrier indices up to (including) max_degree.
    """
    degree_barrier_indices = [0]

    for degree in range(1,max_degree+1):
        degree_barrier_indices.append( total_degree_subspace_dimension(dimension, degree) )

    return degree_barrier_indices

def total_degree_orthogonal_transformation( coefficients, d ):
    """
    Returns an orthogonal matrix transformation that "matches" the input
    coefficients.

    Parameters
    ----------
    coefficients: np.ndarray
        Length-N vector of expansion coefficients
    d: int
        Parametric dimension

    Returns
    -------
    Q: np.ndarray
        A size N x N orthogonal matrix transformation. The first column
        is a unit vector in the direction of coefficients.
    """

    from scipy.linalg import qr

    N = coefficients.size

    degree_barrier_indices = [1]
    max_degree = 0
    while degree_barrier_indices[-1] < N-1:
        max_degree += 1
        degree_barrier_indices.append( total_degree_subspace_dimension(d, max_degree) )

    q = np.zeros([N, N])

    # Assume degree = 0 is just constant
    q[0,0] = 1.

    for degree in range(1,max_degree+1):
        i1 = degree_barrier_indices[degree-1]
        i2 = degree_barrier_indices[degree]

        M = i2-i1
        q[i1:i2,i1:i2] = qr( coefficients[i1:i2].reshape([M, 1]) )[0]

    return q

def get_low_rank_matrix(num_rows,num_cols,rank):
    """
    Construct a matrix of size num_rows x num_cols with a given rank.

    Parameters
    ----------
    num_rows : integer
        The number rows in the matrix

    num_cols : integer
        The number columns in the matrix

    rank : integer
        The rank of the matrix

    Returns
    -------
    Amatrix : np.ndarray (num_rows,num_cols)
        The low-rank matrix generated
    """
    assert rank <= min(num_rows,num_cols)
    # Generate a matrix with normally distributed entries
    N = max(num_rows,num_cols)
    Amatrix = np.random.normal(0,1,(N,N))
    # Make A symmetric positive definite
    Amatrix = np.dot( Amatrix.T, Amatrix )
    # Construct low rank approximation of A
    eigvals, eigvecs = np.linalg.eigh( Amatrix.copy() )
    # Set smallest eigenvalues to zero. Note eigenvals are in
    # ascending order
    eigvals[:(eigvals.shape[0]-rank)] = 0.
    # Construct rank r A matrix
    Amatrix = np.dot(eigvecs,np.dot(np.diag(eigvals),eigvecs.T))
    # Resize matrix to have requested size
    Amatrix = Amatrix[:num_rows,:num_cols]
    return Amatrix


def adjust_sign_svd(U, V, adjust_based_upon_U=True):
    """
    Ensure uniquness of svd by ensuring the first entry of each left singular
    singular vector be positive. Only works for np.linalg.svd
    if full_matrices=False

    Parameters
    ----------
    U : (M x M) matrix
        left singular vectors of a singular value decomposition of a (M x N)
        matrix A.

    V : (N x N) matrix
        right singular vectors of a singular value decomposition of a (M x N)
        matrix A.

    adjust_based_upon_U : boolean (default=True)
        True - make the first entry of each column of U positive
        False - make the first entry of each row of V positive

    Returns
    -------
    U : (M x M) matrix
       left singular vectors with first entry of the first
       singular vector always being positive.

    V : (M x M) matrix
        right singular vectors consistent with sign adjustment applied to U.
    """
    if U.shape[1] != V.shape[0]:
        raise Exception('U.shape[1] must equal V.shape[0]. If using np.linalg.svd set full_matrices=False')

    if adjust_based_upon_U:
        s = np.sign(U[0,:])
    else:
        s = np.sign(V[:,0])
    U *= s
    V *= s[:,np.newaxis]
    return U,V

def adjust_sign_eig(U):
    """
    Ensure uniquness of eigenvalue decompotision by ensuring the first entry
    of the first singular vector of U is positive.

    Parameters
    ----------
    U : (M x M) matrix
        left singular vectors of a singular value decomposition of a (M x M)
        matrix A.

    Returns
    -------
    U : (M x M) matrix
       left singular vectors with first entry of the first
       singular vector always being positive.
    """
    s = np.sign(U[0,:])
    U *= s
    return U

def sorted_eigh(C):
    """
    Compute the eigenvalue decomposition of a matrix C and sort
    the eigenvalues and corresponding eigenvectors by decreasing
    magnitude.

    Warning. This will prioritize large eigenvalues even if they
    are negative. Do not use if need to distinguish between positive
    and negative eigenvalues

    Input

    B: matrix (NxN)
      matrix to decompose

    Output

    e: vector (N)
      absolute values of the eigenvalues of C sorted by decreasing
      magnitude

    W: eigenvectors sorted so that they respect sorting of e
    """
    e, W = np.linalg.eigh(C)
    e = abs(e)
    ind = np.argsort(e)
    e = e[ind[::-1]]
    W = W[:,ind[::-1]]
    s = np.sign(W[0,:])
    s[s==0] = 1
    W = W*s
    return e.reshape((e.size,1)), W

def continue_pivoted_lu_factorization(LU_factor,raw_pivots,current_iter,
                                      max_iters,num_initial_rows=0):
    it = current_iter
    for it in range(current_iter,max_iters):
                    
        # find best pivot
        if np.isscalar(num_initial_rows) and (it<num_initial_rows):
            pivot = np.argmax(np.absolute(LU_factor[it:num_initial_rows,it]))+it
        elif (not np.isscalar(num_initial_rows) and
              (it<num_initial_rows.shape[0])):
            pivot=num_initial_rows[it]
        else:
            pivot = np.argmax(np.absolute(LU_factor[it:,it]))+it


        # update pivots vector
        #swap_rows(pivots,it,pivot)
        raw_pivots[it]=pivot
      
        # apply pivots(swap rows) in L factorization
        swap_rows(LU_factor,it,pivot)

        # check for singularity
        if abs(LU_factor[it,it])<np.finfo(float).eps:
            msg = "pivot %1.2e"%abs(LU_factor[it,it])
            msg += " is to small. Stopping factorization."
            print (msg)
            break

        # update L_factor
        LU_factor[it+1:,it] /= LU_factor[it,it];

        # udpate U_factor
        col_vector = LU_factor[it+1:,it]
        row_vector = LU_factor[it,it+1:]

        update = np.outer(col_vector,row_vector)
        LU_factor[it+1:,it+1:]-= update
    return LU_factor, raw_pivots, it

def unprecondition_LU_factor(LU_factor,precond_weights,num_pivots=None):
    """
    A=LU and WA=XY
    Then WLU=XY
    We also know Y=WU
    So WLU=XWU => WL=XW so L=inv(W)*X*W
    and U = inv(W)Y
    """
    if num_pivots is None:
        num_pivots = np.min(LU_factor.shape)
    assert precond_weights.shape[1]==1
    assert precond_weights.shape[0]==LU_factor.shape[0]
    # left multiply L an U by inv(W), i.e. compute inv(W).dot(L)
    # and inv(W).dot(U)
    LU_factor = LU_factor.copy()/precond_weights
    # right multiply L by W, i.e. compute L.dot(W)
    # Do not overwrite columns past num_pivots. If not all pivots have been
    # performed the columns to the right of this point contain U factor
    for ii in range(num_pivots):
        LU_factor[ii+1:,ii]*=precond_weights[ii,0]
    return LU_factor
    

def split_lu_factorization_matrix(LU_factor,num_pivots=None):
    """
    Return the L and U factors of an inplace LU factorization

    Parameters
    ----------
    num_pivots : integer
        The number of pivots performed. This allows LU in place matrix
        to be split during evolution of LU algorithm
    """
    if num_pivots is None:
        num_pivots = np.min(LU_factor.shape)
    L_factor = np.tril(LU_factor)
    if L_factor.shape[1]<L_factor.shape[0]:
        # if matrix over-determined ensure L is a square matrix
        n0 = L_factor.shape[0]-L_factor.shape[1]
        L_factor=np.hstack([L_factor,np.zeros((L_factor.shape[0],n0))])
    if num_pivots<np.min(L_factor.shape):
        n1 = L_factor.shape[0]-num_pivots
        n2 = L_factor.shape[1]-num_pivots
        L_factor[num_pivots:,num_pivots:] = np.eye(n1,n2)
    np.fill_diagonal(L_factor,1.)
    U_factor = np.triu(LU_factor)
    U_factor[num_pivots:,num_pivots:] = LU_factor[num_pivots:,num_pivots:]
    return L_factor, U_factor

def truncated_pivoted_lu_factorization(A,max_iters,num_initial_rows=0,
                                       truncate_L_factor=True):
    """
    Compute a incomplete pivoted LU decompostion of a matrix.

    Parameters
    ----------
    A np.ndarray (num_rows,num_cols)
        The matrix to be factored

    max_iters : integer
        The maximum number of pivots to perform. Internally max)iters will be 
        set such that max_iters = min(max_iters,K), K=min(num_rows,num_cols)

    num_initial_rows: integer or np.ndarray()
        The number of the top rows of A to be chosen as pivots before
        any remaining rows can be chosen.
        If object is an array then entries are raw pivots which
        will be used in order.
    

    Returns
    -------
    L_factor : np.ndarray (max_iters,K)
        The lower triangular factor with a unit diagonal. 
        K=min(num_rows,num_cols)

    U_factor : np.ndarray (K,num_cols)
        The upper triangular factor

    raw_pivots : np.ndarray (num_rows)
        The sequential pivots used to during algorithm to swap rows of A. 
        pivots can be obtained from raw_pivots using 
        get_final_pivots_from_sequential_pivots(raw_pivots)

    pivots : np.ndarray (max_iters)
        The index of the chosen rows in the original matrix A chosen as pivots
    """
    num_rows,num_cols = A.shape
    min_num_rows_cols = min(num_rows, num_cols)
    max_iters = min(max_iters, min_num_rows_cols)
    if ( A.shape[1] < max_iters ):
        msg = "truncated_pivoted_lu_factorization: "
        msg += " A is inconsistent with max_iters. Try deceasing max_iters or "
        msg += " increasing the number of columns of A"
        raise Exception(msg)

    # Use L to store both L and U during factoriation then copy out U in post
    # processing
    LU_factor = A.copy()
    raw_pivots = np.arange(num_rows)#np.empty(num_rows,dtype=int)
    LU_factor,raw_pivots,it = continue_pivoted_lu_factorization(
        LU_factor,raw_pivots,0,max_iters,num_initial_rows)
        
    if not truncate_L_factor:
        return LU_factor, raw_pivots
    else:
        pivots = get_final_pivots_from_sequential_pivots(
            raw_pivots)[:it+1]
        L_factor, U_factor = split_lu_factorization_matrix(LU_factor,it+1)
        L_factor = L_factor[:it+1,:it+1]
        U_factor = U_factor[:it+1,:it+1]
        return L_factor, U_factor, pivots
    
def add_columns_to_pivoted_lu_factorization(LU_factor,new_cols,raw_pivots):
    """
    Given factorization PA=LU add new columns to A in unpermuted order and update
    LU factorization
    
    raw_pivots : np.ndarray (num_pivots)
    The pivots applied at each iteration of pivoted LU factorization.
    If desired one can use get_final_pivots_from_sequential_pivots to 
    compute final position of rows after all pivots have been applied.
    """
    assert LU_factor.shape[0]==new_cols.shape[0]
    assert raw_pivots.shape[0]<=new_cols.shape[0]
    num_new_cols = new_cols.shape[1]
    num_pivots = raw_pivots.shape[0]
    for it in range(num_pivots):
        pivot = raw_pivots[it]
        swap_rows(new_cols,it,pivot)

        # update U_factor
        # recover state of col vector from permuted LU factor
        # Let  (jj,kk) represent iteration and pivot pairs
        # then if lu factorization produced sequence of pairs
        # (0,4),(1,2),(2,4) then LU_factor[:,0] here will be col_vector
        # in LU algorithm with the second and third permutations
        # so undo these permutations in reverse order
        col_vector = LU_factor[it+1:,it].copy()
        for ii in range(num_pivots-it-1):
            # (it+1) necessary in two lines below because only dealing
            # with compressed col vector which starts at row it in LU_factor
            jj=raw_pivots[num_pivots-1-ii]-(it+1)
            kk=num_pivots-ii-1-(it+1)
            swap_rows(col_vector,jj,kk)
        row_vector = new_cols[it,:]

        update = np.outer(col_vector,row_vector)
        new_cols[it+1:,:] -= update

        #new_cols = add_rows_to_pivoted_lu_factorization(
        #    new_cols[:it+1,:],new_cols[it+1:,:],num_pivots)

    LU_factor = np.hstack((LU_factor,new_cols))
    return LU_factor

def add_rows_to_pivoted_lu_factorization(LU_factor,new_rows,num_pivots):
    assert LU_factor.shape[1]==new_rows.shape[1]
    num_new_rows = new_rows.shape[0]
    LU_factor_extra = new_rows.copy()
    for it in range(num_pivots):
        LU_factor_extra[:,it]/=LU_factor[it,it]
        col_vector = LU_factor_extra[:,it]
        row_vector = LU_factor[it,it+1:]
        update = np.outer(col_vector,row_vector)
        LU_factor_extra[:,it+1:] -= update
        
    return np.vstack([LU_factor,LU_factor_extra])
        
def swap_rows(matrix,ii,jj):
    temp = matrix[ii].copy()
    matrix[ii]=matrix[jj]
    matrix[jj]=temp

def pivot_rows(pivots,matrix,in_place=True):
    if not in_place:
        matrix = matrix.copy()
    num_pivots = pivots.shape[0]
    assert num_pivots <= matrix.shape[0]
    for ii in range(num_pivots):
        swap_rows(matrix,ii,pivots[ii])
    return matrix

def get_final_pivots_from_sequential_pivots(sequential_pivots,num_pivots=None):
    if num_pivots is None:
        num_pivots = sequential_pivots.shape[0]
    assert num_pivots >= sequential_pivots.shape[0]
    pivots = np.arange(num_pivots)
    return pivot_rows(sequential_pivots,pivots,False)

def get_tensor_product_quadrature_rule(
        degrees,num_vars,univariate_quadrature_rules,transform_samples=None,
        density_function=None):
    """
    if get error about outer product failing it may be because 
    univariate_quadrature rule is returning a weights array for every level, 
    i.e. l=0,...level
    """
    degrees = np.atleast_1d(degrees)
    if degrees.shape[0]==1 and num_vars>1:
        degrees = np.array([degrees[0]]*num_vars,dtype=int)
    
    if callable(univariate_quadrature_rules):
        univariate_quadrature_rules = [univariate_quadrature_rules]*num_vars
        
    x_1d = []; w_1d = []
    for ii in range(len(univariate_quadrature_rules)):
        x,w = univariate_quadrature_rules[ii](degrees[ii])
        x_1d.append(x); w_1d.append(w)
    samples = cartesian_product(x_1d,1)
    weights = outer_product(w_1d)            
            
    if density_function is not None:
        weights *= density_function(samples)
    if transform_samples is not None:
        samples = transform_samples(samples)
    return samples, weights

def piecewise_quadratic_interpolation(samples,mesh,mesh_vals,ranges):
    assert mesh.shape[0]==mesh_vals.shape[0]
    vals = np.zeros_like(samples)
    samples = (samples-ranges[0])/(ranges[1]-ranges[0])
    for ii in range(0,mesh.shape[0]-2,2):
        xl=mesh[ii]; xr=mesh[ii+2]
        x=(samples-xl)/(xr-xl)            
        interval_vals = canonical_piecewise_quadratic_interpolation(
            x,mesh_vals[ii:ii+3])
        # to avoid double counting we set left boundary of each interval to zero
        # except for first interval
        if ii==0:
            interval_vals[(x<0)|(x>1)]=0.
        else:
            interval_vals[(x<=0)|(x>1)]=0.
        vals += interval_vals
    return vals

    # I = np.argsort(samples)
    # sorted_samples = samples[I]
    # idx2=0
    # for ii in range(0,mesh.shape[0]-2,2):
    #     xl=mesh[ii]; xr=mesh[ii+2]
    #     for jj in range(idx2,sorted_samples.shape[0]):
    #         if ii==0:
    #             if sorted_samples[jj]>=xl:
    #                 idx1=jj
    #                 break
    #         else:
    #             if sorted_samples[jj]>xl:
    #                 idx1=jj
    #                 break
    #     for jj in range(idx1,sorted_samples.shape[0]):
    #         if sorted_samples[jj]>xr:
    #             idx2=jj-1
    #             break
    #     if jj==sorted_samples.shape[0]-1:
    #         idx2=jj
    #     x=(sorted_samples[idx1:idx2+1]-xl)/(xr-xl)
    #     interval_vals = canonical_piecewise_quadratic_interpolation(
    #         x,mesh_vals[ii:ii+3])
    #     vals[idx1:idx2+1] += interval_vals
    # return vals[np.argsort(I)]

def canonical_piecewise_quadratic_interpolation(x,nodal_vals):
    """
    Piecewise quadratic interpolation of nodes at [0,0.5,1]
    Assumes all values are in [0,1]. 
    """
    assert x.ndim==1
    assert nodal_vals.shape[0]==3
    vals = nodal_vals[0]*(1.0-3.0*x+2.0*x**2)+nodal_vals[1]*(4.0*x-4.0*x**2)+\
      nodal_vals[2]*(-x+2.0*x**2)
    return vals

def discrete_sampling(N,probs,states=None):
    """
    discrete_sampling -- samples iid from a discrete probability measure

    x = discrete_sampling(N, prob, states)

    Generates N iid samples from a random variable X whose probability mass
    function is 

    prob(X = states[j]) = prob[j],    1 <= j <= length(prob).

    If states is not given, the states are gives by 1 <= state <= length(prob)
    """

    p = probs.squeeze()/np.sum(probs)

    bins = np.digitize(
        np.random.uniform(0.,1.,(N,1)), np.hstack((0,np.cumsum(p))))-1

    if states is None:
        x = bins
    else:
        assert(states.shape[0] == probs.shape[0])
        x = states[bins]
        
    return x.squeeze()

def lists_of_arrays_equal(list1,list2):
    if len(list1)!=len(list2):
        return False
    equal = True
    for ll in range(len(list1)):
        if not np.allclose(list1[ll],list2[ll]):
            return False
    return True


def lists_of_lists_of_arrays_equal(list1,list2):
    if len(list1)!=len(list2):
        return False
    equal = True
    for ll in range(len(list1)):
        for kk in range(len(list1[ll])):
            if not np.allclose(list1[ll][kk],list2[ll][kk]):
                return False
    return True

def beta_pdf(alpha_stat,beta_stat,x):
    #scipy implementation is slow
    const = 1./beta_fn(alpha_stat,beta_stat)
    return const*(x**(alpha_stat-1)*(1-x)**(beta_stat-1))

def pdf_under_affine_map(pdf,loc,scale,y):
    return pdf((y-loc)/scale)/scale

def beta_pdf_on_ab(alpha_stat,beta_stat,a,b,x):
    #const = 1./beta_fn(alpha_stat,beta_stat)
    #const /= (b-a)**(alpha_stat+beta_stat-1)
    #return const*((x-a)**(alpha_stat-1)*(b-x)**(beta_stat-1))
    from functools import partial
    pdf = partial(beta_pdf,alpha_stat,beta_stat)
    return pdf_under_affine_map(pdf,a,(b-a),x)

def beta_pdf_derivative(alpha_stat,beta_stat,x):
    """
    x in [0,1]
    """
    #beta_const = gamma_fn(alpha_stat+beta_stat)/(
    # gamma_fn(alpha_stat)*gamma_fn(beta_stat))

    beta_const = 1./beta_fn(alpha_stat,beta_stat)
    deriv=0
    if alpha_stat>1:
        deriv += (alpha_stat-1)*(x**(alpha_stat-2)*(1-x)**(beta_stat-1))
    if beta_stat>1:
        deriv -= (beta_stat -1)*(x**(alpha_stat-1)*(1-x)**(beta_stat-2))
    deriv *= beta_const
    return deriv

from scipy.special import erf
def gaussian_cdf(mean,var,x):
  return 0.5*(1+erf((x-mean)/(np.sqrt(var*2))))

def gaussian_pdf(mean,var,x,package=np):
    """
    set package=sympy if want to use for symbolic calculations
    """
    return package.exp(-(x-mean)**2/(2*var)) / (2*package.pi*var)**.5

def gaussian_pdf_derivative(mean,var,x):
    return -gaussian_pdf(mean,var,x)*(x-mean)/var

def pdf_derivative_under_affine_map(pdf_deriv,loc,scale,y):
    """
    Let y=g(x)=x*scale+loc and x = g^{-1}(y) = v(y) = (y-loc)/scale, scale>0
    p_Y(y)=p_X(v(y))*|dv/dy(y)|=p_X((y-loc)/scale))/scale
    dp_Y(y)/dy = dv/dy(y)*dp_X/dx(v(y))/scale = dp_X/dx(v(y))/scale**2
    """
    return pdf_deriv((y-loc)/scale)/scale**2

def gradient_of_tensor_product_function(univariate_functions,
                                        univariate_derivatives,samples):
    num_samples = samples.shape[1]
    num_vars = len(univariate_functions)
    assert len(univariate_derivatives)==num_vars
    gradient = np.empty((num_vars,num_samples))
    # precompute data which is reused multiple times
    function_values = []
    for ii in range(num_vars):
        function_values.append(univariate_functions[ii](samples[ii,:]))
        
    for ii in range(num_vars):
        gradient[ii,:] = univariate_derivatives[ii](samples[ii,:])
        for jj in range(ii):
            gradient[ii,:] *= function_values[jj]
        for jj in range(ii+1,num_vars):
            gradient[ii,:] *= function_values[jj]
    return gradient

def evaluate_tensor_product_function(univariate_functions,samples):
    num_samples = samples.shape[1]
    num_vars = len(univariate_functions)
    values = np.ones((num_samples))
    for ii in range(num_vars):
        values *= univariate_functions[ii](samples[ii,:])
    return values

def cholesky_decomposition(Amat):
    
    nrows = Amat.shape[0]
    assert Amat.shape[1]==nrows

    L = np.zeros((nrows,nrows))
    for ii in range(nrows):
        temp = Amat[ii,ii]-np.sum(L[ii,:ii]**2)
        if temp <= 0:
            raise Exception ('matrix is not positive definite')
        L[ii,ii]=np.sqrt(temp)
        L[ii+1:,ii]=\
           (Amat[ii+1:,ii]-np.sum(L[ii+1:,:ii]*L[ii,:ii],axis=1))/L[ii,ii]
        
    return L

def pivoted_cholesky_decomposition(A,npivots,init_pivots=None,tol=0.,
                                   error_on_small_tol=False):
    """
    Return a low-rank pivoted Cholesky decomposition of matrix A.

    If A is positive definite and npivots is equal to the number of rows of A
    then L.dot(L.T)==A

    To obtain the pivoted form of L set
    L = L[pivots,:]

    Then P.T.dot(A).P == L.dot(L.T)

    where P is the standrad pivot matrix which can be obtained from the pivot 
    vector using the function 
    """
    Amat = A.copy()
    nrows = Amat.shape[0]
    assert Amat.shape[1]==nrows
    assert npivots<=nrows

    L = np.zeros(((nrows,npivots)))
    diag = np.diag(Amat).copy() # diag returns a view
    pivots = np.arange(nrows)
    init_error = np.absolute(diag).sum()
    for ii in range(npivots):
        if init_pivots is None or ii>=len(init_pivots):
            pivot = np.argmax(diag[pivots[ii:]])+ii
        else:
            pivot = pivots[init_pivots[ii]]
        #print(pivot)
            
        swap_rows(pivots,ii,pivot)
        if diag[pivots[ii]] <= 0:
            raise Exception ('matrix is not positive definite')
        L[pivots[ii],ii] = np.sqrt(diag[pivots[ii]])

        L[pivots[ii+1:],ii]=(Amat[pivots[ii+1:],pivots[ii]]-
            L[pivots[ii+1:],:ii].dot(L[pivots[ii],:ii]))/L[pivots[ii],ii]
        diag[pivots[ii+1:]] -= L[pivots[ii+1:],ii]**2

        # for jj in range(ii+1,nrows):
        #     L[pivots[jj],ii]=(Amat[pivots[ii],pivots[jj]]-
        #         L[pivots[ii],:ii].dot(L[pivots[jj],:ii]))/L[pivots[ii],ii]
        #     diag[pivots[jj]] -= L[pivots[jj],ii]**2
        error = diag[pivots[ii+1:]].sum()/init_error
        #print(ii,'error',error)
        if error<tol:
            msg = 'Tolerance reached at iteration %d. Tol=%1.2e'%(ii,error)
            # If matrix is rank r then then error will be machine precision
            # In such a case exiting without an error is the right thing to do
            if error_on_small_tol:
                raise Exception(msg)
            else:
                print(msg)
            break
        
    pivots = pivots[:ii+1]
    
    return L, pivots, error

def get_pivot_matrix_from_vector(pivots,nrows):
    P = np.eye(nrows)
    P = P[pivots,:]
    return P

def determinant_triangular_matrix(matrix):
    return np.prod(np.diag(matrix))

def get_all_primes_less_than_or_equal_to_n(n):
    primes = list()
    primes.append(2)
    for num in range(3, n+1, 2):
        if all(num % i != 0 for i in range(2, int(num**.5 ) + 1)):
            primes.append(num)
    return np.asarray(primes)


def get_first_n_primes(n):
    primes = list()
    primes.append(2)
    num=3
    while len(primes)<n:
        if all(num % i != 0 for i in range(2, int(num**.5 ) + 1)):
            primes.append(num)
        num+=2
    return np.asarray(primes)

def halton_sequence(num_vars, index1, index2):
    assert index1<index2
    assert num_vars<=100

    primes = get_first_n_primes(num_vars)
    
    try:
        from pyapprox.cython.utilities import halton_sequence_pyx
        return halton_sequence_pyx(primes,index1,index2)
    except:
        print ('halton_sequence extension failed')
        pass

    num_samples = index2-index1
    sequence = np.zeros((num_vars,num_samples))
    ones = np.ones(num_vars)

    kk=0
    for ii in range(index1,index2):
        ff = ii*ones
        prime_inv = 1./primes
        summand = ii*num_vars
        while summand>0:
            remainder = np.remainder(ff,primes)
            sequence[:,kk] += remainder*prime_inv
            prime_inv /= primes
            ff=ff//primes
            summand = ff.sum()
        kk+=1
    return sequence

def transformed_halton_sequence(marginal_icdfs,num_vars,num_samples):
    # sample with index 0 is [0,..0] this can cause problems for icdfs of
    # unbounded random variables so start with index 1 in halton sequence
    samples = halton_sequence(num_vars, 1, num_samples+1)
    if marginal_icdfs is None:
        return samples
        
    if callable(marginal_icdfs):
        marginal_icdfs = [marginal_icdfs]*num_vars
    else:
        assert len(marginal_icdfs)==num_vars
    
    for ii in range(num_vars):
        samples[ii,:] = marginal_icdfs[ii](samples[ii,:])
    return samples

def approx_fprime(x,func,eps=np.sqrt(np.finfo(float).eps)):
    """Approx the gradient of a vector valued function at a single
    sample using finite_difference
    """
    assert x.shape[1]==1
    nvars = x.shape[0]
    fprime = []
    func_at_x = func(x).squeeze()
    assert func_at_x.ndim==1
    for ii in range(nvars):
        x_plus_eps = x.copy()
        x_plus_eps[ii] += eps
        fprime.append((func(x_plus_eps).squeeze()-func_at_x)/eps)
    return np.array(fprime)

def partial_functions_equal(func1, func2):
    if not (isinstance(func1, partial) and isinstance(func2, partial)):
        return False
    are_equal = all([getattr(func1, attr) == getattr(func2, attr)
                     for attr in ['func', 'args', 'keywords']])
    return are_equal

def get_all_sample_combinations(samples1,samples2):
    """
    For two sample sets of different random variables
    loop over all combinations 

    samples1 vary slowest and samples2 vary fastest

    Let samples1 = [[1,2],[2,3]]
        samples2 = [[0, 0, 0],[0, 1, 2]]

    Then samples will be

    ([1, 2, 0, 0, 0])
    ([1, 2, 0, 1, 2])
    ([3, 4, 0, 0, 0])
    ([3, 4, 0, 1, 2])

    """
    import itertools
    samples = []
    for r in itertools.product(*[samples1.T,samples2.T]):
        samples.append(np.concatenate(r))
    return np.asarray(samples).T
