import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg import lapack


def invert_permutation_vector(p, dtype=int):
    r"""
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
    pt = np.zeros(p.size, dtype=dtype)
    pt[p] = np.arange(N, dtype=dtype)
    return pt


def get_low_rank_matrix(num_rows, num_cols, rank):
    r"""
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
    assert rank <= min(num_rows, num_cols)
    # Generate a matrix with normally distributed entries
    N = max(num_rows, num_cols)
    Amatrix = np.random.normal(0, 1, (N, N))
    # Make A symmetric positive definite
    Amatrix = np.dot(Amatrix.T, Amatrix)
    # Construct low rank approximation of A
    eigvals, eigvecs = np.linalg.eigh(Amatrix.copy())
    # Set smallest eigenvalues to zero. Note eigenvals are in
    # ascending order
    eigvals[:(eigvals.shape[0]-rank)] = 0.
    # Construct rank r A matrix
    Amatrix = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    # Resize matrix to have requested size
    Amatrix = Amatrix[:num_rows, :num_cols]
    return Amatrix


def adjust_sign_svd(U, V, adjust_based_upon_U=True):
    r"""
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
        msg = 'U.shape[1] must equal V.shape[0]. If using np.linalg.svd set '
        msg += 'full_matrices=False'
        raise ValueError(msg)

    if adjust_based_upon_U:
        s = np.sign(U[0, :])
    else:
        s = np.sign(V[:, 0])
    U *= s
    V *= s[:, np.newaxis]
    return U, V


def adjust_sign_eig(U):
    r"""
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
    s = np.sign(U[0, :])
    U *= s
    return U


def sorted_eigh(C):
    r"""
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
    W = W[:, ind[::-1]]
    s = np.sign(W[0, :])
    s[s == 0] = 1
    W = W*s
    return e.reshape((e.size, 1)), W


def continue_pivoted_lu_factorization(LU_factor, raw_pivots, current_iter,
                                      max_iters, num_initial_rows=0):
    it = current_iter
    for it in range(current_iter, max_iters):

        # find best pivot
        if np.isscalar(num_initial_rows) and (it < num_initial_rows):
            # pivot=np.argmax(np.absolute(LU_factor[it:num_initial_rows,it]))+it
            pivot = it
        elif (not np.isscalar(num_initial_rows) and
              (it < num_initial_rows.shape[0])):
            pivot = num_initial_rows[it]
        else:
            pivot = np.argmax(np.absolute(LU_factor[it:, it]))+it

        # update pivots vector
        # swap_rows(pivots,it,pivot)
        raw_pivots[it] = pivot

        # apply pivots(swap rows) in L factorization
        swap_rows(LU_factor, it, pivot)

        # check for singularity
        if abs(LU_factor[it, it]) < np.finfo(float).eps:
            msg = "pivot %1.2e" % abs(LU_factor[it, it])
            msg += " is to small. Stopping factorization."
            print(msg)
            break

        # update L_factor
        LU_factor[it+1:, it] /= LU_factor[it, it]

        # udpate U_factor
        col_vector = LU_factor[it+1:, it]
        row_vector = LU_factor[it, it+1:]

        update = np.outer(col_vector, row_vector)
        LU_factor[it+1:, it+1:] -= update
    return LU_factor, raw_pivots, it


def unprecondition_LU_factor(LU_factor, precond_weights, num_pivots=None):
    r"""
    A=LU and WA=XY
    Then WLU=XY
    We also know Y=WU
    So WLU=XWU => WL=XW so L=inv(W)*X*W
    and U = inv(W)Y
    """
    if num_pivots is None:
        num_pivots = np.min(LU_factor.shape)
    assert precond_weights.shape[1] == 1
    assert precond_weights.shape[0] == LU_factor.shape[0]
    # left multiply L an U by inv(W), i.e. compute inv(W).dot(L)
    # and inv(W).dot(U)

    # `np.array` creates a new copy of LU_factor, faster than `.copy()`
    LU_factor = np.array(LU_factor)/precond_weights

    # right multiply L by W, i.e. compute L.dot(W)
    # Do not overwrite columns past num_pivots. If not all pivots have been
    # performed the columns to the right of this point contain U factor
    for ii in range(num_pivots):
        LU_factor[ii+1:, ii] *= precond_weights[ii, 0]

    return LU_factor


def split_lu_factorization_matrix(LU_factor, num_pivots=None):
    r"""
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
    if L_factor.shape[1] < L_factor.shape[0]:
        # if matrix over-determined ensure L is a square matrix
        n0 = L_factor.shape[0]-L_factor.shape[1]
        L_factor = np.hstack([L_factor, np.zeros((L_factor.shape[0], n0))])
    if num_pivots < np.min(L_factor.shape):
        n1 = L_factor.shape[0]-num_pivots
        n2 = L_factor.shape[1]-num_pivots
        L_factor[num_pivots:, num_pivots:] = np.eye(n1, n2)
    np.fill_diagonal(L_factor, 1.)
    U_factor = np.triu(LU_factor)
    U_factor[num_pivots:, num_pivots:] = LU_factor[num_pivots:, num_pivots:]
    return L_factor, U_factor


def truncated_pivoted_lu_factorization(A, max_iters, num_initial_rows=0,
                                       truncate_L_factor=True):
    r"""
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
    num_rows, num_cols = A.shape
    min_num_rows_cols = min(num_rows, num_cols)
    max_iters = min(max_iters, min_num_rows_cols)
    if (A.shape[1] < max_iters):
        msg = "truncated_pivoted_lu_factorization: "
        msg += " A is inconsistent with max_iters. Try deceasing max_iters or "
        msg += " increasing the number of columns of A"
        raise Exception(msg)

    # Use L to store both L and U during factoriation then copy out U in post
    # processing
    # `np.array` creates a new copy of A (faster than `.copy()`)
    LU_factor = np.array(A)
    raw_pivots = np.arange(num_rows)
    LU_factor, raw_pivots, it = continue_pivoted_lu_factorization(
        LU_factor, raw_pivots, 0, max_iters, num_initial_rows)

    if not truncate_L_factor:
        return LU_factor, raw_pivots
    else:
        pivots = get_final_pivots_from_sequential_pivots(
            raw_pivots)[:it+1]
        L_factor, U_factor = split_lu_factorization_matrix(LU_factor, it+1)
        L_factor = L_factor[:it+1, :it+1]
        U_factor = U_factor[:it+1, :it+1]
        return L_factor, U_factor, pivots


def add_columns_to_pivoted_lu_factorization(LU_factor, new_cols, raw_pivots):
    r"""
    Given factorization PA=LU add new columns to A in unpermuted order and
    update LU factorization

    Parameters
    ----------
    raw_pivots : np.ndarray (num_pivots)
        The pivots applied at each iteration of pivoted LU factorization.
        If desired one can use get_final_pivots_from_sequential_pivots to
        compute final position of rows after all pivots have been applied.
    """
    assert LU_factor.shape[0] == new_cols.shape[0]
    assert raw_pivots.shape[0] <= new_cols.shape[0]
    num_pivots = raw_pivots.shape[0]
    for it, pivot in enumerate(raw_pivots):
        # inlined swap_rows() for performance
        new_cols[[it, pivot]] = new_cols[[pivot, it]]

        # update LU_factor
        # recover state of col vector from permuted LU factor
        # Let  (jj,kk) represent iteration and pivot pairs
        # then if lu factorization produced sequence of pairs
        # (0,4),(1,2),(2,4) then LU_factor[:,0] here will be col_vector
        # in LU algorithm with the second and third permutations
        # so undo these permutations in reverse order
        next_idx = it+1

        # `col_vector` is a copy of the LU_factor subset
        col_vector = np.array(LU_factor[next_idx:, it])
        for ii in range(num_pivots-it-1):
            # (it+1) necessary in two lines below because only dealing
            # with compressed col vector which starts at row it in LU_factor
            jj = raw_pivots[num_pivots-1-ii]-next_idx
            kk = num_pivots-ii-1-next_idx

            # inlined swap_rows()
            col_vector[jj], col_vector[kk] = col_vector[kk], col_vector[jj]

        new_cols[next_idx:, :] -= np.outer(col_vector, new_cols[it, :])

    LU_factor = np.hstack((LU_factor, new_cols))

    return LU_factor


def add_rows_to_pivoted_lu_factorization(LU_factor, new_rows, num_pivots):
    assert LU_factor.shape[1] == new_rows.shape[1]
    LU_factor_extra = np.array(new_rows)  # take copy of `new_rows`
    for it in range(num_pivots):
        LU_factor_extra[:, it] /= LU_factor[it, it]
        col_vector = LU_factor_extra[:, it]
        row_vector = LU_factor[it, it+1:]
        update = np.outer(col_vector, row_vector)
        LU_factor_extra[:, it+1:] -= update

    return np.vstack([LU_factor, LU_factor_extra])


def swap_rows(matrix, ii, jj):
    matrix[[ii, jj]] = matrix[[jj, ii]]


def pivot_rows(pivots, matrix, in_place=True):
    if not in_place:
        matrix = matrix.copy()
    num_pivots = pivots.shape[0]
    assert num_pivots <= matrix.shape[0]
    for ii in range(num_pivots):
        swap_rows(matrix, ii, pivots[ii])
    return matrix


def get_final_pivots_from_sequential_pivots(
        sequential_pivots, num_pivots=None):
    if num_pivots is None:
        num_pivots = sequential_pivots.shape[0]
    assert num_pivots >= sequential_pivots.shape[0]
    pivots = np.arange(num_pivots)
    return pivot_rows(sequential_pivots, pivots, False)


def cholesky_decomposition(Amat):

    nrows = Amat.shape[0]
    assert Amat.shape[1] == nrows

    L = np.zeros((nrows, nrows))
    for ii in range(nrows):
        temp = Amat[ii, ii]-np.sum(L[ii, :ii]**2)
        if temp <= 0:
            raise Exception('matrix is not positive definite')
        L[ii, ii] = np.sqrt(temp)
        L[ii+1:, ii] =\
            (Amat[ii+1:, ii]-np.sum(
                L[ii+1:, :ii]*L[ii, :ii], axis=1))/L[ii, ii]

    return L


def pivoted_cholesky_decomposition(A, npivots, init_pivots=None, tol=0.,
                                   error_on_small_tol=False,
                                   pivot_weights=None,
                                   return_full=False,
                                   econ=True):
    r"""
    Return a low-rank pivoted Cholesky decomposition of matrix A.

    If A is positive definite and npivots is equal to the number of rows of A
    then L.dot(L.T)==A

    To obtain the pivoted form of L set
    L = L[pivots,:]

    Then P.T.dot(A).P == L.dot(L.T)

    where P is the standard pivot matrix which can be obtained from the
    pivot vector using the function
    """
    Amat = A.copy()
    nrows = Amat.shape[0]
    assert Amat.shape[1] == nrows
    assert npivots <= nrows

    # L = np.zeros(((nrows,npivots)))
    L = np.zeros(((nrows, nrows)))
    # diag1 = np.diag(Amat).copy() # returns a copy of diag
    diag = Amat.ravel()[::Amat.shape[0]+1]  # returns a view of diag
    # assert np.allclose(diag,diag1)
    pivots = np.arange(nrows)
    init_error = np.absolute(diag).sum()
    L, pivots, diag, chol_flag, ncompleted_pivots, error = \
        continue_pivoted_cholesky_decomposition(
            Amat, L, npivots, init_pivots, tol,
            error_on_small_tol,
            pivot_weights, pivots, diag,
            0, init_error, econ)

    if not return_full:
        return L[:, :ncompleted_pivots], pivots[:ncompleted_pivots], error,\
            chol_flag
    else:
        return L, pivots, error, chol_flag, diag.copy(), init_error, \
            ncompleted_pivots


def continue_pivoted_cholesky_decomposition(Amat, L, npivots, init_pivots, tol,
                                            error_on_small_tol,
                                            pivot_weights, pivots, diag,
                                            ncompleted_pivots, init_error,
                                            econ):
    Amat = Amat.copy()  # Do not overwrite incoming Amat
    if econ is False and pivot_weights is not None:
        msg = 'pivot weights not used when econ is False'
        raise Exception(msg)
    chol_flag = 0
    assert ncompleted_pivots < npivots
    for ii in range(ncompleted_pivots, npivots):
        if init_pivots is None or ii >= len(init_pivots):
            if econ:
                if pivot_weights is None:
                    pivot = np.argmax(diag[pivots[ii:]])+ii
                else:
                    pivot = np.argmax(
                        pivot_weights[pivots[ii:]]*diag[pivots[ii:]])+ii
            else:
                schur_complement = (
                    Amat[np.ix_(pivots[ii:], pivots[ii:])] -
                    L[pivots[ii:], :ii].dot(L[pivots[ii:], :ii].T))
                schur_diag = np.diagonal(schur_complement)
                pivot = np.argmax(
                    np.linalg.norm(schur_complement, axis=0)**2/schur_diag)
                pivot += ii
        else:
            pivot = np.where(pivots == init_pivots[ii])[0][0]
            assert pivot >= ii

        swap_rows(pivots, ii, pivot)
        if diag[pivots[ii]] <= 0:
            msg = 'matrix is not positive definite'
            if error_on_small_tol:
                raise Exception(msg)
            else:
                print(msg)
                chol_flag = 1
                break

        L[pivots[ii], ii] = np.sqrt(diag[pivots[ii]])

        L[pivots[ii+1:], ii] = (
            Amat[pivots[ii+1:], pivots[ii]] -
            L[pivots[ii+1:], :ii].dot(L[pivots[ii], :ii]))/L[pivots[ii], ii]
        diag[pivots[ii+1:]] -= L[pivots[ii+1:], ii]**2

        # for jj in range(ii+1,nrows):
        #     L[pivots[jj],ii]=(Amat[pivots[ii],pivots[jj]]-
        #         L[pivots[ii],:ii].dot(L[pivots[jj],:ii]))/L[pivots[ii],ii]
        #     diag[pivots[jj]] -= L[pivots[jj],ii]**2
        error = diag[pivots[ii+1:]].sum()/init_error
        # print(ii, 'error', error)
        if error < tol:
            msg = 'Tolerance reached. '
            msg += f'Iteration:{ii}. Tol={tol}. Error={error}'
            # If matrix is rank r then then error will be machine precision
            # In such a case exiting without an error is the right thing to do
            if error_on_small_tol:
                raise Exception(msg)
            else:
                chol_flag = 1
                print(msg)
                break

    return L, pivots, diag, chol_flag, ii+1, error


def get_pivot_matrix_from_vector(pivots, nrows):
    P = np.eye(nrows)
    P = P[pivots, :]
    return P


def determinant_triangular_matrix(matrix):
    return np.prod(np.diag(matrix))


def cholesky_solve_linear_system(L, rhs):
    r"""
    Solve LL'x = b using forwards and backwards substitution
    """
    # Use forward subsitution to solve Ly = b
    y = solve_triangular(L, rhs, lower=True)
    # Use backwards subsitution to solve L'x = y
    x = solve_triangular(L.T, y, lower=False)
    return x


def update_cholesky_factorization(L_11, A_12, A_22):
    r"""
    Update a Cholesky factorization.

    Specifically compute the Cholesky factorization of

    .. math:: A=\begin{bmatrix} A_{11} & A_{12}\\ A_{12}^T &
              A_{22}\end{bmatrix}

    where :math:`L_{11}` is the Cholesky factorization of :math:`A_{11}`.
    Noting that

    .. math::

      \begin{bmatrix} A_{11} & A_{12}\\ A_{12}^T & A_{22}\end{bmatrix} =
      \begin{bmatrix} L_{11} & 0\\ L_{12}^T & L_{22}\end{bmatrix}
      \begin{bmatrix} L_{11}^T & L_{12}\\ 0 & L_{22}^T\end{bmatrix}

    we can equate terms to find

    .. math::

        L_{12} = L_{11}^{-1}A_{12}, \quad
        L_{22}L_{22}^T = A_{22}-L_{12}^TL_{12}
    """
    if L_11.shape[0] == 0:
        return np.linalg.cholesky(A_22)

    nrows, ncols = A_12.shape
    assert A_22.shape == (ncols, ncols)
    assert L_11.shape == (nrows, nrows)
    L_12 = solve_triangular(L_11, A_12, lower=True)
    print(A_22 - L_12.T.dot(L_12))
    L_22 = np.linalg.cholesky(A_22 - L_12.T.dot(L_12))
    L = np.block([[L_11, np.zeros((nrows, ncols))], [L_12.T, L_22]])
    return L


def update_cholesky_factorization_inverse(L_11_inv, L_12, L_22):
    nrows, ncols = L_12.shape
    L_22_inv = np.linalg.inv(L_22)
    L_inv = np.block(
        [[L_11_inv, np.zeros((nrows, ncols))],
         [-L_22_inv.dot(L_12.T.dot(L_11_inv)), L_22_inv]])
    return L_inv


def update_trace_involving_cholesky_inverse(L_11_inv, L_12, L_22_inv, B,
                                            prev_trace):
    r"""
    Update the trace of matrix matrix product involving the inverse of a
    matrix with a cholesky factorization.

    That is compute

    .. math:: \mathrm{Trace}\leftA^{inv}B\right}

    where :math:`A=LL^T`
    """
    nrows, ncols = L_12.shape
    assert B.shape == (nrows+ncols, nrows+ncols)
    B_11 = B[:nrows, :nrows]
    B_12 = B[:nrows, nrows:]
    B_21 = B[nrows:, :nrows]
    B_22 = B[nrows:, nrows:]
    # assert np.allclose(B, np.block([[B_11, B_12],[B_21, B_22]]))

    C = -np.dot(L_22_inv.dot(L_12.T), L_11_inv)
    C_T_L_22_inv = C.T.dot(L_22_inv)
    trace = prev_trace + np.sum(C.T.dot(C)*B_11) + \
        np.sum(C_T_L_22_inv*B_12) + np.sum(C_T_L_22_inv.T*B_21) +  \
        np.sum(L_22_inv.T.dot(L_22_inv)*B_22)
    return trace


def num_entries_square_triangular_matrix(N, include_diagonal=True):
    r"""Num entries in upper (or lower) NxN traingular matrix"""
    if include_diagonal:
        return int(N*(N+1)/2)
    else:
        return int(N*(N-1)/2)


def num_entries_rectangular_triangular_matrix(M, N, upper=True):
    r"""Num entries in upper (or lower) MxN traingular matrix.
    This is useful for nested for loops like

    (upper=True)

    for ii in range(M):
        for jj in range(ii+1):

    (upper=False)

    for jj in range(N):
        for ii in range(jj+1):

    """
    assert M >= N
    if upper:
        return num_entries_square_triangular_matrix(N)
    else:
        return num_entries_square_triangular_matrix(M) -\
            num_entries_square_triangular_matrix(M-N)


def flattened_rectangular_lower_triangular_matrix_index(ii, jj, M, N):
    r"""
    Get flattened index kk from row and column indices (ii,jj) of a
    lower triangular part of MxN matrix
    """
    assert M >= N
    assert ii >= jj
    if ii == 0:
        return 0
    T = num_entries_rectangular_triangular_matrix(ii, min(ii, N), upper=False)
    kk = T+jj
    return kk


def qr_solve(Q, R, rhs):
    """
    Find the least squares solution Ax = rhs given a QR factorization of the
    matrix A

    Parameters
    ----------
    Q : np.ndarray (nrows, nrows)
        The unitary/upper triangular Q factor

    R : np.ndarray (nrows, ncols)
        The upper triangular R matrix

    rhs : np.ndarray (nrows, nqoi)
        The right hand side vectors

    Returns
    -------
    x : np.ndarray (nrows, nqoi)
        The solution
    """
    tmp = np.dot(Q.T, rhs)
    return solve_triangular(R, tmp, lower=False)


def equality_constrained_linear_least_squares(A, B, y, z):
    """
    Solve equality constrained least squares regression

    minimize || y - A*x ||_2   subject to   B*x = z

    It is assumed that

    Parameters
    ----------
    A : np.ndarray (M, N)
        P <= N <= M+P, and

    B : np.ndarray (N, P)
        P <= N <= M+P, and

    y : np.ndarray (M, 1)
        P <= N <= M+P, and

    z : np.ndarray (P, 1)
        P <= N <= M+P, and

    Returns
    -------
    x : np.ndarray (N, 1)
        The solution
    """
    return lapack.dgglse(A, B, y, z)[3]
