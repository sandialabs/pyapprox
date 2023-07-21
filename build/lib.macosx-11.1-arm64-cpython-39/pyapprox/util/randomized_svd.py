import numpy as np

from pyapprox.util.linalg import adjust_sign_svd


def get_from_dict_or_apply_default(dictionary, key, default):
    if key in dictionary:
        return dictionary[key]
    else:
        return default


class MatVecOperator(object):
    """
    Operator representing the action of a Matrix on a vector. I.e.
    matrix vector multiplication.
    """

    def __init__(self, matrix=None):
        self.matrix = matrix

    def apply(self, vectors, transpose=True):
        if transpose:
            return np.dot(self.matrix.T, vectors)
        else:
            return np.dot(self.matrix, vectors)

    def num_rows(self):
        return self.matrix.shape[0]

    def num_cols(self):
        return self.matrix.shape[1]


def randomized_range_finder(operator, opts, num_power_iterations):
    """Given an m x n matrix A and an integer r, this scheme computes an m x r
    orthonormal matrix Q whose range approximates the range of A.

    Parameters
    ----------
    operator : MatVecOperator class
        Action of a the matrix A on a vector, i.e op(x)=dot(A,x)

    num_power_iterations : integer (default is 1)
        The number of power iterations.
        A higher number of iterations is useful if the singular spectrum
        of the input matrix may decay slowly.

    opts : dictionary

    Required arguments
    ------------------
    num_singular_values : integer (default=None)
         Number of singular values to extract. Must be specified (not None)
         if range_finder=='standard'. It is ignored otherwise.

    Optional arguments:
    ------------------
    num_extra_samples : integer (default=5)
        The number of columns of Q that the algorithm needs to
        reach a required accuracy is usually slightly larger than the rank r
        of the smallest basis. This discrepancy is the num_extra_samples.
        The more rapid the decay of the singular values, the less
        oversampling is needed. In the extreme case that the matrix has exact
        rank r, it is not necessary to oversample.

    Returns
    -------
    Q : (m x r) matrix
        Orthonormal matrix Q whose range approximates the range of A.

    X : (n x r) matrix
        Gaussian Random matrix used to compute action of A

    Y : (n x r) matrix
        Y = dot(A,X)
    """

    num_singular_values = get_from_dict_or_apply_default(
        opts, "num_singular_values", None)
    num_extra_samples = get_from_dict_or_apply_default(
        opts, "num_extra_samples", 5)
    assert num_singular_values > 0
    assert num_extra_samples > 0
    if num_singular_values is None:
        raise Exception("must specify num_singular_values in opts")
    num_samples = num_singular_values + num_extra_samples

    # Draw an (n x r) Gaussian random matrix X
    X = np.random.normal(0., 1., (operator.num_cols(), num_samples))

    # Construct an (m x r) matrix Q whose columns form an orthonormal
    # basis for the range of Y , e.g., using the QR factorization Y = QR.
    Y = operator.apply(X, transpose=False)
    I = np.where(np.all(np.isfinite(Y), axis=0) == False)[0]
    if I.shape[0] > 0:
        return None, X, Y

    Q, R = np.linalg.qr(Y)

    # Form the (m x r) matrix Y = dot(A,X)
    for i in range(num_power_iterations):
        Q, R = np.linalg.qr(operator.apply(Q, transpose=False))
        Q, R = np.linalg.qr(operator.apply(Q, transpose=True))
    return Q, X, Y


def terminate_adaptive_randomized_range_finder(
        Q, X, Y, Z, it, num_extra_samples,
        method, tolerance, min_singular_value,
        verbosity, num_samples, max_num_samples,
        best_error, num_iter_error_increase,
        max_num_iter_error_increase):
    """
    Evaluate termination condition of random range finder.

    Parameters
    ----------
    Q : (m x r) matrix
        Orthonormal matrix Q whose range approximates the range of A.

    X : (n x r) matrix
        Gaussian Random matrix used to compute action of A

    Y : (n x r) matrix
        Y = dot(A,X)

    Z : (n x r) matrix
        Y projected onto the basis Q

    it : integer
        The current iteration count of the randomized range finder

    num_extra_samples : integer
        The number of columns, k, of Q that required to have a norm less than
        the specified error tolerance.

    method : string
        'error_in_approx_range' - terminate when aposterior-error estimate
                                  of how well Q approximates range of A, is
                                  smaller than the specified tolerance.
        'min_singular_value' - terminate when the smallest singular value
                                     is smaller than the specified tolerance.

    tolerance : double
        The desired error in the approximate range. The error is relative to
        the first error estimate obtained by the algorithm. TODO
        consider having both an absolute and relative tolerance.

    min_singular_value : double
        The size of the minimum singular value.

    verbosity : integer
        The level of verbosity which controls amount of information printed

    num_samples : integer
        The number of samples used to compute svd.

    max_num_samples : integer
        The maximum number of samples used to compute svd using adaptive
        range finder. Note this is not
        the maximum number of matrix vector operations. TODO replace this
        option with one that controls max number of matrix vector operations.

    best_error : integer
        The smallest error in all previous iterations

    num_iter_error_increase : integer
        The number of consecutive times the error has failed to decrease

    max_num_iter_error_increase : integer
        The maximum number of consecutive times the error is allowed to not
        decrease. Warning setting this to a small number may stop
        the algorithm to early.
        test_compute_single_pass_adaptive_randomized_svd_from_file
        shows that error can increase at least 11 times before getting smaller
        again.However making max_num_iter_error_increase to large can cause
        large errors in SVD due to the effect of oversampling on the stability
        of the Gram-Schmidt ortogonalization used by the adaptive range finder
    """
    error = np.max(np.linalg.norm(Z[:, -num_extra_samples:], axis=0))
    terminate = False
    if (error < tolerance):
        terminate = True
        if verbosity > 0:
            print('terminating range finder. tolerance reached')

    if error >= best_error:
        num_iter_error_increase += 1
        if num_iter_error_increase >= max_num_iter_error_increase:
            terminate = True
            if verbosity > 0:
                print(('terminating range finder. error did not ',
                       'decrease in %d iterations' % max_num_iter_error_increase))
    else:
        best_error = error
        num_iter_error_increase = 0

    if method == 'min_singular_value':
        if Q.shape[1] > 0:
            U, S, V = svd_using_orthogonal_basis(
                None, Q, X[:, :Q.shape[1]], Y[:, :Q.shape[1]], True)
            current_min_singular_value = S.min()
            if (current_min_singular_value < min_singular_value):
                terminate = True
                print('terminating range finder. min singular value reached')

    if num_samples >= max_num_samples:
        terminate = True
        if verbosity > 0:
            print('terminating range finder. max num samples reached')

    return terminate, error, best_error, num_iter_error_increase


def adaptive_randomized_range_finder(operator, opts):
    """
    Given an (m x n) matrix A, a tolerance, and an integer p (e.g., p = 10),
    compute an orthonormal matrix Q such that ||(I-Q*Q')A|| < tol holds
    with probability at least 1-min{m, n}*10^{-r}.

    Parameters
    ----------

    operator : MatVecOperator class
        Action of a the matrix A on a vector, i.e op(x)=dot(A,x)

    opts : dictionary

    Required arguments
    ----------------
    max_num_samples : integer
        The maximum number of samples used to compute svd using adaptive
        range finder. Note this is not
        the maximum number of matrix vector operations. TODO replace this
        option with one that controls max number of matrix vector operations.

    Optional arguments
    ------------------
    num_extra_samples : integer (default=10)
        The number of columns, k, of Q that required to have a norm less than
        the specified error tolerance.

    concurrency : integer (default=1)
       The number of matrix vector operations that can be carried out at once.

    verbosity : integer (default=0)
       The level of verbosity which controls amount of information printed

    max_num_iter_error_increase : integer (default=10)
        The maximum number of consecutive times the error is allowed to not
        decrease

    termination_method : string (default="error_in_approx_range")
       The method used to determine termination of the adaptive range finder.
       'error_in_approx_range' - terminate when aposterior-error estimate
                                  of how well Q approximates range of A, is
                                  smaller than the specified tolerance.
       'min_singular_value' - terminate when the smallest singular value
                                    is smaller than the specified tolerance.

    min_singular_value : double (default=None)
        The size of the minimum singular value. Note this must be specified
        (not None) if termination_method=='min_singular_value'

    tolerance : double (default=1e-4)
       The desired error in the approximate range. The error is relative to
       the first error estimate obtained by the algorithm.
       Note this option is only active when
       termination_method=='error_in_approx_range'

    Returns
    -------
    Q : (m x r) matrix
        Orthonormal matrix Q whose range approximates the range of A.

    X : (n x r) matrix
        Gaussian Random matrix used to compute action of A

    Y : (n x r) matrix
        Y = dot(A,X)
    """

    num_extra_samples = get_from_dict_or_apply_default(
        opts, "num_extra_samples", 10)
    assert num_extra_samples > 0
    concurrency = get_from_dict_or_apply_default(opts, "concurrency", 1)
    verbosity = get_from_dict_or_apply_default(opts, "verbosity", 0)
    termination_method = get_from_dict_or_apply_default(
        opts, "termination_method", "error_in_approx_range")
    max_num_iter_error_increase = get_from_dict_or_apply_default(
        opts, "max_num_iter_error_increase", 10)

    min_singular_value = get_from_dict_or_apply_default(
        opts, "min_singular_value", None)
    if termination_method == "min_singular_value" and min_singular is None:
        raise Exception("option 'min_singular value' must be specifed in opts")
    tolerance = get_from_dict_or_apply_default(opts, "tolerance", 1e-4)

    max_num_samples = get_from_dict_or_apply_default(
        opts, "max_num_samples", None)
    if max_num_samples is None:
        raise Exception("'max_num_samples' must be specified")
    # first step applys operator to num_extra_samples then
    # first iteration of loop applies to concurrency samples
    # so lets makes sure that we allow enough samples for these two steps
    assert max_num_samples >= num_extra_samples + concurrency

    num_samples = 0

    X = np.random.normal(0., 1., (operator.num_cols(), num_extra_samples))
    num_samples = X.shape[1]
    Y = operator.apply(X, transpose=False)
    Z = Y.copy()

    it = -1
    best_error = np.finfo(float).max
    num_iter_error_increase = 0
    Q = np.empty((operator.num_cols(), 0), float)
    errors = []
    it_count_since_last_error_decrease = 0

    tol = 0.

    terminate, error, best_error, num_iter_error_increase = \
        terminate_adaptive_randomized_range_finder(
            Q, X, Y, Z, it, num_extra_samples, termination_method, tol,
            min_singular_value, verbosity, num_samples, max_num_samples,
            best_error, num_iter_error_increase, max_num_iter_error_increase)

    # tolerance specifies relative decrease in error from first error estimates
    tol = tolerance*error

    errors.append(error)
    if verbosity > 1:
        print(('iter %d,\terror: %1.2e' % (it+1, error)))

    while not terminate:
        num_new_samples = min(concurrency, max_num_samples-num_samples)
        #Xnew = np.random.normal(0.,1.,(operator.num_cols(),num_new_samples))
        # create transpose of matrix we actually need so that for a given seed
        # the first column of Xnew  will be the same regardless of whether
        # num_new_samples=1 or num_new_samples>1. np.random.normal(a,b,(n,m))
        # produces stores values in cmajor ordering but we need fortran major
        # ordering to be the same.
        Xnew = np.random.normal(
            0., 1., (num_new_samples, operator.num_cols())).T
        Ynew = operator.apply(Xnew, transpose=False)
        for k in range(num_new_samples):
            it += 1
            y = Ynew[:, k]
            Z, Q = adaptive_range_finder_update(Z, Q, y, it, num_extra_samples)
            X = np.hstack((X, Xnew[:, k:k+1]))
            Y = np.hstack((Y, y[:, np.newaxis]))
            num_samples = X.shape[1]

        terminate, error, best_error, num_iter_error_increase = \
            terminate_adaptive_randomized_range_finder(
                Q, X, Y, Z, it, num_extra_samples, termination_method, tol,
                min_singular_value, verbosity, num_samples, max_num_samples,
                best_error, num_iter_error_increase, max_num_iter_error_increase)
        errors.append(error)
        if verbosity > 1:
            print(('iter %d,\terror: %1.2e' % (it+1, error)))

    return Q, X, Y, np.array(errors)


def randomized_svd(operator, opts):
    """
    Given an m x n matrix A this procedure computes an approximate singular
    value decomposition of A using randomized SVD.

    Parameters
    ----------
    operator : MatVecOperator class
        Action of a the matrix A on a vector, i.e op(x)=dot(A,x)

    num_power_iterations : integer (default=0)
        The number of power iterations.
        A higher number of iterations is useful if the singular spectrum
        of the input matrix may decay slowly.

    opts : dictionary
       Options to configure svd.

    Option parameters
    --------------
    single_pass : bool
        True - use one pass algorithm to compute svd
        False - use two pass algorithm to compute svd
        one pass algorithm requires less matrix vector products than two
        pass version, but comes with slightly reduced accuracy
        If True A must be a semi-positive definite matrix
        The single-pass approaches described in this section can degrade
        the approximation error in the final decomposition significantly.
        The situation can be improved by increaseing num_extra_samples.

    history_filename : string (default=None)
        If specified the data used to compute the svd is stored in a .npz file
        of that name

    standard_opts : dictionary (default=None)
        Options for the standard range finder. See documentation of
        randomized_range_finder().
        If not None use the standard randomized svd algorithm where rank is
        specified apriori

    adaptive_opts : dictionary (default={})
        Options for the adaptive range finder. See documentation of
        adaptive_randomized_range_finder()
        If not None adaptively determine rank up to a given tolerance

    Returns
    -------
    U : matrix (num_singular_values x num_singular_values)
        left singular vectors of A = USV

    S : vector (num_singular_values)
        singular values of A = USV

    V : matrix (num_singular_values x num_singular_values)
        right singular vectors of A = USV
    """

    standard_opts = get_from_dict_or_apply_default(opts, 'standard_opts', None)
    adaptive_opts = get_from_dict_or_apply_default(opts, 'adaptive_opts', None)
    if standard_opts is None and adaptive_opts is None:
        raise Exception("must specify 'standard_opts' or 'adaptive_opts'")
    if standard_opts is not None and adaptive_opts is not None:
        raise Exception("must only specify 'standard_opts' or 'adaptive_opts'")
    if standard_opts is not None:
        range_finder = 'standard'
    else:
        range_finder = 'adaptive'

    single_pass = get_from_dict_or_apply_default(opts, "single_pass", True)
    num_power_iterations = get_from_dict_or_apply_default(
        opts, "num_power_iterations", 0)
    history_filename = get_from_dict_or_apply_default(
        opts, "history_filename", None)

    if single_pass:
        assert num_power_iterations == 0
        # operator must be hermitian. This is a weak test but still
        # helpful
        assert operator.num_rows() == operator.num_cols()

    if range_finder == 'standard':
        Q, X, Y = randomized_range_finder(
            operator, standard_opts, num_power_iterations)
        if Q is None:
            # evaluations of Y failed so save data to file for recovery
            np.savez('randomized_svd_recovery_data.npz', X=X, Y=Y)
            raise Exception('evaluations of Y failed')

    elif range_finder == 'adaptive':
        Q, X_all, Y_all, errors = adaptive_randomized_range_finder(
            operator, adaptive_opts)
        # not all X, Y samples are used to compute svd
        # truncate to correct X and Y here
        X = X_all[:, :Q.shape[1]]
        Y = Y_all[:, :Q.shape[1]]
    else:
        raise Exception('incorrect range_finder specified')

    U, S, V = svd_using_orthogonal_basis(operator, Q, X, Y, single_pass)

    # Resize matrices
    if range_finder == 'standard':
        num_singular_values = get_from_dict_or_apply_default(
            standard_opts, "num_singular_values", None)
        U = U[:, :num_singular_values]
        S = S[:num_singular_values]
        V = V[:num_singular_values, :]

    U, V = adjust_sign_svd(U, V)

    if history_filename is not None:
        if range_finder == 'adaptive':
            X = X_all
            Y = Y_all
        np.savez(history_filename, Q=Q, X=X, Y=Y, U=U, S=S, V=V)

    return U, S, V


def svd_using_orthogonal_basis(operator, Q, X, Y, single_pass):
    """
    Given an m x n matrix A, and an m x (r+p) matrix Q whose columns are
    orthonormal and whose range approximates the range of A compute the
    approximate singular value decomposition of A.

    Parameters
    ----------
    operator : MatVecOperator class
        Action of a the matrix A on a vector, i.e op(x)=dot(A,x)

    Q : (m x r) matrix
        Orthonormal matrix Q whose range approximates the range of A.

    X : (n x r) matrix
        Gaussian Random matrix used to compute action of A

    Y : (n x r) matrix
        Y = dot(A,X)

    single_pass : bool
        True - use one pass algorithm to compute svd
        False - use two pass algorithm to compute svd
        one pass algorithm requires less matrix vector products than two
        pass version, but comes with slightly reduced accuracy
        If True A must be a semi-positive definite matrix
        The single-pass approaches described in this section can degrade
        the approximation error in the final decomposition significantly.
        The situation can be improved by increasing num_extra_samples.

    Returns
    -------
    U : matrix (num_singular_values x num_singular_values)
        left singular vectors of A = USV

    S : vector (num_singular_values)
        singular values of A = USV

    V : matrix (num_singular_values x num_singular_values)
        right singular vectors of A = USV
    """
    if not single_pass:
        B = operator.apply(Q, transpose=True).T  # (m x (r+p)) matrix
    else:
        XTQ = np.dot(X.T, Q)
        YTQ = np.dot(Y.T, Q)
        #B = qr_solve(XTQ,YTQ,0)
        B = np.linalg.lstsq(XTQ, YTQ, rcond=None)[0]

    # Compute an SVD of the small matrix B
    U, S, V = np.linalg.svd(B, full_matrices=False)
    U = np.dot(Q, U)

    if single_pass:
        # assumes A is hermitian so V=U.T
        V = U.copy().T

    return U, S, V


def adaptive_range_finder_update(Z, Q, y, it, num_extra_samples):
    # The vectors Q[:,i] become small as the basis starts to capture
    # most of the action of A. In finite-precision arithmetic, their
    # direction is extremely unreliable. To address this problem, we
    # simply reproject the normalized vector Q[:,it] onto
    # range( Q^(it-1) ) complement
    if it > 0:
        Z[:, it] = Z[:, it] - np.dot(Q, np.dot(Q.T, Z[:, it]))
    Qit = Z[:, it]/np.linalg.norm(Z[:, it])

    Q = np.hstack((Q, Qit[:, np.newaxis]))
    z = y-np.dot(Q, np.dot(Q.T, y))
    for i in range(it+1, it+num_extra_samples):
        Z[:, i] = Z[:, i]-Qit*np.dot(Qit, Z[:, i])

    Z = np.hstack((Z, z[:, np.newaxis]))
    return Z, Q


def load_svd_data(history_filename):
    if history_filename[-4:] != '.npz':
        history_filename += '.npz'
    svd_data = np.load(history_filename)
    U = svd_data['U']
    S = svd_data['S']
    V = svd_data['V']
    X = svd_data['X']
    Y = svd_data['Y']
    Q = svd_data['Q']
    return U, S, V, X, Y, Q


def compute_single_pass_adaptive_randomized_svd_from_file(
        history_filename, num_extra_samples, max_num_samples):
    """
    Compute the SVD of a matrix from the history or a previous
    randomized svd computation.

    Parameters
    ----------
    history_filename : string, default=None
        If filename of the file containing the data used to compute the svd 
        (stored in a .npz file)

    num_extra_samples : integer
        The number of columns, k, of Q that required to have a norm less than
        the specified error tolerance. TODO:
        Currently we are not computing an error but rather using max_num_samples
        to limit size of Q. This allows us to compute svd for different 
        num_samples which can help debugging

    max_num_samples : integer, default=None
        The maximum number of samples used to compute svd. This must be less than
        or equal to size of the gaussian random samples X stored in the file.
        If None  all samples will be used
    """
    U, S, V, X, Y, Qfile = load_svd_data(history_filename)
    assert max_num_samples <= X.shape[1]

    if max_num_samples is None:
        num_samples = X.shape[1]
    else:
        num_samples = max_num_samples
    Q = np.empty((Qfile.shape[0], 0), float)
    Z = Y[:, :num_extra_samples].copy()

    for j in range(num_samples-num_extra_samples):

        y = Y[:, num_extra_samples+j]
        Z, Q = adaptive_range_finder_update(Z, Q, y, j, num_extra_samples)

        # Recall not all X, Y samples are used to compute svd when adaptive
        # range finder is used. X.shape[1] = Q.shape[1]+num_extra_samples
        # this step here accounts for this inconsistency
        Xj = X[:, :Q.shape[1]]
        Yj = Y[:, :Q.shape[1]]

    # Compute current svd
    XTQ = np.dot(Xj.T, Q)
    YTQ = np.dot(Yj.T, Q)
    #B = qr_solve(XTQ,YTQ,0)
    B = np.linalg.lstsq(XTQ, YTQ, rcond=None)[0]
    U, S, V = np.linalg.svd(B, full_matrices=False)
    U = np.dot(Q, U)
    V = U.copy().T         # Single pass assumes A is hermitian so V=U.T
    return U, S, V
