from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def qr_solve(Q, R, rhs, bkd=NumpyLinAlgMixin):
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
    return bkd.solve_triangular(R, Q.T @ rhs, lower=False)
