from abc import ABC, abstractmethod


class LinAlgMixin(ABC):
    """Abstract base class for linear algebra operations.

    Designed to not need a call to __init__."""

    @abstractmethod
    def _la_dot(self, Amat, Bmat):
        """Compute the dot product of two matrices."""
        raise NotImplementedError

    @abstractmethod
    def _la_eye(self, nrows: int):
        """Return the identity matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_inv(self, mat):
        """Compute the inverse of a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_cholesky(self, mat):
        """Compute the cholesky factorization of a matrix."""
        raise NotImplementedError

    def _la_cholesky_solve(self, chol, bvec, lower: bool = True):
        """Solve the linear equation A x = b for x,
        using the cholesky factorization of A."""
        raise NotImplementedError

    @abstractmethod
    def _la_solve_triangular(self, Amat, bvec, lower: bool = True):
        """Solve the linear equation A x = b for x,
        when A is a triangular matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_full(self, *args):
        """Return a matrix with all values set to fill_value"""
        raise NotImplementedError

    @abstractmethod
    def _la_empty(self, *args):
        """Return a matrix with uniitialized values"""
        raise NotImplementedError

    @abstractmethod
    def _la_exp(self, matrix):
        """Apply exponential element wise to a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_sqrt(self, matrix):
        """Apply sqrt element wise to a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_cos(self, matrix):
        """Apply cos element wise to a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_arccos(self, matrix):
        """Apply arccos element wise to a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_sin(self, matrix):
        """Apply sin element wise to a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_log(self, matrix):
        """Apply log element wise to a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_multidot(self, matrix_list):
        """Compute the dot product of multiple matrices."""
        raise NotImplementedError

    @abstractmethod
    def _la_prod(self, matrix_list, axis=None):
        """Compute the product of a matrix along a given axis."""
        raise NotImplementedError

    @abstractmethod
    def _la_hstack(self, arrays):
        """Stack arrays horizontally (column wise)."""
        raise NotImplementedError

    @abstractmethod
    def _la_vstack(self, arrays):
        """Stack arrays vertically (row wise)."""
        raise NotImplementedError

    @abstractmethod
    def _la_dstack(self, arrays):
        """Stack arrays along third axis."""
        raise NotImplementedError

    @abstractmethod
    def _la_arange(self, *args):
        """Return equidistant values within a given interval."""
        raise NotImplementedError

    @abstractmethod
    def _la_linspace(self, *args):
        """Return equidistant values within a given interval."""
        raise NotImplementedError

    @abstractmethod
    def _la_ndim(self, mat) -> int:
        """Return the dimension of the tensor."""
        raise NotImplementedError

    @abstractmethod
    def _la_repeat(self, mat, nreps):
        """Makes repeated deep copies of a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_cdist(self, Amat, Bmat):
        """
        Return cthe euclidean distance between elements of two matrices.
        Should be equivalent to
        scipy.spatial.distance.cdist(Amat, Bmat, metric="euclidean")
        """
        raise NotImplementedError

    @abstractmethod
    def _la_einsum(self, *args):
        """Compute Einstein summation on two tensors."""
        raise NotImplementedError

    @abstractmethod
    def _la_trace(self, mat):
        """Compute the trace of a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_copy(self, mat):
        """Return a deep copy of a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_get_diagonal(self, mat):
        """Return the diagonal of a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_isnan(self, mat):
        """Determine what entries are NAN."""
        raise NotImplementedError

    @abstractmethod
    def _la_atleast1d(self, val, dtype=None):
        """Make an object at least a 1D tensor."""
        raise NotImplementedError

    @abstractmethod
    def _la_atleast2d(self, val, dtype=None):
        """Make an object at least a 2D tensor."""
        raise NotImplementedError

    @abstractmethod
    def _la_reshape(self, mat, newshape):
        """Reshape a matrix."""
        raise NotImplementedError

    @abstractmethod
    def _la_where(self, cond):
        """Return whether elements of a matrix satisfy a condition."""
        raise NotImplementedError

    @abstractmethod
    def _la_tointeger(self, mat):
        """Cast a matrix to integers"""
        raise NotImplementedError

    @abstractmethod
    def _la_inf(self):
        """Return native representation of infinity."""
        raise NotImplementedError

    @abstractmethod
    def _la_norm(self, mat, axis=None):
        """Return the norm of a matrix along a given axis."""
        raise NotImplementedError

    @abstractmethod
    def _la_any(self, mat, axis=None):
        """Find if any element of a matrix evaluates to True."""
        raise NotImplementedError

    @abstractmethod
    def _la_all(self, mat, axis=None):
        """Find if all elements of a matrix evaluate to True."""
        raise NotImplementedError

    @abstractmethod
    def _la_kron(self, Amat, Bmat):
        """Compute the Kroneker product of two matrices"""
        raise NotImplementedError

    @abstractmethod
    def _la_slogdet(self, Amat):
        """Compute the log determinant of a matrix"""
        raise NotImplementedError

    def _la_mean(self, mat, axis=None):
        """Compute the mean of a matrix"""
        raise NotImplementedError

    def _la_std(self, mat, axis=None, ddof=0):
        """Compute the standard-deviation of a matrix"""
        raise NotImplementedError

    def _la_cov(self, mat, ddof=0, rowvar=True):
        """Compute the covariance matrix from samples of variables
        in a matrix."""
        raise NotImplementedError

    def _la_abs(self, mat):
        """Compute the absolte values of each entry in a matrix"""
        raise NotImplementedError

    def _la_to_numpy(self, mat):
        """Compute the matrix to a np.ndarray."""
        raise NotImplementedError

    def _la_argsort(self, mat, axis=-1):
        """Compute the indices that sort a matrix in ascending order."""
        raise NotImplementedError

    def _la_sort(self, mat, axis=-1):
        """Return the matrix sorted in ascending order."""
        raise NotImplementedError

    def _la_flip(self, mat, axis=None):
        "Reverse the order of the elements in a matrix."
        raise NotImplementedError

    def _la_allclose(self, Amat, Bmat, **kwargs):
        "Check if two matries are close"
        raise NotImplementedError

    def _la_detach(self, mat):
        """Detach a matrix from the computational graph.
        Override for backends that support automatic differentiation."""
        return mat

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    def _la_block_cholesky_engine(self, L_A, L_A_inv_B, B, D, return_blocks):
        schur_comp = D-self._la_multidot((L_A_inv_B.T, L_A_inv_B))
        L_S = self._la_cholesky(schur_comp)
        chol_blocks = [L_A, L_A_inv_B.T, L_S]
        if return_blocks:
            return chol_blocks
        return self._la_vstack([
            self._la_hstack([chol_blocks[0], 0*L_A_inv_B]),
            self._la_hstack([chol_blocks[1], chol_blocks[2]])])

    def _la_block_cholesky(self, blocks, return_blocks=False):
        A, B = blocks[0]
        D = blocks[1][1]
        L_A = self._la_cholesky(A)
        L_A_inv_B = self._la_solve_triangular(L_A, B)
        return self._la_block_cholesky_engine(
            L_A, L_A_inv_B, B, D, return_blocks)

    def _la_get_correlation_from_covariance(self, cov):
        r"""
        Compute the correlation matrix from a covariance matrix
        """
        stdev_inv = 1/self._la_sqrt(self._la_get_diagonal(cov))
        cor = stdev_inv[None, :]*cov*stdev_inv[:, None]
        return cor
