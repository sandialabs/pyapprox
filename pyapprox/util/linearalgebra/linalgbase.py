from abc import ABC, abstractmethod
import itertools


class LinAlgMixin(ABC):
    """Abstract base class for linear algebra operations.

    Designed to not need a call to __init__."""

    @staticmethod
    @abstractmethod
    def _la_dot(Amat, Bmat):
        """Compute the dot product of two matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_eye(nrows: int, dtype=None):
        """Return the identity matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_inv(mat):
        """Compute the inverse of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_cholesky(mat):
        """Compute the cholesky factorization of a matrix."""
        raise NotImplementedError

    @staticmethod
    def _la_cholesky_solve(chol, bvec, lower: bool = True):
        """Solve the linear equation A x = b for x,
        using the cholesky factorization of A."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_solve_triangular(Amat, bvec, lower: bool = True):
        """Solve the linear equation A x = b for x,
        when A is a triangular matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_full(*args, dtype=None):
        """Return a matrix with all values set to fill_value"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_zeros(*args, dtype=None):
        """Return a matrix with all values set to zero"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_ones(*args, dtype=None):
        """Return a matrix with all values set to one"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_empty(*args):
        """Return a matrix with uniitialized values"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_exp(matrix):
        """Apply exponential element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_sqrt(matrix):
        """Apply sqrt element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_cos(matrix):
        """Apply cos element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_arccos(matrix):
        """Apply arccos element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_sin(matrix):
        """Apply sin element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_log(matrix):
        """Apply log element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_multidot(matrix_list):
        """Compute the dot product of multiple matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_prod(matrix_list, axis=None):
        """Compute the product of a matrix along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_hstack(arrays):
        """Stack arrays horizontally (column wise)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_vstack(arrays):
        """Stack arrays vertically (row wise)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_stack(arrays, axis=0):
        """Stack arrays along a new axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_dstack(arrays):
        """Stack arrays along third axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_arange(*args):
        """Return equidistant values within a given interval."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_linspace(*args):
        """Return equidistant values within a given interval."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_ndim(mat) -> int:
        """Return the dimension of the tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_repeat(mat, nreps):
        """Makes repeated deep copies of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_cdist(Amat, Bmat):
        """
        Return cthe euclidean distance between elements of two matrices.
        Should be equivalent to
        scipy.spatial.distance.cdist(Amat, Bmat, metric="euclidean")
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_einsum(*args):
        """Compute Einstein summation on two tensors."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_trace(mat):
        """Compute the trace of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_copy(mat):
        """Return a deep copy of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_get_diagonal(mat):
        """Return the diagonal of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_diag(array, k=0):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_isnan(mat):
        """Determine what entries are NAN."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_atleast1d(val, dtype=None):
        """Make an object at least a 1D tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_atleast2d(val, dtype=None):
        """Make an object at least a 2D tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_reshape(mat, newshape):
        """Reshape a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_where(cond):
        """Return whether elements of a matrix satisfy a condition."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_tointeger(mat):
        """Cast a matrix to integers"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_inf():
        """Return native representation of infinity."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_norm(mat, axis=None):
        """Return the norm of a matrix along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_any(mat, axis=None):
        """Find if any element of a matrix evaluates to True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_all(mat, axis=None):
        """Find if all elements of a matrix evaluate to True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_kron(Amat, Bmat):
        """Compute the Kroneker product of two matrices"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_slogdet(Amat):
        """Compute the log determinant of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_mean(mat, axis=None):
        """Compute the mean of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_std(mat, axis=None, ddof=0):
        """Compute the standard-deviation of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_cov(mat, ddof=0, rowvar=True):
        """Compute the covariance matrix from samples of variables
        in a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_abs(mat):
        """Compute the absolte values of each entry in a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_to_numpy(mat):
        """Compute the matrix to a np.ndarray."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_argsort(mat, axis=-1):
        """Compute the indices that sort a matrix in ascending order."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_sort(mat, axis=-1):
        """Return the matrix sorted in ascending order."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_flip(mat, axis=None):
        "Reverse the order of the elements in a matrix."
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_allclose(Amat, Bmat, **kwargs):
        "Check if two matries are close"
        raise NotImplementedError

    @staticmethod
    def _la_detach(mat):
        """Detach a matrix from the computational graph.
        Override for backends that support automatic differentiation."""
        return mat

    @staticmethod
    def _la_block_cholesky_engine(L_A, L_A_inv_B, B, D, return_blocks):
        schur_comp = D-LinAlgMixin._la_multidot((L_A_inv_B.T, L_A_inv_B))
        L_S = LinAlgMixin._la_cholesky(schur_comp)
        chol_blocks = [L_A, L_A_inv_B.T, L_S]
        if return_blocks:
            return chol_blocks
        return LinAlgMixin._la_vstack([
            LinAlgMixin._la_hstack([chol_blocks[0], 0*L_A_inv_B]),
            LinAlgMixin._la_hstack([chol_blocks[1], chol_blocks[2]])])

    @staticmethod
    def _la_block_cholesky(blocks, return_blocks=False):
        A, B = blocks[0]
        D = blocks[1][1]
        L_A = LinAlgMixin._la_cholesky(A)
        L_A_inv_B = LinAlgMixin._la_solve_triangular(L_A, B)
        return LinAlgMixin._la_block_cholesky_engine(
            L_A, L_A_inv_B, B, D, return_blocks)

    @staticmethod
    def _la_get_correlation_from_covariance(cov):
        r"""
        Compute the correlation matrix from a covariance matrix
        """
        stdev_inv = 1/LinAlgMixin._la_sqrt(LinAlgMixin._la_get_diagonal(cov))
        cor = stdev_inv[None, :]*cov*stdev_inv[:, None]
        return cor

    @staticmethod
    @abstractmethod
    def _la_lstsq(Amat, Bmat):
        """Solve the linear system Ax=b."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_argmax(array):
        """Return the index of the maximum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_max(array, axis=None):
        """Return the maximum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_min(array, axis=None):
        """Return the minimum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_block(blocks):
        """Assemble 2d-array from nested lists of blocks."""
        raise NotImplementedError

    @staticmethod
    def _la_update_cholesky_factorization(L_11, A_12, A_22):
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
        if L_11 is None:
            return LinAlgMixin._la_cholesky(A_22)

        nrows, ncols = A_12.shape
        if (A_22.shape != (ncols, ncols) or L_11.shape != (nrows, nrows)):
            raise ValueError(
                "A_12 shape {0} and/or A_22 shape {1} insconsistent".format(
                    A_12.shape, A_22.shape))
        L_12 = LinAlgMixin._la_solve_triangular(L_11, A_12, lower=True)
        L_22 = LinAlgMixin._la_cholesky(
            A_22 - LinAlgMixin._la_dot(L_12.T, L_12))
        L = LinAlgMixin._la_block(
            [[L_11, LinAlgMixin._la_full((nrows, ncols), 0.)], [L_12.T, L_22]])
        return L

    @staticmethod
    @abstractmethod
    def _la_sum(matrix, axis=None):
        """Compute the sum of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_count_nonzero(matrix, axis=None):
        """Compute the number of non-zero entries in a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_array(array, **kwargs):
        """Covert an array to native format."""
        raise NotImplementedError

    @staticmethod
    def _la_cartesian_product(input_sets, elem_size=1):
        r"""
        Compute the cartesian product of an arbitray number of sets.

        The sets can consist of numbers or themselves be lists or vectors. All
        the lists or vectors of a given set must have the same number of
        entries (elem_size). However each set can have a different number
        of scalars, lists, or vectors.

        Parameters
        ----------
        input_sets
            The sets to be used in the cartesian product.

        elem_size : integer
            The size of the vectors within each set.

        Returns
        -------
        result : array (num_sets*elem_size, num_elems)
            The cartesian product. num_elems = np.prod(sizes)/elem_size,
            where sizes[ii] = len(input_sets[ii]), ii=0,..,num_sets-1.
            result.dtype will be set to the first entry of the first input_set
        """
        out = []
        # ::-1 reverse order to be backwards compatiable with old
        # function below
        for r in itertools.product(*input_sets[::-1]):
            out.append(LinAlgMixin._la_array(r, dtype=r[0].dtype))
        return LinAlgMixin._la_flip(LinAlgMixin._la_stack(out, axis=1))

    @staticmethod
    def _la_outer_product(input_sets, axis=0):
        r"""
        Construct the outer product of an arbitary number of sets.

        Examples
        --------

        .. math::

            \{1,2\}\times\{3,4\}=\{1\times3, 2\times3, 1\times4, 2\times4\} =
            \{3, 6, 4, 8\}

        Parameters
        ----------
        input_sets
            The sets to be used in the outer product

        Returns
        -------
        result : np.ndarray(np.prod(sizes))
           The outer product of the sets.
           result.dtype will be set to the first entry of the first input_set
        """
        out = LinAlgMixin._la_cartesian_product(input_sets)
        return LinAlgMixin._la_prod(out, axis=axis)

    @staticmethod
    def get_all_sample_combinations(samples1, samples2):
        r"""
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
        samples = []
        for r in itertools.product(*[samples1.T, samples2.T]):
            samples.append(LinAlgMixin._la_hstack(r))
        return LinAlgMixin._la_array(samples).T

    @staticmethod
    @abstractmethod
    def _la_eigh(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_isfinite(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _la_cond(matrix):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    def _la_jacobian(self, fun, inputs, active_opt_params):
        raise NotImplementedError
