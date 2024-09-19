from abc import ABC, abstractmethod
import itertools


class LinAlgMixin(ABC):
    """Abstract base class for linear algebra operations.

    Designed to not need a call to __init__."""

    def __init__(self):
        raise NotImplementedError("Do not instantiate this class")

    @staticmethod
    @abstractmethod
    def dot(Amat, Bmat):
        """Compute the dot product of two matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def eye(nrows: int, dtype=None):
        """Return the identity matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def inv(mat):
        """Compute the inverse of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pinv(mat):
        """Compute the pseudo inverse of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def solve(Amat, Bmat):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cholesky(mat):
        """Compute the cholesky factorization of a matrix."""
        raise NotImplementedError

    @staticmethod
    def cholesky_solve(chol, bvec, lower: bool = True):
        """Solve the linear equation A x = b for x,
        using the cholesky factorization of A."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def solve_triangular(Amat, bvec, lower: bool = True):
        """Solve the linear equation A x = b for x,
        when A is a triangular matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def full(*args, dtype=None):
        """Return a matrix with all values set to fill_value"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def zeros(*args, dtype=None):
        """Return a matrix with all values set to zero"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ones(*args, dtype=None):
        """Return a matrix with all values set to one"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def empty(*args):
        """Return a matrix with uniitialized values"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def exp(matrix):
        """Apply exponential element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sqrt(matrix):
        """Apply sqrt element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cos(matrix):
        """Apply cos element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arccos(matrix):
        """Apply arccos element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tan(matrix):
        """Apply tan element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arctan(matrix):
        """Apply arctan element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sin(matrix):
        """Apply sin element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arcsin(matrix):
        """Apply arcasin element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cosh(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sinh(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arccosh(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arcsinh(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log(matrix):
        """Apply log element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log10(matrix):
        """Apply log base 10 element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def multidot(matrix_list):
        """Compute the dot product of multiple matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prod(matrix_list, axis=None):
        """Compute the product of a matrix along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def hstack(arrays):
        """Stack arrays horizontally (column wise)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vstack(arrays):
        """Stack arrays vertically (row wise)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def stack(arrays, axis=0):
        """Stack arrays along a new axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def dstack(arrays):
        """Stack arrays along third axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arange(*args):
        """Return equidistant values within a given interval."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def linspace(*args, **kwargs):
        """Return equidistant values within a given interval."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def logspace(*args):
        """Return equidistant values within a given interval in logspace."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ndim(mat) -> int:
        """Return the dimension of the tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def repeat(mat, nreps):
        """Makes repeated deep copies of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cdist(Amat, Bmat):
        """
        Return cthe euclidean distance between elements of two matrices.
        Should be equivalent to
        scipy.spatial.distance.cdist(Amat, Bmat, metric="euclidean")
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def einsum(*args):
        """Compute Einstein summation on two tensors."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trace(mat):
        """Compute the trace of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy(mat):
        """Return a deep copy of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_diagonal(mat):
        """Return the diagonal of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def diag(array, k=0):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def isnan(mat):
        """Determine what entries are NAN."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def atleast1d(val, dtype=None):
        """Make an object at least a 1D tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def atleast2d(val, dtype=None):
        """Make an object at least a 2D tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reshape(mat, newshape):
        """Reshape a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def where(cond):
        """Return whether elements of a matrix satisfy a condition."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tointeger(mat):
        """Cast a matrix to integers"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def inf():
        """Return native representation of infinity."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def norm(mat, axis=None):
        """Return the norm of a matrix along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def any(mat, axis=None):
        """Find if any element of a matrix evaluates to True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def all(mat, axis=None):
        """Find if all elements of a matrix evaluate to True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def kron(Amat, Bmat):
        """Compute the Kroneker product of two matrices"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def slogdet(Amat):
        """Compute the log determinant of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mean(mat, axis=None):
        """Compute the mean of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def std(mat, axis=None, ddof=0):
        """Compute the standard-deviation of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cov(mat, ddof=0, rowvar=True):
        """Compute the covariance matrix from samples of variables
        in a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def abs(mat):
        """Compute the absolte values of each entry in a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_numpy(mat):
        """Compute the matrix to a np.ndarray."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argsort(mat, axis=-1):
        """Compute the indices that sort a matrix in ascending order."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sort(mat, axis=-1):
        """Return the matrix sorted in ascending order."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def flip(mat, axis=None):
        "Reverse the order of the elements in a matrix."
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def allclose(Amat, Bmat, **kwargs):
        "Check if two matries are close"
        raise NotImplementedError

    @staticmethod
    def detach(mat):
        """Detach a matrix from the computational graph.
        Override for backends that support automatic differentiation."""
        return mat

    @classmethod
    def block_cholesky_engine(cls, L_A, L_A_inv_B, B, D, return_blocks):
        schur_comp = D-cls.multidot((L_A_inv_B.T, L_A_inv_B))
        L_S = cls.cholesky(schur_comp)
        chol_blocks = [L_A, L_A_inv_B.T, L_S]
        if return_blocks:
            return chol_blocks
        return cls.vstack([
            cls.hstack([chol_blocks[0], 0*L_A_inv_B]),
            cls.hstack([chol_blocks[1], chol_blocks[2]])])

    @classmethod
    def block_cholesky(cls, blocks, return_blocks=False):
        A, B = blocks[0]
        D = blocks[1][1]
        L_A = cls.cholesky(A)
        L_A_inv_B = cls.solve_triangular(L_A, B)
        return cls.block_cholesky_engine(
            L_A, L_A_inv_B, B, D, return_blocks)

    @classmethod
    def get_correlation_from_covariance(cls, cov):
        r"""
        Compute the correlation matrix from a covariance matrix
        """
        stdev_inv = 1/cls.sqrt(cls.get_diagonal(cov))
        cor = stdev_inv[None, :]*cov*stdev_inv[:, None]
        return cor

    @staticmethod
    @abstractmethod
    def lstsq(Amat, Bmat):
        """Solve the linear system Ax=b."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argmax(array):
        """Return the index of the maximum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argmin(array):
        """Return the index of the minimum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def max(array, axis=None):
        """Return the maximum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def min(array, axis=None):
        """Return the minimum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def block(blocks):
        """Assemble 2d-array from nested lists of blocks."""
        raise NotImplementedError

    @classmethod
    def update_cholesky_factorization(cls, L_11, A_12, A_22):
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
            return cls.cholesky(A_22), True

        nrows, ncols = A_12.shape
        if (A_22.shape != (ncols, ncols) or L_11.shape != (nrows, nrows)):
            raise ValueError(
                "A_12 shape {0} and/or A_22 shape {1} insconsistent".format(
                    A_12.shape, A_22.shape))
        L_12 = cls.solve_triangular(L_11, A_12, lower=True)
        if A_22.shape[0] == 1 and A_22 - cls.dot(L_12.T, L_12) < 1e-12:
            return L_11, False
        try:
            L_22 = cls.cholesky(
                A_22 - cls.dot(L_12.T, L_12))
        except:
            return L_11, False
        L = cls.block(
            [[L_11, cls.full((nrows, ncols), 0.)], [L_12.T, L_22]])
        return L, True

    @staticmethod
    @abstractmethod
    def sum(matrix, axis=None):
        """Compute the sum of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def count_nonzero(matrix, axis=None):
        """Compute the number of non-zero entries in a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def array(array, **kwargs):
        """Covert an array to native format."""
        raise NotImplementedError

    @classmethod
    def cartesian_product(cls, input_sets, elem_size=1):
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
        if len(input_sets) == 1:
            return cls.stack(input_sets, axis=0)
        out = []
        # ::-1 reverse order to be backwards compatiable with old
        # function below
        for r in itertools.product(*input_sets[::-1]):
            out.append(cls.array(r, dtype=r[0].dtype))
        return cls.flip(cls.stack(out, axis=1), axis=(0,))

    @classmethod
    def outer_product(cls, input_sets, axis=0):
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
        out = cls.cartesian_product(input_sets)
        return cls.prod(out, axis=axis)

    @classmethod
    def get_all_sample_combinations(cls, samples1, samples2):
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
            samples.append(cls.hstack(r))
        return cls.array(samples).T

    @staticmethod
    @abstractmethod
    def eigh(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def isfinite(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cond(matrix):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)

    @staticmethod
    def jacobian(fun, params):
        raise NotImplementedError

    @staticmethod
    def grad(fun, params):
        raise NotImplementedError

    @staticmethod
    def hessian(fun, params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def up(matrix, indices, submatrix, axis=0):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def moveaxis(array, source, destination):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def floor(array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def asarray(array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def unique(array, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def delete(array, obj, axis=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def jacobian_implemented() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def hessian_implemented() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def meshgrid(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tanh(array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def diff(array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def int():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cumsum(array, axis=0, **kwargs):
        raise NotImplementedError

    @staticmethod
    def bkd_equal(bkd1, bkd2):
        # has to be a comparison that does not require instantiating class
        return bkd1.__name__ == bkd2.__name__

    @staticmethod
    @abstractmethod
    def complex_dtype():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def real(array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def imag(array):
        raise NotImplementedError
