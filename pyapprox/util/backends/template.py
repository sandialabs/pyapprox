from abc import ABC, abstractmethod
import itertools
from typing import List, Optional, Any, Tuple, Protocol, Iterable, Union


# Array = TypeVar("Array")
class Array(Protocol):
    @property
    def shape(self) -> Any: ...

    @property
    def ndim(self) -> Any: ...

    @property
    def T(self) -> "Array": ...

    def __truediv__(self, other: Union[float, "Array"]) -> "Array": ...

    def __rtruediv__(self, other: Union[float, "Array"]) -> "Array": ...

    def __mul__(self, other: Union[float, "Array"]) -> "Array": ...

    def __rmul__(self, other: Union[float, "Array"]) -> "Array": ...

    def __add__(self, other: Union[float, "Array"]) -> "Array": ...

    def __radd__(self, other: Union[float, "Array"]) -> "Array": ...

    def __sub__(self, other: Union[float, "Array"]) -> "Array": ...

    def __rsub__(self, other: Union[float, "Array"]) -> "Array": ...

    def __gt__(self, other: Union[float, "Array"]) -> Union[bool, "Array"]: ...

    def __lt__(self, other: Union[float, "Array"]) -> Union[bool, "Array"]: ...

    def __ge__(self, other: Union[float, "Array"]) -> Union[bool, "Array"]: ...

    def __le__(self, other: Union[float, "Array"]) -> Union[bool, "Array"]: ...

    def __matmul__(self, other: "Array") -> "Array": ...

    # turn off warning It is recommended for "__eq__" to work with arbitrary
    # obejcts which arises because numpy for edxample returns an array not bool
    def __eq__(self, other: "Array") -> "Array": ...  # type: ignore

    def __pow__(self, other: Union[float, int]) -> "Array": ...

    def __setitem__(self, index: Any, value: Any) -> None: ...

    def __neg__(self) -> "Array": ...

    def __getitem__(self, index: Any) -> Any: ...

    def __iter__(self) -> Any: ...


# Define the axisarg type
AxisArg = Union[int, Tuple[int, ...]]


class BackendMixin(ABC):
    """Abstract base class for linear algebra operations.

    Designed to not need a call to __init__."""

    def __init__(self) -> None:
        raise NotImplementedError("Do not instantiate this class")

    @staticmethod
    @abstractmethod
    def dot(Amat: Array, Bmat: Array) -> Array:
        """Compute the dot product of two matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def eye(
        nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> Array:
        """Return the identity matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def inv(mat: Array) -> Array:
        """Compute the inverse of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pinv(mat: Array) -> Array:
        """Compute the pseudo inverse of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def solve(Amat: Array, Bmat: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cholesky(mat: Array) -> Array:
        """Compute the cholesky factorization of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cholesky_solve(chol: Array, bvec: Array, lower: bool = True) -> Array:
        """Solve the linear equation A x = b for x,
        using the cholesky factorization of A."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def qr(mat: Array, mode: str = "complete") -> Tuple[Array, Array]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def solve_triangular(
        Amat: Array, bvec: Array, lower: bool = True
    ) -> Array:
        """Solve the linear equation A x = b for x,
        when A is a triangular matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def full(*args, dtype=None) -> Array:
        """Return a matrix with all values set to fill_value"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def zeros(*args, dtype=None) -> Array:
        """Return a matrix with all values set to zero"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ones(*args, dtype=None) -> Array:
        """Return a matrix with all values set to one"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def empty(*args) -> Array:
        """Return a matrix with uniitialized values"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def empty_like(*args, dtype=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def exp(matrix: Array) -> Array:
        """Apply exponential element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sqrt(matrix: Array) -> Array:
        """Apply sqrt element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cos(matrix: Array) -> Array:
        """Apply cos element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arccos(matrix: Array) -> Array:
        """Apply arccos element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tan(matrix: Array) -> Array:
        """Apply tan element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arctan(matrix: Array) -> Array:
        """Apply arctan element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arctan2(matrix1: Array, matrix2: Array) -> Array:
        """Apply arctan2 element wise to two matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sin(matrix: Array) -> Array:
        """Apply sin element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arcsin(matrix: Array) -> Array:
        """Apply arcasin element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cosh(matrix: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sinh(matrix: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arccosh(matrix: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arcsinh(matrix: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log(matrix: Array) -> Array:
        """Apply log element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log10(matrix: Array) -> Array:
        """Apply log base 10 element wise to a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def multidot(matrix_list: Iterable[Array]) -> Array:
        """Compute the dot product of multiple matrices."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prod(
        matrix_list: Iterable[Array],
        axis: Optional[AxisArg] = None,
    ) -> Array:
        """Compute the product of a matrix along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def hstack(arrays: Iterable[Array]) -> Array:
        """Stack arrays horizontally (column wise)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vstack(arrays: Iterable[Array]) -> Array:
        """Stack arrays vertically (row wise)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def stack(arrays: Iterable[Array], axis: AxisArg = 0) -> Array:
        """Stack arrays along a new axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def dstack(arrays: Iterable[Array]) -> Array:
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
    def ndim(mat: Array) -> int:
        """Return the dimension of the tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def repeat(mat: Array, nreps: int) -> Array:
        """Makes repeated deep copies of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tile(mat: Array, nreps: int) -> Array:
        "Construct an array by repeating A the number of times given by reps."
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cdist(Amat: Array, Bmat: Array) -> Array:
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
    def trace(mat: Array) -> Array:
        """Compute the trace of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy(mat: Array) -> Array:
        """Return a deep copy of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_diagonal(mat: Array) -> Array:
        """Return the diagonal of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def diag(array: Array, k: int = 0) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def isnan(mat: Array) -> bool:
        """Determine what entries are NAN."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def atleast1d(val: Array) -> Array:
        """Make an object at least a 1D tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def atleast2d(val: Array) -> Array:
        """Make an object at least a 2D tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reshape(mat: Array, newshape: tuple) -> Array:
        """Reshape a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def where(
        cond, array1: Optional[Array] = None, array2: Optional[Array] = None
    ) -> Array:
        """Return whether elements of a matrix satisfy a condition."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tointeger(mat: Array) -> Array:
        """Cast a matrix to integers"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def inf() -> float:
        """Return native representation of infinity."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def norm(mat: Array, axis: Optional[AxisArg] = None) -> Array:
        """Return the norm of a matrix along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def any(mat: Array, axis: Optional[AxisArg] = None) -> bool:
        """Find if any element of a matrix evaluates to True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def all(mat: Array, axis: Optional[AxisArg] = None) -> bool:
        """Find if all elements of a matrix evaluate to True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def kron(Amat: Array, Bmat: Array) -> Array:
        """Compute the Kroneker product of two matrices"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def slogdet(Amat: Array) -> float:
        """Compute the log determinant of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mean(mat: Array, axis: Optional[AxisArg] = None) -> Array:
        """Compute the mean of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def var(
        mat: Array, axis: Optional[AxisArg] = None, ddof: int = 0
    ) -> Array:
        """Compute the variance of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def std(
        mat: Array, axis: Optional[AxisArg] = None, ddof: int = 0
    ) -> Array:
        """Compute the standard-deviation of a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cov(
        mat: Array,
        ddof: int = 0,
        rowvar: bool = True,
        aweights: Optional[Array] = None,
    ) -> Array:
        """Compute the covariance matrix from samples of variables
        in a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def abs(mat: Array) -> Array:
        """Compute the absolte values of each entry in a matrix"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_numpy(mat: Array) -> Array:
        """Compute the matrix to a np.ndarray."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argsort(mat: Array, axis: AxisArg = -1) -> Array:
        """Compute the indices that sort a matrix in ascending order."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sort(mat: Array, axis: AxisArg = -1) -> Array:
        """Return the matrix sorted in ascending order."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def flip(
        mat: Array, axis: Optional[Union[Tuple[int, ...], int]] = None
    ) -> Array:
        "Reverse the order of the elements in a matrix."
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def allclose(Amat: Array, Bmat: Array, **kwargs) -> bool:
        "Check if two matries are close"
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def isclose(Amat: Array, Bmat: Array, **kwargs) -> bool:
        raise NotImplementedError

    @staticmethod
    def detach(mat: Array) -> Array:
        """Detach a matrix from the computational graph.
        Override for backends that support automatic differentiation."""
        return mat

    @classmethod
    def block_cholesky_engine(cls, L_A, L_A_inv_B, B, D, return_blocks):
        schur_comp = D - cls.multidot((L_A_inv_B.T, L_A_inv_B))
        L_S = cls.cholesky(schur_comp)
        chol_blocks = [L_A, L_A_inv_B.T, L_S]
        if return_blocks:
            return chol_blocks
        return cls.vstack(
            [
                cls.hstack([chol_blocks[0], 0 * L_A_inv_B]),
                cls.hstack([chol_blocks[1], chol_blocks[2]]),
            ]
        )

    @classmethod
    def block_cholesky(cls, blocks, return_blocks: bool = False):
        A, B = blocks[0]
        D = blocks[1][1]
        L_A = cls.cholesky(A)
        L_A_inv_B = cls.solve_triangular(L_A, B)
        return cls.block_cholesky_engine(L_A, L_A_inv_B, B, D, return_blocks)

    @classmethod
    def covariance_to_correlation(cls, cov: Array) -> Array:
        r"""
        Compute the correlation matrix from a covariance matrix
        """
        stdev_inv = 1.0 / cls.sqrt(cls.get_diagonal(cov))
        cor = stdev_inv[None, :] * cov * stdev_inv[:, None]
        return cor

    @staticmethod
    @abstractmethod
    def lstsq(Amat: Array, Bmat: Array) -> Array:
        """Solve the linear system Ax=b."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argmax(array: Array) -> float:
        """Return the index of the maximum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def argmin(array: Array) -> float:
        """Return the index of the minimum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def max(array: Array, axis: Optional[AxisArg] = None) -> Array:
        """Return the maximum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def maximum(array1: Array, array2: Array) -> float:
        """Return the elementwise maximum values of two arrays."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def minimum(array1: Array, array2: Array) -> Array:
        """Return the elementwise minimum values of two arrays."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def min(array: Array, axis=None) -> Array:
        """Return the minimum value in an array."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def block(blocks) -> Array:
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
        if A_22.shape != (ncols, ncols) or L_11.shape != (nrows, nrows):
            raise ValueError(
                "A_12 shape {0} and/or A_22 shape {1} insconsistent".format(
                    A_12.shape, A_22.shape
                )
            )
        L_12 = cls.solve_triangular(L_11, A_12, lower=True)
        if A_22.shape[0] == 1 and A_22 - cls.dot(L_12.T, L_12) < 1e-12:
            return L_11, False
        try:
            L_22 = cls.cholesky(A_22 - cls.dot(L_12.T, L_12))
        except:
            return L_11, False
        L = cls.block([[L_11, cls.full((nrows, ncols), 0.0)], [L_12.T, L_22]])
        return L, True

    @staticmethod
    @abstractmethod
    def sum(matrix: Array, axis=None) -> Array:
        """Compute the sum of a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def count_nonzero(matrix: Array, axis: Optional[AxisArg] = None) -> int:
        """Compute the number of non-zero entries in a matrix."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def array(array: Iterable, **kwargs) -> Array:
        """Covert an array to native format."""
        raise NotImplementedError

    @classmethod
    def cartesian_product(cls, input_sets: List[Array]) -> Array:
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
        # function
        for r in itertools.product(*input_sets[::-1]):
            out.append(cls.array(r, dtype=r[0].dtype))
        return cls.flip(cls.stack(out, axis=1), axis=(0,))

    @classmethod
    def outer_product(cls, input_sets, axis=0) -> Array:
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
    def eigh(matrix: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def svd(matrix):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def isfinite(matrix: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cond(matrix: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def rank(matrix: Array) -> int:
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
    def jvp(fun, params, vec):
        raise NotImplementedError

    @staticmethod
    def hvp(fun, params, vec):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def up(matrix: Array, indices: Array, submatrix: Array, axis: AxisArg = 0):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def moveaxis(array: Array, source, destination):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def floor(array: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ceil(array: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def asarray(array: Iterable) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def unique(array: Array, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def delete(array: Array, obj, axis=None):
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
    def jvp_implemented() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def hvp_implemented() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def meshgrid(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tanh(array: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def diff(array: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def int_dtype():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cumsum(array: Array, axis: AxisArg = 0, **kwargs):
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
    def array_type():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def real(array: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def imag(array: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def round(array):
        raise NotImplementedError

    def assert_isarray(cls, array: Array):
        if not isinstance(array, cls.array_type()):
            raise ValueError(
                "array must be an instance of {0}".format(cls.array_type())
            )

    @staticmethod
    @abstractmethod
    def flatten(array: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def double_type():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def bool_type():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gammaln(array: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def split(mat: Array, splits: Array, axis: AxisArg = 0):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def chunks(mat: Array, nchunks: int, axis: AxisArg = 0) -> List[Array]:
        raise NotImplementedError

    @staticmethod
    def isbackend():
        return True

    @staticmethod
    @abstractmethod
    def sign(mat: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_scalar_array(array: Array) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def quantile(array: Array, q: float, axis=None) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tril(array: Array, k: int = 0) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tril_indices(n: int, k: int = 0, m: Optional[int] = None) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def triu(array: Array, k: int = 0) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def triu_indices(n: int, k: int = 0, m: Optional[int] = None) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def digamma(array: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def erf(array: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def erfinv(array: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reshape_fortran(array: Array, shape) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gammainc(a: Array, x: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def factorial(array: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def clip(array: Array, minval: float, maxval: float) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def swapaxes(array: Array, axis1: int, axis2: int) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def block_diag(arrays: List[Array]) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def searchsorted(array: Array, values: Array, side: str = "left") -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fft(mat: Array, axis=None, **kwargs) -> Array:
        """Compute fast Fourier transform of mat."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ifft(mat: Array, axis=None, **kwargs) -> Array:
        """Compute inverse fast Fourier transform of mat."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fftshift(mat: Array, axis=None, **kwargs) -> Array:
        """Re-index FFT so that mode 0 is in the center"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ifftshift(mat: Array, axis=None, **kwargs) -> Array:
        """Re-index inverse FFT"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cfloat():
        """Returns native complex dtype (64-bit for each part)"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def transpose(mat: Array, axis=None) -> Array:
        """Returns transpose of mat along axis"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def size(mat: Array) -> int:
        """Returns number of elements in mat"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def nan():
        """Return native representation of nan."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_slices(mat: Array, slices) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def concatenate(mats: List[Array], axis: AxisArg = 0) -> Array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def assert_allclose(
        actual: Array,
        desired: Array,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> bool:
        raise NotImplementedError
