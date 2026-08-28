r"""The metric a projection is taken in.

Every empirical basis -- a KLE from field samples, a POD basis, a PCA
encoder -- answers the same eigenproblem,

.. math:: C M \phi_i = \lambda_i \phi_i, \qquad \phi_i^T M \phi_j = \delta_{ij}

and what varies between them is rarely the covariance :math:`C`. It is
the metric :math:`M`: the identity, a diagonal of quadrature weights, or
an assembled FEM mass matrix. That distinction decides whether a
projection is a best approximation, because the coordinates of a state
are :math:`z = \phi^T M (s - \bar{s})` and *not* :math:`\phi^T (s -
\bar{s})` unless :math:`M = I`. Pairing an :math:`M`-orthonormal basis
with a Euclidean projection is idempotent in neither direction, and it
silently returns something that is not the nearest point in the
subspace.

The same concept currently appears across the library as a 1-D weight
array, an assembled matrix argument, and nothing at all. This module is
the one object, so a basis and the metric it was built in can travel
together instead of being two arguments a caller may mismatch.

It belongs in ``util`` rather than ``surrogates`` because the
``util-isolation`` import contract forbids ``util -> surrogates`` and
nothing here needs anything from that layer -- which leaves it reachable
from ``surrogates``, ``pde`` and ``ode`` alike.

**There is deliberately no square root.** A symmetrized formulation
wants :math:`M^{1/2}`, but the metric that motivates this abstraction --
a sparse FEM mass matrix -- cannot supply one without CHOLMOD or
densifying, and densifying is the :math:`O(N^2)` cost the design exists
to avoid. Algorithms needing a weighted basis should use the method of
snapshots, which asks only for :meth:`apply`.
"""

from typing import Any, Generic, Optional, Protocol, runtime_checkable

from scipy.sparse import csc_matrix, issparse, spmatrix
from scipy.sparse.linalg import SuperLU, splu

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class InnerProductProtocol(Protocol, Generic[Array]):
    r"""A weighted inner product :math:`\langle x, y \rangle_M = x^T M y`.

    Arrays follow the samples-are-columns convention: a single vector is
    ``(nstates, 1)`` and a batch is ``(nstates, ncols)``.
    """

    def nstates(self) -> int:
        """Dimension of the space the metric acts on."""
        ...

    def apply(self, x: Array) -> Array:
        """Return ``M x``, shape ``(nstates, ncols)``."""
        ...

    def dot(self, x: Array, y: Array) -> Array:
        """Return ``x^T M y``, shape ``(ncols_x, ncols_y)``."""
        ...

    def norm(self, x: Array) -> Array:
        """Column-wise ``sqrt(x^T M x)``, shape ``(ncols,)``."""
        ...

    def is_diagonal(self) -> bool:
        """Whether ``M`` is diagonal.

        Diagonal metrics admit a cheap elementwise square root, so a
        solver may symmetrize rather than fall back to the method of
        snapshots. Callers gate on this rather than on the concrete
        type, so a future banded or block-diagonal metric answering
        ``True`` needs no change at the call site.
        """
        ...


def m_orthonormality_drift(
    basis: Array,
    inner_product: InnerProductProtocol[Array],
    bkd: Backend[Array],
) -> float:
    r"""Return :math:`\|V^T M V - I\|_F`, zero iff ``basis`` is
    :math:`M`-orthonormal.

    The reduced mass matrix is the identity exactly when this vanishes,
    which is the condition reduced physics and basis builders check
    before trusting a projection.
    """
    ncodes = int(basis.shape[1])
    gram = bkd.dot(basis.T, inner_product.apply(basis))
    return bkd.to_float(bkd.norm(gram - bkd.eye(ncodes)))


class EuclideanInnerProduct(Generic[Array]):
    r"""The unweighted inner product, :math:`M = I`.

    ``apply`` returns its argument untouched rather than multiplying by
    an identity, so using this in place of a special-cased ``metric is
    None`` branch costs nothing.
    """

    def __init__(self, nstates: int, bkd: Backend[Array]) -> None:
        if nstates < 1:
            raise ValueError(f"nstates={nstates} must be positive")
        self._nstates = nstates
        self._bkd = bkd

    def nstates(self) -> int:
        return self._nstates

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def apply(self, x: Array) -> Array:
        return x

    def dot(self, x: Array, y: Array) -> Array:
        return self._bkd.dot(x.T, y)

    def norm(self, x: Array) -> Array:
        return self._bkd.sqrt(self._bkd.sum(x * x, axis=0))

    def is_diagonal(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self._nstates})"


class DiagonalInnerProduct(Generic[Array]):
    r"""A diagonal metric, :math:`M = \mathrm{diag}(w)`.

    This is what a ``quad_weights`` argument has always meant: the
    quadrature rule that turns a sum over mesh points into an integral,
    so that eigenvectors are orthonormal in :math:`L^2` rather than in
    the Euclidean norm of whatever discretization happened to be used.
    Without it a basis is biased toward wherever the mesh is refined.

    Parameters
    ----------
    weights : Array
        Shape ``(nstates,)``, strictly positive. Zero or negative
        weights would make the form indefinite, so it would no longer
        be an inner product and the "orthonormal" basis built in it
        would not be a basis.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, weights: Array, bkd: Backend[Array]) -> None:
        if weights.ndim != 1:
            raise ValueError(
                f"weights must be 1D (nstates,), got ndim={weights.ndim}"
            )
        if not bool(bkd.all_bool(weights > 0.0)):
            raise ValueError(
                "weights must be strictly positive; a non-positive weight "
                "makes x^T M x indefinite, so it is not an inner product"
            )
        self._weights = weights
        self._bkd = bkd

    def nstates(self) -> int:
        return int(self._weights.shape[0])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def weights(self) -> Array:
        """Return the diagonal, shape ``(nstates,)``."""
        return self._weights

    def apply(self, x: Array) -> Array:
        return self._weights[:, None] * x

    def dot(self, x: Array, y: Array) -> Array:
        return self._bkd.dot(x.T, self.apply(y))

    def norm(self, x: Array) -> Array:
        return self._bkd.sqrt(self._bkd.sum(x * self.apply(x), axis=0))

    def is_diagonal(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"


class MassInnerProduct(Generic[Array]):
    r"""A metric weighted by a sparse SPD matrix, typically an FEM mass
    matrix.

    The matrix stays scipy-sparse and :meth:`apply` round-trips through
    numpy, so torch autograd is not preserved across it. That follows
    the ``ConstantSparseMassMatrix`` precedent in :mod:`pyapprox.ode`
    and is acceptable for the same reason: a sparse metric comes from
    Galerkin assembly, where it is constant data rather than something
    differentiated through.

    Parameters
    ----------
    matrix : scipy sparse matrix
        Shape ``(nstates, nstates)``, symmetric positive definite. Only
        the shape is checked; verifying definiteness would cost a
        factorization on every construction.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, matrix: spmatrix, bkd: Backend[Array]) -> None:
        if not issparse(matrix):
            raise ValueError(
                "MassInnerProduct requires a scipy sparse matrix, got "
                f"{type(matrix).__name__}. Use DiagonalInnerProduct for "
                "quadrature weights, or EuclideanInnerProduct for none."
            )
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError(
                f"matrix must be square, got shape {matrix.shape}"
            )
        self._matrix = matrix
        self._bkd = bkd
        self._lu: Optional[SuperLU] = None

    def nstates(self) -> int:
        return int(self._matrix.shape[0])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def matrix(self) -> Any:
        """Return the sparse weight matrix."""
        return self._matrix

    def apply(self, x: Array) -> Array:
        return self._bkd.asarray(self._matrix @ self._bkd.to_numpy(x))

    def dot(self, x: Array, y: Array) -> Array:
        return self._bkd.dot(x.T, self.apply(y))

    def norm(self, x: Array) -> Array:
        return self._bkd.sqrt(self._bkd.sum(x * self.apply(x), axis=0))

    def is_diagonal(self) -> bool:
        return False

    def solve(self, x: Array) -> Array:
        """Return ``M^{-1} x``, reusing one cached sparse LU.

        Needed by dual-norm error functionals. Deliberately outside
        :class:`InnerProductProtocol`: a metric is defined by the form it
        induces, and requiring every implementation to invert itself
        would exclude ones that legitimately cannot.
        """
        if self._lu is None:
            self._lu = splu(csc_matrix(self._matrix))
        return self._bkd.asarray(self._lu.solve(self._bkd.to_numpy(x)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"
