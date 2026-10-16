r"""A dense linear-algebra stack generic over the computational backend.

Vectors are backend arrays of shape ``(size, ncols)``. Every operation
is a backend array expression, so a torch graph through the operands is
preserved; for that reason ``out`` is ignored and every result is a new
array (writing in place would break automatic differentiation). Meant
for small dense problems and as a verification oracle, not for speed.
"""

from typing import Any, Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators.layout import Layout
from pyapprox.util.linalg.operators.protocols import OperatorProperties

_SYMMETRY_RTOL = 1e-12


def _require_backend_dtype(layout: Layout, bkd: Backend[Array], what: str) -> Any:
    """Return the backend's configured dtype, refusing a layout of another.

    The dtype name is read from a one-element array, so any backend that
    reports its default dtype works, whatever its dtype objects are.
    """
    dtype = bkd.default_dtype()
    name = bkd.to_numpy(bkd.zeros((1,), dtype=dtype)).dtype.name
    if layout.dtype != name:
        raise ValueError(
            f"{what}: layout dtype {layout.dtype!r} differs from the backend's "
            f"configured dtype {name!r}; configure the backend or change the layout"
        )
    return dtype


class BackendVectorSpace(Generic[Array]):
    """Blocks of vectors stored as backend arrays of shape ``(size, ncols)``.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.VectorSpaceProtocol`
    with backend arrays as both the vector and the reduction type.
    ``out`` is accepted and ignored, so callers must use the returned
    value. The aliasing rule is still enforced, by identity, so code
    tested here does not break on a stack that writes in place.

    Parameters
    ----------
    layout : Layout
        The space of the vectors. Its dtype must be the backend's
        configured dtype (``bkd.default_dtype()``).
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If the layout's dtype is not the backend's. From any operation,
        if a vector is not 2-D with ``layout.size`` rows, or if ``out``
        is an operand it may not alias.
    """

    def __init__(self, layout: Layout, bkd: Backend[Array]) -> None:
        self._dtype: Any = _require_backend_dtype(layout, bkd, "BackendVectorSpace")
        self._layout = layout
        self._bkd = bkd

    def layout(self) -> Layout:
        return self._layout

    def _check(self, x: Array, name: str) -> None:
        if x.ndim != 2 or x.shape[0] != self._layout.size:
            raise ValueError(
                f"{name} must have shape ({self._layout.size}, ncols), "
                f"got {tuple(x.shape)}"
            )

    @staticmethod
    def _check_out(out: Optional[Array], others: tuple[Array, ...]) -> None:
        """Refuse an ``out`` that is one of ``others``, the operands that
        are not accumulated into."""
        if out is not None and any(o is out for o in others):
            raise ValueError("out is an operand it may not alias")

    def zeros(self, ncols: int = 1) -> Array:
        return self._bkd.zeros((self._layout.size, ncols), dtype=self._dtype)

    def zeros_like(self, x: Array) -> Array:
        self._check(x, "x")
        return self._bkd.zeros((self._layout.size, x.shape[1]), dtype=self._dtype)

    def ncols(self, x: Array) -> int:
        self._check(x, "x")
        return int(x.shape[1])

    def copy(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, "x")
        self._check_out(out, (x,))
        return self._bkd.copy(x)

    def scale(self, a: float, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, "x")
        self._check_out(out, ())
        return a * x

    def axpby(
        self, a: float, x: Array, b: float, y: Array, out: Optional[Array] = None
    ) -> Array:
        self._check(x, "x")
        self._check(y, "y")
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {tuple(x.shape)} and "
                f"{tuple(y.shape)}"
            )
        self._check_out(out, (x,))
        return a * x + b * y

    def dot(self, x: Array, y: Array) -> Array:
        self._check(x, "x")
        self._check(y, "y")
        return self._bkd.dot(x.T, y)

    def norm(self, x: Array) -> Array:
        self._check(x, "x")
        return self._bkd.sqrt(self._bkd.sum(x * x, axis=0, keepdims=True))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._layout!r})"


class DenseFormat(Generic[Array]):
    """Dense backend storage, the format of :class:`BackendDenseOperator`.

    Parameters
    ----------
    matrix : Array
        The stored matrix.
    """

    def __init__(self, matrix: Array) -> None:
        self._matrix = matrix

    def matrix(self) -> Array:
        """The stored matrix. Not to be modified: operators are immutable."""
        return self._matrix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self._matrix.shape)})"


class BackendDenseOperator(Generic[Array]):
    """A linear operator stored as a dense backend array.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.DiagonalCapableProtocol`
    with backend arrays as vectors and :class:`DenseFormat` as its format.
    ``apply`` is a backend product, so a torch graph through the input
    (and through the matrix, if it carries one) is preserved; ``out`` is
    ignored, as in :class:`BackendVectorSpace`.

    Parameters
    ----------
    matrix : Array
        Shape ``(range.size, domain.size)``.
    domain : Layout
        The space of the input of ``apply``. Its dtype, and the range's,
        must be the backend's configured dtype.
    bkd : Backend[Array]
        Computational backend.
    range : Layout, optional
        The space of the output of ``apply``. Defaults to ``domain``.
    properties : OperatorProperties, optional
        Declared properties. ``symmetric``, ``diagonal`` and ``identity``
        are verified; ``spd`` is not (that would cost a factorization).

    Raises
    ------
    ValueError
        If the shape disagrees with the layouts, a layout's dtype is not
        the backend's,
        a symmetric operator's domain and range differ, or a verified
        property does not hold.
    """

    def __init__(
        self,
        matrix: Array,
        domain: Layout,
        bkd: Backend[Array],
        range: Optional[Layout] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> None:
        self._domain = domain
        self._range = domain if range is None else range
        self._properties = OperatorProperties() if properties is None else properties
        self._bkd = bkd
        self._dtype: Any = _require_backend_dtype(
            self._domain, bkd, "BackendDenseOperator"
        )
        _require_backend_dtype(self._range, bkd, "BackendDenseOperator")
        shape = tuple(matrix.shape)
        if shape != (self._range.size, self._domain.size):
            raise ValueError(
                f"matrix must have shape (range.size, domain.size) = "
                f"({self._range.size}, {self._domain.size}), got {shape}"
            )
        if self._properties.symmetric:
            self._domain.require_same(self._range, "range of a symmetric operator")
        self._verify_properties(matrix)
        self._format = DenseFormat(matrix)

    def _verify_properties(self, matrix: Array) -> None:
        if matrix.shape[0] == 0:
            return
        bkd, props = self._bkd, self._properties
        # Checks only: converting the reductions to floats detaches
        # nothing that is returned.
        if props.symmetric:
            asymmetry = bkd.to_float(bkd.max(bkd.abs(matrix - matrix.T)))
            scale = bkd.to_float(bkd.max(bkd.abs(matrix)))
            if asymmetry > _SYMMETRY_RTOL * max(scale, 1.0):
                raise ValueError(
                    f"matrix declared symmetric is not: max |A - A^T| = "
                    f"{asymmetry:.3e} against max |A| = {scale:.3e}"
                )
        if props.diagonal:
            off_diagonal = matrix - bkd.diag(bkd.diag(matrix))
            if bkd.to_float(bkd.max(bkd.abs(off_diagonal))) > 0.0:
                raise ValueError("matrix declared diagonal has off-diagonal nonzeros")
        if props.identity:
            n = matrix.shape[0]
            deviation = matrix - bkd.eye(n, dtype=self._dtype)
            if bkd.to_float(bkd.max(bkd.abs(deviation))) > 0.0:
                raise ValueError(
                    "matrix declared identity has a diagonal entry other than 1"
                )

    @classmethod
    def identity(
        cls, layout: Layout, bkd: Backend[Array]
    ) -> "BackendDenseOperator[Array]":
        """The identity on ``layout``, stored densely so it combines with
        other dense operators without leaving the format."""
        return cls(
            bkd.eye(layout.size, dtype=bkd.default_dtype()),
            layout,
            bkd,
            properties=OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True
            ),
        )

    @classmethod
    def from_diagonal(
        cls, d: Array, layout: Layout, bkd: Backend[Array]
    ) -> "BackendDenseOperator[Array]":
        """The diagonal operator with diagonal ``d``, a single vector of
        ``layout``, stored densely. Declared ``spd`` when ``d`` is
        positive."""
        if tuple(d.shape) != (layout.size, 1):
            raise ValueError(
                f"d must be a single vector of shape ({layout.size}, 1), "
                f"got {tuple(d.shape)}"
            )
        spd = layout.size == 0 or bkd.to_float(bkd.min(d)) > 0.0
        return cls(
            bkd.diag(bkd.reshape(d, (layout.size,))),
            layout,
            bkd,
            properties=OperatorProperties(symmetric=True, spd=spd, diagonal=True),
        )

    def domain(self) -> Layout:
        return self._domain

    def range(self) -> Layout:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> DenseFormat[Array]:
        return self._format

    @staticmethod
    def _check_apply(x: Array, out: Optional[Array], src: Layout) -> None:
        if x.ndim != 2 or x.shape[0] != src.size:
            raise ValueError(
                f"input must have shape ({src.size}, ncols), got {tuple(x.shape)}"
            )
        if out is x:
            raise ValueError("out must not alias the input of apply")

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check_apply(x, out, self._domain)
        return self._bkd.dot(self._format.matrix(), x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        self._check_apply(y, out, self._range)
        return self._bkd.dot(self._format.matrix().T, y)

    def diagonal(self) -> Array:
        self._domain.require_same(
            self._range, "range of an operator asked for its diagonal"
        )
        return self._bkd.reshape(
            self._bkd.diag(self._format.matrix()), (self._domain.size, 1)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain={self._domain!r}, "
            f"range={self._range!r}, {self._properties!r})"
        )
