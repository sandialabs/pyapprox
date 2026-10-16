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
        The space of the vectors. Its dtype must be ``"float64"``.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If the layout's dtype is not ``"float64"``. From any operation,
        if a vector is not 2-D with ``layout.size`` rows, or if ``out``
        is an operand it may not alias.
    """

    def __init__(self, layout: Layout, bkd: Backend[Array]) -> None:
        if layout.dtype != "float64":
            raise ValueError(
                "BackendVectorSpace supports float64 layouts only, got "
                f"{layout.dtype!r}"
            )
        self._layout = layout
        self._bkd = bkd
        self._dtype: Any = bkd.double_dtype()

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
