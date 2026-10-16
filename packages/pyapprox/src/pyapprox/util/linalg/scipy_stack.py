r"""The NumPy/SciPy linear-algebra stack.

A single-process stack whose vectors are NumPy arrays of shape
``(size, ncols)``: a single vector is ``(size, 1)``, never 1-D. It is a
bound implementation: it works on NumPy arrays only, writes in place
whenever ``out`` is given, and is not differentiable.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.linalg.operators.layout import Layout

_Vector = NDArray[Any]


class NumpyVectorSpace:
    """Blocks of vectors stored as NumPy arrays of shape ``(size, ncols)``.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.VectorSpaceProtocol`
    with NumPy arrays as both the vector and the reduction type. Every
    operation given ``out`` writes into it and returns it.

    Parameters
    ----------
    layout : Layout
        The space of the vectors.

    Raises
    ------
    ValueError
        From any operation, if a vector is not 2-D with ``layout.size``
        rows and the layout's dtype, if an ``out`` has a different shape
        from the result, or if ``out`` aliases an operand it may not.
    """

    def __init__(self, layout: Layout) -> None:
        self._layout = layout
        self._dtype = np.dtype(layout.dtype)

    def layout(self) -> Layout:
        return self._layout

    def _check(self, x: _Vector, name: str) -> None:
        if x.ndim != 2 or x.shape[0] != self._layout.size:
            raise ValueError(
                f"{name} must have shape ({self._layout.size}, ncols), got {x.shape}"
            )
        if x.dtype != self._dtype:
            raise ValueError(f"{name} must have dtype {self._dtype}, got {x.dtype}")

    def _check_out(
        self,
        out: _Vector,
        ncols: int,
        accumulator: Optional[_Vector],
        others: tuple[_Vector, ...],
    ) -> None:
        """Check ``out``: it may be ``accumulator`` itself, and must share
        no storage with anything else (including ``others`` that happen
        to be the accumulator)."""
        self._check(out, "out")
        if out.shape[1] != ncols:
            raise ValueError(
                f"out must have shape ({self._layout.size}, {ncols}), got {out.shape}"
            )
        operands = others
        if accumulator is not None and out is not accumulator:
            operands = (*others, accumulator)
        for operand in operands:
            if np.may_share_memory(out, operand):
                raise ValueError("out shares storage with an operand it may not alias")

    def zeros(self, ncols: int = 1) -> _Vector:
        return np.zeros((self._layout.size, ncols), dtype=self._dtype)

    def zeros_like(self, x: _Vector) -> _Vector:
        self._check(x, "x")
        return np.zeros_like(x)

    def ncols(self, x: _Vector) -> int:
        self._check(x, "x")
        return int(x.shape[1])

    def copy(self, x: _Vector, out: Optional[_Vector] = None) -> _Vector:
        self._check(x, "x")
        if out is None:
            return x.copy()
        self._check_out(out, x.shape[1], None, (x,))
        np.copyto(out, x)
        return out

    def scale(self, a: float, x: _Vector, out: Optional[_Vector] = None) -> _Vector:
        self._check(x, "x")
        if out is None:
            return a * x
        self._check_out(out, x.shape[1], x, ())
        np.multiply(x, a, out=out)
        return out

    def axpby(
        self,
        a: float,
        x: _Vector,
        b: float,
        y: _Vector,
        out: Optional[_Vector] = None,
    ) -> _Vector:
        self._check(x, "x")
        self._check(y, "y")
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}"
            )
        if out is None:
            return a * x + b * y
        self._check_out(out, x.shape[1], y, (x,))
        # out is y or disjoint from both, so scaling y first is safe.
        np.multiply(y, b, out=out)
        out += a * x
        return out

    def dot(self, x: _Vector, y: _Vector) -> _Vector:
        self._check(x, "x")
        self._check(y, "y")
        return np.conj(x).T @ y

    def norm(self, x: _Vector) -> _Vector:
        self._check(x, "x")
        # numpy's stubs type a norm along an axis as Any.
        norm: _Vector = np.linalg.norm(x, axis=0, keepdims=True)
        return norm

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._layout!r})"
