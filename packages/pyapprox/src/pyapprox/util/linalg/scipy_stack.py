r"""The NumPy/SciPy linear-algebra stack.

A single-process stack whose vectors are NumPy arrays of shape
``(size, ncols)``: a single vector is ``(size, 1)``, never 1-D. It is a
bound implementation: it works on NumPy arrays only, writes in place
whenever ``out`` is given, and is not differentiable.
"""

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
from scipy.sparse import identity as sparse_identity

from pyapprox.util.linalg.operators.layout import Layout
from pyapprox.util.linalg.operators.protocols import OperatorProperties

_Vector = NDArray[Any]

_SYMMETRY_RTOL = 1e-12


def _check_block(x: _Vector, size: int, dtype: np.dtype[Any], name: str) -> None:
    if x.ndim != 2 or x.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size}, ncols), got {x.shape}")
    if x.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {x.dtype}")


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
        _check_block(x, self._layout.size, self._dtype, name)

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


class CSRFormat:
    """Compressed sparse row storage, the format of :class:`ScipyCSROperator`.

    A direct solver that factors a CSR matrix accepts operators of this
    format and reads the matrix through :meth:`matrix`.

    Parameters
    ----------
    matrix : csr_matrix
        The stored matrix.
    """

    def __init__(self, matrix: csr_matrix) -> None:
        self._matrix = matrix

    def matrix(self) -> csr_matrix:
        """The stored matrix. Not to be modified: operators are immutable."""
        return self._matrix

    def scaled_sum(
        self, coefficient: float, others: Sequence[tuple[float, "CSRFormat"]]
    ) -> "CSRFormat":
        total = coefficient * self._matrix
        for c, other in others:
            total = total + c * other._matrix
        return CSRFormat(total.tocsr())

    def transposed(self) -> "CSRFormat":
        return CSRFormat(self._matrix.T.tocsr())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._matrix.shape})"


class ScipyCSROperator:
    """A linear operator stored as a SciPy CSR matrix.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.DiagonalCapableProtocol`
    with NumPy arrays as vectors and :class:`CSRFormat` as its format.
    ``apply`` and ``apply_transpose`` copy into ``out`` when it is given.

    Parameters
    ----------
    matrix : scipy sparse matrix
        Shape ``(range.size, domain.size)``. Stored as CSR.
    domain : Layout
        The space of the input of ``apply``.
    range : Layout, optional
        The space of the output of ``apply``. Defaults to ``domain``.
    properties : OperatorProperties, optional
        Declared properties. ``symmetric``, ``diagonal`` and ``identity``
        are verified, in time proportional to the number of nonzeros;
        ``spd`` is not (that would cost a factorization).

    Raises
    ------
    TypeError
        If ``matrix`` is not a SciPy sparse matrix.
    ValueError
        If the shape or dtype disagrees with the layouts, if a symmetric
        operator's domain and range differ, or if a verified property
        does not hold.
    """

    def __init__(
        self,
        matrix: spmatrix,
        domain: Layout,
        range: Optional[Layout] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> None:
        if not issparse(matrix):
            raise TypeError(
                "ScipyCSROperator requires a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        self._domain = domain
        self._range = domain if range is None else range
        self._properties = OperatorProperties() if properties is None else properties
        csr = matrix.tocsr()
        self._dtype = np.dtype(self._domain.dtype)
        _check_operator_layouts(
            tuple(csr.shape), csr.dtype, self._domain, self._range, self._properties
        )
        _verify_csr_properties(csr, self._properties)
        self._format = CSRFormat(csr)

    @classmethod
    def identity(cls, layout: Layout) -> "ScipyCSROperator":
        """The identity on ``layout``, stored as CSR so it combines with
        other CSR operators without leaving the format."""
        matrix = sparse_identity(
            layout.size, dtype=np.dtype(layout.dtype), format="csr"
        )
        return cls(
            matrix,
            layout,
            properties=OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True
            ),
        )

    @classmethod
    def from_diagonal(cls, d: _Vector, layout: Layout) -> "ScipyCSROperator":
        """The diagonal operator with diagonal ``d``, a single vector of
        ``layout``, stored as CSR. Declared ``spd`` when ``d`` is real and
        positive."""
        _check_block(d, layout.size, np.dtype(layout.dtype), "d")
        if d.shape[1] != 1:
            raise ValueError(f"d must be a single vector, got shape {d.shape}")
        spd = bool(np.isrealobj(d) and np.all(d > 0))
        return cls(
            diags(d[:, 0], format="csr"),
            layout,
            properties=OperatorProperties(symmetric=True, spd=spd, diagonal=True),
        )

    def domain(self) -> Layout:
        return self._domain

    def range(self) -> Layout:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> CSRFormat:
        return self._format

    def _into(self, result: _Vector, out: Optional[_Vector]) -> _Vector:
        if out is None:
            return result
        np.copyto(out, result)
        return out

    def _check_apply(
        self, x: _Vector, out: Optional[_Vector], src: Layout, dst: Layout
    ) -> None:
        _check_block(x, src.size, self._dtype, "input")
        if out is None:
            return
        _check_block(out, dst.size, self._dtype, "out")
        if out.shape[1] != x.shape[1]:
            raise ValueError(
                f"out must have {x.shape[1]} columns, got shape {out.shape}"
            )
        if np.may_share_memory(out, x):
            raise ValueError("out must not alias the input of apply")

    def apply(self, x: _Vector, out: Optional[_Vector] = None) -> _Vector:
        self._check_apply(x, out, self._domain, self._range)
        result: _Vector = self._format.matrix() @ x
        return self._into(result, out)

    def apply_transpose(self, y: _Vector, out: Optional[_Vector] = None) -> _Vector:
        self._check_apply(y, out, self._range, self._domain)
        result: _Vector = self._format.matrix().T @ y
        return self._into(result, out)

    def diagonal(self) -> _Vector:
        self._domain.require_same(
            self._range, "range of an operator asked for its diagonal"
        )
        diagonal: _Vector = self._format.matrix().diagonal()
        return diagonal.reshape(-1, 1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain={self._domain!r}, "
            f"range={self._range!r}, {self._properties!r})"
        )


def _check_operator_layouts(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    domain: Layout,
    range_: Layout,
    properties: OperatorProperties,
) -> None:
    if shape != (range_.size, domain.size):
        raise ValueError(
            f"matrix must have shape (range.size, domain.size) = "
            f"({range_.size}, {domain.size}), got {shape}"
        )
    for name, layout in (("domain", domain), ("range", range_)):
        if np.dtype(layout.dtype) != dtype:
            raise ValueError(
                f"matrix dtype {dtype} differs from the {name} dtype {layout.dtype}"
            )
    if properties.symmetric:
        domain.require_same(range_, "range of a symmetric operator")


def _verify_csr_properties(csr: csr_matrix, properties: OperatorProperties) -> None:
    if csr.nnz == 0:
        if properties.identity and csr.shape[0] > 0:
            raise ValueError("matrix declared identity is zero")
        return
    if properties.symmetric:
        # Sparse A - A^T stays sparse: an O(nnz) check, never densified.
        asymmetry = float(abs(csr - csr.T).max())
        scale = float(abs(csr).max())
        if asymmetry > _SYMMETRY_RTOL * max(scale, 1.0):
            raise ValueError(
                f"matrix declared symmetric is not: max |A - A^T| = "
                f"{asymmetry:.3e} against max |A| = {scale:.3e}"
            )
    if properties.diagonal:
        off_diagonal = csr - diags(csr.diagonal(), format="csr")
        if off_diagonal.count_nonzero() > 0:
            raise ValueError("matrix declared diagonal has off-diagonal nonzeros")
    if properties.identity and not np.all(csr.diagonal() == 1):
        raise ValueError("matrix declared identity has a diagonal entry other than 1")
