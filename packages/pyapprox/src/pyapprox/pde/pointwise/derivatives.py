r"""Derivatives of a pointwise physics, by symbolic differentiation.

For a test field :math:`a` and trial field :math:`b` the Jacobian of the
residual is assembled from four pointwise blocks (PETSc's
:math:`g_0, \ldots, g_3`):

.. math::

    g_0 = \frac{\partial f_{0,a}}{\partial u_b}, \quad
    g_1 = \frac{\partial f_{0,a}}{\partial \nabla u_b}, \quad
    g_2 = \frac{\partial f_{1,a}}{\partial u_b}, \quad
    g_3 = \frac{\partial f_{1,a}}{\partial \nabla u_b},

so that :math:`J_{ij} = \int \phi_i (g_0 \psi_j + g_1 \cdot \nabla
\psi_j) + \nabla \phi_i \cdot (g_2 \psi_j + g_3 \nabla \psi_j)\, dx`,
with ``g3[i][j]`` the derivative of ``f1[i]`` by ``u_b``'s ``j``-th
gradient component. The parameter derivatives are
:math:`\partial f_{0,a} / \partial p_k` and
:math:`\partial f_{1,a} / \partial p_k`.

A block, or a whole field pair, is ``None`` when it is symbolically zero:
when sympy's derivative is exactly ``0`` because the integrand does not
depend on that input. Zeros that need simplification to appear are not
looked for; they are evaluated like any other entry.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import sympy as sp

from pyapprox.pde.pointwise.spec import PointwisePhysics


@dataclass(frozen=True)
class SymbolicJacobianBlock:
    """The pointwise Jacobian blocks of one (test, trial) field pair.

    Each is ``None`` when symbolically zero.

    Attributes
    ----------
    g0 : sympy.Expr, optional
        ``∂f0/∂u``.
    g1 : tuple of sympy.Expr, optional
        ``∂f0/∂∇u``, ``ndim`` entries.
    g2 : tuple of sympy.Expr, optional
        ``∂f1/∂u``, ``ndim`` entries.
    g3 : tuple of tuple of sympy.Expr, optional
        ``∂f1/∂∇u``, ``ndim × ndim``, ``g3[i][j] = ∂f1[i]/∂(∂_j u)``.
    """

    g0: Optional[sp.Expr]
    g1: Optional[Tuple[sp.Expr, ...]]
    g2: Optional[Tuple[sp.Expr, ...]]
    g3: Optional[Tuple[Tuple[sp.Expr, ...], ...]]


@dataclass(frozen=True)
class SymbolicTerms:
    """Derivatives of one field's integrands by one scalar input.

    Attributes
    ----------
    f0 : sympy.Expr, optional
        The derivative of ``f0``; ``None`` when symbolically zero.
    f1 : tuple of sympy.Expr, optional
        The derivative of ``f1``, ``ndim`` entries; ``None`` when
        symbolically zero.
    """

    f0: Optional[sp.Expr]
    f1: Optional[Tuple[sp.Expr, ...]]


def _diff(expression: sp.Expr, symbol: sp.Symbol) -> sp.Expr:
    result: sp.Expr = sp.diff(expression, symbol)
    unevaluated = result.atoms(sp.Derivative, sp.Subs)
    if unevaluated:
        raise ValueError(
            f"cannot differentiate {expression} by {symbol}: sympy leaves "
            f"{sorted(map(str, unevaluated))}; declare a differentiable function"
        )
    return result


def _scalar(expression: sp.Expr) -> Optional[sp.Expr]:
    return None if expression == 0 else expression


def _vector(entries: Sequence[sp.Expr]) -> Optional[Tuple[sp.Expr, ...]]:
    return None if all(e == 0 for e in entries) else tuple(entries)


def _matrix(
    rows: Sequence[Sequence[sp.Expr]],
) -> Optional[Tuple[Tuple[sp.Expr, ...], ...]]:
    if all(e == 0 for row in rows for e in row):
        return None
    return tuple(tuple(row) for row in rows)


class PointwiseDerivatives:
    """The Jacobian blocks and parameter derivatives of a physics.

    All derivatives are computed when this is built, so differentiation
    errors are raised here rather than when a kernel is evaluated.

    Parameters
    ----------
    physics : PointwisePhysics
        The physics to differentiate.

    Raises
    ------
    TypeError
        If ``physics`` is not a :class:`PointwisePhysics`.
    ValueError
        If an integrand uses a function sympy cannot differentiate.
    """

    def __init__(self, physics: PointwisePhysics) -> None:
        if not isinstance(physics, PointwisePhysics):
            raise TypeError(
                f"physics must be a PointwisePhysics, got {type(physics).__name__}"
            )
        self._physics = physics
        names = physics.field_names()
        self._jacobian: Dict[Tuple[str, str], Optional[SymbolicJacobianBlock]] = {}
        for test in names:
            for trial in names:
                self._jacobian[(test, trial)] = self._block(test, trial)
        self._params: Dict[Tuple[str, str], Optional[SymbolicTerms]] = {}
        for test in names:
            for param, symbol in zip(physics.params(), physics.param_symbols()):
                self._params[(test, param)] = self._terms(test, symbol)

    def _block(self, test: str, trial: str) -> Optional[SymbolicJacobianBlock]:
        physics = self._physics
        f0, f1 = physics.f0(test), physics.f1(test)
        value = physics.value_symbol(trial)
        grads = physics.gradient_symbols(trial)
        block = SymbolicJacobianBlock(
            g0=_scalar(_diff(f0, value)),
            g1=_vector([_diff(f0, g) for g in grads]),
            g2=_vector([_diff(c, value) for c in f1]),
            g3=_matrix([[_diff(c, g) for g in grads] for c in f1]),
        )
        if all(t is None for t in (block.g0, block.g1, block.g2, block.g3)):
            return None
        return block

    def _terms(self, test: str, symbol: sp.Symbol) -> Optional[SymbolicTerms]:
        physics = self._physics
        terms = SymbolicTerms(
            f0=_scalar(_diff(physics.f0(test), symbol)),
            f1=_vector([_diff(c, symbol) for c in physics.f1(test)]),
        )
        if terms.f0 is None and terms.f1 is None:
            return None
        return terms

    def physics(self) -> PointwisePhysics:
        """The physics differentiated."""
        return self._physics

    def jacobian_block(self, test: str, trial: str) -> Optional[SymbolicJacobianBlock]:
        """The blocks of field ``test``'s residual by field ``trial``, or
        ``None`` when all four are symbolically zero.

        Raises
        ------
        KeyError
            If either name is not a field.
        """
        return self._jacobian[(test, trial)]

    def zero_jacobian_blocks(self) -> Tuple[Tuple[str, str], ...]:
        """The (test, trial) field pairs whose Jacobian is symbolically
        zero, in block order."""
        return tuple(pair for pair, block in self._jacobian.items() if block is None)

    def parameter_derivative(self, test: str, param: str) -> Optional[SymbolicTerms]:
        """The derivative of field ``test``'s integrands by parameter
        ``param``, or ``None`` when symbolically zero.

        Raises
        ------
        KeyError
            If ``test`` is not a field or ``param`` not a parameter.
        """
        return self._params[(test, param)]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._physics!r}, "
            f"zero_jacobian_blocks={list(self.zero_jacobian_blocks())})"
        )
