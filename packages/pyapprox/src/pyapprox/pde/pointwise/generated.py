"""The expression sets every code target prints, in one fixed layout.

:func:`generate` turns the integrands of a physics and their derivatives
into three :class:`GeneratedSet` (residual, Jacobian, parameter
derivatives). Each set is a flat, ordered list of outputs with a named
entry per term, the input variables it reads, and common subexpressions
eliminated once (``sympy.cse``). A target binds a set without reordering
it: the backend kernel lambdifies it with backend functions, and a
compiled target prints the same replacements and outputs as C or C++, so
every target agrees on which outputs exist and where.

Only sympy is used here: no arrays and no backend.

Named constants are replaced by their values and exact number constants
(``pi``, ``E``) by 17-digit floats, so no target needs a name for them.
Symbolically zero terms have no entry.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import sympy as sp

from pyapprox.pde.pointwise.derivatives import PointwiseDerivatives
from pyapprox.pde.pointwise.spec import PointwisePhysics

# Prefix of the common-subexpression symbols. Field and coordinate names
# start with a letter, so only a parameter or constant can clash, which
# generate refuses.
_CSE_PREFIX = "_cse"


@dataclass(frozen=True)
class GeneratedEntry:
    """One term of a set: where its outputs start, and their shape.

    Attributes
    ----------
    key : tuple of str
        The term's name followed by field and parameter names:
        ``("f0", a)`` and ``("f1", a)`` in the residual set,
        ``("g0", a, b)`` to ``("g3", a, b)`` for test field ``a`` and trial
        field ``b`` in the Jacobian set, ``("f0", a, p)`` and
        ``("f1", a, p)`` in the parameter set.
    start : int
        The index of its first output.
    shape : tuple of int
        ``()`` for one output, ``(ndim,)`` or ``(ndim, ndim)`` for
        row-major outputs.
    """

    key: Tuple[str, ...]
    start: int
    shape: Tuple[int, ...]

    def size(self) -> int:
        """The number of outputs."""
        size = 1
        for n in self.shape:
            size *= n
        return size


@dataclass(frozen=True)
class GeneratedSet:
    """An ordered set of outputs with common subexpressions eliminated.

    Attributes
    ----------
    entries : tuple of GeneratedEntry
        The terms, in output order.
    variables : tuple of sympy.Symbol
        The inputs the outputs read, in the physics' name order.
    replacements : tuple of (sympy.Symbol, sympy.Expr)
        The common subexpressions, each over the variables and the earlier
        replacements, to be computed in order.
    outputs : tuple of sympy.Expr
        The outputs over the variables and replacements.
    """

    entries: Tuple[GeneratedEntry, ...]
    variables: Tuple[sp.Symbol, ...]
    replacements: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    outputs: Tuple[sp.Expr, ...]

    def noutputs(self) -> int:
        """The number of outputs."""
        return len(self.outputs)

    def entry(self, key: Tuple[str, ...]) -> GeneratedEntry:
        """The entry named ``key``.

        Raises
        ------
        KeyError
            If the set has no such entry (it is symbolically zero or not
            a term of this set).
        """
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def has(self, key: Tuple[str, ...]) -> bool:
        """Whether the set has an entry named ``key``."""
        return any(entry.key == key for entry in self.entries)

    def expanded(self) -> Tuple[sp.Expr, ...]:
        """The outputs with the replacements substituted back, over the
        variables alone."""
        outputs = list(self.outputs)
        for symbol, expression in reversed(self.replacements):
            outputs = [o.xreplace({symbol: expression}) for o in outputs]
        return tuple(outputs)


@dataclass(frozen=True)
class GeneratedSets:
    """The sets of one physics.

    Attributes
    ----------
    residual : GeneratedSet
        ``f0`` and ``f1`` of every field, zero ones included.
    jacobian : GeneratedSet
        The nonzero Jacobian blocks.
    parameters : GeneratedSet
        The nonzero parameter derivatives.
    """

    residual: GeneratedSet
    jacobian: GeneratedSet
    parameters: GeneratedSet


class _Builder:
    """Collects the entries of one set before it is finalized."""

    def __init__(self, physics: PointwisePhysics) -> None:
        self._physics = physics
        self._entries: List[GeneratedEntry] = []
        self._expressions: List[sp.Expr] = []

    def add(
        self, key: Tuple[str, ...], flat: Sequence[sp.Expr], shape: Tuple[int, ...]
    ) -> None:
        entry = GeneratedEntry(key, len(self._expressions), shape)
        if len(flat) != entry.size():
            raise ValueError(f"{key} has {len(flat)} expressions, not {entry.size()}")
        self._entries.append(entry)
        self._expressions.extend(flat)

    def build(self) -> GeneratedSet:
        names = self._physics.names()
        numeric: List[sp.Expr] = []
        for expression in self._expressions:
            expression = names.substitute_constants(expression)
            numeric.append(
                expression.xreplace(
                    {
                        c: sp.Float(c.evalf(17))
                        for c in expression.atoms(sp.NumberSymbol)
                    }
                )
            )
        used: "set[sp.Basic]" = set()
        for expression in numeric:
            used |= expression.free_symbols
        variables = tuple(s for s in names.variables() if s in used)
        replacements, outputs = sp.cse(
            numeric, symbols=sp.numbered_symbols(_CSE_PREFIX)
        )
        return GeneratedSet(
            entries=tuple(self._entries),
            variables=variables,
            replacements=tuple((s, e) for s, e in replacements),
            outputs=tuple(outputs),
        )


def _check_cse_prefix(physics: PointwisePhysics) -> None:
    names = physics.names()
    declared = [s.name for s in names.variables()] + [s.name for s in names.constants()]
    clashes = [n for n in declared if n.startswith(_CSE_PREFIX)]
    if clashes:
        raise ValueError(
            f"names starting with {_CSE_PREFIX!r} are reserved for generated "
            f"code, got {clashes}"
        )


def generate(derivatives: PointwiseDerivatives) -> GeneratedSets:
    """The residual, Jacobian and parameter-derivative sets of a physics.

    Parameters
    ----------
    derivatives : PointwiseDerivatives
        The physics' symbolic derivatives.

    Returns
    -------
    GeneratedSets
        Fields in block order; within the Jacobian set, test field outer
        and trial field inner, terms ``g0`` to ``g3``; within the parameter
        set, field outer and parameter inner, terms ``f0`` then ``f1``.

    Raises
    ------
    TypeError
        If ``derivatives`` is not a :class:`PointwiseDerivatives`.
    ValueError
        If a parameter or constant name starts with ``_cse``.
    """
    if not isinstance(derivatives, PointwiseDerivatives):
        raise TypeError(
            "derivatives must be a PointwiseDerivatives, got "
            f"{type(derivatives).__name__}"
        )
    physics = derivatives.physics()
    _check_cse_prefix(physics)
    ndim, fields = physics.ndim(), physics.field_names()

    residual = _Builder(physics)
    for name in fields:
        residual.add(("f0", name), [physics.f0(name)], ())
        residual.add(("f1", name), physics.f1(name), (ndim,))

    jacobian = _Builder(physics)
    for test in fields:
        for trial in fields:
            block = derivatives.jacobian_block(test, trial)
            if block is None:
                continue
            if block.g0 is not None:
                jacobian.add(("g0", test, trial), [block.g0], ())
            if block.g1 is not None:
                jacobian.add(("g1", test, trial), block.g1, (ndim,))
            if block.g2 is not None:
                jacobian.add(("g2", test, trial), block.g2, (ndim,))
            if block.g3 is not None:
                flat = [entry for row in block.g3 for entry in row]
                jacobian.add(("g3", test, trial), flat, (ndim, ndim))

    parameters = _Builder(physics)
    for test in fields:
        for param in physics.params():
            terms = derivatives.parameter_derivative(test, param)
            if terms is None:
                continue
            if terms.f0 is not None:
                parameters.add(("f0", test, param), [terms.f0], ())
            if terms.f1 is not None:
                parameters.add(("f1", test, param), terms.f1, (ndim,))

    return GeneratedSets(residual.build(), jacobian.build(), parameters.build())
