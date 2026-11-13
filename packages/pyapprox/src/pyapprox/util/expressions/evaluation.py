"""Expressions evaluated on backend arrays.

Each expression is lambdified with its functions bound to the backend's
own methods, so it runs on backend arrays throughout: values land on the
backend's dtype and device, and a differentiable backend differentiates
through them. A function without a backend method is refused when the
evaluator is built, unless declared as a
:class:`~pyapprox.util.expressions.names.CustomFunction`.

Every result has one value per point: an expression constant in some or
all variables is broadcast.
"""

import functools
from typing import Any, Callable, Dict, Generic, List, Sequence

import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import Value, parse

# Backend methods for sympy function classes, by class name.
_BACKEND_METHODS = {
    "sin": "sin",
    "cos": "cos",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "atan2": "arctan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "Abs": "abs",
    "sign": "sign",
    "erf": "erf",
}
# Where lambdify prints a class under another name.
_PRINTED = {"Abs": "abs"}


def _namespace(bkd: Backend[Any], names: Names) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        _PRINTED.get(cls, cls): getattr(bkd, method)
        for cls, method in _BACKEND_METHODS.items()
    }
    # sqrt(x) is a power; lambdify still prints it as sqrt.
    namespace["sqrt"] = bkd.sqrt
    for function in names.functions().values():
        namespace[function.symbolic.__name__] = functools.partial(function.numeric, bkd)
    return namespace


class ExpressionEvaluator(Generic[Array]):
    """Expressions evaluated at points with ``bkd``.

    Parameters
    ----------
    expressions : Sequence
        The expressions, as text or sympy, over ``names``.
    names : Names
        Their variables, constants and functions.
    bkd : Backend
        The backend of the points and values.

    Raises
    ------
    ValueError
        If an expression fails to parse, or uses a function with neither a
        backend method nor a custom definition.
    """

    def __init__(
        self, expressions: Sequence[Value], names: Names, bkd: Backend[Array]
    ) -> None:
        self._names, self._bkd = names, bkd
        supported = set(_BACKEND_METHODS) | {
            f.symbolic.__name__ for f in names.functions().values()
        }
        numeric: List[sp.Expr] = []
        for value in expressions:
            expression = names.substitute_constants(parse(value, names))
            # Exact constants (pi, E, ...) become floats, so lambdify needs
            # no name for them.
            expression = expression.xreplace(
                {c: sp.Float(c.evalf(17)) for c in expression.atoms(sp.NumberSymbol)}
            )
            used = {type(f).__name__ for f in expression.atoms(sp.Function)}
            if not used <= supported:
                raise ValueError(
                    f"no backend method for {sorted(used - supported)}; "
                    "declare them as CustomFunction"
                )
            numeric.append(expression)
        self._used: "set[sp.Basic]" = set()
        for expression in numeric:
            self._used |= expression.free_symbols
        namespace = _namespace(bkd, names)
        self._functions: List[Callable[..., Any]] = [
            sp.lambdify(names.variables(), expression, modules=[namespace])
            for expression in numeric
        ]

    def nexpressions(self) -> int:
        """The number of expressions."""
        return len(self._functions)

    def nvars(self) -> int:
        """The number of variables."""
        return len(self._names.variables())

    def bkd(self) -> Backend[Array]:
        """The backend of the points and values."""
        return self._bkd

    def uses(self, symbol: sp.Symbol) -> bool:
        """Whether an expression depends on ``symbol``."""
        return symbol in self._used

    def __call__(self, values: Array) -> Array:
        """The expressions at points.

        Parameters
        ----------
        values : Array
            The variables at the points, shape ``(nvars, npts)``.

        Returns
        -------
        Array
            Shape ``(nexpressions, npts)``.

        Raises
        ------
        ValueError
            If ``values`` is not ``(nvars, npts)``.
        """
        if values.ndim != 2 or values.shape[0] != self.nvars():
            raise ValueError(
                f"values must have shape ({self.nvars()}, npts), got "
                f"{tuple(values.shape)}"
            )
        zeros = self._bkd.zeros((values.shape[1],))
        rows = [values[ii] for ii in range(self.nvars())]
        return self._bkd.stack(
            [function(*rows) + zeros for function in self._functions], axis=0
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nexpressions={self.nexpressions()}, "
            f"names={self._names!r})"
        )
