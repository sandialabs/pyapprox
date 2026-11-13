"""What an expression may refer to.

A :class:`Names` declares variables (in order), named constants with
their values, and extra functions. Declared names win over sympy's own:
a declared ``E`` is the caller's symbol, an undeclared one Euler's number.
"""

import keyword
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

import sympy as sp


@dataclass(frozen=True)
class CustomFunction:
    r"""A function beyond the built-in ones.

    Parameters
    ----------
    symbolic : type
        The sympy function class the parser builds (``sympy.tan``,
        ``sympy.besselj``, or ``sympy.Function("f")`` for one sympy does
        not know, which then cannot be differentiated).
    numeric : Callable
        ``numeric(bkd, *args)``, the function on backend arrays. A
        module-level function keeps the declarations picklable.
    """

    symbolic: Type[sp.Function]
    numeric: Callable[..., Any]


def numbered(prefix: str, count: int) -> Tuple[str, ...]:
    """``prefix1, ..., prefix<count>``: names for anonymous variables.

    Raises
    ------
    ValueError
        If ``count`` is not positive.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    return tuple(f"{prefix}{ii}" for ii in range(1, count + 1))


def _check_name(name: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{name!r} is not a name an expression can use")


class Names:
    r"""What expressions may refer to.

    Parameters
    ----------
    variables : Sequence[str]
        The variables, in the order an evaluator takes them.
    constants : Mapping[str, float], optional
        Named constants and their values. They stay symbols in parsed
        expressions and are replaced by their values for evaluation.
    functions : Mapping[str, CustomFunction], optional
        Extra functions, by the name text calls them.

    Raises
    ------
    ValueError
        If a name is not an identifier, is a Python keyword, or is
        declared twice.
    TypeError
        If a function is not a :class:`CustomFunction`.
    """

    def __init__(
        self,
        variables: Sequence[str],
        constants: Optional[Mapping[str, float]] = None,
        functions: Optional[Mapping[str, CustomFunction]] = None,
    ) -> None:
        constants = {} if constants is None else dict(constants)
        functions = {} if functions is None else dict(functions)
        declared = list(variables) + list(constants) + list(functions)
        for name in declared:
            _check_name(name)
        repeated = sorted({name for name in declared if declared.count(name) > 1})
        if repeated:
            raise ValueError(f"names declared more than once: {repeated}")
        for name, function in functions.items():
            if not isinstance(function, CustomFunction):
                raise TypeError(
                    f"function {name!r} must be a CustomFunction, got "
                    f"{type(function).__name__}"
                )
        self._variables = tuple(sp.Symbol(name) for name in variables)
        self._constants = {
            sp.Symbol(name): float(value) for name, value in constants.items()
        }
        self._functions = functions
        self._symbols = {s.name: s for s in self._variables + tuple(self._constants)}

    def variables(self) -> Tuple[sp.Symbol, ...]:
        """The variable symbols, in order."""
        return self._variables

    def constants(self) -> Dict[sp.Symbol, float]:
        """The constant symbols and their values."""
        return dict(self._constants)

    def functions(self) -> Dict[str, CustomFunction]:
        """The extra functions, by name."""
        return dict(self._functions)

    def symbol(self, name: str) -> Optional[sp.Symbol]:
        """The declared variable or constant ``name``, or ``None``."""
        return self._symbols.get(name)

    def substitute_constants(self, expression: sp.Expr) -> sp.Expr:
        """``expression`` with the constants replaced by their values."""
        replacements = {s: sp.Float(v) for s, v in self._constants.items()}
        result: sp.Expr = expression.xreplace(replacements)
        return result

    def __repr__(self) -> str:
        constants = {s.name: v for s, v in self._constants.items()}
        return (
            f"{type(self).__name__}(variables={[s.name for s in self._variables]}, "
            f"constants={constants}, functions={sorted(self._functions)})"
        )
