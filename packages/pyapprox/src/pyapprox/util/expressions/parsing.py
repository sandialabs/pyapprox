"""Text to sympy, without running the text.

``sympy.sympify`` and ``parse_expr`` call ``eval``, so a crafted string
can execute code. :func:`parse` instead walks Python's syntax tree,
accepts only numbers, names, ``+ - * / **``, unary signs and calls of
functions by name, and builds the sympy expression itself. Strings from
users or agents are therefore safe to parse.

Names resolve in order: a declared variable, constant or function; then
sympy's own numeric constants (``pi``, ``E``) and function classes
(``sin``, ``exp``, ``besselj``, ...), plus ``sqrt``, ``cbrt``, ``abs``,
and ``Max``/``Min`` (also spelled ``max``/``min``).
Anything else is refused.
"""

import ast
from typing import Callable, Dict, Union

import sympy as sp

from pyapprox.util.expressions.names import Names

Value = Union[sp.Expr, str, float, int]

# Longest text accepted, and largest integer power of a number, so that a
# short string cannot request an enormous exact computation.
_MAX_LENGTH = 10000
_MAX_POWER = 1000

# sympy helpers that are functions, not Function classes, but plain
# mathematics (``sqrt(x)`` builds ``x**(1/2)``).
_HELPERS: Dict[str, Callable[..., sp.Expr]] = {
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    # Max and Min are not Function classes, so are listed by name.
    "Max": sp.Max,
    "Min": sp.Min,
    "max": sp.Max,
    "min": sp.Min,
}


class _Builder:
    """Builds a sympy expression from an allowed syntax tree."""

    def __init__(self, names: Names) -> None:
        self._names = names

    def build(self, node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            return self._number(node.value)
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self.build(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise ValueError(f"expressions may not contain {type(node).__name__}")

    def _number(self, value: object) -> sp.Expr:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"only real numbers may appear, got {value!r}")
        if isinstance(value, int):
            return sp.Integer(value)
        return sp.Float(value)

    def _name(self, name: str) -> sp.Expr:
        symbol = self._names.symbol(name)
        if symbol is not None:
            return symbol
        value = getattr(sp, name, None)
        if isinstance(value, sp.Expr) and value.is_number:
            return value
        raise ValueError(f"unknown name {name!r}")

    def _binary(self, node: ast.BinOp) -> sp.Expr:
        left, right = self.build(node.left), self.build(node.right)
        result: sp.Expr
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.Pow):
            if left.is_number and right.is_Integer and abs(int(right)) > _MAX_POWER:
                raise ValueError(
                    f"integer powers of numbers are limited to {_MAX_POWER}"
                )
            result = left**right
        else:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return result

    def _call(self, node: ast.Call) -> sp.Expr:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("only functions called by name, without keywords")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise ValueError("function arguments may not be unpacked")
        function = self._function(node.func.id)
        result = function(*(self.build(arg) for arg in node.args))
        if not isinstance(result, sp.Expr):
            raise ValueError(f"{node.func.id}(...) is not an expression")
        return result

    def _function(self, name: str) -> Callable[..., sp.Expr]:
        custom = self._names.functions().get(name)
        if custom is not None:
            return custom.symbolic
        if name in _HELPERS:
            return _HELPERS[name]
        value = getattr(sp, name, None)
        if (
            isinstance(value, type)
            and issubclass(value, sp.Function)
            and value is not sp.Function
        ):
            return value
        raise ValueError(f"unknown function {name!r}")


def parse(value: Value, names: Names) -> sp.Expr:
    """``value`` as a sympy expression over ``names``.

    Text is read without being run; sympy objects and numbers pass
    through, checked for undeclared symbols.

    Raises
    ------
    ValueError
        If the text is too long, is not one expression, or uses anything
        but numbers, known names, arithmetic and calls of known functions;
        or a sympy object has undeclared symbols.
    """
    if isinstance(value, str):
        if len(value) > _MAX_LENGTH:
            raise ValueError(f"expressions are limited to {_MAX_LENGTH} characters")
        try:
            tree = ast.parse(value.strip(), mode="eval")
        except SyntaxError as error:
            raise ValueError(f"cannot parse {value!r}: {error.msg}") from None
        return _Builder(names).build(tree.body)
    expression: sp.Expr = sp.sympify(value)
    declared = set(names.variables()) | set(names.constants())
    unknown = expression.free_symbols - declared
    if unknown:
        raise ValueError(f"undeclared symbols {sorted(map(str, unknown))}")
    return expression
