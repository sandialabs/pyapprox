r"""Expressions from text, evaluated with a backend.

Parsing, against sympy built by hand:

- ``"2*x1**2 - sin(pi*x2)/3 + sqrt(x1) + abs(x2)"`` over declared ``x1,
  x2``;
- declared names win: with ``E`` declared, ``"E*x"`` is that symbol; with
  it undeclared, Euler's number;
- named constants stay symbols and are substituted for evaluation;
- hostile or unsupported text is refused without being run: imports,
  attributes, subscripts, lambdas, comprehensions, keywords, comparisons,
  undeclared names and functions, huge powers, overlong text.

Evaluation, on both backends at three points, against the same values
written with backend methods (1e-12): built-in functions, a constant and
a one-variable expression broadcast to every point, a custom ``tan``, the
backend's dtype, and on autodiff backends the gradient of
:math:`\sum x_1^2`. ``sign``, ``Heaviside`` (1/2 at zero, as sympy),
``Max`` and ``Min`` (variadic, with number arguments) on arrays, against
hand values at three points including zeros, and on autodiff backends the
gradient of
:math:`\max(x_1, 0)^2`.
"""

import pickle
import subprocess
import sys
from typing import Any, Callable

import pytest
import sympy as sp

from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.evaluation import ExpressionEvaluator
from pyapprox.util.expressions.names import CustomFunction, Names, numbered
from pyapprox.util.expressions.parsing import parse

x1, x2 = sp.symbols("x1 x2")


def _tan(bkd: Backend[Any], value: Any) -> Any:
    return bkd.sin(value) / bkd.cos(value)


def _grad_of_sum(
    bkd: AutodiffBackend[Array], fun: Callable[[Array], Array], samples: Array
) -> Array:
    """Autodiff gradient of ``sum(fun(samples))``, shaped like ``samples``."""
    shape = tuple(samples.shape)

    def total(flat: Array) -> Array:
        return bkd.reshape(bkd.sum(fun(bkd.reshape(flat, shape))), (1,))

    return bkd.reshape(bkd.jacobian(total, bkd.flatten(samples)), shape)


class TestNames:
    def test_numbered(self) -> None:
        assert numbered("x", 3) == ("x1", "x2", "x3")
        with pytest.raises(ValueError, match="positive"):
            numbered("x", 0)

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="not a name"):
            Names(["lambda"])
        with pytest.raises(ValueError, match="not a name"):
            Names(["2x"])
        with pytest.raises(ValueError, match="more than once"):
            Names(["a"], constants={"a": 1.0})
        with pytest.raises(TypeError, match="CustomFunction"):
            Names(["a"], functions={"f": sp.tan})

    def test_pickles(self) -> None:
        names = Names(["x"], {"k": 2.0}, {"tan": CustomFunction(sp.tan, _tan)})
        copy = pickle.loads(pickle.dumps(names))
        assert copy.variables() == names.variables()
        assert copy.constants() == names.constants()


class TestParse:
    def test_matches_sympy(self) -> None:
        names = Names(["x1", "x2"])
        parsed = parse("2*x1**2 - sin(pi*x2)/3 + sqrt(x1) + abs(x2)", names)
        expected = 2 * x1**2 - sp.sin(sp.pi * x2) / 3 + sp.sqrt(x1) + sp.Abs(x2)
        assert sp.simplify(parsed - expected) == 0

    def test_declared_names_win(self) -> None:
        x, e = sp.symbols("x E")
        assert parse("E*x", Names(["x"], constants={"E": 200.0})) == e * x
        assert parse("E*x", Names(["x"])) == sp.E * x

    def test_constants_substituted(self) -> None:
        names = Names(["x"], constants={"k": 2.0})
        parsed = parse("k*x", names)
        assert parsed == sp.Symbol("k") * sp.Symbol("x")
        assert names.substitute_constants(parsed) == 2.0 * sp.Symbol("x")

    def test_sympy_objects_pass_through(self) -> None:
        names = Names(["x1", "x2"])
        assert parse(x1 * x2, names) == x1 * x2
        assert parse(3, names) == 3
        with pytest.raises(ValueError, match="undeclared"):
            parse(sp.Symbol("w"), names)

    def test_custom_function(self) -> None:
        names = Names(["x"], functions={"tan": CustomFunction(sp.tan, _tan)})
        assert parse("tan(x)", names) == sp.tan(sp.Symbol("x"))

    @pytest.mark.parametrize(
        "text",
        [
            "__import__('os').system('true')",
            "().__class__",
            "x.real",
            "[x][0]",
            "lambda: 0",
            "[a for a in (1,)]",
            "sin(x=1)",
            "sin(*[x])",
            "x < 1",
            "x if x else 1",
            "'text'",
            "1j*x",
            "True",
            "w + x",
            "open(x)",
            "Function('f')(x)",
            "9**9**9",
            "x;x",
            "x +",
        ],
    )
    def test_refuses(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse(text, Names(["x"]))

    def test_refuses_overlong_text(self) -> None:
        with pytest.raises(ValueError, match="limited"):
            parse("+".join(["x"] * 6000), Names(["x"]))


def _points(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[0.1, 0.5, 0.9], [-0.2, 0.4, 0.7]])


class TestEvaluate:
    def test_matches_backend_methods(self, bkd: Backend[Array]) -> None:
        names = Names(["x1", "x2"], constants={"k": 3.0})
        pts = _points(bkd)
        values = ExpressionEvaluator(
            ["sin(pi*x1)*exp(-k) + sqrt(1 + x1**2) + abs(x2) + atan(x1)"], names, bkd
        )(pts)
        xs, ys = pts[0], pts[1]
        expected = (
            bkd.sin(3.141592653589793 * xs) * bkd.exp(bkd.asarray([-3.0]))
            + bkd.sqrt(1 + xs**2)
            + bkd.abs(ys)
            + bkd.arctan(xs)
        )
        assert values.shape == (1, 3)
        bkd.assert_allclose(values[0], expected, rtol=1e-12)

    def test_constants_broadcast(self, bkd: Backend[Array]) -> None:
        pts = _points(bkd)
        values = ExpressionEvaluator(["2", "x1", "E"], Names(["x1", "x2"]), bkd)(pts)
        assert values.shape == (3, 3)
        bkd.assert_allclose(values[0], bkd.full((3,), 2.0))
        bkd.assert_allclose(values[1], pts[0])
        bkd.assert_allclose(values[2], bkd.full((3,), 2.718281828459045), rtol=1e-12)

    def test_custom_function(self, bkd: Backend[Array]) -> None:
        names = Names(["x1", "x2"], functions={"tan": CustomFunction(sp.tan, _tan)})
        pts = _points(bkd)
        values = ExpressionEvaluator(["tan(x1)"], names, bkd)(pts)
        bkd.assert_allclose(values[0], bkd.sin(pts[0]) / bkd.cos(pts[0]), rtol=1e-12)

    def test_dtype_follows_the_backend(self, bkd: Backend[Array]) -> None:
        values = ExpressionEvaluator(["x1*x2"], Names(["x1", "x2"]), bkd)(_points(bkd))
        assert values.dtype == bkd.zeros((1,)).dtype

    def test_nonsmooth_functions(self, bkd: Backend[Array]) -> None:
        pts = bkd.asarray([[-1.5, 0.0, 0.5], [0.2, -0.3, 0.0]])
        texts = [
            "sign(x1)",
            "Heaviside(x1)",
            "max(x1, x2, 0)",
            "Min(x1, 0.25)",
            "Max(x1, 0)*x2",
        ]
        values = ExpressionEvaluator(texts, Names(["x1", "x2"]), bkd)(pts)
        bkd.assert_allclose(values[0], bkd.asarray([-1.0, 0.0, 1.0]))
        # sympy's Heaviside(0) = 1/2.
        bkd.assert_allclose(values[1], bkd.asarray([0.0, 0.5, 1.0]))
        bkd.assert_allclose(values[2], bkd.asarray([0.2, 0.0, 0.5]))
        bkd.assert_allclose(values[3], bkd.asarray([-1.5, 0.0, 0.25]))
        bkd.assert_allclose(values[4], bkd.asarray([0.0, 0.0, 0.0]))

    def test_graph_through_max(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        bkd = autodiff_bkd
        pts = bkd.asarray([[-1.0, 0.5], [0.0, 0.0]])
        evaluator = ExpressionEvaluator(["Max(x1, 0)**2"], Names(["x1", "x2"]), bkd)
        gradient = _grad_of_sum(bkd, evaluator, pts)
        bkd.assert_allclose(gradient[0], bkd.asarray([0.0, 1.0]), rtol=1e-12)

    def test_uses(self, numpy_bkd: Backend[Array]) -> None:
        evaluator = ExpressionEvaluator(["x1"], Names(["x1", "x2"]), numpy_bkd)
        assert evaluator.uses(x1) and not evaluator.uses(x2)

    def test_graph_survives(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        bkd = autodiff_bkd
        pts = bkd.asarray([[0.1, 0.5, 0.9], [0.2, 0.4, 0.7]])
        evaluator = ExpressionEvaluator(["x1**2"], Names(["x1", "x2"]), bkd)
        gradient = _grad_of_sum(bkd, evaluator, pts)
        bkd.assert_allclose(gradient[0], 2 * pts[0], rtol=1e-12)
        bkd.assert_allclose(gradient[1], bkd.zeros((3,)), rtol=1e-12)

    def test_refusals(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match=r"no backend method for \['tan'\]"):
            ExpressionEvaluator(["tan(x1)"], Names(["x1"]), bkd)
        with pytest.raises(ValueError, match="values must have shape"):
            ExpressionEvaluator(["x1"], Names(["x1"]), bkd)(_points(bkd))


def test_parsing_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import pyapprox.util.expressions.parsing\n"
        "heavy = [m for m in ('pyapprox.util.backends.numpy', 'torch')\n"
        "         if m in sys.modules]\n"
        "print(heavy)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
