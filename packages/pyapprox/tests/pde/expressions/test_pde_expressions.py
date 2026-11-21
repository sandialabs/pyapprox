r"""PDE expression symbols and point evaluation.

Symbols: jets up to order 4 (35 per field in 3D, 5 in 1D), sorted
directions, reserved names refused as fields, constants or functions.

Evaluation, on both backends at three points of the unit square, against
the same values written with backend methods (1e-12):

- :math:`k\sin(\pi x) e^{-t} + |y|` with the constant :math:`k = 2`, at
  :math:`t = 0.3`;
- a constant and an expression in :math:`t` alone give one value per
  point;
- a normal flux :math:`x\, n_x + y\, n_y` and a state law
  :math:`u^2 + u_x` with ``u`` and ``u_x`` passed by name;
- the backend's dtype; on an autodiff backend the Jacobian of
  :math:`\sum x^2` with respect to the points.

Refusals: wrong shapes, missing normals or jets, undeclared symbols.
"""

import pytest
import sympy as sp

from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.expressions.symbols import (
    derivative,
    jets,
    normal,
    pde_names,
    standard,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend

_X, _XY = standard(1), standard(2)


def _points(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[0.1, 0.5, 0.9], [-0.2, 0.4, 0.7]])


class TestSymbols:
    def test_jets(self) -> None:
        assert len(jets("u", standard(3))) == 35
        assert list(jets("u", _X).values()) == ["", "x", "xx", "xxx", "xxxx"]
        assert derivative("u", "yx") == sp.Symbol("u_xy")
        # Phase space: directions in coordinate order, normals n + name.
        assert derivative("f", "vx", ("x", "v")) == sp.Symbol("f_xv")
        assert len(jets("f", "xyzuvw")) == 210
        assert normal(("x", "v")) == sp.symbols("nx nv")

    def test_coordinate_refusals(self) -> None:
        for coords in (("t",), ("n",), ("vx",), ("x", "x")):
            with pytest.raises(ValueError, match="coordinates"):
                pde_names(coords)
        with pytest.raises(ValueError, match="is a coordinate"):
            pde_names(("x", "v"), ["v"])

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="field name"):
            pde_names(_XY, ["t"])
        with pytest.raises(ValueError, match="field name"):
            pde_names(_XY, ["u_1"])
        with pytest.raises(ValueError, match="other names"):
            pde_names(_XY, ["u"], constants={"u_x": 1.0})
        with pytest.raises(ValueError, match="other names"):
            pde_names(_XY, constants={"nx": 1.0})

    def test_declared_constant_wins(self) -> None:
        from pyapprox.util.expressions.parsing import parse

        names = pde_names(_X, ["u"], constants={"E": 2.0})
        assert parse("E*u_x", names) == sp.Symbol("E") * sp.Symbol("u_x")


class TestValues:
    def test_matches_backend_methods(self, bkd: Backend[Array]) -> None:
        names = pde_names(_XY, constants={"k": 2.0})
        pts = _points(bkd)
        values = PointExpressions(["k*sin(pi*x)*exp(-t) + abs(y)"], names, _XY, bkd)(
            pts, t=0.3
        )
        expected = 2.0 * bkd.sin(3.141592653589793 * pts[0]) * bkd.exp(
            bkd.asarray([-0.3])
        ) + bkd.abs(pts[1])
        assert values.shape == (1, 3)
        bkd.assert_allclose(values[0], expected, rtol=1e-12)

    def test_constants_broadcast(self, bkd: Backend[Array]) -> None:
        pts = _points(bkd)
        values = PointExpressions(["2", "exp(-t)"], pde_names(_XY), _XY, bkd)(
            pts, t=0.5
        )
        assert values.shape == (2, 3)
        bkd.assert_allclose(values[0], bkd.full((3,), 2.0))
        bkd.assert_allclose(values[1], bkd.full((3,), 0.6065306597126334), rtol=1e-12)

    def test_normals_and_state(self, bkd: Backend[Array]) -> None:
        pts = _points(bkd)
        normals = bkd.asarray([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]])
        u, u_x = bkd.asarray([1.0, 2.0, 3.0]), bkd.asarray([0.5, 0.0, -1.0])
        evaluator = PointExpressions(
            ["x*nx + y*ny", "u**2 + u_x"], pde_names(_XY, ["u"]), _XY, bkd
        )
        assert evaluator.needs_normal()
        assert evaluator.states() == ["u", "u_x"]
        values = evaluator(pts, normals=normals, state={"u": u, "u_x": u_x})
        bkd.assert_allclose(
            values[0], pts[0] * normals[0] + pts[1] * normals[1], rtol=1e-12
        )
        bkd.assert_allclose(values[1], u**2 + u_x, rtol=1e-12)

    def test_dtype_follows_the_backend(self, bkd: Backend[Array]) -> None:
        values = PointExpressions(["x*y"], pde_names(_XY), _XY, bkd)(_points(bkd))
        assert values.dtype == bkd.zeros((1,)).dtype

    def test_autodiff_graph_survives(self, torch_bkd: AutodiffBackend[Array]) -> None:
        bkd = torch_bkd
        evaluator = PointExpressions(["x**2"], pde_names(_XY), _XY, bkd)

        def total(flat_points: Array) -> Array:
            values = evaluator(bkd.reshape(flat_points, (2, 3)))
            return bkd.reshape(bkd.sum(values), (1,))

        x = bkd.asarray([0.1, 0.5, 0.9])
        flat = bkd.concatenate([x, bkd.asarray([0.2, 0.4, 0.7])])
        gradient = bkd.jacobian(total, flat)
        expected = bkd.concatenate([2.0 * x, bkd.zeros((3,))])
        bkd.assert_allclose(gradient, bkd.reshape(expected, (1, 6)), rtol=1e-12)


class TestRefusals:
    def test_construction(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="unknown name 'w'"):
            PointExpressions(["w"], pde_names(_XY), _XY, numpy_bkd)
        with pytest.raises(ValueError, match="unknown name 'z'"):
            PointExpressions(["z"], pde_names(_XY), _XY, numpy_bkd)
        with pytest.raises(ValueError, match="pde_names"):
            PointExpressions(["x"], pde_names(_X), _XY, numpy_bkd)

    def test_call(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        pts = _points(bkd)
        names = pde_names(_XY, ["u"])
        with pytest.raises(ValueError, match="coords must have shape"):
            PointExpressions(["x"], pde_names(_X), _X, bkd)(pts)
        with pytest.raises(ValueError, match="outward normals"):
            PointExpressions(["nx"], names, _XY, bkd)(pts)
        with pytest.raises(ValueError, match="state u_x"):
            PointExpressions(["u_x"], names, _XY, bkd)(pts)
        with pytest.raises(ValueError, match="normals must have shape"):
            PointExpressions(["nx"], names, _XY, bkd)(pts, normals=bkd.zeros((2, 2)))
        with pytest.raises(ValueError, match="u must have shape"):
            PointExpressions(["u"], names, _XY, bkd)(pts, state={"u": bkd.zeros((2,))})
