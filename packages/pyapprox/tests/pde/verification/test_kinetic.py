r"""Kinetic problems on phase space: Vlasov and Vlasov-Poisson.

Phase-space boxes name their coordinates (physical ``x, y, z``, velocity
``v``; ``v, w``; ``u, v, w``) and their sides (``xmin``, ``vmax``, ...).

- Hand-derived, 1D1V with a constant field :math:`E = 1`: :math:`f^* =
  \sin x\, e^{-v^2} e^{-t}` in :math:`\partial_t f + v f_x + E f_v` gives
  :math:`e^{-t} e^{-v^2} (-\sin x + v\cos x - 2v\sin x)`; periodic in
  :math:`x` (no data), Dirichlet in :math:`v`.
- Derivation-free, 1D1V, 1D2V, 2D2V and 3D3V with a prescribed field
  :math:`E(x, t)`: the phase-space flow :math:`a = (v, E)` is divergence
  free, so the conservative form :math:`\nabla_{x,v}\cdot(a f)` and the
  advective form :math:`a\cdot\nabla_{x,v} f` give the same forcing (at
  five points, on the backend, to 1e-12). The advective row still takes
  boundary conditions.
- Vlasov-Poisson 1D1V (electrons, :math:`\partial_t f + v f_x - E f_v`,
  :math:`E = -\phi_x`, :math:`-\phi_{xx} + \int f\, dv = f_\phi`):
  :math:`f^* = (1 + a\cos(x - t)) e^{-v^2/2}/\sqrt{2\pi}`, :math:`\phi^*
  = a\cos(x - t)`, :math:`v \in [-6, 6]`, :math:`a = 0.1`; the moment
  gives :math:`f_\phi = a\cos(x - t) + (1 + a\cos(x - t))\,
  \mathrm{erf}(3\sqrt 2)`.
- A moment over a coordinate the domain is not a product over is
  refused.
"""

import math
from typing import Dict, List, Tuple

import pytest
import sympy as sp

from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.expressions.symbols import derivative
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Periodic,
)
from pyapprox.pde.verification.domains import Box, Disk
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Moment,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.protocols import Array, Backend

x, v, t = sp.symbols("x v t")

_PHASE_SPACES = {
    "1d1v": (("x",), ("v",)),
    "1d2v": (("x",), ("v", "w")),
    "2d2v": (("x", "y"), ("v", "w")),
    "3d3v": (("x", "y", "z"), ("u", "v", "w")),
}


def _phase_box(physical: Tuple[str, ...], velocity: Tuple[str, ...]) -> Box:
    bounds = [(0.0, 2 * math.pi)] * len(physical) + [(-6.0, 6.0)] * len(velocity)
    return Box(bounds, physical + velocity)


def _conditions(box: Box, physical: Tuple[str, ...]) -> Dict[str, ConditionProtocol]:
    conditions: Dict[str, ConditionProtocol] = {}
    for side in box.boundaries():
        conditions[side] = Periodic() if side[0] in physical else Dirichlet()
    return conditions


def _same(actual: sp.Expr, expected: sp.Expr) -> bool:
    return bool(sp.simplify(actual - expected) == 0)


class TestVlasov:
    def test_hand_derived(self) -> None:
        box = _phase_box(("x",), ("v",))
        assert box.boundaries() == ("xmin", "xmax", "vmin", "vmax")
        problem = ManufacturedProblem(
            {"f": "sin(x)*exp(-v**2)*exp(-t)"},
            [TimeDerivative("f"), Source("f", "v*f_x + E*f_v")],
            box,
            _conditions(box, ("x",)),
            constants={"E": 1.0},
        )
        expected = (
            sp.exp(-t)
            * sp.exp(-(v**2))
            * (-sp.sin(x) + v * sp.cos(x) - 2 * v * sp.sin(x))
        )
        assert _same(problem.forcing("f"), expected)
        assert problem.boundary_rows() == ("f",)
        assert problem.boundary_data("xmin") == {}
        assert _same(
            problem.boundary_data("vmax")["f"], sp.sin(x) * sp.exp(-(v**2) - t)
        )

    @pytest.mark.parametrize("space", list(_PHASE_SPACES))
    def test_conservative_equals_advective(
        self, numpy_bkd: Backend[Array], space: str
    ) -> None:
        physical, velocity = _PHASE_SPACES[space]
        coords = physical + velocity
        box = _phase_box(physical, velocity)
        xs = [sp.Symbol(c) for c in physical]
        # Advection in x by the first velocity components; E(x, t) per
        # velocity direction, independent of velocity.
        flow = [sp.Symbol(c) for c in velocity[: len(physical)]] + [
            sp.cos(xs[0] + k) * sp.sin(t) + sum(xs) / 4 for k in range(len(velocity))
        ]
        grad = [derivative("f", c, coords) for c in coords]
        f = derivative("f")
        solution = "*".join(
            [f"(2 + cos({c}))" for c in physical]
            + [f"exp(-{c}**2/2)" for c in velocity]
            + ["(1 + t**2)"]
        )
        conservative: List[TermProtocol] = [
            TimeDerivative("f"),
            Divergence("f", [a * f for a in flow]),
        ]
        advective: List[TermProtocol] = [
            TimeDerivative("f"),
            Source("f", sum(a * g for a, g in zip(flow, grad))),
        ]
        problems = [
            ManufacturedProblem({"f": solution}, terms, box, _conditions(box, physical))
            for terms in (conservative, advective)
        ]
        assert problems[1].boundary_rows() == ("f",)
        evaluate = PointExpressions(
            [p.forcing("f") for p in problems],
            problems[0].names(),
            coords,
            numpy_bkd,
        )
        points = numpy_bkd.asarray(
            [[0.3 + 0.7 * k + 0.11 * j for k in range(5)] for j in range(len(coords))]
        )
        values = evaluate(points, t=0.3)
        numpy_bkd.assert_allclose(values[0], values[1], rtol=1e-12, atol=1e-12)


class TestVlasovPoisson:
    def _problem(self) -> ManufacturedProblem:
        box = _phase_box(("x",), ("v",))
        terms: List[TermProtocol] = [
            TimeDerivative("f"),
            Source("f", "v*f_x + phi_x*f_v"),
            Divergence("phi", ["-phi_x", "0"]),
            Moment("phi", "f", over=["v"]),
        ]
        x_periodic: Dict[str, ConditionProtocol] = {
            "f": Periodic(),
            "phi": Periodic(),
        }
        v_side: Dict[str, ConditionProtocol] = {"f": Dirichlet(), "phi": Neumann()}
        return ManufacturedProblem(
            {
                "f": "(1 + a*cos(x - t))*exp(-v**2/2)/sqrt(2*pi)",
                "phi": "a*cos(x - t)",
            },
            terms,
            box,
            {"xmin": x_periodic, "xmax": x_periodic, "vmin": v_side, "vmax": v_side},
            constants={"a": 0.1},
        )

    def test_moment_forcing(self, numpy_bkd: Backend[Array]) -> None:
        problem = self._problem()
        expected = "a*cos(x - t) + (1 + a*cos(x - t))*erf(3*sqrt(2))"
        evaluate = PointExpressions(
            [problem.forcing("phi"), expected],
            problem.names(),
            problem.coordinates(),
            numpy_bkd,
        )
        points = numpy_bkd.asarray([[0.1, 1.3, 2.9, 4.4], [-5.0, -1.0, 0.5, 3.0]])
        values = evaluate(points, t=0.7)
        numpy_bkd.assert_allclose(values[0], values[1], rtol=1e-12)
        assert problem.boundary_rows() == ("f", "phi")

    def test_moment_needs_a_product_domain(self) -> None:
        with pytest.raises(ValueError, match="not a product over 'x'"):
            ManufacturedProblem(
                {"u": "x*y"},
                [Divergence("u", ["-u_x", "-u_y"]), Moment("u", "u", over=["x"])],
                Disk(),
                {"circle": Dirichlet()},
            )
