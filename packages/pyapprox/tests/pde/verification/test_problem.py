r"""Symbolic verification problems.

Hand-derived values on one case, :math:`u^* = x^2 y` on the unit square
with :math:`D = 2`, :math:`v = (x, 0)`, :math:`R(u) = u^2`: gradient
:math:`(2xy, x^2)`, so

- conservative: :math:`q = (-4xy + x^3 y, -2x^2)`,
  :math:`f = -4y + 3x^2 y - x^4 y^2`;
- non-conservative: :math:`q = (-4xy, -2x^2)`,
  :math:`f = -4y + 2x^2 y - x^4 y^2`.

Boundary data on each side (Dirichlet, Neumann, Robin with a constant and
a variable :math:`\alpha`) equal :math:`u^*`, :math:`q n` and
:math:`\alpha u^* + q n` exactly. Transient: :math:`u^* = x(1-x)e^{-t}`
with :math:`D = 1` has :math:`f = (2 - x(1-x)) e^{-t}`. Domains and
refusals.
"""

from typing import Dict

import pytest
import sympy as sp

from pyapprox.pde.constitutive.advection import (
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import AdvectionDiffusionReactionProblem

x, y, t, nx, ny = sp.symbols("x y t nx ny")
_SQUARE = Box([(0.0, 1.0), (0.0, 1.0)])


def _conditions() -> Dict[str, ConditionProtocol]:
    return {
        "left": Dirichlet(),
        "right": Neumann(),
        "bottom": Robin(3.0),
        "top": Robin("x"),
    }


def _problem(conservative: bool) -> AdvectionDiffusionReactionProblem:
    form = ConservativeAdvection() if conservative else NonConservativeAdvection()
    return AdvectionDiffusionReactionProblem(
        "x**2*y",
        _SQUARE,
        _conditions(),
        diffusion="2",
        reaction="u**2",
        velocity=["x", "0"],
        advection_form=form,
    )


def _same(actual: sp.Expr, expected: sp.Expr) -> bool:
    return bool(sp.simplify(actual - expected) == 0)


class TestDomains:
    @pytest.mark.parametrize(
        "ndim, names",
        [
            (1, ("left", "right")),
            (2, ("left", "right", "bottom", "top")),
            (3, ("left", "right", "bottom", "top", "back", "front")),
        ],
    )
    def test_box_names_its_sides(self, ndim: int, names: tuple[str, ...]) -> None:
        box = Box([(0.0, 1.0)] * ndim)
        assert isinstance(box, DomainProtocol)
        assert box.ndim() == ndim and box.boundaries() == names

    def test_disk(self) -> None:
        disk = Disk((1.0, 2.0), 0.5)
        assert isinstance(disk, DomainProtocol)
        assert disk.ndim() == 2 and disk.boundaries() == ("circle",)
        assert disk.center() == (1.0, 2.0) and disk.radius() == 0.5

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="one to three"):
            Box([(0.0, 1.0)] * 4)
        with pytest.raises(ValueError, match="lower bound"):
            Box([(1.0, 0.0)])
        with pytest.raises(ValueError, match="positive"):
            Disk(radius=0.0)


class TestAdvectionDiffusionReaction:
    @pytest.mark.parametrize("conservative", [True, False])
    def test_forcing_and_flux(self, conservative: bool) -> None:
        problem = _problem(conservative)
        assert problem.ndim() == 2 and problem.ncomponents() == 1
        advection = 3 * x**2 * y if conservative else 2 * x**2 * y
        assert _same(problem.forcing()[0], -4 * y + advection - x**4 * y**2)
        q = problem.flux()[0]
        advective = x**3 * y if conservative else 0
        assert _same(q[0], -4 * x * y + advective) and _same(q[1], -2 * x**2)
        assert _same(problem.normal_flux()[0], q[0] * nx + q[1] * ny)
        assert not problem.is_transient()

    def test_boundary_data(self) -> None:
        problem = _problem(True)
        u = x**2 * y
        flux = (-4 * x * y + x**3 * y) * nx - 2 * x**2 * ny
        assert _same(problem.boundary_data("left")[0], u)
        assert _same(problem.boundary_data("right")[0], flux)
        assert _same(problem.boundary_data("bottom")[0], 3 * u + flux)
        assert _same(problem.boundary_data("top")[0], x * u + flux)

    def test_coefficients(self) -> None:
        problem = _problem(True)
        u = sp.Symbol("u")
        assert _same(problem.diffusion(), sp.Integer(2))
        assert _same(problem.reaction(), u**2)
        assert problem.velocity() == (x, sp.Integer(0))
        assert isinstance(problem.advection_form(), ConservativeAdvection)

    def test_default_velocity_is_zero(self) -> None:
        problem = AdvectionDiffusionReactionProblem("x**2*y", _SQUARE, _conditions())
        assert problem.velocity() == (0, 0)
        assert _same(problem.forcing()[0], -2 * y)

    def test_transient_forcing_includes_the_time_derivative(self) -> None:
        conditions: Dict[str, ConditionProtocol] = {
            "left": Dirichlet(),
            "right": Dirichlet(),
        }
        problem = AdvectionDiffusionReactionProblem(
            "x*(1-x)*exp(-t)", Box([(0.0, 1.0)]), conditions
        )
        assert problem.is_transient()
        expected = (2 - x * (1 - x)) * sp.exp(-t)
        assert _same(problem.forcing()[0], expected)

    def test_time_in_boundary_data_alone_is_transient(self) -> None:
        conditions: Dict[str, ConditionProtocol] = {
            "left": Dirichlet(),
            "right": Robin("1 + t"),
        }
        problem = AdvectionDiffusionReactionProblem(
            "x*(1-x)", Box([(0.0, 1.0)]), conditions
        )
        assert problem.is_transient()


class TestRefusals:
    def test_conditions_must_cover_the_boundary(self) -> None:
        missing = _conditions()
        del missing["top"]
        with pytest.raises(ValueError, match=r"missing \['top'\]"):
            AdvectionDiffusionReactionProblem("x", _SQUARE, missing)
        unknown = {**_conditions(), "circle": Dirichlet()}
        with pytest.raises(ValueError, match=r"unknown \['circle'\]"):
            AdvectionDiffusionReactionProblem("x", _SQUARE, unknown)

    def test_wrong_types(self) -> None:
        with pytest.raises(TypeError, match="DomainProtocol"):
            AdvectionDiffusionReactionProblem(
                "x",
                3.0,  # type: ignore[arg-type]
                _conditions(),
            )
        bad = {**_conditions(), "top": 3.0}
        with pytest.raises(TypeError, match="ConditionProtocol"):
            AdvectionDiffusionReactionProblem(
                "x",
                _SQUARE,
                bad,  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="AdvectionFormProtocol"):
            AdvectionDiffusionReactionProblem(
                "x",
                _SQUARE,
                _conditions(),
                advection_form=3.0,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"solution": "x*z"}, "expressions may depend only on"),
            ({"diffusion": "1 + u"}, "diffusion may depend only on"),
            ({"reaction": "w*u"}, "reaction may depend only on"),
            ({"velocity": ["1"]}, "velocity needs 2 entries"),
            ({"velocity": ["u", "0"]}, "velocity may depend only on"),
        ],
    )
    def test_symbols_and_shapes(self, kwargs: Dict[str, object], match: str) -> None:
        arguments: Dict[str, object] = {"solution": "x*y", **kwargs}
        with pytest.raises(ValueError, match=match):
            AdvectionDiffusionReactionProblem(
                domain=_SQUARE,
                conditions=_conditions(),
                **arguments,  # type: ignore[arg-type]
            )

    def test_robin_alpha_symbols(self) -> None:
        with pytest.raises(ValueError, match="alpha may depend only on"):
            Robin("w")

    def test_unknown_boundary_part(self) -> None:
        with pytest.raises(ValueError, match="no boundary part 'circle'"):
            _problem(True).boundary_data("circle")
