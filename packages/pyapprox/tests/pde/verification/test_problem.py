r"""Symbolic verification problems built from terms.

Hand-derived values:

- Advection-diffusion-reaction, :math:`u^* = x^2 y` on the unit square
  with the constant :math:`D = 2`, :math:`v = (x, 0)`, :math:`R(u) =
  u^2`, gradient :math:`(2xy, x^2)`:

  - conservative, :math:`q = -D\nabla u + v u = (-4xy + x^3 y, -2x^2)`,
    :math:`f = -4y + 3x^2 y - x^4 y^2`;
  - non-conservative, :math:`q = -D\nabla u = (-4xy, -2x^2)`, source
    :math:`v\cdot\nabla u - u^2`, :math:`f = -4y + 2x^2 y - x^4 y^2`.

  Boundary data (Dirichlet, Neumann, Robin with constant and variable
  :math:`\alpha`) equal :math:`u^*`, :math:`q n` and :math:`q n - \alpha
  u^*`; ``Robin(0)`` gives ``Neumann()``'s data.
- Transient: :math:`u^* = x(1-x)e^{-t}`, :math:`\partial_t u -
  u_{xx}`, :math:`f = (2 - x(1-x)) e^{-t}`.
- Stokes, :math:`\mu = 1`, :math:`u^* = (y^2, x^2)`, :math:`p^* = xy`:
  :math:`f = (-2 + y, -2 + x)`, continuity row :math:`0`; only the
  velocity rows take boundary conditions, per row on one side (roller).
- ODE :math:`y' + k y` with :math:`k = 2`, :math:`y^* = e^{-t}`:
  :math:`f = e^{-t}`; second order :math:`y'' + y`, :math:`y^* =
  \sin 2t`: :math:`f = -3\sin 2t`; index-1 DAE :math:`y' - z = f_1`,
  :math:`z - y^2 = f_2`, :math:`(e^t, e^{2t})`: :math:`f = (e^t -
  e^{2t}, 0)`.
- A declared constant ``E = 3`` is that constant, not Euler's number.

Domains and refusals.
"""

from typing import Dict, List, Mapping, Union

import pytest
import sympy as sp

from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)

x, y, t, nx, ny = sp.symbols("x y t nx ny")
_SQUARE = Box([(0.0, 1.0), (0.0, 1.0)])
_LINE = Box([(0.0, 1.0)])


def _conditions() -> Dict[str, ConditionProtocol]:
    return {
        "left": Dirichlet(),
        "right": Neumann(),
        "bottom": Robin(3.0),
        "top": Robin("x"),
    }


def _adr(conservative: bool) -> ManufacturedProblem:
    terms: List[TermProtocol]
    if conservative:
        terms = [
            Divergence("u", ["-D*u_x + x*u", "-D*u_y"]),
            Source("u", "-u**2"),
        ]
    else:
        terms = [
            Divergence("u", ["-D*u_x", "-D*u_y"]),
            Source("u", "x*u_x - u**2"),
        ]
    return ManufacturedProblem(
        {"u": "x**2*y"}, terms, _SQUARE, _conditions(), constants={"D": 2.0}
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
        problem = _adr(conservative)
        assert problem.ndim() == 2 and problem.fields() == ("u",)
        advection = 3 * x**2 * y if conservative else 2 * x**2 * y
        assert _same(problem.forcing("u"), -4 * y + advection - x**4 * y**2)
        q = problem.flux("u")
        assert q is not None
        advective = x**3 * y if conservative else 0
        assert _same(q[0], -4 * x * y + advective) and _same(q[1], -2 * x**2)
        assert _same(problem.normal_flux("u"), q[0] * nx + q[1] * ny)
        assert not problem.is_transient()
        assert problem.time_order("u") == 0

    def test_boundary_data(self) -> None:
        problem = _adr(True)
        u = x**2 * y
        flux = (-4 * x * y + x**3 * y) * nx - 2 * x**2 * ny
        assert _same(problem.boundary_data("left")["u"], u)
        assert _same(problem.boundary_data("right")["u"], flux)
        assert _same(problem.boundary_data("bottom")["u"], flux - 3 * u)
        assert _same(problem.boundary_data("top")["u"], flux - x * u)
        assert isinstance(problem.conditions("left")["u"], Dirichlet)

    def test_robin_zero_is_neumann(self) -> None:
        problem = _adr(True)
        conditions = {**_conditions(), "bottom": Robin(0.0), "top": Neumann()}
        other = ManufacturedProblem(
            {"u": "x**2*y"},
            problem.terms("u"),
            _SQUARE,
            conditions,
            constants={"D": 2.0},
        )
        data = other.boundary_data
        assert _same(data("bottom")["u"], data("top")["u"])

    def test_transient_forcing_includes_the_time_derivative(self) -> None:
        problem = ManufacturedProblem(
            {"u": "x*(1-x)*exp(-t)"},
            [TimeDerivative("u"), Divergence("u", ["-u_x"])],
            _LINE,
            {"left": Dirichlet(), "right": Dirichlet()},
        )
        assert problem.is_transient() and problem.time_order("u") == 1
        assert _same(problem.forcing("u"), (2 - x * (1 - x)) * sp.exp(-t))

    def test_time_in_boundary_data_alone_is_transient(self) -> None:
        problem = ManufacturedProblem(
            {"u": "x*(1-x)"},
            [Divergence("u", ["-u_x"])],
            _LINE,
            {"left": Dirichlet(), "right": Robin("1 + t")},
        )
        assert problem.is_transient()


class TestConstants:
    def test_declared_e_is_the_constant(self) -> None:
        problem = ManufacturedProblem(
            {"u": "t"}, [Source("u", "E*u")], constants={"E": 3.0}
        )
        assert _same(problem.forcing("u"), 3 * t)

    def test_undeclared_e_is_eulers_number(self) -> None:
        problem = ManufacturedProblem({"u": "t"}, [Source("u", "E*u")])
        assert _same(problem.forcing("u"), sp.E * t)


class TestStokes:
    def _problem(self) -> ManufacturedProblem:
        terms: List[TermProtocol] = [
            Divergence("ux", ["-mu*ux_x + p", "-mu*ux_y"]),
            Divergence("uy", ["-mu*uy_x", "-mu*uy_y + p"]),
            Source("p", "ux_x + uy_y"),
        ]
        roller: Dict[str, ConditionProtocol] = {"ux": Dirichlet(), "uy": Neumann()}
        conditions: Dict[str, Union[ConditionProtocol, Mapping[str, ConditionProtocol]]]
        conditions = {
            "left": roller,
            "right": Dirichlet(),
            "bottom": Dirichlet(),
            "top": Dirichlet(),
        }
        return ManufacturedProblem(
            {"ux": "y**2", "uy": "x**2", "p": "x*y"},
            terms,
            _SQUARE,
            conditions,
            constants={"mu": 1.0},
        )

    def test_forcing(self) -> None:
        problem = self._problem()
        assert _same(problem.forcing("ux"), -2 + y)
        assert _same(problem.forcing("uy"), -2 + x)
        assert _same(problem.forcing("p"), sp.Integer(0))
        assert problem.flux("p") is None
        assert problem.boundary_rows() == ("ux", "uy")
        assert problem.time_order("p") == 0

    def test_roller(self) -> None:
        data = self._problem().boundary_data("left")
        assert _same(data["ux"], y**2)
        # q of uy is (-uy_x, -uy_y + p) = (-2x, xy).
        assert _same(data["uy"], -2 * x * nx + x * y * ny)


class TestTime:
    def test_ode(self) -> None:
        problem = ManufacturedProblem(
            {"w": "exp(-t)"},
            [TimeDerivative("w"), Source("w", "k*w")],
            constants={"k": 2.0},
        )
        assert problem.ndim() == 0 and problem.boundary_rows() == ()
        assert problem.is_transient()
        assert _same(problem.forcing("w"), sp.exp(-t))

    def test_second_order(self) -> None:
        problem = ManufacturedProblem(
            {"w": "sin(2*t)"}, [TimeDerivative("w", order=2), Source("w", "w")]
        )
        assert problem.time_order("w") == 2
        assert _same(problem.forcing("w"), -3 * sp.sin(2 * t))

    def test_dae(self) -> None:
        problem = ManufacturedProblem(
            {"w": "exp(t)", "v": "exp(2*t)"},
            [TimeDerivative("w"), Source("w", "-v"), Source("v", "v - w**2")],
        )
        assert _same(problem.forcing("w"), sp.exp(t) - sp.exp(2 * t))
        assert _same(problem.forcing("v"), sp.Integer(0))
        assert problem.time_order("v") == 0

    def test_field_names_free_without_space(self) -> None:
        problem = ManufacturedProblem({"y": "exp(t)"}, [Source("y", "y")])
        assert _same(problem.forcing("y"), sp.exp(t))
        with pytest.raises(ValueError, match="is a coordinate"):
            ManufacturedProblem({"y": "x"}, [Source("y", "y")], _SQUARE)


class TestRefusals:
    def test_conditions_must_cover_the_boundary(self) -> None:
        terms = [Divergence("u", ["-u_x", "-u_y"])]
        missing = _conditions()
        del missing["top"]
        with pytest.raises(ValueError, match=r"missing \['top'\]"):
            ManufacturedProblem({"u": "x"}, terms, _SQUARE, missing)
        unknown = {**_conditions(), "circle": Dirichlet()}
        with pytest.raises(ValueError, match=r"unknown \['circle'\]"):
            ManufacturedProblem({"u": "x"}, terms, _SQUARE, unknown)
        with pytest.raises(ValueError, match=r"unknown \['left'\]"):
            ManufacturedProblem(
                {"w": "t"}, [Source("w", "w")], conditions={"left": Dirichlet()}
            )

    def test_conditions_must_cover_the_boundary_rows(self) -> None:
        terms = [Divergence("u", ["-u_x"]), Divergence("v", ["-v_x"])]
        with pytest.raises(ValueError, match="each boundary row"):
            ManufacturedProblem(
                {"u": "x", "v": "x"},
                terms,
                _LINE,
                {"left": Dirichlet(), "right": {"u": Dirichlet()}},
            )

    def test_wrong_types(self) -> None:
        terms = [Divergence("u", ["-u_x", "-u_y"])]
        with pytest.raises(TypeError, match="DomainProtocol"):
            ManufacturedProblem(
                {"u": "x"},
                terms,
                3.0,  # type: ignore[arg-type]
                _conditions(),
            )
        bad = {**_conditions(), "top": 3.0}
        with pytest.raises(TypeError, match="ConditionProtocol"):
            ManufacturedProblem(
                {"u": "x"},
                terms,
                _SQUARE,
                bad,  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="TermProtocol"):
            ManufacturedProblem({"u": "t"}, [3.0])  # type: ignore[list-item]

    def test_rows_and_terms(self) -> None:
        with pytest.raises(ValueError, match="acts on row 'v'"):
            ManufacturedProblem({"u": "t"}, [Source("u", "u"), Source("v", "u")])
        with pytest.raises(ValueError, match=r"rows without terms: \['v'\]"):
            ManufacturedProblem({"u": "t", "v": "t"}, [Source("u", "u")])
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            TimeDerivative("u", order=3)
        with pytest.raises(ValueError, match="field name"):
            Source("u_x", "u")

    def test_symbols(self) -> None:
        with pytest.raises(ValueError, match="may depend only on"):
            ManufacturedProblem({"u": "u_x"}, [Source("u", "u")], _LINE)
        with pytest.raises(ValueError, match="unknown name 'w'"):
            ManufacturedProblem({"u": "x"}, [Source("u", "w*u")], _LINE)
        with pytest.raises(ValueError, match="has 1 entries"):
            ManufacturedProblem(
                {"u": "x"}, [Divergence("u", ["-u_x"])], _SQUARE, _conditions()
            )
        robin_in_normal = {**_conditions(), "top": Robin("nx")}
        with pytest.raises(ValueError, match="may depend only on"):
            ManufacturedProblem(
                {"u": "x"},
                [Divergence("u", ["-u_x", "-u_y"])],
                _SQUARE,
                robin_in_normal,
            )

    def test_lookups(self) -> None:
        problem = ManufacturedProblem({"w": "t"}, [Source("w", "w")])
        with pytest.raises(ValueError, match="no boundary part 'left'"):
            problem.boundary_data("left")
        with pytest.raises(ValueError, match="no divergence terms"):
            problem.normal_flux("w")
        with pytest.raises(ValueError, match="no row 'v'"):
            problem.forcing("v")
