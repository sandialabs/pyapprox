r"""Classical exact solutions give zero forcing.

An independent check of the term machinery: each solution below solves
its homogeneous equation by a classical result, not by our derivation,
so the problem's forcing must vanish. Checked at five points of the unit
interval or square at :math:`t = 0.3`, evaluated on the backend, to
1e-10.

- advection-diffusion boundary layer :math:`(e^{Pe\,x} - 1)/(e^{Pe} -
  1)` for :math:`-u'' + Pe\,u' = 0`, :math:`Pe = 5`; its Dirichlet data
  are 0 and 1 at the ends;
- decaying heat mode :math:`e^{-D\pi^2 t}\sin\pi x`, :math:`D = 0.5`;
- advected heat kernel :math:`(4\pi D t)^{-1/2} e^{-(x - ct)^2/(4Dt)}`
  for :math:`\partial_t u + (c u - D u_x)_x = 0`, :math:`c = 0.7`;
- Helmholtz plane wave :math:`\cos(3x + 4y)`, :math:`k^2 = 25`, and the
  radial Bessel wave :math:`J_0(k r)`, :math:`k = 5`, through a custom
  function;
- Burgers viscous shock :math:`c - a\tanh(a(x - ct)/(2\nu))`,
  :math:`a = 0.8`, :math:`c = 0.3`, :math:`\nu = 0.1`;
- Vlasov free streaming :math:`e^{-v^2}(2 + \sin(x - vt))` and the
  equilibrium :math:`e^{-v^2/2 + \cos x}` in the potential
  :math:`\phi = \cos x`;
- Stokes Poiseuille flow :math:`(y(1 - y), 0)`, :math:`p = -2x`;
  Navier-Stokes Taylor-Green vortex, :math:`\nu = 0.1`; Kovasznay flow,
  :math:`Re = 40`;
- linear elasticity rigid motion and uniform strain (variable
  :math:`\mu`); neo-Hookean homogeneous deformation.
"""

import math
from typing import Any, Dict, List, Mapping, Optional

import pytest
import sympy as sp

from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.expressions.symbols import gradient, standard
from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.names import CustomFunction


def _bessel(bkd: Backend[Any], order: Any, argument: Any) -> Any:
    from scipy.special import jv

    return jv(order, argument)


def _dirichlet(box: Box) -> Dict[str, ConditionProtocol]:
    return {side: Dirichlet() for side in box.boundaries()}


def _assert_zero_forcing(bkd: Backend[Array], problem: ManufacturedProblem) -> None:
    coords = problem.coordinates()
    evaluate = PointExpressions(
        [problem.forcing(row) for row in problem.fields()],
        problem.names(),
        coords,
        bkd,
    )
    rows = [[0.1, 0.3, 0.5, 0.7, 0.9], [0.2, 0.8, 0.4, 0.6, 0.35]]
    points = bkd.asarray(rows[: len(coords)])
    values = evaluate(points, t=0.3)
    bkd.assert_allclose(values, bkd.zeros(tuple(values.shape)), atol=1e-10)


def _problem(
    solution: Mapping[str, str],
    terms: List[TermProtocol],
    ndim: int,
    constants: Optional[Mapping[str, float]] = None,
    functions: Optional[Mapping[str, CustomFunction]] = None,
) -> ManufacturedProblem:
    box = Box([(0.0, 1.0)] * ndim)
    return ManufacturedProblem(
        solution, terms, box, _dirichlet(box), constants, functions
    )


class TestScalar:
    def test_boundary_layer(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"u": "(exp(Pe*x) - 1)/(exp(Pe) - 1)"},
            [Divergence("u", ["-u_x"]), Source("u", "Pe*u_x")],
            1,
            {"Pe": 5.0},
        )
        _assert_zero_forcing(numpy_bkd, problem)
        x = sp.Symbol("x")
        left = problem.boundary_data("left")["u"].subs(x, 0)
        right = problem.boundary_data("right")["u"].subs(x, 1)
        assert abs(float(left)) < 1e-14 and abs(float(right) - 1.0) < 1e-14

    def test_heat_mode(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"u": "exp(-D*pi**2*t)*sin(pi*x)"},
            [TimeDerivative("u"), Divergence("u", ["-D*u_x"])],
            1,
            {"D": 0.5},
        )
        _assert_zero_forcing(numpy_bkd, problem)

    def test_advected_heat_kernel(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"u": "exp(-(x - c*t)**2/(4*D*t))/sqrt(4*pi*D*t)"},
            [TimeDerivative("u"), Divergence("u", ["c*u - D*u_x"])],
            1,
            {"D": 0.5, "c": 0.7},
        )
        _assert_zero_forcing(numpy_bkd, problem)

    def test_helmholtz_plane_wave(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"u": "cos(3*x + 4*y)"},
            [Divergence("u", ["-u_x", "-u_y"]), Source("u", "-ksq*u")],
            2,
            {"ksq": 25.0},
        )
        _assert_zero_forcing(numpy_bkd, problem)

    def test_helmholtz_bessel_wave(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"u": "besselj(0, k*sqrt(x**2 + y**2))"},
            [Divergence("u", ["-u_x", "-u_y"]), Source("u", "-k**2*u")],
            2,
            {"k": 5.0},
            {"besselj": CustomFunction(sp.besselj, _bessel)},
        )
        _assert_zero_forcing(numpy_bkd, problem)

    def test_burgers_viscous_shock(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"u": "c - a*tanh(a*(x - c*t)/(2*nu))"},
            [TimeDerivative("u"), Divergence("u", ["u**2/2 - nu*u_x"])],
            1,
            {"a": 0.8, "c": 0.3, "nu": 0.1},
        )
        _assert_zero_forcing(numpy_bkd, problem)


class TestVlasov:
    def _phase_problem(self, solution: str, source: str) -> ManufacturedProblem:
        box = Box([(0.0, 1.0), (-1.0, 1.0)], ("x", "v"))
        return ManufacturedProblem(
            {"f": solution},
            [TimeDerivative("f"), Source("f", source)],
            box,
            _dirichlet(box),
        )

    def test_free_streaming(self, numpy_bkd: Backend[Array]) -> None:
        problem = self._phase_problem("exp(-v**2)*(2 + sin(x - v*t))", "v*f_x")
        _assert_zero_forcing(numpy_bkd, problem)

    def test_equilibrium(self, numpy_bkd: Backend[Array]) -> None:
        # Electrons: v f_x - E f_v with E = -phi_x = sin(x).
        problem = self._phase_problem("exp(-v**2/2 + cos(x))", "v*f_x - sin(x)*f_v")
        _assert_zero_forcing(numpy_bkd, problem)


def _flow_terms(nu: str, transient: bool) -> List[TermProtocol]:
    terms: List[TermProtocol] = [
        Divergence("ux", [f"-{nu}*ux_x + p", f"-{nu}*ux_y"]),
        Divergence("uy", [f"-{nu}*uy_x", f"-{nu}*uy_y + p"]),
        Source("p", "ux_x + uy_y"),
    ]
    if transient:
        terms += [TimeDerivative("ux"), TimeDerivative("uy")]
    return terms


def _advection() -> List[TermProtocol]:
    return [
        Source("ux", "ux*ux_x + uy*ux_y"),
        Source("uy", "ux*uy_x + uy*uy_y"),
    ]


class TestFlow:
    def test_poiseuille(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem(
            {"ux": "y*(1 - y)", "uy": "0", "p": "-2*x"}, _flow_terms("1", False), 2
        )
        _assert_zero_forcing(numpy_bkd, problem)

    def test_taylor_green(self, numpy_bkd: Backend[Array]) -> None:
        decay = "exp(-2*nu*t)"
        problem = _problem(
            {
                "ux": f"cos(x)*sin(y)*{decay}",
                "uy": f"-sin(x)*cos(y)*{decay}",
                "p": f"-(cos(2*x) + cos(2*y))/4*{decay}**2",
            },
            _flow_terms("nu", True) + _advection(),
            2,
            {"nu": 0.1},
        )
        _assert_zero_forcing(numpy_bkd, problem)

    def test_kovasznay(self, numpy_bkd: Backend[Array]) -> None:
        reynolds = 40.0
        lam = reynolds / 2 - math.sqrt(reynolds**2 / 4 + 4 * math.pi**2)
        problem = _problem(
            {
                "ux": "1 - exp(lam*x)*cos(2*pi*y)",
                "uy": "lam/(2*pi)*exp(lam*x)*sin(2*pi*y)",
                "p": "(1 - exp(2*lam*x))/2",
            },
            _flow_terms("nu", False) + _advection(),
            2,
            {"lam": lam, "nu": 1 / reynolds},
        )
        _assert_zero_forcing(numpy_bkd, problem)


def _stress_terms(stress: sp.Matrix) -> List[TermProtocol]:
    return [
        Divergence(name, [-stress[i, j] for j in range(2)])
        for i, name in enumerate(("ux", "uy"))
    ]


class TestSolids:
    @pytest.mark.parametrize(
        "solution, mu",
        [
            # A rigid motion has no strain, so no stress for any moduli.
            ({"ux": "0.3 - 0.2*y", "uy": "-0.1 + 0.2*x"}, "1 + x"),
            # Uniform strain is stress-free in the divergence only for
            # uniform moduli.
            ({"ux": "0.01*x", "uy": "-0.02*y"}, "1"),
        ],
    )
    def test_linear_elasticity(
        self, numpy_bkd: Backend[Array], solution: Dict[str, str], mu: str
    ) -> None:
        coords = standard(2)
        grad = sp.Matrix([list(gradient(n, coords)) for n in ("ux", "uy")])
        strain = (grad + grad.T) / 2
        stress = 2 * strain.trace() * sp.eye(2) + 2 * sp.sympify(mu) * strain
        _assert_zero_forcing(numpy_bkd, _problem(solution, _stress_terms(stress), 2))

    def test_neo_hookean_homogeneous(self, numpy_bkd: Backend[Array]) -> None:
        coords = standard(2)
        grad = sp.Matrix([list(gradient(n, coords)) for n in ("ux", "uy")])
        deformation = sp.eye(2) + grad
        det = deformation.det()
        stress = deformation + (2 * sp.log(det) - 1) * deformation.adjugate().T / det
        problem = _problem(
            {"ux": "0.1*x + 0.05*y", "uy": "0.02*x - 0.1*y"},
            _stress_terms(stress),
            2,
        )
        _assert_zero_forcing(numpy_bkd, problem)
