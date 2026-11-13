r"""Every steady Galerkin physics written as terms, against the existing
manufactured-solution generators, in 1, 2 and 3 dimensions.

A consistency check, not an independent one: both sides are derivations.
Each strong form is written in its textbook sign convention, built from
jet symbols with sympy so one case covers every dimension; every
generator's code agrees with it. Forcing is compared at five points of
the unit interval, square or cube at :math:`t = 0.3`, evaluated on the
backend, to 1e-10.

- advection-diffusion-reaction, :math:`D = 1 + x^2`, :math:`v = (1, y,
  xz)`, :math:`R = u^2`, both advection forms, steady and transient;
- Helmholtz, :math:`-\Delta u - k^2 u` with :math:`k^2 = 1 + x` (the
  generator's docstring writes :math:`+k^2 u`, the screened Poisson
  operator; its code is the Helmholtz one); the screened sign is caught;
- linear elasticity (plane strain in 2D), :math:`\lambda = 2`, :math:`\mu
  = 1 + x`: :math:`-\nabla\cdot\sigma`, :math:`\sigma = \lambda\,
  \mathrm{tr}(\varepsilon) I + 2\mu\varepsilon`;
- hyperelasticity, compressible neo-Hookean :math:`P = \mu F + (\lambda
  \ln J - \mu) F^{-T}`, :math:`F = I + \nabla u`, :math:`\lambda = 2`,
  :math:`\mu = 1`: :math:`-\nabla\cdot P`;
- Stokes and Navier-Stokes, :math:`\mu = 1`: :math:`-\Delta u + (u\cdot
  \nabla) u + \nabla p`, continuity :math:`\nabla\cdot u`;
- Burgers (1D only, as is its generator): :math:`\partial_t u + (u^2/2 -
  \nu u_x)_x`, :math:`\nu = 0.1`.

Euler-Bernoulli has no generator; its analytic cantilever under a uniform
load, :math:`w = q_0 x^2 (6L^2 - 4Lx + x^2) / (24 EI)`, has forcing
:math:`EI\, w'''' = q_0` (:math:`EI = 2`, :math:`q_0 = 3`, :math:`L = 1`).
"""

from typing import Dict, List, Sequence

import pytest
import sympy as sp

from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.expressions.symbols import derivative, gradient, standard
from pyapprox.pde.manufactured.advection_diffusion import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.pde.manufactured.burgers import ManufacturedBurgers1D
from pyapprox.pde.manufactured.helmholtz import ManufacturedHelmholtz
from pyapprox.pde.manufactured.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.manufactured.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.manufactured.stokes import ManufacturedStokes
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

_GENERATOR_TIME = sp.Symbol("T")
_NDIMS = [1, 2, 3]
_COMPONENTS = ("ux", "uy", "uz")


def _box(ndim: int) -> Box:
    return Box([(0.0, 1.0)] * ndim)


def _dirichlet(ndim: int) -> Dict[str, ConditionProtocol]:
    return {side: Dirichlet() for side in _box(ndim).boundaries()}


def _scalar_solution(ndim: int) -> str:
    return "*".join(["sin(pi*x)", "(1 + y**2)", "cos(z)"][:ndim])


def _vector_solution(ndim: int, scale: str = "1") -> List[str]:
    factors = ["", "*(1 + y**2)", "*cos(z)"][:ndim]
    bases = ["x**2", "sin(x)", "exp(x)"][:ndim]
    return [f"{scale}*{base}{''.join(factors)}" for base in bases]


def _gradient_matrix(fields: Sequence[str], ndim: int) -> sp.Matrix:
    return sp.Matrix([list(gradient(name, standard(ndim))) for name in fields])


def _points(bkd: Backend[Array], ndim: int) -> Array:
    values = [
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.8, 0.4, 0.6, 0.35],
        [0.15, 0.45, 0.95, 0.25, 0.65],
    ]
    return bkd.asarray(values[:ndim])


def _assert_forcing_agrees(
    bkd: Backend[Array], problem: ManufacturedProblem, generated: Sequence[sp.Expr]
) -> None:
    ndim, rows = problem.ndim(), problem.fields()
    ours = [problem.forcing(row) for row in rows]
    theirs = [sp.sympify(f).subs(_GENERATOR_TIME, sp.Symbol("t")) for f in generated]
    evaluate = PointExpressions(
        ours + theirs, problem.names(), problem.coordinates(), bkd
    )
    values = evaluate(_points(bkd, ndim), t=0.3)
    nrows = len(rows)
    bkd.assert_allclose(values[:nrows], values[nrows:], rtol=1e-10, atol=1e-10)


class TestScalar:
    @pytest.mark.parametrize("ndim", _NDIMS)
    @pytest.mark.parametrize("conservative", [True, False])
    @pytest.mark.parametrize("transient", [False, True])
    def test_advection_diffusion_reaction(
        self,
        numpy_bkd: Backend[Array],
        ndim: int,
        conservative: bool,
        transient: bool,
    ) -> None:
        x, y, z = sp.symbols("x y z")
        diffusion = 1 + x**2
        velocity = [sp.Integer(1), y, x * z][:ndim]
        u, grad = derivative("u"), gradient("u", standard(ndim))
        terms: List[TermProtocol]
        form: AdvectionFormProtocol
        if conservative:
            flux = [-diffusion * g + v * u for g, v in zip(grad, velocity)]
            terms = [Divergence("u", flux), Source("u", -(u**2))]
            form = ConservativeAdvection()
        else:
            advection = sum(v * g for v, g in zip(velocity, grad))
            terms = [
                Divergence("u", [-diffusion * g for g in grad]),
                Source("u", advection - u**2),
            ]
            form = NonConservativeAdvection()
        solution = _scalar_solution(ndim)
        if transient:
            terms.append(TimeDerivative("u"))
            solution += "*exp(-{})"
        problem = ManufacturedProblem(
            {"u": solution.format("t")}, terms, _box(ndim), _dirichlet(ndim)
        )
        generator = ManufacturedAdvectionDiffusionReaction(
            solution.format("T"),
            ndim,
            "1 + x**2",
            "u**2",
            [str(v) for v in velocity],
            numpy_bkd,
            advection_form=form,
        )
        _assert_forcing_agrees(numpy_bkd, problem, [generator.expressions()["forcing"]])

    def _helmholtz(self, ndim: int, sign: int) -> ManufacturedProblem:
        x = sp.Symbol("x")
        terms: List[TermProtocol] = [
            Divergence("u", [-g for g in gradient("u", standard(ndim))]),
            Source("u", -sign * (1 + x) * derivative("u")),
        ]
        return ManufacturedProblem(
            {"u": _scalar_solution(ndim)}, terms, _box(ndim), _dirichlet(ndim)
        )

    @pytest.mark.parametrize("ndim", _NDIMS)
    def test_helmholtz(self, numpy_bkd: Backend[Array], ndim: int) -> None:
        generator = ManufacturedHelmholtz(
            _scalar_solution(ndim), ndim, "1 + x", numpy_bkd
        )
        forcing = [generator.expressions()["forcing"]]
        _assert_forcing_agrees(numpy_bkd, self._helmholtz(ndim, 1), forcing)
        with pytest.raises(AssertionError):
            _assert_forcing_agrees(numpy_bkd, self._helmholtz(ndim, -1), forcing)

    def test_burgers(self, numpy_bkd: Backend[Array]) -> None:
        problem = ManufacturedProblem(
            {"u": "sin(pi*x)*exp(-t)"},
            [TimeDerivative("u"), Divergence("u", ["u**2/2 - nu*u_x"])],
            _box(1),
            _dirichlet(1),
            constants={"nu": 0.1},
        )
        generator = ManufacturedBurgers1D("sin(pi*x)*exp(-T)", "0.1", numpy_bkd)
        _assert_forcing_agrees(numpy_bkd, problem, [generator.expressions()["forcing"]])

    def test_euler_bernoulli_cantilever(self, numpy_bkd: Backend[Array]) -> None:
        problem = ManufacturedProblem(
            {"u": "q0/(24*EI)*x**2*(6*L**2 - 4*L*x + x**2)"},
            [Divergence("u", ["EI*u_xxx"])],
            _box(1),
            _dirichlet(1),
            constants={"EI": 2.0, "q0": 3.0, "L": 1.0},
        )
        _assert_forcing_agrees(numpy_bkd, problem, [sp.Integer(3)])


class TestVector:
    @pytest.mark.parametrize("ndim", _NDIMS)
    def test_linear_elasticity(self, numpy_bkd: Backend[Array], ndim: int) -> None:
        fields = _COMPONENTS[:ndim]
        grad = _gradient_matrix(fields, ndim)
        strain = (grad + grad.T) / 2
        lam, mu = sp.Symbol("lam"), 1 + sp.Symbol("x")
        stress = lam * strain.trace() * sp.eye(ndim) + 2 * mu * strain
        terms: List[TermProtocol] = [
            Divergence(name, [-stress[i, j] for j in range(ndim)])
            for i, name in enumerate(fields)
        ]
        solution = _vector_solution(ndim)
        problem = ManufacturedProblem(
            dict(zip(fields, solution)),
            terms,
            _box(ndim),
            _dirichlet(ndim),
            constants={"lam": 2.0},
        )
        generator = ManufacturedLinearElasticityEquations(
            solution, ndim, "2", "1 + x", numpy_bkd
        )
        _assert_forcing_agrees(numpy_bkd, problem, generator.expressions()["forcing"])

    @pytest.mark.parametrize("ndim", _NDIMS)
    def test_hyperelasticity(self, numpy_bkd: Backend[Array], ndim: int) -> None:
        fields = _COMPONENTS[:ndim]
        deformation = sp.eye(ndim) + _gradient_matrix(fields, ndim)
        det = deformation.det()
        lam, mu = sp.symbols("lam mu")
        inverse_transpose = deformation.adjugate().T / det
        stress = mu * deformation + (lam * sp.log(det) - mu) * inverse_transpose
        terms: List[TermProtocol] = [
            Divergence(name, [-stress[i, j] for j in range(ndim)])
            for i, name in enumerate(fields)
        ]
        solution = _vector_solution(ndim, scale="1/10")
        problem = ManufacturedProblem(
            dict(zip(fields, solution)),
            terms,
            _box(ndim),
            _dirichlet(ndim),
            constants={"lam": 2.0, "mu": 1.0},
        )
        generator = ManufacturedHyperelasticityEquations(
            solution, ndim, NeoHookeanStress(2.0, 1.0), numpy_bkd
        )
        _assert_forcing_agrees(numpy_bkd, problem, generator.expressions()["forcing"])

    @pytest.mark.parametrize("ndim", _NDIMS)
    @pytest.mark.parametrize("navier_stokes", [False, True])
    def test_stokes(
        self, numpy_bkd: Backend[Array], ndim: int, navier_stokes: bool
    ) -> None:
        fields = _COMPONENTS[:ndim]
        grad = _gradient_matrix(fields, ndim)
        pressure = derivative("p")
        terms: List[TermProtocol] = []
        for i, name in enumerate(fields):
            flux = [-grad[i, j] + (pressure if i == j else 0) for j in range(ndim)]
            terms.append(Divergence(name, flux))
            if navier_stokes:
                velocity = [derivative(f) for f in fields]
                advection = sum(velocity[j] * grad[i, j] for j in range(ndim))
                terms.append(Source(name, advection))
        terms.append(Source("p", grad.trace()))
        x = sp.Symbol("x")
        solution = _vector_solution(ndim) + [str(x * (1 + x))]
        problem = ManufacturedProblem(
            dict(zip(list(fields) + ["p"], solution)),
            terms,
            _box(ndim),
            _dirichlet(ndim),
        )
        generator = ManufacturedStokes(solution, ndim, navier_stokes, numpy_bkd)
        _assert_forcing_agrees(numpy_bkd, problem, generator.expressions()["forcing"])
