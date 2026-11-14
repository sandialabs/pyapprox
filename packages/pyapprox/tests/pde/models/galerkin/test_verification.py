r"""Galerkin advection-diffusion-reaction verified by manufactured solutions.

The problem is stated as terms, its strong form
:math:`\nabla\cdot(-D\nabla u + [v u]) + [v\cdot\nabla u] - R(u) = f`
(the bracket matching the advection form), and the physics under test,
:class:`~pyapprox.pde.galerkin.physics.AdvectionDiffusionReaction`, is
built from the same coefficient strings by a system factory. Errors are
true :math:`L^2` errors, integrated by skfem quadrature against
:math:`u^*`.

Cases (fitted rate over the last three levels, tolerance 0.1):

- P1 in 1D (cells 8-64), 2D quads (4-32) and 3D hexes (2-8), and P2 in
  1D and 2D (one order higher): Dirichlet, Neumann and Robin sides
  together, variable :math:`D`, variable :math:`v`, cubic reaction,
  both advection forms;
- the unit disk, P1 triangles (refinements 2-5), Robin everywhere: rate
  2 despite the :math:`O(h^2)` polygonal boundary;
- a variable Robin coefficient :math:`1 + x` on a square side;
- P2 exact recovery (error below 1e-10) of quadratics in 1D and 2D,
  quads and triangles, with natural conditions;
- wrong-model control: natural data with the normal reversed stalls
  (fitted rate below 0.5).

Refusals: periodic sides, transient and multi-field problems, a box
with named coordinates.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import sympy as sp
from numpy.typing import NDArray

from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC, NeumannBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    GalerkinDiscretizer,
    Lagrange,
    MeshFamilyProtocol,
    SkfemDisk,
    SkfemPoints,
    StructuredBox,
    boundary_conditions,
    field,
    forcing,
    state_law,
    vector_field,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.study import (
    Converges,
    L2Error,
    Recovers,
    Stalls,
    run_study,
    verify,
)
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd

_Np = NDArray[Any]


class _ADR:
    """The physics under test and its strong form, from the same strings.

    Parameters
    ----------
    diffusivity : str
        :math:`D(x)`.
    velocity : Sequence[str], optional
        :math:`v(x)`, one entry per coordinate.
    reaction : str, optional
        :math:`R(x, u)`, positive a source.
    conservative : bool
        The advection form.
    """

    def __init__(
        self,
        diffusivity: str,
        velocity: Optional[Sequence[str]] = None,
        reaction: Optional[str] = None,
        conservative: bool = False,
    ) -> None:
        self._diffusivity, self._velocity = diffusivity, velocity
        self._reaction, self._conservative = reaction, conservative

    def terms(self, coords: Sequence[str]) -> List[TermProtocol]:
        """The strong form as terms on row ``u``."""
        fluxes = [f"-({self._diffusivity})*u_{c}" for c in coords]
        terms: List[TermProtocol] = []
        if self._velocity is not None:
            if self._conservative:
                fluxes = [f"{q} + ({v})*u" for q, v in zip(fluxes, self._velocity)]
            else:
                transport = [f"({v})*u_{c}" for v, c in zip(self._velocity, coords)]
                terms.append(Source("u", " + ".join(transport)))
        terms.append(Divergence("u", fluxes))
        if self._reaction is not None:
            terms.append(Source("u", f"-({self._reaction})"))
        return terms

    def physics(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> AdvectionDiffusionReaction[_Np]:
        form: AdvectionFormProtocol = (
            ConservativeAdvection()
            if self._conservative
            else NonConservativeAdvection()
        )
        return AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=field(problem, self._diffusivity),
            bkd=NumpyBkd(),
            velocity=(
                None
                if self._velocity is None
                else vector_field(problem, self._velocity)
            ),
            reaction=(
                None
                if self._reaction is None
                else state_law(problem, self._reaction, "u")
            ),
            forcing=forcing(problem, "u"),
            advection_form=form,
        )

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        return compose_galerkin_system(
            self.physics(problem, basis), boundary_conditions(problem, basis, "u")
        )

    def __repr__(self) -> str:
        return (
            f"_ADR({self._diffusivity!r}, {self._velocity!r}, {self._reaction!r}, "
            f"conservative={self._conservative})"
        )


def _problem(
    solution: str,
    adr: _ADR,
    domain: DomainProtocol,
    conditions: Dict[str, ConditionProtocol],
) -> ManufacturedProblem:
    return ManufacturedProblem(
        {"u": solution}, adr.terms(domain.coordinates()), domain, conditions
    )


_MIXED_1D: Dict[str, ConditionProtocol] = {"left": Dirichlet(), "right": Neumann()}
_MIXED_2D: Dict[str, ConditionProtocol] = {
    "left": Dirichlet(),
    "right": Neumann(),
    "bottom": Robin(1.5),
    "top": Dirichlet(),
}
_MIXED_3D: Dict[str, ConditionProtocol] = {
    "left": Dirichlet(),
    "right": Neumann(),
    "bottom": Robin(2.0),
    "top": Dirichlet(),
    "back": Dirichlet(),
    "front": Neumann(),
}


def _converges(
    problem: ManufacturedProblem,
    meshes: MeshFamilyProtocol,
    degree: int,
    adr: _ADR,
    rate: float,
) -> None:
    discretizer = GalerkinDiscretizer(meshes, Lagrange(degree), adr)
    result = verify(problem, discretizer, L2Error(), Converges(rate, last=3))
    assert result.passed, result.message


@pytest.mark.parametrize("conservative", [False, True])
class TestRates:
    def test_1d(self, conservative: bool) -> None:
        """1D, Dirichlet left and Neumann right, variable D and v, cubic
        reaction: L2 rate 2 for P1 and 3 for P2."""
        adr = _ADR("1 + x/2", ["1 + x"], "-u**3", conservative)
        box = Box([(0.0, 1.0)])
        problem = _problem("sin(pi*x) + x", adr, box, _MIXED_1D)
        _converges(problem, StructuredBox([8, 16, 32, 64]), 1, adr, 2.0)
        _converges(problem, StructuredBox([4, 8, 16, 32]), 2, adr, 3.0)

    def test_2d(self, conservative: bool) -> None:
        """Unit square, quads, Dirichlet, Neumann and Robin sides,
        variable D and rotating v, nonlinear reaction: L2 rate 2 for P1
        and 3 for P2."""
        adr = _ADR("1 + x*y/4", ["1 - y", "x/2"], "-u - u**3/4", conservative)
        box = Box([(0.0, 1.0), (0.0, 1.0)])
        problem = _problem("exp(x/2)*sin(pi*y) + x*y", adr, box, _MIXED_2D)
        _converges(problem, StructuredBox([4, 8, 16, 32]), 1, adr, 2.0)
        _converges(problem, StructuredBox([2, 4, 8, 16]), 2, adr, 3.0)

    def test_3d(self, conservative: bool) -> None:
        """Unit cube, hexes, all three condition kinds: P1 L2 rate 2."""
        adr = _ADR("1 + z/2", ["1", "0", "x/2"], "-u", conservative)
        box = Box([(0.0, 1.0)] * 3)
        problem = _problem("sin(x)*cos(y)*exp(z)", adr, box, _MIXED_3D)
        _converges(problem, StructuredBox([2, 4, 8]), 1, adr, 2.0)

    def test_disk(self, conservative: bool) -> None:
        """Unit disk, Robin on the curved boundary with the mesh's own
        facet normals: P1 L2 rate 2, so the O(h^2) polygon error does not
        spoil it."""
        adr = _ADR("1 + x**2/2", ["y", "-x"], "-u", conservative)
        problem = _problem("exp(x)*cos(y)", adr, Disk(), {"circle": Robin(2.0)})
        _converges(problem, SkfemDisk([2, 3, 4, 5]), 1, adr, 2.0)


def test_variable_robin() -> None:
    """A Robin coefficient that varies along the side (1 + x on the top)
    reaches the stack as a function: P1 L2 rate 2."""
    adr = _ADR("1", ["1", "0"], "-u")
    box = Box([(0.0, 1.0), (0.0, 1.0)])
    conditions: Dict[str, ConditionProtocol] = {
        **_MIXED_2D,
        "top": Robin("1 + x"),
    }
    problem = _problem("cos(x)*exp(y)", adr, box, conditions)
    _converges(problem, StructuredBox([4, 8, 16, 32]), 1, adr, 2.0)


@pytest.mark.parametrize(
    "ndim, simplex, solution",
    [
        (1, False, "1 + x - x**2/2"),
        (2, False, "1 + x*y - y**2/2"),
        (2, True, "1 + x*y - y**2/2"),
    ],
)
def test_p2_recovers_quadratics(ndim: int, simplex: bool, solution: str) -> None:
    """A quadratic lies in the P2 space, so with constant coefficients
    and natural conditions P2 reproduces it to round-off on every mesh:
    any wrong sign or missing boundary term shows as an O(1) error."""
    adr = _ADR("2", ["1"] * ndim, "-u")
    box = Box([(0.0, 1.0)] * ndim)
    conditions = _MIXED_1D if ndim == 1 else _MIXED_2D
    problem = _problem(solution, adr, box, conditions)
    discretizer = GalerkinDiscretizer(StructuredBox([2, 3], simplex), Lagrange(2), adr)
    result = verify(problem, discretizer, L2Error(), Recovers(1e-10))
    assert result.passed, result.message


class _ReversedNormal(_ADR):
    """Natural data with the outward normal reversed: the core's flux
    data :math:`g` passed as this stack's :math:`-q\\,n` without the
    sign change."""

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        bkd = NumpyBkd()
        conditions: List[BoundaryConditionRole[_Np]] = []
        for part in ("left", "right"):
            data = problem.boundary_data(part)["u"]
            if isinstance(problem.conditions(part)["u"], Neumann):
                conditions.append(
                    NeumannBC(basis, part, _Unnegated(data, problem), bkd)
                )
            else:
                conditions.append(DirichletBC(basis, part, field(problem, data), bkd))
        return compose_galerkin_system(self.physics(problem, basis), conditions)


class _Unnegated:
    def __init__(self, data: sp.Expr, problem: ManufacturedProblem) -> None:
        self._points = SkfemPoints([data], problem)

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        values: _Np = self._points(coords, normals=normal)[0]
        return values

    def is_time_dependent(self) -> bool:
        return False


def test_reversed_normal_stalls() -> None:
    """Wrong-model control: Neumann data with the normal reversed make
    the error stop falling, so the rate tests above can detect a sign
    error in natural data."""
    adr = _ReversedNormal("1", ["1"], "-u")
    problem = _problem("sin(pi*x) + x", adr, Box([(0.0, 1.0)]), _MIXED_1D)
    discretizer = GalerkinDiscretizer(StructuredBox([8, 16, 32, 64]), Lagrange(1), adr)
    study = run_study(problem, discretizer, L2Error())
    result = Stalls(below=0.5).verify(study)
    assert result.passed, result.message


def test_time_dependence_is_declared_per_supplier() -> None:
    """A steady forcing and time-varying boundary data can be mixed: each
    supplier declares time dependence exactly when its own expression
    uses t, so the stack caches the steady ones and re-evaluates the
    others; values at t = 0.5 match the expression."""
    problem = _problem("x", _ADR("1"), Box([(0.0, 1.0)]), _MIXED_1D)
    steady, varying = field(problem, "1 + x"), field(problem, "x*exp(-t)")
    assert not isinstance(steady, float) and not isinstance(varying, float)
    assert not steady.is_time_dependent() and varying.is_time_dependent()
    pts = np.array([[0.25, 0.75]])
    np.testing.assert_allclose(varying(pts, 0.5), pts[0] * np.exp(-0.5), rtol=1e-14)
    assert isinstance(field(problem, "2*pi"), float)


class TestRefusals:
    def test_periodic(self) -> None:
        """This stack has no periodic conditions, so a periodic side is
        refused with a clear error rather than silently dropped."""
        adr = _ADR("1")
        problem = _problem(
            "sin(2*pi*x)",
            adr,
            Box([(0.0, 1.0)]),
            {"left": Periodic(), "right": Periodic()},
        )
        discretizer = GalerkinDiscretizer(StructuredBox([4]), Lagrange(1), adr)
        with pytest.raises(ValueError, match="no periodic conditions"):
            discretizer.approximations(problem)

    def test_transient(self) -> None:
        """This discretizer solves only the steady equation F(u) = 0; on a
        transient problem it would drop du/dt and report a meaningless
        error, so it refuses. Transient problems will get their own
        discretizer, reusing the same helpers."""
        adr = _ADR("1")
        discretizer = GalerkinDiscretizer(StructuredBox([4]), Lagrange(1), adr)
        transient = ManufacturedProblem(
            {"u": "exp(-t)*x"},
            [TimeDerivative("u"), *adr.terms(("x",))],
            Box([(0.0, 1.0)]),
            _MIXED_1D,
        )
        with pytest.raises(ValueError, match="steady"):
            discretizer.approximations(transient)

    def test_multi_field(self) -> None:
        """Errors are measured for one field, so a two-field problem is
        refused until vector and mixed bases are supported."""
        adr = _ADR("1")
        box = Box([(0.0, 1.0)])
        discretizer = GalerkinDiscretizer(StructuredBox([4]), Lagrange(1), adr)
        two = ManufacturedProblem(
            {"u": "x", "w": "x"},
            [Divergence("u", ["-u_x"]), Divergence("w", ["-w_x"])],
            box,
            {"left": Dirichlet(), "right": Dirichlet()},
        )
        with pytest.raises(ValueError, match="single-field"):
            discretizer.approximations(two)

    def test_named_box(self) -> None:
        """Structured meshes name their sides left, right, ...; a box with
        named coordinates (sides xmin, vmax, ...) would not match them, so
        it is refused."""
        with pytest.raises(ValueError, match="names the sides"):
            StructuredBox([4]).meshes(Box([(0.0, 1.0), (-1.0, 1.0)], ("x", "v")))

    def test_helpers(self) -> None:
        """Coefficients may use only coordinates, t and constants (not
        the state); a vector needs one entry per coordinate; a state law
        needs a declared field and may not use t, which this stack's
        reaction laws do not take."""
        box = Box([(0.0, 1.0)])
        problem = _problem("x", _ADR("1"), box, _MIXED_1D)
        with pytest.raises(ValueError, match="may depend only on"):
            field(problem, "u_x")
        with pytest.raises(ValueError, match="one expression per coordinate"):
            vector_field(problem, ["1", "2"])
        with pytest.raises(ValueError, match="not a field"):
            state_law(problem, "q**2", "q")
        with pytest.raises(ValueError, match="may depend only on"):
            state_law(problem, "u*t", "u")
