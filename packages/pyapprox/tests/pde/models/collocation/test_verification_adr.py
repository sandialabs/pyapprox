r"""Collocation advection-diffusion-reaction verified by manufactured
solutions.

The problem is stated as terms, its strong form
:math:`\nabla\cdot(-D\nabla u + [v u]) + [v\cdot\nabla u] - r u = f`
(the bracket matching the advection form), and the physics under test,
:class:`~pyapprox.pde.collocation.physics.advection_diffusion.
AdvectionDiffusionReaction`, is built from the same coefficient strings
by a system factory. Its reaction is linear, :math:`r(x) u`. Errors are
measured at the Chebyshev-Gauss-Lobatto nodes (L2 by the collocation
quadrature).

Cases:

- recovery (L2 error below 1e-10) of polynomial solutions of degree
  below the node count, where collocation is exact: 1D cubics on [0, 1]
  (8 nodes) under every pairing of Dirichlet, Neumann and Robin ends,
  and 2D cubics on [0, 1] x [-1, 1] (6 x 6 nodes) with all three kinds
  on its sides; variable D, v and r, both advection forms, so the
  Neumann and Robin data carry the advective flux exactly when the form
  is conservative;
- spectral decay (fitted decay of log error against 1/h) of analytic
  solutions: 1D and 2D under mixed conditions, and periodic ends in 1D
  and a 2D box periodic in x, in both advection forms;
- wrong-model controls, fitted algebraic rate below 0.5: Neumann data
  with the normal reversed; Robin ends without their spring term;
- the adapter's own pieces: the decay check on synthetic studies, and
  the quadrature L2 error of :math:`u^* = x` in 1D, 2D and 3D boxes
  against the closed form.

Refusals: transient and multi-field problems, a domain that is not a
box, one periodic side of a pair, a flux condition without a flux, and
a row that takes no conditions.
"""

import math
from typing import Dict, List, Optional, Sequence

import pytest

from pyapprox.pde.collocation.boundary.closure_rows import RobinPart, RobinRows
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import FluxNormalOperator
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    ChebyshevLevel,
    CollocationApproximation,
    CollocationDiscretizer,
    DecaysExponentially,
    boundary_conditions,
    collocation_system,
    field,
    forcing,
    nodal_vector,
    side_nodes,
    side_values,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Periodic,
    Robin,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.study import Study
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import coded, decays, recovers, stalls


class _ADR:
    """The physics under test and its strong form, from the same strings.

    Parameters
    ----------
    diffusivity : str
        :math:`D(x)`.
    velocity : Sequence[str], optional
        :math:`v(x)`, one entry per coordinate, fixed in time.
    reaction : str, optional
        :math:`r(x)`, the coefficient of the linear term :math:`r u`.
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
            terms.append(Source("u", f"-({self._reaction})*u"))
        return terms

    def physics(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> AdvectionDiffusionReaction[Array]:
        form: AdvectionFormProtocol = (
            ConservativeAdvection()
            if self._conservative
            else NonConservativeAdvection()
        )
        return AdvectionDiffusionReaction(
            level.basis(),
            level.bkd(),
            diffusion=field(problem, self._diffusivity, level),
            velocity=(
                None
                if self._velocity is None
                else nodal_vector(problem, self._velocity, level)
            ),
            reaction=(
                None
                if self._reaction is None
                else field(problem, self._reaction, level)
            ),
            forcing=forcing(problem, "u", level),
            advection_form=form,
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        return collocation_system(physics, problem, level, "u", flux=physics)

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


_UNIT = Box([(0.0, 1.0)])
_RECTANGLE = Box([(0.0, 1.0), (-1.0, 1.0)])


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("codes", ["DD", "DN", "ND", "RD", "DR", "RR", "NR"])
def test_recovers_cubic_1d(
    numpy_bkd: Backend[Array], codes: str, conservative: bool
) -> None:
    """u = 1 + x - x^2/2 + x^3 on [0, 1], D = 1 + x/2, v = 1 + x,
    r = -1: 8 nodes reproduce it to 1e-10 under every pairing of ends."""
    adr = _ADR("1 + x/2", ["1 + x"], "-1", conservative)
    problem = _problem("1 + x - x**2/2 + x**3", adr, _UNIT, coded(codes, _UNIT))
    recovers(problem, adr, 8, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("codes", ["DDDD", "DRDN", "NRDR", "RRRR"])
def test_recovers_cubic_2d(bkd: Backend[Array], codes: str, conservative: bool) -> None:
    """u = x y + y^2 - x^2 y on [0, 1] x [-1, 1] (sides left, right,
    bottom, top), D = 2 + x y, v = (1 - y, x/2), r = -1: 6 x 6 nodes
    reproduce it to 1e-10, on both backends."""
    adr = _ADR("2 + x*y", ["1 - y", "x/2"], "-1", conservative)
    problem = _problem(
        "x*y + y**2 - x**2*y", adr, _RECTANGLE, coded(codes, _RECTANGLE)
    )
    recovers(problem, adr, 6, bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_1d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """u = exp(x) sin(3 x) on [0, 1], Dirichlet left and Neumann right,
    variable D and v: the L2 error falls exponentially over 4-12 nodes
    (0.2 to 4e-9; observed decay 3.3 non-conservative, 3.4 conservative,
    required 1.5)."""
    adr = _ADR("1 + x/2", ["1 + x"], "-1", conservative)
    problem = _problem("exp(x)*sin(3*x)", adr, _UNIT, coded("DN", _UNIT))
    decays(problem, adr, [4, 6, 8, 10, 12], 1.5, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_2d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """u = exp(x/2) sin(2 y) + x y on [0, 1] x [-1, 1], Dirichlet,
    Robin and Neumann sides, rotating v: exponential decay over 4-12
    nodes per direction (observed 6.8 in both forms, required 3)."""
    adr = _ADR("1 + x*y/4", ["1 - y", "x/2"], "-1", conservative)
    problem = _problem(
        "exp(x/2)*sin(2*y) + x*y", adr, _RECTANGLE, coded("DRDN", _RECTANGLE)
    )
    decays(problem, adr, [4, 6, 8, 10, 12], 3.0, numpy_bkd)


_PERIODIC_1D: Dict[str, ConditionProtocol] = {"left": Periodic(), "right": Periodic()}


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_periodic_1d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """Both ends periodic, periodic D and v, absorption r = -1: the
    value and derivative match close the system, exponential decay over
    8-20 nodes (2.1 to 3e-7 L2; observed 2.07 non-conservative, 1.98
    conservative, required 1)."""
    adr = _ADR("1 + sin(2*pi*x)/2", ["1 + cos(2*pi*x)/4"], "-1", conservative)
    problem = _problem(
        "sin(2*pi*x) + cos(4*pi*x)/2 + 1", adr, _UNIT, _PERIODIC_1D
    )
    decays(problem, adr, [8, 12, 16, 20], 1.0, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_periodic_2d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """[0, 1] x [-1, 1] periodic in x, Dirichlet bottom and Robin top:
    exponential decay over 8-16 nodes per direction (1e-3 to 3e-11 L2;
    observed 6.8 in both forms, required 3); the bottom Dirichlet values
    agree on the paired corners."""
    adr = _ADR("1", ["1", "x/4 + 1/2"], "-1", conservative)
    conditions: Dict[str, ConditionProtocol] = {
        "left": Periodic(),
        "right": Periodic(),
        "bottom": Dirichlet(),
        "top": Robin(1.0),
    }
    problem = _problem("sin(2*pi*x)*exp(y/2) + y", adr, _RECTANGLE, conditions)
    decays(problem, adr, [8, 10, 12, 14, 16], 3.0, numpy_bkd)


class _NoSpring(_ADR):
    """Robin ends built with the spring term alpha u dropped (alpha = 0)
    but the problem's data kept."""

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        bkd = level.bkd()
        parts = []
        for part in ("left", "right"):
            indices, normals = side_nodes(level, part)
            parts.append(
                RobinPart(
                    indices,
                    FluxNormalOperator(bkd, indices, normals, physics),
                    0.0,
                    1.0,
                    side_values(problem, level, part, problem.boundary_data(part)["u"]),
                )
            )
        rows = RobinRows(bkd, level.npts(), 1, parts)
        return CollocationSystem(physics, None, [rows])


def test_robin_without_spring_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: Robin ends with alpha u dropped stall, so the
    Robin cases above can see the spring term."""
    adr = _NoSpring("1", ["1"], "-1")
    problem = _problem("sin(pi*x) + x + 1", adr, _UNIT, coded("RR", _UNIT))
    stalls(problem, adr, [6, 8, 10, 12], numpy_bkd)


class _ReversedNormal(_ADR):
    """Neumann data with the outward normal reversed: the core's flux
    data :math:`g` passed as :math:`-g`."""

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        bkd = level.bkd()
        left, _ = side_nodes(level, "left")
        right, normals = side_nodes(level, "right")
        data = problem.boundary_data("right")["u"]
        left_values = side_values(
            problem, level, "left", problem.boundary_data("left")["u"]
        )
        reversed_rows = RobinRows(
            bkd,
            level.npts(),
            1,
            [
                RobinPart(
                    right,
                    FluxNormalOperator(bkd, right, normals, physics),
                    0.0,
                    1.0,
                    side_values(problem, level, "right", -data),
                )
            ],
        )
        essential = DirichletRows(
            bkd, level.npts(), 1, [DirichletPart(left, left_values)]
        )
        return CollocationSystem(physics, essential, [reversed_rows])


def test_reversed_normal_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: Neumann data with the normal reversed make
    the error stop falling, so the cases above can detect a sign error
    in flux data."""
    adr = _ReversedNormal("1", ["1"], "-1")
    problem = _problem("sin(pi*x) + x", adr, _UNIT, coded("DN", _UNIT))
    stalls(problem, adr, [6, 8, 10, 12], numpy_bkd)


def test_time_dependence_is_declared_per_coefficient(
    numpy_bkd: Backend[Array],
) -> None:
    """A steady coefficient and a time-varying one can be mixed: each
    declares time dependence exactly when its own expression uses t;
    values at t = 0.5 match the expression, and a constant is a float."""
    bkd = numpy_bkd
    problem = _problem("x", _ADR("1"), _UNIT, coded("DN", _UNIT))
    (level,) = ChebyshevBox([5], bkd).levels(_UNIT)
    steady, varying = field(problem, "1 + x", level), field(problem, "x*exp(-t)", level)
    assert not isinstance(steady, float) and not isinstance(varying, float)
    assert not steady.is_time_dependent() and varying.is_time_dependent()
    x = level.mesh().points()[0]
    bkd.assert_allclose(varying.values(0.5), x * bkd.exp(bkd.asarray(-0.5)), rtol=1e-14)
    assert isinstance(field(problem, "2*pi", level), float)


class TestAdapterPieces:
    def test_decay_check(self) -> None:
        """Errors exp(-2/h) decay at 2 per unit 1/h: required 1.5 passes,
        2.5 fails; a zero error or a single level is refused."""
        h = (0.5, 0.25, 0.125)
        study = Study(h, tuple(math.exp(-2.0 / value) for value in h))
        result = DecaysExponentially(1.5).verify(study)
        assert result.passed and abs(result.observed - 2.0) < 1e-12
        assert not DecaysExponentially(2.5).verify(study).passed
        assert not DecaysExponentially(1.0).verify(Study(h, (1.0, 0.1, 0.0))).passed
        with pytest.raises(ValueError, match="at least two levels"):
            DecaysExponentially(1.0, last=1).verify(study)

    @pytest.mark.parametrize(
        "box, expected",
        [
            (Box([(0.0, 1.0)]), 1.0 / 3.0),
            (Box([(0.0, 1.0), (-1.0, 1.0)]), 2.0 / 3.0),
            (Box([(0.0, 1.0), (0.0, 2.0), (0.0, 1.0)]), 2.0 / 3.0),
        ],
        ids=["1d", "2d", "3d"],
    )
    def test_l2_error_by_quadrature(
        self, bkd: Backend[Array], box: Box, expected: float
    ) -> None:
        """u* = x against a zero state: the L2 error is the box integral
        of x^2, exact by the tensor quadrature and the affine map's
        Jacobian; the nodal error is max x = 1."""
        conditions: Dict[str, ConditionProtocol] = {
            side: Dirichlet() for side in box.boundaries()
        }
        problem = _problem("x", _ADR("1"), box, conditions)
        (level,) = ChebyshevBox([4], bkd).levels(box)
        approximation = CollocationApproximation(
            problem, "u", level, bkd.zeros((level.npts(),))
        )
        assert abs(approximation.l2_error() - math.sqrt(expected)) < 1e-13
        assert abs(approximation.nodal_error() - 1.0) < 1e-13


class TestRefusals:
    def test_transient(self, numpy_bkd: Backend[Array]) -> None:
        """This discretizer solves only the steady equation; on a
        transient problem it would drop du/dt, so it refuses."""
        adr = _ADR("1")
        discretizer = CollocationDiscretizer(ChebyshevBox([4], numpy_bkd), adr)
        transient = ManufacturedProblem(
            {"u": "exp(-t)*x"},
            [TimeDerivative("u"), *adr.terms(("x",))],
            _UNIT,
            coded("DN", _UNIT),
        )
        with pytest.raises(ValueError, match="steady"):
            discretizer.approximations(transient)

    def test_multi_field(self, numpy_bkd: Backend[Array]) -> None:
        """The discretizer solves one field; a two-field problem is
        refused rather than measured on its first field only."""
        adr = _ADR("1")
        discretizer = CollocationDiscretizer(ChebyshevBox([4], numpy_bkd), adr)
        two = ManufacturedProblem(
            {"u": "x", "w": "x"},
            [Divergence("u", ["-u_x"]), Divergence("w", ["-w_x"])],
            _UNIT,
            {"left": Dirichlet(), "right": Dirichlet()},
        )
        with pytest.raises(ValueError, match="one field"):
            discretizer.approximations(two)

    def test_not_a_box(self, numpy_bkd: Backend[Array]) -> None:
        """Chebyshev boxes mesh boxes only; a disk waits on a polar
        family."""
        with pytest.raises(TypeError, match="meshes a Box"):
            ChebyshevBox([4], numpy_bkd).levels(Disk())

    def test_half_periodic(self, numpy_bkd: Backend[Array]) -> None:
        """A periodic side needs its partner across the same coordinate
        periodic too: one alone has nothing to match."""
        adr = _ADR("1", None, "-1")
        conditions: Dict[str, ConditionProtocol] = {
            "left": Periodic(),
            "right": Dirichlet(),
        }
        problem = _problem("sin(2*pi*x)", adr, _UNIT, conditions)
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="'right', across x, is not"):
            boundary_conditions(problem, level, "u")

    def test_flux_condition_needs_a_flux(self, numpy_bkd: Backend[Array]) -> None:
        """Neumann and Robin rows prescribe a normal flux, so they need
        the flux whose normal component they prescribe."""
        adr = _ADR("1")
        problem = _problem("x", adr, _UNIT, coded("DN", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="prescribes a flux"):
            boundary_conditions(problem, level, "u")

    def test_row_without_conditions(self, numpy_bkd: Backend[Array]) -> None:
        adr = _ADR("1")
        problem = _problem("x", adr, _UNIT, coded("DD", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="takes no boundary conditions"):
            boundary_conditions(problem, level, "w")

    def test_helpers(self, numpy_bkd: Backend[Array]) -> None:
        """Coefficients may use only coordinates, t and constants (not
        the state); a fixed vector may not use t and needs one entry per
        coordinate."""
        problem = _problem("x", _ADR("1"), _UNIT, coded("DN", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="may depend only on"):
            field(problem, "u_x", level)
        with pytest.raises(ValueError, match="may depend only on"):
            nodal_vector(problem, ["x*t"], level)
        with pytest.raises(ValueError, match="one expression per coordinate"):
            nodal_vector(problem, ["1", "2"], level)
