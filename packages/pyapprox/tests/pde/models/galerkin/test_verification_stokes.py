r"""Galerkin Stokes and Navier-Stokes verified by manufactured solutions.

:class:`~pyapprox.pde.galerkin.physics.stokes.StokesPhysics` solves
:math:`-\nu \Delta u + (u\cdot\nabla) u + \nabla p = f` (the convection
for Navier-Stokes only) with :math:`\nabla\cdot u = f_p` on Taylor-Hood
elements, velocity :math:`(u, v)` and pressure :math:`p` as fields. The
problem states each momentum row as :math:`\nabla\cdot(-\nu \nabla u_i +
p e_i)` plus the convection as a source, so its natural data are
:math:`-(\nu \partial_n u - p n)`, which the adapter negates into this
stack's pseudo-traction; the continuity row is the source
:math:`\nabla\cdot u` and takes no boundary conditions. Where the
velocity is prescribed on the whole boundary, the pressure is fixed by
one DOF set to the exact value. Velocity and pressure converge at
different rates, so they are measured apart.

Cases:

- exact recovery (L2 below 1e-10) of the old suite's four steady cases,
  1D and 2D, Stokes and Navier-Stokes, with all-Dirichlet velocity (the
  old suite also prescribed the pressure on every side, which the
  continuity row does not take), and the 2D Stokes case with traction
  and Robin sides;
- rates (fitted over the last three levels, within 0.1): velocity 3 and
  pressure 2, on 1D, 2D quads and triangles, all-Dirichlet and with
  traction, Robin and mixed-component sides; Navier-Stokes in 1D and 2D
  from a zero initial guess;
- transient: the old suite's time-linear cases, 1D and 2D, Stokes and
  Navier-Stokes, stepped exactly (below 1e-10) by backward Euler and
  Crank-Nicolson;
- wrong-model controls, which stall (fitted rate below 0.5): the
  continuity forcing given with the wrong sign, and the Navier-Stokes
  problem solved without convection;
- refusals: Taylor-Hood below degree 2, a measurement splitting the
  velocity, a basis with more components than the problem has fields,
  and the mixed-field helpers given the wrong kind of field.
"""

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import ButcherTableau, backward_euler, crank_nicolson
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.galerkin.basis import LagrangeBasis, MixedBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.boundary.implementations import NeumannBC, RobinBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_system import (
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import (
    GalerkinDiscretizer,
    StructuredBox,
    TaylorHood,
    exact_dof_values,
    forcing,
    galerkin_l2_error,
    pinned_value,
    vector_boundary_conditions,
    vector_forcing,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.study import Converges, run_study
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_verification import (
    converges_on,
    recovers_on,
    stalls_on,
)

_Np = NDArray[Any]
_VELOCITY = ("u", "v")
_COORDS = ("x", "y")
_Conditions = Dict[str, Union[ConditionProtocol, Dict[str, ConditionProtocol]]]


class _VelocityForcing:
    """The momentum forcing in ``StokesPhysics``'s layout: points
    ``(npts,)`` in 1D and ``(ndim, npts)`` otherwise, values ``(npts,
    nvars)``."""

    def __init__(self, supplier: Any) -> None:
        self._supplier = supplier

    def __call__(self, points: _Np, time: float) -> _Np:
        return np.asarray(self._supplier(np.atleast_2d(points), time)).T

    def is_time_dependent(self) -> bool:
        return bool(self._supplier.is_time_dependent())


class _ContinuityForcing:
    """The continuity forcing in ``StokesPhysics``'s layout, values
    ``(npts,)``, times ``sign``."""

    def __init__(self, supplier: Any, sign: float) -> None:
        self._supplier, self._sign = supplier, sign

    def __call__(self, points: _Np, time: float) -> _Np:
        return self._sign * np.asarray(self._supplier(np.atleast_2d(points), time))

    def is_time_dependent(self) -> bool:
        return bool(self._supplier.is_time_dependent())


def _terms(ndim: int, viscosity: float, convection: bool) -> List[TermProtocol]:
    """Momentum rows :math:`\\nabla\\cdot(-\\nu \\nabla u_i + p e_i)`, plus
    :math:`(u\\cdot\\nabla) u_i` for Navier-Stokes, and the continuity
    row :math:`\\nabla\\cdot u`."""
    fields, coords = _VELOCITY[:ndim], _COORDS[:ndim]
    terms: List[TermProtocol] = []
    for i, row in enumerate(fields):
        fluxes = [
            f"-({viscosity})*{row}_{c}" + (" + p" if i == j else "")
            for j, c in enumerate(coords)
        ]
        terms.append(Divergence(row, fluxes))
        if convection:
            transport = " + ".join(f"{f}*{row}_{c}" for f, c in zip(fields, coords))
            terms.append(Source(row, transport))
    terms.append(Source("p", " + ".join(f"{f}_{c}" for f, c in zip(fields, coords))))
    return terms


class _Stokes:
    """The physics under test on a Taylor-Hood basis, with the velocity's
    conditions from the problem and, when they are all Dirichlet, the
    pressure fixed at one DOF.

    Parameters
    ----------
    viscosity : float
        :math:`\\nu`.
    navier_stokes : bool
        Include the convection in the physics.
    continuity_sign : float
        Multiplies the continuity forcing (-1 is the wrong-model control).
    """

    def __init__(
        self,
        viscosity: float = 1.0,
        navier_stokes: bool = False,
        continuity_sign: float = 1.0,
    ) -> None:
        self._viscosity, self._navier_stokes = viscosity, navier_stokes
        self._sign = continuity_sign

    def system(
        self, problem: ManufacturedProblem, basis: MixedBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        velocity = problem.fields()[:-1]
        physics = StokesPhysics(
            basis,
            NumpyBkd(),
            navier_stokes=self._navier_stokes,
            viscosity=self._viscosity,
            vel_forcing=_VelocityForcing(vector_forcing(problem, velocity)),
            pres_forcing=_ContinuityForcing(forcing(problem, "p"), self._sign),
        )
        conditions: List[BoundaryConditionRole[_Np]] = vector_boundary_conditions(
            problem, basis, velocity, field_name="velocity"
        )
        if not any(isinstance(bc, (NeumannBC, RobinBC)) for bc in conditions):
            conditions.append(pinned_value(problem, basis, "pressure", "p"))
        return compose_galerkin_system(physics, conditions)

    def __repr__(self) -> str:
        return (
            f"_Stokes({self._viscosity}, navier_stokes={self._navier_stokes}, "
            f"continuity_sign={self._sign})"
        )


def _problem(
    velocity: Sequence[str],
    pressure: str,
    box: Box,
    conditions: _Conditions,
    viscosity: float = 1.0,
    convection: bool = False,
) -> ManufacturedProblem:
    ndim = box.ndim()
    solution = dict(zip(_VELOCITY[:ndim], velocity))
    solution["p"] = pressure
    return ManufacturedProblem(
        solution, _terms(ndim, viscosity, convection), box, conditions
    )


def _sides(box: Box, kinds: str) -> _Conditions:
    """One velocity condition per side: D, N (traction) or R (alpha 2)."""
    kind: Dict[str, ConditionProtocol] = {
        "D": Dirichlet(),
        "N": Neumann(),
        "R": Robin(2.0),
    }
    return {side: kind[k] for k, side in zip(kinds, box.boundaries())}


_UNIT_INTERVAL = Box([(0.0, 1.0)])
_UNIT_SQUARE = Box([(0.0, 1.0)] * 2)
_SMOOTH_1D = ("sin(2*x) + x", "cos(3*x)")
_SMOOTH_2D = (("sin(2*x)*cos(y)", "cos(x)*sin(3*y)/2"), "sin(x + 2*y)")
_LEVELS_1D = StructuredBox([8, 16, 32, 64])
_LEVELS_2D = StructuredBox([4, 8, 16, 32])
# Each field with its expected L2 rate.
_FIELDS_MEASURED: List[Tuple[str, float]] = [("velocity", 3.0), ("pressure", 2.0)]


def _rows(field: str, ndim: int) -> Tuple[str, ...]:
    """The rows of a Taylor-Hood field."""
    return _VELOCITY[:ndim] if field == "velocity" else ("p",)


class TestRecovery:
    @pytest.mark.parametrize(
        "velocity,pressure,navier_stokes",
        [
            (("(2*x - 1)**2",), "x", False),
            (("2*x - 1",), "x", True),
        ],
        ids=["1d_stokes_DD", "1d_ns_DD"],
    )
    def test_old_1d_cases(
        self, velocity: Tuple[str, ...], pressure: str, navier_stokes: bool
    ) -> None:
        """The old suite's 1D cases, a quadratic or linear velocity with a
        linear pressure, both ends Dirichlet, are reproduced by P2-P1 on
        ten cells. The old solutions' 1e-16 perturbations, there only to
        keep the old symbolic code from simplifying, are dropped."""
        problem = _problem(
            velocity,
            pressure,
            _UNIT_INTERVAL,
            _sides(_UNIT_INTERVAL, "DD"),
            convection=navier_stokes,
        )
        recovers_on(problem, TaylorHood(), _Stokes(navier_stokes=navier_stokes), 10)

    @pytest.mark.parametrize(
        "velocity,pressure,navier_stokes",
        [
            (("y", "(x + 1)*(y + 1)"), "x", False),
            (("x**2*y**2", "(x + 1)*(y + 1)"), "x*y", True),
        ],
        ids=["2d_stokes_DDDD", "2d_ns_DDDD"],
    )
    def test_old_2d_cases(
        self, velocity: Tuple[str, ...], pressure: str, navier_stokes: bool
    ) -> None:
        """The old suite's 2D cases, all sides Dirichlet, are reproduced
        by Q2-Q1 on five quads per side: their velocities (x^2 y^2
        included) lie in Q2 and their pressures in Q1."""
        problem = _problem(
            velocity,
            pressure,
            _UNIT_SQUARE,
            _sides(_UNIT_SQUARE, "DDDD"),
            convection=navier_stokes,
        )
        recovers_on(problem, TaylorHood(), _Stokes(navier_stokes=navier_stokes), 5)

    @pytest.mark.parametrize("sides", ["DNDD", "DRDN"])
    def test_2d_with_traction(self, sides: str) -> None:
        """The old 2D Stokes case with a traction side, and with a Robin
        and a traction side: the natural data, built from the exact
        velocity and pressure, determine the pressure, so none is
        pinned, and Q2-Q1 on five quads per side reproduces the
        solution."""
        problem = _problem(
            ("y", "(x + 1)*(y + 1)"), "x", _UNIT_SQUARE, _sides(_UNIT_SQUARE, sides)
        )
        recovers_on(problem, TaylorHood(), _Stokes(), 5)


class TestRates:
    @pytest.mark.parametrize("field,rate", _FIELDS_MEASURED)
    @pytest.mark.parametrize("sides", ["DD", "DN", "RD"])
    def test_1d(self, sides: str, field: str, rate: float) -> None:
        """1D P2-P1: velocity converges at 3 and pressure at 2, with
        Dirichlet, traction and Robin ends."""
        velocity, pressure = _SMOOTH_1D
        problem = _problem(
            (velocity,), pressure, _UNIT_INTERVAL, _sides(_UNIT_INTERVAL, sides)
        )
        converges_on(
            problem, _LEVELS_1D, TaylorHood(), _Stokes(), rate, _rows(field, 1)
        )

    def test_velocity_and_pressure_from_one_solve(self) -> None:
        """One study, one solve per level, three columns: Q2-Q1 velocity
        at 3, pressure at 2, and the pressure with its mean removed at 2
        (all-Dirichlet, so the pressure is pinned at one DOF)."""
        velocity, pressure = _SMOOTH_2D
        problem = _problem(
            velocity, pressure, _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DDDD"), 0.5
        )
        discretizer = GalerkinDiscretizer(
            StructuredBox([4, 8, 16, 32]), TaylorHood(), _Stokes(viscosity=0.5)
        )
        study = run_study(
            problem,
            discretizer,
            [
                galerkin_l2_error(["u", "v"], label="velocity"),
                galerkin_l2_error(["p"], label="pressure"),
                galerkin_l2_error(["p"], zero_mean=True, label="pressure, mean 0"),
            ],
        )
        for column, rate in [
            ("velocity", 3.0),
            ("pressure", 2.0),
            ("pressure, mean 0", 2.0),
        ]:
            result = Converges(rate, last=3, column=column).verify(study)
            assert result.passed, result.message
        norms = study.norms("velocity")
        assert norms is not None and min(norms) > 0.0

    @pytest.mark.parametrize("field,rate", _FIELDS_MEASURED)
    @pytest.mark.parametrize("simplex", [False, True])
    @pytest.mark.parametrize("sides", ["DDDD", "DNDR"])
    def test_2d(self, sides: str, simplex: bool, field: str, rate: float) -> None:
        """2D Taylor-Hood on quads (Q2-Q1) and triangles (P2-P1), viscosity
        0.5: velocity converges at 3 and pressure at 2, all-Dirichlet with
        the pressure pinned, and with a traction and a Robin side."""
        velocity, pressure = _SMOOTH_2D
        problem = _problem(
            velocity, pressure, _UNIT_SQUARE, _sides(_UNIT_SQUARE, sides), 0.5
        )
        meshes = StructuredBox([4, 8, 16, 32], simplex=simplex)
        converges_on(
            problem,
            meshes,
            TaylorHood(),
            _Stokes(viscosity=0.5),
            rate,
            _rows(field, 2),
        )

    @pytest.mark.parametrize("field,rate", _FIELDS_MEASURED)
    def test_2d_mixed_components(self, field: str, rate: float) -> None:
        """Sides holding one velocity component and leaving the other
        traction free (u fixed on the left, v on the bottom), the others
        Dirichlet or Robin: velocity 3, pressure 2."""
        neumann, dirichlet = Neumann(), Dirichlet()
        conditions: _Conditions = {
            "left": {"u": dirichlet, "v": neumann},
            "right": dirichlet,
            "bottom": {"u": neumann, "v": dirichlet},
            "top": Robin(2.0),
        }
        velocity, pressure = _SMOOTH_2D
        problem = _problem(velocity, pressure, _UNIT_SQUARE, conditions)
        converges_on(
            problem, _LEVELS_2D, TaylorHood(), _Stokes(), rate, _rows(field, 2)
        )

    @pytest.mark.parametrize("field,rate", _FIELDS_MEASURED)
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_navier_stokes(self, ndim: int, field: str, rate: float) -> None:
        """Navier-Stokes from a zero initial guess, viscosity 0.5, with a
        traction side: velocity 3, pressure 2, in 1D and 2D."""
        velocity: Tuple[str, ...]
        if ndim == 1:
            velocity, pressure = (_SMOOTH_1D[0],), _SMOOTH_1D[1]
            box, sides, meshes = _UNIT_INTERVAL, "DN", _LEVELS_1D
        else:
            velocity, pressure = _SMOOTH_2D
            box, sides, meshes = _UNIT_SQUARE, "DNDD", _LEVELS_2D
        problem = _problem(
            velocity, pressure, box, _sides(box, sides), 0.5, convection=True
        )
        converges_on(
            problem,
            meshes,
            TaylorHood(),
            _Stokes(viscosity=0.5, navier_stokes=True),
            rate,
            _rows(field, ndim),
        )


class TestControls:
    def test_continuity_sign_flipped(self) -> None:
        """The continuity forcing given as -div u (the sign the physics'
        docstring states) on a non-solenoidal velocity solves another
        problem: the error stalls."""
        velocity, pressure = _SMOOTH_2D
        problem = _problem(
            velocity, pressure, _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DDDD")
        )
        stalls_on(problem, _LEVELS_2D, TaylorHood(), _Stokes(continuity_sign=-1.0))

    def test_navier_stokes_without_convection(self) -> None:
        """The Navier-Stokes problem solved by the Stokes physics, which
        omits the convection: the error stalls."""
        velocity, pressure = _SMOOTH_2D
        problem = _problem(
            velocity,
            pressure,
            _UNIT_SQUARE,
            _sides(_UNIT_SQUARE, "DNDD"),
            0.5,
            convection=True,
        )
        stalls_on(problem, _LEVELS_2D, TaylorHood(), _Stokes(viscosity=0.5))


class TestTransient:
    @pytest.mark.parametrize("navier_stokes", [False, True], ids=["stokes", "ns"])
    @pytest.mark.parametrize(
        "tableau", [backward_euler, crank_nicolson], ids=["BE", "CN"]
    )
    @pytest.mark.parametrize(
        "velocity,pressure,box,nx",
        [
            (("(2*x - 1)**2*(1 + t)",), "x*(1 + t)", _UNIT_INTERVAL, 20),
            (
                ("y*(1 + t)", "(x + 1)*(y + 1)*(1 + t)"),
                "x*(1 + t)",
                _UNIT_SQUARE,
                5,
            ),
        ],
        ids=["1d", "2d"],
    )
    def test_time_linear_solution_is_stepped_exactly(
        self,
        velocity: Tuple[str, ...],
        pressure: str,
        box: Box,
        nx: int,
        tableau: Callable[[], ButcherTableau],
        navier_stokes: bool,
    ) -> None:
        """The old suite's transient cases, :math:`\\partial_t u` added to
        each momentum row, all sides Dirichlet and the pressure pinned at
        one DOF: a velocity and pressure in Taylor-Hood space times
        :math:`1 + t` are stepped exactly by backward Euler and
        Crank-Nicolson, since the rate is constant in time (the
        convection included). The velocity data and the pinned pressure
        step with their exact rates."""
        ndim = box.ndim()
        rows = _VELOCITY[:ndim]
        solution = dict(zip(rows, velocity))
        solution["p"] = pressure
        terms: List[TermProtocol] = [TimeDerivative(row) for row in rows]
        terms += _terms(ndim, 1.0, navier_stokes)
        problem = ManufacturedProblem(solution, terms, box, _sides(box, "D" * 2 * ndim))
        mesh = StructuredBox([nx]).meshes(box)[0]
        velocity_basis = VectorLagrangeBasis(mesh, degree=2)
        pressure_basis = LagrangeBasis(mesh, degree=1)
        fields: List[Tuple[str, MixedFieldBasisProtocol[_Np]]] = [
            ("velocity", velocity_basis),
            ("pressure", pressure_basis),
        ]
        system = _Stokes(navier_stokes=navier_stokes).system(
            problem, MixedBasis(fields)
        )
        transient = galerkin_transient_system(
            skfem_discretization(system), ScipySparseLU()
        )

        def exact(time: float) -> _Np:
            return np.concatenate(
                [
                    exact_dof_values(problem, velocity_basis, rows, time),
                    exact_dof_values(problem, pressure_basis, ["p"], time),
                ]
            )

        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            transient,
            RungeKutta(tableau(), Newton(atol=1e-12, rtol=0.0, maxiters=50)),
            RecomputeAlways(transient.solved.solver),
            FixedStep(0.1, (0.0, 0.5)),
            exact(0.0)[:, None],
            store,
        )
        last = store.ntimes() - 1
        error = np.max(np.abs(store.state(last)[:, 0] - exact(store.times()[last])))
        assert error < 1e-10


class TestRefusals:
    def test_taylor_hood_below_degree_two(self) -> None:
        """P1-P0 has no continuous pressure space: refused."""
        with pytest.raises(ValueError, match="velocity degree 2"):
            TaylorHood(1)

    def test_measurement_splitting_the_velocity(self) -> None:
        """Measuring u without v splits the velocity field: refused."""
        velocity, pressure = _SMOOTH_2D
        problem = _problem(
            velocity, pressure, _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DDDD")
        )
        discretizer = GalerkinDiscretizer(StructuredBox([2]), TaylorHood(), _Stokes())
        with pytest.raises(ValueError, match="measure whole basis fields"):
            run_study(problem, discretizer, [galerkin_l2_error(["u"])])

    def test_too_few_fields(self) -> None:
        """A 1D problem (u, p) on a 2D Taylor-Hood basis has fewer fields
        than the basis has components: refused before solving."""
        problem = _problem(("x",), "x", _UNIT_INTERVAL, _sides(_UNIT_INTERVAL, "DD"))

        class _Square:
            def meshes(self, domain: Any) -> Any:
                return StructuredBox([2]).meshes(_UNIT_SQUARE)

        discretizer = GalerkinDiscretizer(_Square(), TaylorHood(), _Stokes())
        with pytest.raises(ValueError, match="one field per component"):
            run_study(problem, discretizer, [galerkin_l2_error()])

    def test_mixed_field_helpers_check_the_field_kind(self) -> None:
        """Velocity conditions for the scalar pressure field, and a pinned
        value for the vector velocity field, are refused."""
        velocity, pressure = _SMOOTH_2D
        problem = _problem(
            velocity, pressure, _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DDDD")
        )
        basis = TaylorHood().build(StructuredBox([2]).meshes(_UNIT_SQUARE)[0])
        with pytest.raises(ValueError, match="not a vector"):
            vector_boundary_conditions(
                problem, basis, ("u", "v"), field_name="pressure"
            )
        with pytest.raises(ValueError, match="is a vector"):
            pinned_value(problem, basis, "velocity", "u")
