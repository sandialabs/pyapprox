"""Galerkin probe and outflux observations, alone and through a model.

- Probes reproduce fields in the element space exactly (linear on P1,
  quadratic on P2), at interior, edge and vertex points.
- The outflux is ``u * (beta . n)`` at the given DOFs, in their order.
- Through ``TransientImplicitMap`` and ``ImplicitMapFunction`` with a KLE
  diffusivity, the new path's ``probe_observations`` and
  ``nodal_outflux_observations`` at stored states 2 and 5: the Jacobian
  by tangent columns passes central finite differences under random
  weights (error ratio at most 1e-6), and the one by adjoint rows
  reproduces it (1e-9). Probes on the Dirichlet boundary are included:
  Dirichlet values do not depend on the parameters, so both strategies
  must agree there too.
- A steady functional is refused by the transient map; a functional
  whose term declares no derivatives gives values and no Jacobian.

The first two cases check this module's old functionals directly.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
    TangentColumns,
)
from pyapprox.implicitfunction.observation_terms import (
    ObservedValue,
    StateEntries,
    WeightedEntries,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TrajectoryFunctionalSourceProtocol,
    TransientImplicitMap,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldVelocity,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.observations import (
    nodal_outflux_functional,
    probe_observation_functional,
)
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_observations import (
    nodal_outflux_observations,
    probe_observations,
)
from pyapprox.pde.models.galerkin.transient_system import galerkin_transient_system
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.trajectory_cases import EachTruncated
from tests._helpers.spde_fields import (
    spde_lognormal_field_map,
)

_Np = NDArray[Any]

# Interior, on an edge, and on a vertex (incl. the domain corner).
_POINTS = np.array([[0.31, 0.5, 0.25, 1.0, 0.0], [0.47, 0.13, 0.5, 1.0, 0.5]])


def _field(coords: _Np, degree: int) -> _Np:
    x, y = coords[0], coords[1]
    linear: _Np = 0.3 + 1.2 * x - 0.7 * y
    if degree == 1:
        return linear
    quadratic: _Np = linear + 0.9 * x * y - 0.4 * x**2 + 0.6 * y**2
    return quadratic


class TestProbesAndOutflux:
    @pytest.mark.parametrize("degree", [1, 2])
    def test_probes_exact_in_element_space(
        self, bkd: Backend[Array], degree: int
    ) -> None:
        mesh = StructuredMesh2D(4, 4, [(0.0, 1.0), (0.0, 1.0)], bkd, element_type="tri")
        basis = LagrangeBasis(mesh, degree=degree)
        coeffs = bkd.asarray(_field(bkd.to_numpy(basis.dof_coordinates()), degree))
        times = [0, 2]
        sol = bkd.stack([coeffs, 2.0 * coeffs, -coeffs], axis=1)
        func = probe_observation_functional(
            basis, bkd.asarray(_POINTS), times, nparams=1
        )
        exact = _field(_POINTS, degree)
        expected = np.concatenate([exact, -exact])[:, None]
        bkd.assert_allclose(
            func(sol, bkd.zeros((1, 1))),
            bkd.asarray(expected),
            rtol=1e-12,
            atol=1e-13,
        )

    def test_outflux_values(self, bkd: Backend[Array]) -> None:
        dofs = bkd.asarray(np.array([7, 2, 5]), dtype=bkd.index_dtype())
        normal_velocity = bkd.asarray(np.array([1.0, 0.5, -2.0]))
        sol = bkd.asarray(np.arange(30.0).reshape(10, 3))
        func = nodal_outflux_functional(dofs, normal_velocity, [1, 2], 10, 2, bkd)
        sol_np = bkd.to_numpy(sol)
        vn = np.array([1.0, 0.5, -2.0])
        expected = np.concatenate(
            [sol_np[[7, 2, 5], 1] * vn, sol_np[[7, 2, 5], 2] * vn]
        )
        bkd.assert_allclose(
            func(sol, bkd.zeros((2, 1))),
            bkd.asarray(expected[:, None]),
            rtol=1e-14,
        )

    def test_outflux_shape_mismatch_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="nbdofs"):
            nodal_outflux_functional(
                bkd.asarray(np.array([1, 2]), dtype=bkd.index_dtype()),
                bkd.ones((3,)),
                [1],
                10,
                2,
                bkd,
            )


_STORED = [2, 5]


def _implicit_map(
    bkd: NumpyBkd, functional: TrajectoryFunctionalSourceProtocol[_Np, _Np]
) -> TransientImplicitMap[_Np, _Np, Any]:
    """Transient ADR on the unit square: KLE log-diffusivity, velocity
    (1, 0), Dirichlet u = 0 on the left only (outflow elsewhere), a
    Gaussian bump initial state, backward Euler with dt = 0.02 to
    T = 0.1 (Newton atol 1e-12)."""
    basis, system, parameterization = _problem(bkd)
    transient = galerkin_transient_system(
        skfem_discretization(system),
        ScipySparseLU(),
        ParameterState(parameterization),
    )
    coords = bkd.to_numpy(basis.dof_coordinates())
    bump = np.exp(-50.0 * ((coords[0] - 0.4) ** 2 + (coords[1] - 0.5) ** 2))
    return TransientImplicitMap(
        transient,
        transient,
        RungeKutta(backward_euler(), Newton(atol=1e-12, rtol=0.0, maxiters=20)),
        RecomputeAlways(transient.solved.solver),
        FixedStep(0.02, (0.0, 0.1)),
        bkd.asarray(bump[:, None]),
        functional,
    )


def _observations(bkd: NumpyBkd, which: str) -> TrajectoryFunctional[_Np, _Np]:
    """Probes (two on the Dirichlet boundary) or the nodal outflux through
    the right side, at the stored states ``_STORED``."""
    basis, _, _ = _problem(bkd)
    if which == "probes":
        # (0, 0.5) and (0, 0.2) lie on the Dirichlet boundary.
        points = bkd.asarray(np.array([[0.0, 0.0, 0.42, 0.8], [0.5, 0.2, 0.55, 0.3]]))
        return probe_observations(basis, points, _STORED)
    coords = bkd.to_numpy(basis.dof_coordinates())
    right = bkd.to_numpy(basis.get_dofs("right"))
    right = right[np.argsort(coords[1, right])]
    return nodal_outflux_observations(
        bkd,
        basis.ndofs(),
        bkd.asarray(right, dtype=bkd.index_dtype()),
        bkd.ones((right.shape[0],)),
        _STORED,
    )


def _functions(
    bkd: NumpyBkd, which: str
) -> tuple[ImplicitMapFunction[Any, _Np, _Np], ImplicitMapFunction[Any, _Np, _Np]]:
    """The observations' function with its Jacobian by tangent columns and
    by adjoint rows."""
    implicit = _implicit_map(bkd, FixedTrajectoryFunctional(_observations(bkd, which)))
    parameters = ArrayParameters(bkd)
    return (
        ImplicitMapFunction(implicit, parameters, TangentColumns()),
        ImplicitMapFunction(implicit, parameters, AdjointRows()),
    )


def _problem(bkd: NumpyBkd) -> tuple[Any, Any, Any]:
    """The basis, composed system and KLE parameterization."""
    mesh = StructuredMesh2D(6, 6, [(0.0, 1.0), (0.0, 1.0)], bkd, element_type="tri")
    basis = LagrangeBasis(mesh, degree=1)
    vel_basis = VectorLagrangeBasis(mesh, degree=1)
    vel_dofs = np.zeros(vel_basis.ndofs())
    vel_dofs[0::2] = 1.0
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis, dofs=0.1 * np.ones(basis.ndofs())),
        bkd=bkd,
        velocity=NodalFieldVelocity(vel_basis, vel_dofs),
    )
    system = compose_galerkin_system(physics, [DirichletBC(basis, "left", 0.0, bkd)])
    diffusivity_map = spde_lognormal_field_map(
        basis,
        bkd.asarray(np.log(0.1) * np.ones(basis.ndofs())),
        bkd,
        n_modes=4,
        gamma=1.0,
        delta=4.0,
        sigma=0.3,
    )
    parameterization = AdvectionDiffusionParameterization(
        physics, bkd=bkd, diffusivity_map=diffusivity_map
    )
    return basis, system, parameterization


def _sample(bkd: NumpyBkd, nvars: int) -> _Np:
    return bkd.asarray(0.4 * np.random.default_rng(7).normal(size=(nvars, 1)))


class TestObservationJacobians:
    @pytest.mark.parametrize("which", ["probes", "outflux"])
    def test_tangent_columns_match_fd(self, numpy_bkd: NumpyBkd, which: str) -> None:
        bkd = numpy_bkd
        columns, _ = _functions(bkd, which)
        checker = DerivativeChecker(columns)
        errors = checker.check_derivatives(
            _sample(bkd, columns.nvars()),
            relative=True,
            central=True,
            weights=bkd.asarray(
                np.random.default_rng(8).normal(size=(columns.nqoi(), 1))
            ),
        )
        assert float(bkd.to_numpy(checker.error_ratio(errors[0]))) <= 1e-6

    @pytest.mark.parametrize("which", ["probes", "outflux"])
    def test_adjoint_rows_match_tangent_columns(
        self, numpy_bkd: NumpyBkd, which: str
    ) -> None:
        bkd = numpy_bkd
        columns, rows = _functions(bkd, which)
        sample = _sample(bkd, columns.nvars())
        by_columns, by_rows = (
            columns.derivatives().jacobian,
            rows.derivatives().jacobian,
        )
        assert by_columns is not None and by_rows is not None
        jacobian = by_columns(sample)
        assert jacobian.shape == (columns.nqoi(), columns.nvars())
        bkd.assert_allclose(by_rows(sample), jacobian, rtol=1e-9, atol=1e-12)


class TestFunctionalKinds:
    def test_transient_map_rejects_a_steady_functional(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A steady functional (of one state, with no sink) is refused at
        construction."""
        bkd = numpy_bkd
        basis, _, _ = _problem(bkd)
        steady: StateFunctional[_Np, _Np] = StateFunctional.from_terms(
            bkd,
            1,
            state_terms=(ObservedValue(StateEntries(bkd, basis.ndofs(), [0])),),
        )
        with pytest.raises(TypeError, match="TrajectoryFunctionalSourceProtocol"):
            _implicit_map(bkd, steady)  # type: ignore[arg-type]

    def test_value_only_functional_is_evaluated_not_differentiated(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A functional with no derivatives (the final state's sum through
        a term declaring none) gives a function with values and no
        Jacobian."""
        bkd = numpy_bkd
        basis, _, _ = _problem(bkd)
        n = basis.ndofs()
        total = WeightedEntries(
            bkd,
            n,
            bkd.asarray(np.arange(n)[None, :], dtype=bkd.index_dtype()),
            bkd.ones((1, n)),
        )
        functional = TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EachTruncated(EndpointTerm(total), 0),)
        )
        function = ImplicitMapFunction(
            _implicit_map(bkd, FixedTrajectoryFunctional(functional)),
            ArrayParameters(bkd),
        )
        assert function.derivatives().jacobian is None
        assert function(bkd.zeros((function.nvars(), 1))).shape == (1, 1)
