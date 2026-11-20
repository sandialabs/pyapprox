"""First-order transient paths refuse collocation elasticity.

Elasticity is second order in time (rho u_tt = div sigma + f). The
collocation transient paths integrate ``du/dt = f(u, t)``, which for it
is a viscous relaxation with no inertia or density, so
``CollocationModel.solve_transient`` and ``TransientForwardModel`` refuse
both ``LinearElasticityPhysics`` and ``HyperelasticityPhysics``. Steady
solves and the physics' mass matrix stay available.
"""

from typing import Any, Callable, Generic, List, Tuple, Union

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics import (
    HyperelasticityPhysics,
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.physics.coefficients import (
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.pde.models.collocation.transient import TransientForwardModel
from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    ChebyshevLevel,
    field,
    stacked_forcing,
)
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.pde.parameterizations.collocation_hyperelasticity import (
    CollocationHyperelasticityParameterization,
)
from pyapprox.pde.time_order import SecondOrderInTimeProtocol
from pyapprox.pde.verification.conditions import Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence
from pyapprox.util.backends.protocols import Array, Backend

_NPTS_1D = 8
_REFUSAL = "second order in time"
_PHYSICS_NAMES = ["LinearElasticityPhysics", "HyperelasticityPhysics"]

_Physics = Union[LinearElasticityPhysics[Any], HyperelasticityPhysics[Any]]


def _time_config() -> TimeIntegrationConfig[Any]:
    return TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.01,
        newton_tol=1e-10,
        newton_maxiter=20,
        verbosity=0,
    )


def _basis(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    return ChebyshevBasis2D(TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd), bkd)


def _nodes(bkd: Backend[Array]) -> Array:
    """The collocation nodes of ``_basis``. Shape: (2, npts)."""
    basis = _basis(bkd)
    xx, yy = bkd.meshgrid(basis.nodes_x(), basis.nodes_y(), indexing="xy")
    return bkd.stack([bkd.flatten(xx), bkd.flatten(yy)], axis=0)


def _set_zero_dirichlet(physics: _Physics, bkd: Backend[Array]) -> None:
    """Homogeneous Dirichlet on all four sides for both components."""
    mesh = create_uniform_mesh_2d((_NPTS_1D, _NPTS_1D), (-1.0, 1.0, -1.0, 1.0), bkd)
    npts = physics.basis().npts()
    bcs: List[BoundaryConditionProtocol[Array]] = []
    for side in range(4):
        boundary_idx = mesh.boundary_indices(side)
        bcs.append(zero_dirichlet_bc(bkd, boundary_idx))
        bcs.append(zero_dirichlet_bc(bkd, boundary_idx + npts))
    physics.set_boundary_conditions(bcs)


_SQUARE = Box([(-1.0, 1.0), (-1.0, 1.0)])
_DISPLACEMENT = ("(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x")


def _manufactured(
    bkd: Backend[Array],
) -> Tuple[ManufacturedProblem, ChebyshevLevel[Array]]:
    """A steady displacement vanishing on the boundary of [-1, 1]^2, for
    -div(sigma) = f with lambda = mu = 1, and the level on ``_basis``'s
    nodes."""
    trace = "u_x + v_y"
    problem = ManufacturedProblem(
        dict(zip(("u", "v"), _DISPLACEMENT)),
        [
            Divergence("u", [f"-({trace} + 2*u_x)", "-(u_y + v_x)"]),
            Divergence("v", ["-(v_x + u_y)", f"-({trace} + 2*v_y)"]),
        ],
        _SQUARE,
        {side: Dirichlet() for side in _SQUARE.boundaries()},
    )
    (level,) = ChebyshevBox([_NPTS_1D], bkd).levels(_SQUARE)
    return problem, level


def _physics(bkd: Backend[Array], physics_name: str) -> _Physics:
    basis = _basis(bkd)
    physics: _Physics
    if physics_name == "LinearElasticityPhysics":
        problem, level = _manufactured(bkd)
        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=stacked_forcing(problem, level)
        )
    else:
        physics = HyperelasticityPhysics(
            basis, bkd, NeoHookeanLaw(bkd), mu=1.0, lamda=1.5
        )
    _set_zero_dirichlet(physics, bkd)
    return physics


def _parameterization(
    physics: _Physics, bkd: Backend[Array]
) -> Union[
    CollocationElasticityParameterization[Array],
    CollocationHyperelasticityParameterization[Array],
]:
    """A two-mode Young's-modulus map; never applied, the model refuses."""
    npts = physics.basis().npts()
    youngs_modulus_map = MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.full(npts, 2.5)),
        bkd.asarray(np.ones((npts, 2))),
    )
    if isinstance(physics, LinearElasticityPhysics):
        return CollocationElasticityParameterization(
            physics,
            youngs_modulus_map=youngs_modulus_map,
            poisson_ratio=0.25,
            bkd=bkd,
        )
    return CollocationHyperelasticityParameterization(
        physics,
        youngs_modulus_map=youngs_modulus_map,
        poisson_ratio=0.25,
        bkd=bkd,
    )


class TestCollocationElasticityTimeOrder:
    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_declares_second_order(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        assert isinstance(physics, SecondOrderInTimeProtocol)
        message = physics.first_order_transient_refusal()
        assert physics_name in message
        assert _REFUSAL in message

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_solve_transient_is_refused(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        model = CollocationModel(physics, bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            model.solve_transient(bkd.zeros((physics.nstates(),)), _time_config())

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_transient_forward_model_is_refused(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        with pytest.raises(TypeError, match=_REFUSAL):
            TransientForwardModel(
                physics,
                bkd,
                bkd.zeros((physics.nstates(),)),
                _time_config(),
                parameterization=_parameterization(physics, bkd),
            )

    def test_steady_solve_still_works(self, bkd: Backend[Array]) -> None:
        physics = _physics(bkd, "LinearElasticityPhysics")
        problem, level = _manufactured(bkd)
        exact = []
        for component in _DISPLACEMENT:
            values = field(problem, component, level)
            assert not isinstance(values, float)
            exact.append(values.values(0.0))
        solution = CollocationModel(physics, bkd).solve_steady(
            bkd.zeros((physics.nstates(),)), tol=1e-10
        )
        bkd.assert_allclose(solution, bkd.concatenate(exact), atol=1e-8)

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_mass_matrix_still_available(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        nstates = physics.nstates()
        bkd.assert_allclose(physics.mass_matrix(), bkd.eye(nstates))


def _growing_modulus(coords: Array, time: float) -> Array:
    return (1.0 + time) * (1.5 + 0.5 * coords[0] * coords[1])


class _GrowingLoad(Generic[Array]):
    """A stacked two-component load growing in time. Shape: (2*npts,)"""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, coords: Array, time: float) -> Array:
        return (1.0 + time) * self._bkd.concatenate([coords[0], coords[1]])


def _growing(
    bkd: Backend[Array], func: Callable[[Array, float], Array]
) -> NodalSupplier[Array]:
    supplier: TimeDependent[Array] = TimeDependent(func)
    return NodalSupplier(supplier, _nodes(bkd))


def _elasticity_state(bkd: Backend[Array]) -> Array:
    rng = np.random.default_rng(7)
    return bkd.asarray(0.1 * rng.standard_normal(2 * _NPTS_1D * _NPTS_1D))


class TestLinearElasticityInterior:
    def test_time_dependent_forcing(self, bkd: Backend[Array]) -> None:
        """A time-dependent load leaves the Jacobian time-invariant but not
        the operator; the linear bundle passes the checker."""
        physics = LinearElasticityPhysics(
            _basis(bkd),
            bkd,
            lamda=1.5,
            mu=NodalField(1.0 + 0.2 * _nodes(bkd)[0] ** 2),
            forcing=_growing(bkd, _GrowingLoad(bkd)),
        )
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None and jacobian.affine()
        report = ODEResidualBundleChecker(rhs).check(_elasticity_state(bkd), 0.3)
        assert report.failures() == []

    @pytest.mark.parametrize("constant", [True, False])
    def test_jacobian_is_the_residuals(
        self, bkd: Backend[Array], constant: bool
    ) -> None:
        """The residual is affine in the displacement, so J u = R(u) - R(0)
        exactly, for constant and variable Lame parameters alike."""
        if constant:
            physics = LinearElasticityPhysics(_basis(bkd), bkd, lamda=1.5, mu=0.8)
        else:
            x, y = _nodes(bkd)[0], _nodes(bkd)[1]
            physics = LinearElasticityPhysics(
                _basis(bkd),
                bkd,
                lamda=NodalField(1.5 + 0.3 * x * y),
                mu=NodalField(0.8 + 0.2 * x * x),
            )
        state = _elasticity_state(bkd)
        zero = bkd.zeros((physics.nstates(),))
        difference = physics.residual(state, 0.0) - physics.residual(zero, 0.0)
        applied = physics.jacobian(state, 0.0) @ state
        scale = float(bkd.max(bkd.abs(difference)))
        bkd.assert_allclose(applied, difference, rtol=0.0, atol=1e-12 * scale)

    def test_time_dependent_modulus(self, bkd: Backend[Array]) -> None:
        """A time-dependent shear modulus makes the Jacobian time-dependent,
        is read at the evaluation time, and is refused by the interface
        traction, which takes no time."""
        time = 0.7
        moving = LinearElasticityPhysics(
            _basis(bkd), bkd, lamda=1.5, mu=_growing(bkd, _growing_modulus)
        )
        frozen = LinearElasticityPhysics(
            _basis(bkd),
            bkd,
            lamda=1.5,
            mu=NodalField(_growing(bkd, _growing_modulus).values(time)),
        )
        assert not moving.interior_jacobian_is_time_invariant()
        state = _elasticity_state(bkd)
        bkd.assert_allclose(
            moving.jacobian(state, time), frozen.jacobian(state, 0.0), rtol=1e-12
        )
        with pytest.raises(ValueError, match="takes no time"):
            moving.compute_interface_flux(
                state, bkd.array([0], dtype=int), bkd.array([1.0, 0.0])
            )
