"""The row treatment through the parameterized adapters, and the adjoint
gradient under the mass-consistent projection.

1D heat equation on [0, 1], P1 on 10 cells, diffusivity exp(W p) from a
three-term log-KLE, forcing (1 + x)(1 + sin 5t), u(0) = 0 and a moving
right end 0.5 sin 4t (its rate supplied), from 0.3 sin(pi x), dt = 0.001
to T = 0.05. The factory and the transient forward model pass the
symmetric row treatment to the tier they build. With it, forward Euler
and Heun stepped under the mass-consistent projection (exactly
conservative for explicit steppers) have adjoint gradients of the value
at the DOF next to the moving end that match finite differences.

The projection's tangent differs from the pin's only in directions with
boundary components, Pi' d = (d_I + M_II^{-1} M_ID d_D, 0), and parameter
sensitivities have none here: the boundary rows carry g_dot and the
initial boundary values, both independent of the parameters, so
dz_D = 0 at every step. A control swapping in the pin's tangent would
therefore pass too; the tangent in boundary directions is checked
exactly in test_state_constraints.py instead.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeDependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    SymmetricNeutralizedRows,
    TimeIntegrationConfig,
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    MassConsistentProjection,
)
from pyapprox.pde.models.galerkin import GalerkinTransientForwardModel
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.newton import NewtonSolver

_Np = NDArray[Any]
_NPARAMS = 3
_PARAM = np.array([[0.4], [-0.3], [0.2]])
_DELTAT, _FINAL_TIME = 0.001, 0.05


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


class _Heat:
    """The parameterized heat system, its initial state and the DOF next
    to the moving right end."""

    def __init__(self, bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        self.system: GalerkinSystem[_Np] = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        x = np.asarray(basis.dof_coordinates())[0]
        modes = np.stack(
            [0.4 * np.sin((k + 1) * np.pi * x) / (k + 1) for k in range(_NPARAMS)],
            axis=1,
        )
        exp = _ExpTransform(bkd)
        self.parameterization = AdvectionDiffusionParameterization(
            physics,
            diffusivity_map=TransformedFieldMap(
                MeshKLEFieldMap(bkd, bkd.zeros((x.shape[0],)), bkd.asarray(modes)),
                exp,
                exp,
                bkd,
                transform_deriv2=exp,
            ),
            bkd=bkd,
        )
        self.init_state: _Np = 0.3 * np.sin(np.pi * x)
        interior = np.setdiff1d(
            np.arange(x.shape[0]), np.asarray(self.system.constraint_set().dofs())
        )
        self.near_right = int(interior[np.argmax(x[interior])])

    def adapter(self) -> GalerkinPhysicsToODEResidualWithSetParamAdapter[_Np, Any]:
        return create_galerkin_physics_ode_residual(
            skfem_discretization(self.system),
            ScipySparseLU(),
            self.parameterization,
            row_treatment=SymmetricNeutralizedRows,
        )


def _is_symmetric(matrix: Any) -> bool:
    dense = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
    return bool(np.abs(dense - dense.T).max() < 1e-14)


class TestPlumbing:
    def test_factory_passes_the_treatment_to_its_tier(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        adapter = _Heat(numpy_bkd).adapter()
        assert _is_symmetric(adapter.mass_matrix().as_matrix())

    def test_set_param_bumps_the_parameter_version(self, numpy_bkd: NumpyBkd) -> None:
        """Every set_param increments the version, even with an unchanged
        value."""
        adapter = _Heat(numpy_bkd).adapter()
        assert adapter.parameters_version() == 0
        adapter.set_param(numpy_bkd.asarray(_PARAM[:, 0]))
        adapter.set_param(numpy_bkd.asarray(_PARAM[:, 0]))
        assert adapter.parameters_version() == 2

    def test_forward_model_passes_it_on(self, numpy_bkd: NumpyBkd) -> None:
        heat = _Heat(numpy_bkd)
        config: TimeIntegrationConfig[_Np] = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=_FINAL_TIME,
            deltat=_DELTAT,
            newton_tol=1e-12,
            newton_maxiter=20,
            verbosity=0,
        )
        model = GalerkinTransientForwardModel(
            skfem_discretization(heat.system),
            ScipySparseLU(),
            heat.parameterization,
            heat.init_state,
            config,
            row_treatment=SymmetricNeutralizedRows,
        )
        assert _is_symmetric(
            model.transient_model().adapter().mass_matrix().as_matrix()
        )


def _checker(heat: _Heat, method: str) -> DerivativeChecker[_Np]:
    """The value at the DOF next to the moving end against the parameters,
    by constrained solves, and its adjoint gradient."""

    def operator() -> TimeAdjointOperatorWithHVP[_Np]:
        adapter = heat.adapter()
        newton = NewtonSolver(create_stepper(method, adapter))
        newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
        integrator = TimeIntegrator(
            0.0,
            _FINAL_TIME,
            _DELTAT,
            newton,
            constraint=MassConsistentProjection(adapter, _NPARAMS),
        )
        functional = EndpointFunctional(
            heat.near_right, heat.system.nstates(), _NPARAMS, NumpyBkd()
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def values(samples: _Np) -> _Np:
        columns = [
            operator()(heat.init_state, samples[:, ii : ii + 1])
            for ii in range(samples.shape[1])
        ]
        return np.hstack(columns)

    def gradient(sample: _Np) -> _Np:
        return operator().jacobian(heat.init_state, sample)

    return DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=1, nvars=_NPARAMS, fun=values, jacobian=gradient, bkd=NumpyBkd()
        )
    )


class TestGradientUnderTheProjection:
    @pytest.mark.parametrize("method", ["forward_euler", "heun"])
    def test_matches_finite_differences(self, numpy_bkd: NumpyBkd, method: str) -> None:
        errors = _checker(_Heat(numpy_bkd), method).check_derivatives(
            numpy_bkd.asarray(_PARAM), relative=True
        )[0]
        assert numpy_bkd.to_float(numpy_bkd.min(errors)) <= 1e-6
