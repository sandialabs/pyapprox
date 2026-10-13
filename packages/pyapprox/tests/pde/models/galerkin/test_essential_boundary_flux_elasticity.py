"""The essential-boundary flux as a support reaction: 2D linear elasticity.

Plate [0, 1]^2, P1 vector elements on 6 x 6 cells, Lame values from
(E, nu) = (1, 0.3) through ``create_galerkin_lame_parameterization``:

- clamped on the left (u = 0);
- an elastic support on the right: Robin, sigma n + a u = g, with a = 2
  and g = (0.1, -0.05);
- a traction t = (0.2, 0.05) on the top;
- a body force f = (0.3, -0.1); the bottom is traction-free.

The QoI is the total support force on the clamped edge in one direction,
the flux functional with unit weights on that edge's component DOFs.

Force balance (no derivation of the flux needed): rigid translations
are in the elasticity stiffness's kernel, so at the solution the support
force cancels exactly the applied forces, each computed independently
of the assembly:

    Q_c = -(f_c |Omega| + t_c |top| + integral_right (g_c - a u_c) ds),

with the Robin integral taken by the trapezoid rule along the right
edge, which is exact for P1 traces.
"""

from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    EssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.models.galerkin.steady import GalerkinSteadyForwardModel
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import (
    NumpyArray,
    solve_based_derivatives_pass_fd,
)

_BODY_FORCE = np.array([0.3, -0.1])
_TRACTION = np.array([0.2, 0.05])
_ROBIN_ALPHA = 2.0
_ROBIN_DATA = np.array([0.1, -0.05])
_PARAM = np.array([[1.0], [0.3]])  # (E, nu)


class _ConstantVector:
    """A constant vector field, shape (2, npts); picklable."""

    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value, dtype=float)

    def __call__(self, coords: Any) -> Any:
        return np.outer(self._value, np.ones(coords.shape[1]))


def _plate(bkd: NumpyBkd) -> Tuple[
    GalerkinSystem[NumpyArray], Any, VectorLagrangeBasis[NumpyArray]
]:
    mesh = StructuredMesh2D(
        nx=6, ny=6, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
    )
    basis = VectorLagrangeBasis(mesh, degree=1)
    physics = CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        bkd=bkd,
        body_force=TimeIndependent(_ConstantVector(_BODY_FORCE)),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            RobinBC(
                basis,
                "right",
                _ROBIN_ALPHA,
                TimeIndependent(_ConstantVector(_ROBIN_DATA)),
                bkd,
            ),
            NeumannBC(
                basis, "top", TimeIndependent(_ConstantVector(_TRACTION)), bkd
            ),
        ],
    )
    return system, create_galerkin_lame_parameterization(physics, bkd), basis


def _support_force(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: Any,
    basis: VectorLagrangeBasis[NumpyArray],
    component: int,
) -> EssentialBoundaryFluxFunctional[NumpyArray]:
    dofs = basis.get_dofs("left", components=(component,))
    return EssentialBoundaryFluxFunctional(
        GalerkinParameterizedResidual(system.spatial_operator(), param_obj),
        system.steady(),
        dofs,
        bkd.ones((1, dofs.shape[0])),
    )


def _robin_edge_integral(
    bkd: NumpyBkd,
    basis: VectorLagrangeBasis[NumpyArray],
    state: NumpyArray,
    component: int,
) -> float:
    """integral over the right edge of u_c ds, by the trapezoid rule on
    the edge nodes (exact for a P1 trace)."""
    dofs = bkd.to_numpy(basis.get_dofs("right", components=(component,)))
    y = bkd.to_numpy(basis.dof_coordinates())[1, dofs]
    order = np.argsort(y)
    values = bkd.to_numpy(state)[dofs[order], 0]
    return float(np.trapezoid(values, y[order]))


class TestSupportReaction:
    @pytest.mark.parametrize("component", [0, 1])
    def test_force_balance(self, numpy_bkd: NumpyBkd, component: int) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _plate(bkd)
        functional = _support_force(bkd, system, param_obj, basis, component)
        model = GalerkinSteadyForwardModel(
            system.steady(),
            param_obj,
            bkd.zeros((system.nstates(),)),
            bkd,
            functional=functional,
        )
        param = bkd.asarray(_PARAM)
        state = model.state_equation().solve(
            bkd.zeros((system.nstates(), 1)), param
        )
        robin_force = _ROBIN_DATA[component] - _ROBIN_ALPHA * (
            _robin_edge_integral(bkd, basis, state, component)
        )
        applied = _BODY_FORCE[component] + _TRACTION[component] + robin_force
        bkd.assert_allclose(
            model(param), bkd.asarray([[-applied]]), rtol=1e-10
        )

    def test_robin_spring_carries_load(self, numpy_bkd: NumpyBkd) -> None:
        """Guard against a vacuous balance: the spring's share of the
        horizontal load is not negligible, so a flux that dropped the
        Robin term would fail the balance."""
        bkd = numpy_bkd
        system, param_obj, basis = _plate(bkd)
        model = GalerkinSteadyForwardModel(
            system.steady(), param_obj, bkd.zeros((system.nstates(),)), bkd
        )
        state = model.state_equation().solve(
            bkd.zeros((system.nstates(), 1)), bkd.asarray(_PARAM)
        )
        spring = _ROBIN_ALPHA * _robin_edge_integral(bkd, basis, state, 0)
        assert abs(spring) > 1e-3

    def test_gradient_and_hvp_wrt_e_nu_match_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _plate(bkd)
        model = GalerkinSteadyForwardModel(
            system.steady(),
            param_obj,
            bkd.zeros((system.nstates(),)),
            bkd,
            functional=_support_force(bkd, system, param_obj, basis, 0),
        )
        derivs = model.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is not None
        assert all(
            solve_based_derivatives_pass_fd(model, bkd.asarray(_PARAM))
        )
