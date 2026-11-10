r"""The stack-generic steady adjoint equals the method-based one on a
Galerkin problem.

Steady ADR on [0, 1] (P1 on 10 cells, unit forcing, u = 0 at both ends,
reaction :math:`u^2` or none), the diffusivity :math:`\exp(W\theta)` with
three smooth modes :math:`0.4 \sin((k+1)\pi x)/(k+1)`, at
:math:`\theta = (0.4, -0.3, 0.2)`; :math:`Q = w^\top u` with :math:`w`
the indicator of the first free node (``WeightedSumFunctional``).

Old: ``AdjointOperator`` and ``ForwardSensitivityOperator`` on
``GalerkinParameterizedStateEquation``. New:
``SteadyDerivatives`` on ``SteadySystem`` (stack ``Newton``, atol 1e-13,
SuperLU) of the pinned unconstrained right-hand side at t = 0, with the
functional wrapped as a state term (``SteadyFunctionalTerm``). The
gradient and the forward product along v = (0.5, 0.7, -0.6) agree to
1e-12 and the HVP along v to 1e-10, relative; one factorization per
Newton step plus one at the solution serves the adjoint, the forward
product and the HVP's two solves (one in all for the linear problem).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.functionals.as_steady_term import (
    SteadyFunctionalTerm,
)
from pyapprox.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.implicitfunction.operator.adjoint_operator import AdjointOperator
from pyapprox.implicitfunction.operator.sensitivities import (
    ForwardSensitivityOperator,
)
from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import RowPin
from pyapprox.implicitfunction.setups import KeepAll
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.pinned_rows import values_at_time
from pyapprox.ode.residual_at_time import RightHandSideAtTime
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.linalg.counting_solver import CountingSolver

_Np = NDArray[Any]
_NPARAMS = 3
_PARAM = np.array([[0.4], [-0.3], [0.2]])


def _unit_forcing(x: _Np) -> _Np:
    return np.ones(x.shape[1])


class _Fixed:
    """Supplies the parameters the problem is solved at."""

    def __init__(self, param: _Np) -> None:
        self._param = param

    def __call__(self) -> _Np:
        return self._param


def _problem(
    bkd: NumpyBkd, nonlinear: bool
) -> Tuple[
    AdjointOperator[_Np],
    ForwardSensitivityOperator[_Np],
    SteadyDerivatives[_Np, _Np, Any],
    Any,
    Any,
    int,
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    reaction = (
        CallableReaction(
            lambda x, u: u**2,
            lambda x, u: 2.0 * u,
            lambda x, u: np.full_like(u, 2.0),
        )
        if nonlinear
        else None
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=reaction,
        forcing=TimeIndependent(_unit_forcing),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(_NPARAMS)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    parameterization = AdvectionDiffusionParameterization(
        physics,
        diffusivity_map=TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp),
        bkd=bkd,
    )
    nstates = system.nstates()
    constrained = set(int(d) for d in bkd.to_numpy(system.constraint_set().dofs()))
    weights = np.zeros((nstates, 1))
    weights[next(i for i in range(nstates) if i not in constrained)] = 1.0
    functional = WeightedSumFunctional(bkd.asarray(weights), _NPARAMS, bkd)
    state_equation = GalerkinParameterizedStateEquation(
        system.steady(), parameterization, bkd
    )
    old = AdjointOperator(state_equation, functional)
    forward = ForwardSensitivityOperator(state_equation, functional)

    discretization = skfem_discretization(system)
    adapter = create_galerkin_physics_ode_residual(
        discretization, ScipySparseLU(), parameterization
    )
    adapter.set_param(_PARAM[:, 0])
    rhs: GalerkinUnconstrainedRightHandSide[_Np, Any] = (
        GalerkinUnconstrainedRightHandSide(
            discretization, adapter.parameter_state(), initial_state=STATED_ZERO
        )
    )
    pinned = PinnedResidual(
        RightHandSideAtTime(rhs, 0.0, BackendColumnForm(bkd)),
        RowPin(values_at_time(EssentialPinnedRows(discretization), 0.0)),
        discretization.space(),
    )
    space = discretization.space()
    assert isinstance(space, InnerProductSpaceProtocol)
    lu: CountingSolver[_Np, Any] = CountingSolver(ScipySparseLU())
    steady = SteadySystem(pinned, space, KeepAll(lu))
    new = SteadyDerivatives(
        steady,
        StateFunctional.from_terms(
            bkd, 1, state_terms=(SteadyFunctionalTerm(functional, _Fixed(_PARAM)),)
        ),
    )
    return old, forward, new, steady, lu, nstates


def _close(actual: _Np, expected: _Np, rtol: float) -> None:
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


class TestAgainstTheMethodBasedAdjoint:
    @pytest.mark.parametrize("nonlinear", [False, True])
    def test_gradient_and_hvp(self, numpy_bkd: NumpyBkd, nonlinear: bool) -> None:
        bkd = numpy_bkd
        old, forward, new, steady, lu, nstates = _problem(bkd, nonlinear)
        init = np.zeros((nstates, 1))
        vvec = np.array([[0.5], [0.7], [-0.6]])
        old_gradient = old.jacobian(init, _PARAM)
        old_hvp = old.derivatives().hvp
        assert old_hvp is not None
        expected_hvp = old_hvp(init, _PARAM, vvec)

        point = steady.solve(Newton(atol=1e-13, rtol=0.0, maxiters=30), init)
        assert point.converged()
        iterations = point.info().iterations
        assert (iterations == 1) == (not nonlinear)
        adjoint = new.adjoint(point)
        _close(new.gradient(adjoint), old_gradient.T, 1e-12)
        hessian = new.hessian()
        assert hessian is not None
        _close(hessian.hvp(adjoint, vvec), expected_hvp, 1e-10)
        _close(new.value(point), old(init, _PARAM), 1e-12)
        _close(new.tangent(point, vvec), forward.jacobian(init, _PARAM) @ vvec, 1e-12)
        assert lu.setups == (1 if not nonlinear else iterations + 1)
