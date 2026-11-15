"""Derivative checks of a parameterized Galerkin steady problem, on the
steady system path.

Two checks cover what a component-wise check of a steady state equation
did:

- :func:`residual_bundle_report`: the residual's own derivatives (state
  and parameter Jacobians, the four adjoint-weighted second derivatives,
  and their dot tests) by ``ODEResidualBundleChecker``, on the
  unconstrained right-hand side the steady residual pins;
- :func:`observed_steady_function`: the solved functional (chosen state
  entries) as a ``SteadyImplicitFunction``, whose gradient and HVP the
  caller checks with ``DerivativeChecker``.

Requires skfem; import only after skipping when it is absent.
"""

from typing import Any, Optional, Sequence

import numpy as np

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.setups import KeepAll
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.bundle_checks import (
    BundleCheckReport,
)
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray


def residual_bundle_report(
    system: GalerkinSystem[NumpyArray],
    parameterization: ParameterizationProtocol[NumpyArray],
    state: NumpyArray,
    adj: NumpyArray,
    param: NumpyArray,
    time: float = 0.0,
) -> BundleCheckReport:
    """The bundle checker's report on the residual's derivatives at
    ``state``, weighted by ``adj``, at ``param`` (all 1-D)."""
    rhs: GalerkinUnconstrainedRightHandSide[NumpyArray, Any] = (
        GalerkinUnconstrainedRightHandSide(
            skfem_discretization(system),
            ParameterState(parameterization),
            initial_state=STATED_ZERO,
        )
    )
    return ODEResidualBundleChecker(rhs).check(state, time, adj=adj, param=param)


def observed_steady_function(
    system: GalerkinSystem[NumpyArray],
    parameterization: ParameterizationProtocol[NumpyArray],
    rows: Sequence[int],
    bkd: NumpyBkd,
    time: Optional[float] = None,
) -> SteadyImplicitFunction[NumpyArray, Any]:
    """The state at ``rows`` as a function of the parameters: the steady
    residual solved by Newton (atol 1e-13) from zero, sparse LU kept per
    factorization."""
    nstates = system.nstates()
    discretization = skfem_discretization(system)
    space = discretization.space()
    if not isinstance(space, InnerProductSpaceProtocol):
        raise TypeError("the discretization's space has no inner product")
    parameters = ParameterState(parameterization)
    functional: StateFunctional[NumpyArray, NumpyArray] = StateFunctional.from_terms(
        bkd,
        len(rows),
        state_terms=(ObservedValue(StateEntries(bkd, nstates, list(rows))),),
    )
    return SteadyImplicitFunction(
        galerkin_steady_residual(discretization, parameters, time),
        space,
        parameters,
        KeepAll(ScipySparseLU()),
        Newton(atol=1e-13, rtol=0.0, maxiters=30),
        functional,
        np.zeros((nstates, 1)),
        bkd,
    )


def first_free_row(system: GalerkinSystem[NumpyArray], bkd: NumpyBkd) -> int:
    """The first state row without an essential constraint."""
    constrained = {int(d) for d in bkd.to_numpy(system.constraint_set().dofs())}
    return next(ii for ii in range(system.nstates()) if ii not in constrained)
