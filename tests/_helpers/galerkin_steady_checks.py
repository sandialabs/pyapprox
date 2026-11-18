"""Derivative checks of a parameterized Galerkin steady problem, on the
steady system path.

Two checks cover what a component-wise check of a steady state equation
did:

- :func:`residual_bundle_report`: the residual's own derivatives (state
  and parameter Jacobians, the four adjoint-weighted second derivatives,
  and their dot tests) by ``ODEResidualBundleChecker``, on the
  unconstrained right-hand side the steady residual pins;
- :func:`observed_steady_function` and :func:`flux_steady_function`: a
  solved functional (chosen state entries, or an essential-boundary
  flux) as a ``SteadyImplicitFunction``, whose gradient and HVP the
  caller checks with ``DerivativeChecker``.

Requires skfem; import only after skipping when it is absent.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.setups import KeepAll, RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.bundle_checks import (
    BundleCheckReport,
)
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    essential_boundary_flux,
)
from pyapprox.pde.models.galerkin.steady_residual import (
    galerkin_steady_residual,
    solve_galerkin_steady,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    InnerProductSpaceProtocol,
    LinearizationPoint,
)
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


class _ObservedRows:
    """Builds the functional of the state at chosen rows; a class, not a
    closure, so whatever holds it pickles."""

    def __init__(self, bkd: NumpyBkd, nstates: int, rows: Sequence[int]) -> None:
        self._bkd, self._nstates, self._rows = bkd, nstates, list(rows)

    def __call__(
        self, residual: PinnedResidual[NumpyArray, Any]
    ) -> StateFunctional[NumpyArray, NumpyArray]:
        entries = StateEntries(self._bkd, self._nstates, self._rows)
        return StateFunctional.from_terms(
            self._bkd, len(self._rows), state_terms=(ObservedValue(entries),)
        )


class _EssentialFlux:
    """Builds the essential-boundary flux functional from the solved
    residual; a class, not a closure, so whatever holds it pickles."""

    def __init__(
        self, bkd: NumpyBkd, dofs: Sequence[int], weights: NumpyArray
    ) -> None:
        self._bkd, self._weights = bkd, weights
        self._dofs = bkd.asarray(list(dofs), dtype=bkd.int64_dtype())

    def __call__(
        self, residual: PinnedResidual[NumpyArray, Any]
    ) -> StateFunctional[NumpyArray, NumpyArray]:
        flux = essential_boundary_flux(self._bkd, residual, self._dofs, self._weights)
        return StateFunctional.from_terms(
            self._bkd, int(self._weights.shape[0]), state_terms=(flux,)
        )


def _solved_function(
    system: GalerkinSystem[NumpyArray],
    parameterization: ParameterizationProtocol[NumpyArray],
    bkd: NumpyBkd,
    time: Optional[float],
    functional_of: Callable[
        [PinnedResidual[NumpyArray, Any]], StateFunctional[NumpyArray, NumpyArray]
    ],
) -> SteadyImplicitFunction[NumpyArray, Any]:
    """A functional of the steady solution as a function of the
    parameters: the steady residual solved by Newton (atol 1e-13) from
    zero, sparse LU kept per factorization; ``functional_of`` builds the
    functional from the solved residual."""
    discretization = skfem_discretization(system)
    space = discretization.space()
    if not isinstance(space, InnerProductSpaceProtocol):
        raise TypeError("the discretization's space has no inner product")
    parameters = ParameterState(parameterization)
    residual = galerkin_steady_residual(discretization, parameters, time)
    return SteadyImplicitFunction(
        residual,
        space,
        parameters,
        KeepAll(ScipySparseLU()),
        Newton(atol=1e-13, rtol=0.0, maxiters=30),
        functional_of(residual),
        np.zeros((system.nstates(), 1)),
        bkd,
    )


def observed_steady_function(
    system: GalerkinSystem[NumpyArray],
    parameterization: ParameterizationProtocol[NumpyArray],
    rows: Sequence[int],
    bkd: NumpyBkd,
    time: Optional[float] = None,
) -> SteadyImplicitFunction[NumpyArray, Any]:
    """The state at ``rows`` as a function of the parameters."""
    builder = _ObservedRows(bkd, system.nstates(), rows)
    return _solved_function(system, parameterization, bkd, time, builder)


def flux_steady_function(
    system: GalerkinSystem[NumpyArray],
    parameterization: ParameterizationProtocol[NumpyArray],
    dofs: Sequence[int],
    weights: NumpyArray,
    bkd: NumpyBkd,
    time: Optional[float] = None,
) -> SteadyImplicitFunction[NumpyArray, Any]:
    """The essential-boundary flux on ``dofs`` (one QoI per row of
    ``weights``, shape ``(nqoi, len(dofs))``) as a function of the
    parameters, built from the solved residual."""
    builder = _EssentialFlux(bkd, dofs, weights)
    return _solved_function(system, parameterization, bkd, time, builder)


def solved_steady(
    system: GalerkinSystem[NumpyArray],
    parameterization: ParameterizationProtocol[NumpyArray],
    param: NumpyArray,
    bkd: NumpyBkd,
) -> Tuple[PinnedResidual[NumpyArray, Any], NumpyArray]:
    """The steady residual at ``param`` (1-D) and its solution, shape
    ``(nstates, 1)``: Newton (atol 1e-12) from zero, sparse LU each
    iteration. A flux of the solution is built from the residual
    returned, so it reads the same parameters and time."""
    parameters = ParameterState(parameterization)
    parameters.set_param(param)
    residual, point = solve_galerkin_steady(
        skfem_discretization(system),
        RecomputeAlways(ScipySparseLU()),
        Newton(atol=1e-12, rtol=0.0, maxiters=50),
        parameters,
    )
    if not point.converged():
        raise RuntimeError("the steady solve did not converge")
    return residual, np.array(point.solution())


class DenseSteadyResidual:
    """The steady residual of a parameter-free system, read row by row.

    ``galerkin_steady_residual`` at ``time``: the spatial residual with
    its essential rows replaced by :math:`u_d - g`. Values are 1-D and the
    Jacobian dense, assembled by applying the residual's Jacobian to the
    identity, for tests on small meshes that read single rows (pinned
    rows, finite differences).
    """

    def __init__(
        self, system: GalerkinSystem[NumpyArray], time: Optional[float] = None
    ) -> None:
        self._residual = galerkin_steady_residual(
            skfem_discretization(system), time=time
        )
        self._nstates = system.nstates()

    def nstates(self) -> int:
        return self._nstates

    def value(self, state: NumpyArray) -> NumpyArray:
        """The residual at ``state``, both shape ``(nstates,)``."""
        column: NumpyArray = self._residual(np.reshape(state, (-1, 1)))
        return np.array(column[:, 0])

    def jacobian(self, state: NumpyArray) -> NumpyArray:
        """The Jacobian at ``state`` (shape ``(nstates,)``), dense."""
        jacobian = self._residual.derivatives().jacobian
        if jacobian is None:
            raise TypeError("the steady residual has no Jacobian")
        point = LinearizationPoint(self._residual.parameters_version(), object())
        linearization = jacobian(np.reshape(state, (-1, 1)), point)
        return np.array(linearization.apply(np.eye(self._nstates)))


def first_free_row(system: GalerkinSystem[NumpyArray], bkd: NumpyBkd) -> int:
    """The first state row without an essential constraint."""
    constrained = {int(d) for d in bkd.to_numpy(system.constraint_set().dofs())}
    return next(ii for ii in range(system.nstates()) if ii not in constrained)
