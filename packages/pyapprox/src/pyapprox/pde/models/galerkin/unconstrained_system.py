r"""A transient Galerkin system with no row treated: the unconstrained one.

:math:`M \dot u = F(u, t; p)` as assembled: the residual, its Jacobian
and mass with the essential rows as integration by parts left them, and
the parameterization's derivatives unchanged. It is not solved: its
essential rows lack the boundary flux, so they are not valid evolution
equations. A quantity read on the essential rows is evaluated on it: a
stepper of it, evaluated at a solved trajectory, gives the physical
residual of those rows, the boundary flux
(:class:`~pyapprox.ode.step_residual_terms.StepResidualTerm`).

It reads what the solved system reads: the same discretization, the
same :class:`ParameterState` (one ``set_param`` moves both, version
included), the same Jacobian classification and the same conversion of
the parameterization's derivatives; it treats no row.
"""

from typing import Generic, Optional, TypeVar, Union

from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.residual_derivatives import ODEResidualDerivatives, StatedZero
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.galerkin.protocols.discretization import (
    SteadyDiscretizationProtocol,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    GalerkinJacobianOperator,
    classified_jacobian,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.pde.models.galerkin.parameter_fields import residual_derivatives
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearSolverProtocol,
)

F = TypeVar("F", bound=FormatProtocol)


class GalerkinUnconstrainedRightHandSide(Generic[Array, F]):
    r""":math:`f = F(u, t; p)`, the spatial residual as assembled, and its
    derivatives bundle with no row treated.

    The bundle's Jacobian is the assembled stack Jacobian, declared by
    what it depends on as the solved system's is; its parameter fields
    are the parameterization's own, read at ``parameters``.

    Parameters
    ----------
    discretization : SteadyDiscretizationProtocol
        The system on a stack; no mass is read.
    parameters : ParameterState, optional
        The parameters, shared with the solved system; none for a system
        without them.
    initial_state : ParamDerivatives or STATED_ZERO, optional
        Where the initial state's parameter derivatives come from: the
        parameterization's own bundle by default (a transient problem),
        :data:`~pyapprox.ode.residual_derivatives.STATED_ZERO` for none (a
        steady problem, or an initial state independent of the
        parameters).

    Raises
    ------
    TypeError
        If ``discretization`` does not satisfy
        ``SteadyDiscretizationProtocol`` or ``parameters`` is not a
        ``ParameterState``.
    ValueError
        If the parameterization writes objects the system does not hold,
        or has a parameter Jacobian and the initial state's bundle no
        ``initial_param_jacobian``.
    """

    def __init__(
        self,
        discretization: SteadyDiscretizationProtocol[Array, Array, F],
        parameters: Optional[ParameterState[Array]] = None,
        initial_state: Optional[Union[ParamDerivatives[Array], StatedZero]] = None,
    ) -> None:
        if not isinstance(discretization, SteadyDiscretizationProtocol):
            raise TypeError(
                "discretization must satisfy SteadyDiscretizationProtocol, got "
                f"{type(discretization).__name__}"
            )
        if parameters is not None and not isinstance(parameters, ParameterState):
            raise TypeError(
                "parameters must be a ParameterState, got "
                f"{type(parameters).__name__}"
            )
        system = discretization.system()
        self._spatial_operator = system.spatial_operator()
        self._bkd = system.bkd()
        self._parameters = parameters
        jacobian = classified_jacobian(
            system, GalerkinJacobianOperator(discretization, None)
        )
        self._derivatives: ODEResidualDerivatives[Array, F]
        if parameters is None:
            self._derivatives = ODEResidualDerivatives.state_only(jacobian)
            return
        parameterization = parameters.parameterization()
        require_owned_targets(parameterization, system)
        param_derivatives = parameterization.param_derivatives()
        self._derivatives = residual_derivatives(
            jacobian,
            param_derivatives,
            self._spatial_operator.state_derivatives().state_state_hvp,
            parameters,
            discretization.space(),
            discretization.parameter_space(parameters.nparams()),
            system.bkd(),
            initial_state=(
                param_derivatives if initial_state is None else initial_state
            ),
        )

    def bkd(self) -> Backend[Array]:
        """The system's backend."""
        return self._bkd

    def __call__(self, state: Array, time: float) -> Array:
        """:math:`F(u, t; p)`. Shape: (nstates,)"""
        return self._spatial_operator.spatial_residual(state, time)

    def set_param(self, param: Array) -> None:
        """Set the shared parameters: every system built on the same
        :class:`ParameterState` moves with them.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,)

        Raises
        ------
        ValueError
            If the system has no parameters, or ``param`` is not 1-D.
        """
        if self._parameters is None:
            raise ValueError("this system was built without parameters")
        self._parameters.set_param(param)

    def derivatives(self) -> ODEResidualDerivatives[Array, F]:
        """The derivatives of :math:`F`, no row treated."""
        return self._derivatives

    def parameters_version(self) -> int:
        """The shared parameter state's version; zero, unchanging, without
        parameters."""
        if self._parameters is None:
            return 0
        return self._parameters.parameters_version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(parameters={self._parameters!r})"


def galerkin_unconstrained_system(
    discretization: GalerkinStackDiscretization[Array, Array, F],
    solver: LinearSolverProtocol[Array, F],
    parameters: Optional[ParameterState[Array]] = None,
) -> MassFormSystem[Array, Array, F]:
    r"""The unconstrained system as a mass-form stepper reads it:
    :class:`GalerkinUnconstrainedRightHandSide`, the mass as assembled
    (consistent or lumped, as the discretization's assembler chooses),
    the discretization's space and ``solver``.

    The mass is the one assembled now; a discretization whose mass
    changes afterwards needs a new system.

    Parameters
    ----------
    discretization : GalerkinStackDiscretization
        The system on a stack.
    solver : LinearSolverProtocol
        Solves with the mass and the stage operators; pairs with the
        stack's format.
    parameters : ParameterState, optional
        The parameters, shared with the solved system.
    """
    return MassFormSystem(
        rhs=GalerkinUnconstrainedRightHandSide(discretization, parameters),
        mass=discretization.assemble_mass(),
        space=discretization.space(),
        solver=solver,
        form=BackendColumnForm(discretization.system().bkd()),
    )
