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

from typing import Generic, Optional, TypeVar

from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.residual_derivatives import ODEResidualDerivatives
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    GalerkinJacobianOperator,
    classified_jacobian,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.pde.models.galerkin.parameter_fields import residual_derivatives
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
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
    discretization : GalerkinStackDiscretization
        The system on a stack.
    parameters : ParameterState, optional
        The parameters, shared with the solved system; none for a system
        without them.

    Raises
    ------
    TypeError
        If ``discretization`` is not a ``GalerkinStackDiscretization`` or
        ``parameters`` not a ``ParameterState``.
    ValueError
        If the parameterization writes objects the system does not hold.
    """

    def __init__(
        self,
        discretization: GalerkinStackDiscretization[Array, Array, F],
        parameters: Optional[ParameterState[Array]] = None,
    ) -> None:
        if not isinstance(discretization, GalerkinStackDiscretization):
            raise TypeError(
                "discretization must be a GalerkinStackDiscretization, got "
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
        self._derivatives = residual_derivatives(
            jacobian,
            parameterization.param_derivatives(),
            self._spatial_operator.state_derivatives().state_state_hvp,
            parameters,
            discretization.space(),
            discretization.parameter_space(parameters.nparams()),
            system.bkd(),
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

    Raises
    ------
    TypeError
        If the discretization's space has no inner product.
    """
    space = discretization.space()
    if not isinstance(space, InnerProductSpaceProtocol):
        raise TypeError(
            "the discretization's space must be an inner-product space, "
            f"got {type(space).__name__}"
        )
    return MassFormSystem(
        rhs=GalerkinUnconstrainedRightHandSide(discretization, parameters),
        mass=discretization.assemble_mass(),
        space=space,
        solver=solver,
        form=BackendColumnForm(discretization.system().bkd()),
    )
