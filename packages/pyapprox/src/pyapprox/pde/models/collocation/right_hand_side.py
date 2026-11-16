r"""A collocation physics' interior operator as an ODE right-hand side.

:math:`f = f(u, t; p)`, the interior operator collocated at every node,
and its derivatives bundle, with no boundary row treated: the boundary
rows are the business of the system that holds the physics. The bundle's
Jacobian is classified by what the physics declares it depends on, and
its parameter fields are the parameterization's own, read at a shared
:class:`ParameterState`.
"""

from typing import Generic, Optional, Union

from pyapprox.ode.dense_jacobian_operator import (
    DenseJacobianOperator,
    DenseJacobianSnapshot,
)
from pyapprox.ode.residual_derivatives import (
    Jacobian,
    ODEResidualDerivatives,
    StatedZero,
    backend_spaces,
    backend_state_space,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.models.galerkin.parameter_fields import residual_derivatives
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor


class _InteriorJacobianMatrix(Generic[Array]):
    """The interior Jacobian as ``DenseJacobianOperator`` reads it."""

    def __init__(self, interior: CollocationInteriorOperatorProtocol[Array]) -> None:
        self._interior = interior

    def jacobian(self, state: Array, time: float) -> Array:
        return self._interior.interior_jacobian(state, time)


class _StateFreeJacobian(Generic[Array]):
    """The Jacobian of an affine :math:`f` at a time, taken at a fixed
    reference state: an affine operator ignores the state."""

    def __init__(
        self, operator: DenseJacobianOperator[Array], reference: Array
    ) -> None:
        self._operator, self._reference = operator, reference

    def __call__(
        self, time: float, point: LinearizationPoint
    ) -> DenseJacobianSnapshot[Array]:
        return self._operator.jacobian_operator(self._reference, time, point)


class _ConstantJacobian(Generic[Array]):
    """The Jacobian of an affine :math:`f` declared time-invariant: taken
    at the reference state and time zero."""

    def __init__(self, at_time: _StateFreeJacobian[Array]) -> None:
        self._at_time = at_time

    def __call__(self, point: LinearizationPoint) -> DenseJacobianSnapshot[Array]:
        return self._at_time(0.0, point)


def classified_interior_jacobian(
    interior: CollocationInteriorOperatorProtocol[Array],
    descriptor: SpaceDescriptor,
) -> Jacobian[Array, DenseFormat[Array]]:
    r"""The interior Jacobian as a bundle's Jacobian, given only what it
    depends on.

    Of the state and time unless the physics declares :math:`f` affine in
    the state (zero curvature); then of the time only, and of neither
    when the Jacobian is also declared time-invariant.

    Parameters
    ----------
    interior : CollocationInteriorOperatorProtocol
        The interior operator and its declarations.
    descriptor : SpaceDescriptor
        The state space.
    """
    bkd = interior.bkd()
    operator = DenseJacobianOperator(_InteriorJacobianMatrix(interior), descriptor, bkd)
    if not interior.interior_state_derivatives().affine():
        return Jacobian.general(operator.jacobian_operator)
    at_time = _StateFreeJacobian(operator, bkd.zeros((interior.nstates(),)))
    if interior.interior_jacobian_is_time_invariant():
        return Jacobian.constant(_ConstantJacobian(at_time))
    return Jacobian.state_free(at_time)


class CollocationRightHandSide(Generic[Array]):
    r""":math:`f = f(u, t; p)`, a collocation physics' interior operator,
    and its derivatives bundle with no row treated.

    Parameters
    ----------
    interior : CollocationInteriorOperatorProtocol
        The physics' interior operator.
    parameters : ParameterState, optional
        The parameters, shared with every system built on them; none for
        a system without them.
    initial_state : ParamDerivatives or STATED_ZERO, optional
        Where the initial state's parameter derivatives come from: the
        parameterization's own bundle by default (a transient problem),
        :data:`~pyapprox.ode.residual_derivatives.STATED_ZERO` for none (a
        steady problem, or an initial state independent of the
        parameters).

    Raises
    ------
    TypeError
        If ``interior`` does not satisfy
        ``CollocationInteriorOperatorProtocol`` or ``parameters`` is not a
        ``ParameterState``.
    ValueError
        If the parameterization writes objects other than the interior
        operator, or has a parameter Jacobian and the initial state's
        bundle no ``initial_param_jacobian``.
    """

    def __init__(
        self,
        interior: CollocationInteriorOperatorProtocol[Array],
        parameters: Optional[ParameterState[Array]] = None,
        initial_state: Optional[Union[ParamDerivatives[Array], StatedZero]] = None,
    ) -> None:
        if not isinstance(interior, CollocationInteriorOperatorProtocol):
            raise TypeError(
                "interior must satisfy CollocationInteriorOperatorProtocol, got "
                f"{type(interior).__name__}"
            )
        if parameters is not None and not isinstance(parameters, ParameterState):
            raise TypeError(
                f"parameters must be a ParameterState, got {type(parameters).__name__}"
            )
        self._interior = interior
        self._bkd = interior.bkd()
        self._parameters = parameters
        nstates = interior.nstates()
        self._derivatives: ODEResidualDerivatives[Array, DenseFormat[Array]]
        if parameters is None:
            states = backend_state_space(self._bkd, nstates)
            self._derivatives = ODEResidualDerivatives.state_only(
                classified_interior_jacobian(interior, states.descriptor())
            )
            return
        parameterization = parameters.parameterization()
        require_owned_targets(parameterization, interior)
        states, params = backend_spaces(self._bkd, nstates, parameters.nparams())
        param_derivatives = parameterization.param_derivatives()
        self._derivatives = residual_derivatives(
            classified_interior_jacobian(interior, states.descriptor()),
            param_derivatives,
            interior.interior_state_derivatives().state_state_hvp,
            parameters,
            states,
            params,
            self._bkd,
            initial_state=(
                param_derivatives if initial_state is None else initial_state
            ),
        )

    def bkd(self) -> Backend[Array]:
        """The physics' backend."""
        return self._bkd

    def __call__(self, state: Array, time: float) -> Array:
        """:math:`f(u, t; p)`. Shape: (nstates,)"""
        return self._interior.interior_residual(state, time)

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
            If the right-hand side has no parameters.
        """
        if self._parameters is None:
            raise ValueError("this right-hand side was built without parameters")
        self._parameters.set_param(param)

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        """The derivatives of :math:`f`, no row treated."""
        return self._derivatives

    def parameters_version(self) -> int:
        """The shared parameter state's version; zero, unchanging, without
        parameters."""
        if self._parameters is None:
            return 0
        return self._parameters.parameters_version()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(interior={self._interior!r}, "
            f"parameters={self._parameters!r})"
        )
