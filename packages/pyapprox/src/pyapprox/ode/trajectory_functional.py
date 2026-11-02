r"""A whole-trajectory functional, as the sweeps read it.

The transient functionals take a whole trajectory as one backend array
(:class:`~pyapprox.ode.functionals.protocols.TransientFunctionalProtocol`).
The sweeps of :mod:`pyapprox.ode.trajectory_derivatives` read a
functional's terms per stored state, as stack vectors
(:class:`~pyapprox.ode.trajectory_derivatives.StateTermProtocol`).
:class:`TransientFunctionalTerm` is the boundary between the two on
the backend stack: a state term that forms the trajectory array once and
evaluates the functional's Jacobians on it once, so it is built per
recorded trajectory; its bundle carries the second derivatives when the
functional's does.

It is transitional: a functional that reads the store one state at a
time is a state term itself, needs no trajectory array and replaces it.
"""

from dataclasses import dataclass
from typing import Generic, Optional

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalProtocol,
    TransientParamHVPFn,
    TransientStepHVPFn,
)
from pyapprox.ode.trajectory_derivatives import (
    StateTermDerivatives,
    TrajectoryFunctional,
)
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol, trajectory_array
from pyapprox.util.backends.protocols import Array


class TransientFunctionalTerm(Generic[Array]):
    r"""A scalar transient functional's derivatives on one recorded
    trajectory, as single-column backend arrays: a state term.

    Its first derivatives are always given; its bundle carries the four
    second derivatives when the functional's bundle has its HVPs (whose
    per-step HVPs index stored states, so its second derivative in the
    states is block diagonal over them, as the Hessian-vector sweep
    requires).

    Parameters
    ----------
    functional : TransientFunctionalProtocol
        :math:`Q(y, p)`, one QoI, every parameter shared with the
        residual; its bundle must have ``state_jacobian`` and
        ``param_jacobian``. A functional integrating over time must
        already hold the scheme's quadrature.
    store : TrajectoryStoreProtocol
        The recorded trajectory.
    param : Array
        The parameters the trajectory was recorded at. Shape:
        ``(nparams, 1)``.

    Raises
    ------
    TypeError
        If ``functional`` does not satisfy its protocol or lacks the
        Jacobians.
    ValueError
        If it has more than one QoI or parameters of its own, or
        ``param`` is not a single column.
    """

    def __init__(
        self,
        functional: TransientFunctionalProtocol[Array],
        store: TrajectoryStoreProtocol[Array],
        param: Array,
    ) -> None:
        if not isinstance(functional, TransientFunctionalProtocol):
            raise TypeError(
                "functional must satisfy TransientFunctionalProtocol, got "
                f"{type(functional).__name__}"
            )
        if functional.nqoi() != 1:
            raise ValueError(
                f"the sweeps take a scalar functional, got nqoi={functional.nqoi()}"
            )
        if functional.nunique_params() != 0:
            raise ValueError(
                "the functional has parameters of its own "
                f"({functional.nunique_params()}); the sweeps differentiate "
                "the residual's parameters only"
            )
        if param.ndim != 2 or param.shape[1] != 1:
            raise ValueError(
                f"param must have shape (nparams, 1), got {tuple(param.shape)}"
            )
        derivs = functional.derivatives()
        if derivs.state_jacobian is None or derivs.param_jacobian is None:
            raise TypeError(
                f"{type(functional).__name__}'s derivatives() has no Jacobians"
            )
        self._bkd, self._param = functional.bkd(), param
        self._trajectory = trajectory_array(store, self._bkd)
        # dQ/dy at every stored state, (nstates, ntimes), and the direct
        # dQ/dp, (1, nparams).
        self._state_jacobian = derivs.state_jacobian(self._trajectory, param)
        self._param_jacobian = derivs.param_jacobian(self._trajectory, param)
        self._hvps: Optional[_Hvps[Array]] = (
            None
            if derivs.state_state_hvp is None
            or derivs.state_param_hvp is None
            or derivs.param_state_hvp is None
            or derivs.param_param_hvp is None
            else _Hvps(
                derivs.state_state_hvp,
                derivs.state_param_hvp,
                derivs.param_state_hvp,
                derivs.param_param_hvp,
            )
        )
        self._derivatives: StateTermDerivatives[Array] = (
            StateTermDerivatives.first_order()
            if self._hvps is None
            else StateTermDerivatives.second_order(
                self._state_state,
                self._state_param,
                self._param_state,
                self._param_param,
            )
        )

    def trajectory(self) -> Array:
        """The trajectory array the Jacobians were evaluated on. Shape:
        ``(nstates, ntimes)``. Not to be modified."""
        return self._trajectory

    def state_cotangent(self, index: int) -> Array:
        r""":math:`\partial Q/\partial y_{\mathrm{index}}`. Shape:
        ``(nstates, 1)``; ``-1`` is the last state."""
        return self._state_jacobian[:, self._column_of(index)]

    def param_cotangent(self) -> Array:
        r"""The direct :math:`\partial Q/\partial p`. Shape:
        ``(nparams, 1)``."""
        return self._param_jacobian.T

    def derivatives(self) -> StateTermDerivatives[Array]:
        """The second derivatives when the functional has its HVPs."""
        return self._derivatives

    def _column_of(self, index: int) -> slice:
        # Normalize so the slice of the last state is not empty.
        column = index % self._trajectory.shape[1]
        return slice(column, column + 1)

    def _time_index(self, index: int) -> int:
        return index % self._trajectory.shape[1]

    def _column(self, x: Array) -> Array:
        """``x`` as a column: the HVPs may return 1-D arrays."""
        return self._bkd.reshape(x, (-1, 1))

    def _checked_hvps(self) -> "_Hvps[Array]":
        if self._hvps is None:
            raise RuntimeError(
                "the functional has no HVPs; check derivatives() before calling"
            )
        return self._hvps

    def _state_state(self, index: int, w: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial y_m^2)\, w`. Shape:
        ``(nstates, 1)``."""
        hvp, time = self._checked_hvps().state_state, self._time_index(index)
        return self._column(hvp(self._trajectory, self._param, time, w))

    def _state_param(self, index: int, v: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial y_m \partial p)\, v`. Shape:
        ``(nstates, 1)``."""
        hvp, time = self._checked_hvps().state_param, self._time_index(index)
        return self._column(hvp(self._trajectory, self._param, time, v))

    def _param_state(self, index: int, w: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial p\, \partial y_m)\, w`. Shape:
        ``(nparams, 1)``."""
        hvp, time = self._checked_hvps().param_state, self._time_index(index)
        return self._column(hvp(self._trajectory, self._param, time, w))

    def _param_param(self, v: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial p^2)\, v`. Shape:
        ``(nparams, 1)``."""
        hvp = self._checked_hvps().param_param
        return self._column(hvp(self._trajectory, self._param, v))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class _Hvps(Generic[Array]):
    """A transient functional's four HVPs, known present."""

    state_state: TransientStepHVPFn[Array]
    state_param: TransientStepHVPFn[Array]
    param_state: TransientStepHVPFn[Array]
    param_param: TransientParamHVPFn[Array]


def from_transient_functional(
    functional: TransientFunctionalProtocol[Array],
    store: TrajectoryStoreProtocol[Array],
    param: Array,
) -> TrajectoryFunctional[Array]:
    """A transient functional, which takes a whole trajectory array, as
    the sweeps read it on the trajectory ``store`` recorded: one state
    term, a :class:`TransientFunctionalTerm`; see it for the parameters
    and the errors raised."""
    return TrajectoryFunctional(
        state_terms=(TransientFunctionalTerm(functional, store, param),)
    )
