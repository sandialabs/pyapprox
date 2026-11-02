r"""A whole-trajectory functional, as the sweeps read it.

The transient functionals take a whole trajectory as one backend array
(:class:`~pyapprox.ode.functionals.protocols.TransientFunctionalProtocol`).
The sweeps of :mod:`pyapprox.ode.trajectory_derivatives` read a
functional per stored state, as stack vectors
(:class:`~pyapprox.ode.trajectory_derivatives.TrajectoryFunctionalProtocol`).
:class:`BackendTrajectoryFunctional` is the boundary between the two on
the backend stack: it forms the trajectory array once and evaluates the
functional's Jacobians on it once, so it is built per recorded
trajectory.

It is transitional: a functional that reads the store one state at a
time satisfies ``TrajectoryFunctionalProtocol`` itself, needs no
trajectory array and replaces it.
"""

from typing import Generic

from pyapprox.ode.functionals.protocols import TransientFunctionalProtocol
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol, trajectory_array
from pyapprox.util.backends.protocols import Array


class BackendTrajectoryFunctional(Generic[Array]):
    r"""A scalar transient functional's first derivatives on one recorded
    trajectory, as single-column backend arrays.

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
        bkd = functional.bkd()
        self._trajectory = trajectory_array(store, bkd)
        # dQ/dy at every stored state, (nstates, ntimes), and the direct
        # dQ/dp, (1, nparams).
        self._state_jacobian = derivs.state_jacobian(self._trajectory, param)
        self._param_jacobian = derivs.param_jacobian(self._trajectory, param)

    def trajectory(self) -> Array:
        """The trajectory array the Jacobians were evaluated on. Shape:
        ``(nstates, ntimes)``. Not to be modified."""
        return self._trajectory

    def state_cotangent(self, index: int) -> Array:
        r""":math:`\partial Q/\partial y_{\mathrm{index}}`. Shape:
        ``(nstates, 1)``; ``-1`` is the last state."""
        # Normalize so the slice of the last state is not empty.
        column = index % self._state_jacobian.shape[1]
        return self._state_jacobian[:, column : column + 1]

    def param_cotangent(self) -> Array:
        r"""The direct :math:`\partial Q/\partial p`. Shape:
        ``(nparams, 1)``."""
        return self._param_jacobian.T

    def step_terms(self) -> None:
        """None: a whole-trajectory functional reads the stored states
        only."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BackendSecondOrderTrajectoryFunctional(Generic[Array]):
    r"""A scalar transient functional's first and second derivatives on
    one recorded trajectory, as single-column backend arrays.

    The functional's per-step HVPs index stored states, so its second
    derivative in the states is block diagonal over them, as the
    Hessian-vector sweep requires.

    Parameters
    ----------
    functional : TransientFunctionalProtocol
        As for :class:`BackendTrajectoryFunctional`; its bundle must also
        have the four HVPs.
    store : TrajectoryStoreProtocol
        The recorded trajectory.
    param : Array
        The parameters the trajectory was recorded at. Shape:
        ``(nparams, 1)``.

    Raises
    ------
    TypeError
        As for :class:`BackendTrajectoryFunctional`, or if the bundle has
        no HVPs.
    ValueError
        As for :class:`BackendTrajectoryFunctional`.
    """

    def __init__(
        self,
        functional: TransientFunctionalProtocol[Array],
        store: TrajectoryStoreProtocol[Array],
        param: Array,
    ) -> None:
        self._first = BackendTrajectoryFunctional(functional, store, param)
        derivs = functional.derivatives()
        if (
            derivs.state_state_hvp is None
            or derivs.state_param_hvp is None
            or derivs.param_state_hvp is None
            or derivs.param_param_hvp is None
        ):
            raise TypeError(f"{type(functional).__name__}'s derivatives() has no HVPs")
        self._state_state = derivs.state_state_hvp
        self._state_param = derivs.state_param_hvp
        self._param_state = derivs.param_state_hvp
        self._param_param = derivs.param_param_hvp
        self._bkd, self._param = functional.bkd(), param

    def _column(self, x: Array) -> Array:
        """``x`` as a column: the HVPs may return 1-D arrays."""
        return self._bkd.reshape(x, (-1, 1))

    def _time_index(self, index: int) -> int:
        return index % self._first.trajectory().shape[1]

    def state_cotangent(self, index: int) -> Array:
        """As :meth:`BackendTrajectoryFunctional.state_cotangent`."""
        return self._first.state_cotangent(index)

    def param_cotangent(self) -> Array:
        """As :meth:`BackendTrajectoryFunctional.param_cotangent`."""
        return self._first.param_cotangent()

    def step_terms(self) -> None:
        """None: a whole-trajectory functional reads the stored states
        only."""
        return None

    def state_state(self, index: int, w: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial y_m^2)\, w`. Shape:
        ``(nstates, 1)``."""
        trajectory, time = self._first.trajectory(), self._time_index(index)
        return self._column(self._state_state(trajectory, self._param, time, w))

    def state_param(self, index: int, v: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial y_m \partial p)\, v`. Shape:
        ``(nstates, 1)``."""
        trajectory, time = self._first.trajectory(), self._time_index(index)
        return self._column(self._state_param(trajectory, self._param, time, v))

    def param_state(self, index: int, w: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial p\, \partial y_m)\, w`. Shape:
        ``(nparams, 1)``."""
        trajectory, time = self._first.trajectory(), self._time_index(index)
        return self._column(self._param_state(trajectory, self._param, time, w))

    def param_param(self, v: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial p^2)\, v`. Shape:
        ``(nparams, 1)``."""
        trajectory = self._first.trajectory()
        return self._column(self._param_param(trajectory, self._param, v))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
