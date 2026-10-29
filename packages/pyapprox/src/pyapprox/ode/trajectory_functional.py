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
        trajectory = trajectory_array(store, bkd)
        # dQ/dy at every stored state, (nstates, ntimes), and the direct
        # dQ/dp, (1, nparams).
        self._state_jacobian = derivs.state_jacobian(trajectory, param)
        self._param_jacobian = derivs.param_jacobian(trajectory, param)

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
