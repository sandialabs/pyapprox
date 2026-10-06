"""
Protocols for transient problem functionals.

Functionals define the quantity of interest (QoI) Q(y, p) for time-dependent
problems, along with derivatives needed for adjoint-based gradient computation.

The design follows the implicitfunction functionals pattern but adapted for
time-dependent problems where:
- State is a trajectory: (nstates, ntimes) instead of (nstates, 1)
- Quadrature weights are needed for path-integrated functionals
- State Jacobian returns (nstates, ntimes) for adjoint accumulation
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TransientFunctionalWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for transient functionals with Jacobian support.

    This protocol defines the interface for computing Q(y(t), p) and its
    derivatives for adjoint-based gradient computation.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend used for computations."""
        ...

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def nstates(self) -> int:
        """Return the number of state variables."""
        ...

    def nparams(self) -> int:
        """Return the total number of parameters."""
        ...

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        ...

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate the functional.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, 1)
        """
        ...

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy for the solution trajectory.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates, ntimes)
        """
        ...

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nqoi, nparams)
        """
        ...


@runtime_checkable
class TimeQuadratureAwareFunctionalProtocol(Protocol, Generic[Array]):
    """
    Protocol for functionals integrating over the time trajectory.

    Such functionals must use the quadrature implied by the
    time-integration scheme (each stepper reports its rule as a
    ``TrajectoryQuadratureProtocol``), or their quadrature order will
    not match the scheme's convergence order. The component that owns
    the scheme — e.g. ``GalerkinTransientForwardModel`` after each
    forward solve — injects the rule; users never construct weights.
    """

    def set_time_quadrature(
        self, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        """
        Inject the scheme-implied trajectory quadrature.

        Parameters
        ----------
        quadrature : TrajectoryQuadratureProtocol
            The rule of the stepper that produced the trajectory.
        """
        ...


@runtime_checkable
class TransientFunctionalWithStateJacobianActionProtocol(
    Protocol, Generic[Array]
):
    """
    Protocol for transient functionals whose state Jacobian is applied
    one time step at a time.

    By the chain rule, a functional of the trajectory has the parameter
    Jacobian

    .. math::

        \\frac{dQ}{dp} = \\sum_n \\frac{\\partial Q}{\\partial y_n} W_n
        + \\frac{\\partial Q}{\\partial p},
        \\qquad W_n = \\frac{dy_n}{dp},

    for any number of QoIs. A tangent-linear sweep holds :math:`W_n` at
    step :math:`n` and hands it to ``apply_state_jacobian``, so the sum
    is accumulated as the sweep runs and no :math:`W_n` is stored. The
    functional never forms :math:`\\partial Q/\\partial y_n` unless it
    chooses to; a sparse observation operator applies directly.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        ...

    def nstates(self) -> int:
        """Return the number of state variables."""
        ...

    def nparams(self) -> int:
        """Return the total number of parameters."""
        ...

    def nunique_params(self) -> int:
        """Return number of parameters unique to the functional."""
        ...

    def __call__(self, sol: Array, param: Array) -> Array:
        """Evaluate the functional. Shape: (nqoi, 1)."""
        ...

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute the direct dQ/dp. Shape: (nqoi, nparams)."""
        ...

    def apply_state_jacobian(
        self, sol: Array, param: Array, time_idx: int, wmat: Array
    ) -> Array:
        """
        Apply :math:`\\partial Q/\\partial y_n` at one time to a matrix.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index :math:`n`, in ``[0, ntimes)``.
        wmat : Array
            Matrix to apply to, typically :math:`W_n`.
            Shape: (nstates, ncols)

        Returns
        -------
        Array
            :math:`(\\partial Q/\\partial y_n)` ``wmat``; zero at times
            the functional does not depend on. Shape: (nqoi, ncols)
        """
        ...


@runtime_checkable
class TransientFunctionalWithRowsProtocol(Protocol, Generic[Array]):
    """
    Protocol for vector transient functionals that expose each QoI as a
    scalar functional.

    The adjoint method computes one Jacobian row per backward sweep and
    needs that row as a scalar functional (its ``state_jacobian`` is the
    gradient shape ``(nstates, ntimes)``).
    """

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        ...

    def row_functional(
        self, qoi_idx: int
    ) -> TransientFunctionalWithJacobianProtocol[Array]:
        """
        Return QoI ``qoi_idx`` as a scalar functional.

        Parameters
        ----------
        qoi_idx : int
            QoI index, in ``[0, nqoi)``.

        Returns
        -------
        TransientFunctionalWithJacobianProtocol
            Scalar functional (nqoi = 1) with the same parameters.
        """
        ...


@runtime_checkable
class TransientFunctionalWithJacobianAndHVPProtocol(Protocol, Generic[Array]):
    """
    Protocol for transient functionals with Jacobian and HVP support.

    Extends TransientFunctionalWithJacobianProtocol with second-order
    derivative methods for Hessian-vector products.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        ...

    def nstates(self) -> int:
        """Return the number of state variables."""
        ...

    def nparams(self) -> int:
        """Return the total number of parameters."""
        ...

    def nunique_params(self) -> int:
        """Return number of parameters unique to the functional."""
        ...

    def __call__(self, sol: Array, param: Array) -> Array:
        """Evaluate the functional."""
        ...

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute dQ/dy."""
        ...

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute dQ/dp."""
        ...

    def state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy^2)·w at a specific time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        wvec : Array
            Direction vector. Shape: (nstates, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nstates, 1)
        """
        ...

    def state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy dp)·v at a specific time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nstates, 1)
        """
        ...

    def param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dp dy)·w at a specific time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        wvec : Array
            Direction vector. Shape: (nstates, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nparams, 1)
        """
        ...

    def param_param_hvp(self, sol: Array, param: Array, vvec: Array) -> Array:
        """
        Compute (d^2Q/dp^2)·v.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nparams, 1)
        """
        ...
