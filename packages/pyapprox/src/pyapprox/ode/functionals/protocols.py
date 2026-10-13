"""
Protocols for transient problem functionals.

Functionals define the quantity of interest (QoI) Q(y, p) for time-dependent
problems, along with derivatives needed for adjoint-based gradient computation.

The design follows the implicitfunction functionals pattern but adapted for
time-dependent problems where:
- State is a trajectory: (nstates, ntimes) instead of (nstates, 1)
- Quadrature weights are needed for path-integrated functionals
- State Jacobian returns (nstates, ntimes) for adjoint accumulation

Optional derivatives are not protocol members; they are listed in the
``TransientFunctionalDerivatives`` bundle returned by ``derivatives()``.
"""

from typing import Generic, Literal, Protocol, runtime_checkable

from pyapprox.ode.functionals.derivatives import (
    TransientFunctionalDerivatives,
)
from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TransientFunctionalProtocol(Protocol, Generic[Array]):
    """
    Protocol for transient functionals.

    A functional computes Q(y(t), p) from a trajectory. Its derivatives,
    which adjoint-based gradients need, are optional and come from
    ``derivatives()``.
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

    def time_dependence(self) -> Literal["transient"]:
        """Return the kind of state this functional reads.

        A trajectory of states, shape ``(nstates, ntimes)``.
        """
        ...

    def derivatives(self) -> TransientFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        ...

    def set_time_quadrature(
        self, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        """
        Accept the trajectory quadrature of the scheme that produced the
        trajectory.

        The component that owns the scheme injects it after each solve.
        A functional that integrates over time must use it, so that its
        quadrature order matches the scheme's; one that does not ignores
        it.
        """
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
