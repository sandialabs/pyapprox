"""Parameter Jacobians of transient quantities of interest.

Two methods compute the same :math:`dQ/dp` from a solved trajectory and
are chosen by the caller, never by a rule inside the library:

- ``adjoint_jacobian`` (this module): one backward adjoint sweep per
  QoI row.
- ``forward_sensitivity_jacobian``
  (``pyapprox.ode.operator.forward_sensitivity``): one tangent-linear
  sweep with a column per parameter.

Both share the ``TransientQoIJacobianMethod`` signature, so a forward
model takes either one (or a user-written method) as an argument;
``default_qoi_jacobian_method`` states the forward models' default.
"""

from typing import Generic, Optional, Protocol

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianProtocol,
    TransientFunctionalWithRowsProtocol,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.forward_sensitivity import (
    forward_sensitivity_jacobian,
)
from pyapprox.util.backends.protocols import Array


class TransientQoIJacobianMethod(Protocol, Generic[Array]):
    """A method computing :math:`dQ/dp` from a solved trajectory."""

    def __call__(
        self,
        integrator: TimeIntegrator[Array],
        functional: TransientFunctionalWithJacobianProtocol[Array],
        fwd_sols: Array,
        times: Array,
        param: Array,
    ) -> Array:
        """
        Compute the parameter Jacobian of ``functional``.

        Parameters
        ----------
        integrator : TimeIntegrator
            Integrator that produced ``fwd_sols``.
        functional : TransientFunctionalWithJacobianProtocol
            Quantity of interest.
        fwd_sols : Array
            Forward trajectory. Shape: ``(nstates, ntimes)``.
        times : Array
            Time points. Shape: ``(ntimes,)``.
        param : Array
            Parameters. Shape: ``(nparams, 1)``.

        Returns
        -------
        Array
            :math:`dQ/dp`. Shape: ``(nqoi, nparams)``.
        """
        ...


def adjoint_jacobian(
    integrator: TimeIntegrator[Array],
    functional: TransientFunctionalWithJacobianProtocol[Array],
    fwd_sols: Array,
    times: Array,
    param: Array,
) -> Array:
    """Compute a functional's parameter Jacobian by the adjoint method.

    Each QoI row costs one backward sweep, independent of ``nparams``.
    A scalar functional is its own row; a vector functional must
    satisfy ``TransientFunctionalWithRowsProtocol``.

    The integrator's functional is reset to each row in turn.

    Parameters
    ----------
    integrator : TimeIntegrator
        Integrator that produced ``fwd_sols``.
    functional : TransientFunctionalWithJacobianProtocol
        Quantity of interest.
    fwd_sols : Array
        Forward trajectory. Shape: ``(nstates, ntimes)``.
    times : Array
        Time points. Shape: ``(ntimes,)``.
    param : Array
        Parameters. Shape: ``(nparams, 1)``.

    Returns
    -------
    Array
        :math:`dQ/dp`. Shape: ``(nqoi, nparams)``.
    """
    if functional.nqoi() == 1:
        integrator.set_functional(functional)
        return integrator.gradient(fwd_sols, times, param)
    if not isinstance(functional, TransientFunctionalWithRowsProtocol):
        raise TypeError(
            "the adjoint jacobian of a vector QoI requires a functional "
            "satisfying TransientFunctionalWithRowsProtocol, got "
            f"{type(functional).__name__}"
        )
    bkd = integrator.bkd()
    rows = []
    for qoi_idx in range(functional.nqoi()):
        integrator.set_functional(functional.row_functional(qoi_idx))
        rows.append(integrator.gradient(fwd_sols, times, param))
    return bkd.vstack(rows)


def default_qoi_jacobian_method(
    functional: TransientFunctionalWithJacobianProtocol[Array],
    jacobian_method: Optional[TransientQoIJacobianMethod[Array]],
) -> TransientQoIJacobianMethod[Array]:
    """Return the caller's method, or the default when none was given.

    The default is ``adjoint_jacobian`` for a scalar QoI and
    ``forward_sensitivity_jacobian`` for a vector QoI.

    Parameters
    ----------
    functional : TransientFunctionalWithJacobianProtocol
        Quantity of interest the method will be applied to.
    jacobian_method : TransientQoIJacobianMethod, optional
        The caller's choice; returned unchanged when given.

    Returns
    -------
    TransientQoIJacobianMethod
        The method to use.
    """
    if jacobian_method is not None:
        return jacobian_method
    if functional.nqoi() == 1:
        return adjoint_jacobian
    return forward_sensitivity_jacobian
