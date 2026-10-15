r"""Reduced residuals: the full-order residual seen from a trial space.

A reduced model replaces :math:`F(u, t, p) = 0` (or the right-hand side
of :math:`M \dot{u} = F`) by its Galerkin projection onto a trial space
:math:`u = g(t) + \phi(z)`,

.. math:: \hat{F}(z, t, p) = J(z)^T F(g(t) + \phi(z), t, p),
          \qquad J(z) = \partial \phi / \partial z.

The raw residual is projected, before its essential rows are replaced
by constraints: a trial space fitted to homogenized snapshots vanishes
on the constrained DOFs, so :math:`J^T` never reads those rows, and the
lift carries the boundary data exactly for every :math:`z`.

:class:`ReducedResidualProtocol` is what every way of evaluating
:math:`\hat{F}` provides: exactly (:class:`ProjectedResidual`, the
reference every approximation is checked against), from precomputed
affine pieces, or from a sampled subset of elements. Consumers take the
protocol, so switching between them is one argument.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.pde.models.galerkin.parameterized_residual import (
    ParameterizedResidualDerivatives,
)
from pyapprox.pde.rom.lift import LiftProtocol
from pyapprox.pde.rom.trial_space import (
    LinearTrialSpaceProtocol,
    TangentOperatorProtocol,
    TrialSpaceProtocol,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

A = TypeVar("A", bound=ArrayProtocol)

ReducedParamJacobianFn = Callable[[Array, float, Array], Array]


@dataclass(frozen=True)
class ReducedResidualDerivatives(Generic[Array]):
    """Optional derivatives of a reduced residual.

    - ``param_jacobian``: ``(codes (ncodes,), time, params (nparams,))
      -> (ncodes, nparams)``, :math:`\\partial \\hat{F}/\\partial p`.
    """

    param_jacobian: Optional[ReducedParamJacobianFn[Array]] = None

    def __post_init__(self) -> None:
        if self.param_jacobian is not None and not callable(self.param_jacobian):
            raise TypeError(
                "ReducedResidualDerivatives field 'param_jacobian' must be "
                f"callable or None; got {type(self.param_jacobian).__name__}"
            )

    @staticmethod
    def none() -> "ReducedResidualDerivatives[A]":
        """No parameter derivatives."""
        return ReducedResidualDerivatives()

    @staticmethod
    def first_order(
        param_jacobian: ReducedParamJacobianFn[A],
    ) -> "ReducedResidualDerivatives[A]":
        """The parameter Jacobian."""
        return ReducedResidualDerivatives(param_jacobian=param_jacobian)


@runtime_checkable
class ReducedResidualProtocol(Protocol, Generic[Array]):
    r""":math:`\hat{F}(z, t, p)` and its state Jacobian."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ncodes(self) -> int:
        """Return the reduced dimension."""
        ...

    def nparams(self) -> int:
        """Return the number of parameters."""
        ...

    def residual(self, codes: Array, time: float, params: Array) -> Array:
        """Return the reduced residual. Shape: (ncodes,)."""
        ...

    def state_jacobian(self, codes: Array, time: float, params: Array) -> Array:
        """Return its derivative in the codes. Shape: (ncodes, ncodes)."""
        ...

    def derivatives(self) -> ReducedResidualDerivatives[Array]:
        """Return the optional parameter derivatives."""
        ...


@runtime_checkable
class ParameterizedResidualProtocol(Protocol, Generic[Array]):
    """A raw full-order residual ``F(u, t, p)`` with its derivatives.

    Satisfied by ``GalerkinParameterizedResidual``.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def nparams(self) -> int:
        """Return the number of parameters."""
        ...

    def residual(self, state: Array, time: float, params: Array) -> Array:
        """Compute ``F(u, t, p)``. Shape: (nstates,)."""
        ...

    def state_jacobian(self, state: Array, time: float, params: Array) -> Array:
        """Compute ``dF/du``. Shape: (nstates, nstates); may be sparse."""
        ...

    def derivatives(self) -> ParameterizedResidualDerivatives[Array]:
        """Return the optional parameter derivatives."""
        ...


def _project_param_jacobian(
    tangent: TangentOperatorProtocol[Array], param_jacobian: Array
) -> Array:
    r"""Return :math:`J^T \partial F/\partial p`. Shape: (ncodes, nparams).

    The full-order parameter Jacobian arrives as a dense
    (nstates, nparams) array, the only ambient-by-parameter array the
    reduced model forms. It is contracted here, in one place, so that a
    transpose action of the parameter Jacobian, once the parameter
    derivatives offer one, replaces this function alone.
    """
    return tangent.apply_transpose(param_jacobian)


class ProjectedResidual(Generic[Array]):
    r"""The exact Galerkin projection of a raw full-order residual.

    :math:`\hat{F}(z, t, p) = J(z)^T F(g(t) + \phi(z), t, p)`, with

    .. math::
        \frac{\partial \hat{F}}{\partial z}
          = J^T \frac{\partial F}{\partial u} J
          + \frac{\partial^2 (w^T \phi)}{\partial z^2}\Big|_{w = F},
        \qquad
        \frac{\partial \hat{F}}{\partial p}
          = J^T \frac{\partial F}{\partial p}.

    The curvature term vanishes for a linear trial space. Every
    evaluation does full-order work, so this is the reference against
    which cheaper reduced residuals are checked, not an online model.

    Parameters
    ----------
    residual : ParameterizedResidualProtocol
        The raw residual of the full-order system, before row
        replacement.
    trial_space : TrialSpaceProtocol
        The decoder, fitted to homogenized snapshots.
    lift : LiftProtocol
        The field carrying the boundary data at each time.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the dimensions disagree.
    """

    def __init__(
        self,
        residual: ParameterizedResidualProtocol[Array],
        trial_space: TrialSpaceProtocol[Array],
        lift: LiftProtocol[Array],
    ) -> None:
        if not isinstance(residual, ParameterizedResidualProtocol):
            raise TypeError(
                "residual must satisfy ParameterizedResidualProtocol, got "
                f"{type(residual).__name__}"
            )
        if not isinstance(trial_space, TrialSpaceProtocol):
            raise TypeError(
                "trial_space must satisfy TrialSpaceProtocol, got "
                f"{type(trial_space).__name__}"
            )
        if not isinstance(lift, LiftProtocol):
            raise TypeError(
                f"lift must satisfy LiftProtocol, got {type(lift).__name__}"
            )
        nstates = residual.nstates()
        if trial_space.nstates() != nstates or lift.nstates() != nstates:
            raise ValueError(
                f"residual has {nstates} states, trial space "
                f"{trial_space.nstates()}, lift {lift.nstates()}"
            )
        self._residual = residual
        self._trial_space = trial_space
        self._lift = lift
        self._bkd = residual.bkd()
        # A linear decoder has zero curvature by definition, so its
        # Jacobian skips the curvature term and the extra full-order
        # residual evaluation that term would need.
        self._curved = not isinstance(trial_space, LinearTrialSpaceProtocol)
        # Capability fixed at construction, as for the full residual.
        self._full_param_jacobian = residual.derivatives().param_jacobian
        self._derivs: ReducedResidualDerivatives[Array] = (
            ReducedResidualDerivatives.none()
            if self._full_param_jacobian is None
            else ReducedResidualDerivatives.first_order(self._param_jacobian)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ncodes(self) -> int:
        """Return the reduced dimension."""
        return self._trial_space.ncodes()

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._residual.nparams()

    def trial_space(self) -> TrialSpaceProtocol[Array]:
        """Return the trial space."""
        return self._trial_space

    def lift(self) -> LiftProtocol[Array]:
        """Return the lift."""
        return self._lift

    def full_state(self, codes: Array, time: float) -> Array:
        r"""Return :math:`g(t) + \phi(z)`. Shape: (ncodes,) -> (nstates,)."""
        return self._lift.lift(time) + self._trial_space.decode(codes[:, None])[:, 0]

    def residual(self, codes: Array, time: float, params: Array) -> Array:
        """Return the reduced residual. Shape: (ncodes,)."""
        full = self._residual.residual(self.full_state(codes, time), time, params)
        return self._trial_space.tangent(codes).apply_transpose(full[:, None])[:, 0]

    def state_jacobian(self, codes: Array, time: float, params: Array) -> Array:
        """Return its derivative in the codes. Shape: (ncodes, ncodes)."""
        state = self.full_state(codes, time)
        tangent = self._trial_space.tangent(codes)
        columns = tangent.apply(self._bkd.eye(self.ncodes()))
        # The full-order Jacobian is a matrix today; once it is an
        # operator this product becomes ncodes Jacobian actions.
        jac_columns = self._residual.state_jacobian(state, time, params) @ columns
        projected = tangent.apply_transpose(self._bkd.asarray(jac_columns))
        if not self._curved:
            return projected
        full = self._residual.residual(state, time, params)
        return projected + self._trial_space.weighted_curvature(codes, full)

    def derivatives(self) -> ReducedResidualDerivatives[Array]:
        """Return the optional parameter derivatives."""
        return self._derivs

    def _param_jacobian(self, codes: Array, time: float, params: Array) -> Array:
        fn = self._full_param_jacobian
        if fn is None:
            raise RuntimeError(
                "param_jacobian is unavailable; check derivatives() first"
            )
        full = fn(self.full_state(codes, time), time, params)
        return _project_param_jacobian(self._trial_space.tangent(codes), full)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ncodes={self.ncodes()}, "
            f"nparams={self.nparams()}, "
            f"trial_space={type(self._trial_space).__name__})"
        )
