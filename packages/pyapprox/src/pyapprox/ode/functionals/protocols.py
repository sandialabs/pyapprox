"""
Protocols for transient problem functionals.

Functionals define the quantity of interest (QoI) Q(y, p) for time-dependent
problems, along with derivatives needed for adjoint-based gradient computation.

The design follows the implicitfunction functionals pattern but adapted for
time-dependent problems where:
- State is a trajectory: (nstates, ntimes) instead of (nstates, 1)
- Quadrature weights are needed for path-integrated functionals
- State Jacobian returns (nstates, ntimes) for adjoint accumulation

Optional derivatives are not protocol members. They are listed in the
frozen ``TransientFunctionalDerivatives`` bundle returned by
``derivatives()``, so a consumer reads capability from data rather than
from which methods a functional has
(``docs/conventions/optional_methods.md``). Absence is ``None``; a
functional with no derivatives is evaluated but not differentiated. The
protocol and the bundle share this module because each refers to the
other: ``derivatives()`` returns the bundle, and the bundle's
``row_functional`` returns a functional.

Bundle field signatures (``sol`` the trajectory, ``(nstates, ntimes)``;
``param`` ``(nparams, 1)``; ``time_idx`` a stored-column index):

- ``state_jacobian``: ``(sol, param) -> (nstates, ntimes)`` for a scalar
  QoI — ``dQ/dy`` at every stored column.
- ``param_jacobian``: ``(sol, param) -> (nqoi, nparams)`` — the direct
  ``dQ/dp``.
- ``state_state_hvp``: ``(sol, param, time_idx, wvec (nstates, 1))
  -> (nstates, 1)`` — ``(d^2 Q/dy_n^2) w``.
- ``state_param_hvp``: ``(sol, param, time_idx, vvec (nparams, 1))
  -> (nstates, 1)`` — ``(d^2 Q/dy_n dp) v``.
- ``param_state_hvp``: ``(sol, param, time_idx, wvec (nstates, 1))
  -> (nparams, 1)`` — ``(d^2 Q/dp dy_n) w``.
- ``param_param_hvp``: ``(sol, param, vvec (nparams, 1))
  -> (nparams, 1)`` — ``(d^2 Q/dp^2) v``.
- ``apply_state_jacobian``: ``(sol, param, time_idx, wmat (nstates,
  ncols)) -> (nqoi, ncols)`` — ``(dQ/dy_n) W``, zero at times the
  functional does not depend on. By the chain rule
  ``dQ/dp = sum_n (dQ/dy_n) W_n + dQ/dp|_direct`` with
  ``W_n = dy_n/dp``, for any number of QoIs, so a tangent-linear sweep
  hands it ``W_n`` at each step and accumulates the sum without storing
  ``W_n`` or forming ``dQ/dy_n`` (a sparse observation operator applies
  directly).
- ``row_functional``: ``(qoi_idx) -> functional`` — QoI ``qoi_idx`` as a
  scalar functional, for the adjoint method's one-row sweeps.

The two Jacobians come together, and every other field requires them:
the HVPs, the action and the rows are all built on first derivatives.
The four HVPs come together or not at all: the second-order adjoint
needs every one, so a partial set is not a capability. Per-step HVPs
index stored columns, so they require a quadrature whose samples are the
columns themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

A = TypeVar("A", bound=ArrayProtocol)

TransientJacobianFn = Callable[[Array, Array], Array]
TransientStepHVPFn = Callable[[Array, Array, int, Array], Array]
TransientParamHVPFn = Callable[[Array, Array, Array], Array]
TransientStateJacobianActionFn = Callable[[Array, Array, int, Array], Array]
RowFunctionalFn = Callable[
    [int], "TransientFunctionalProtocol[Array]"
]

_HVP_FIELDS = (
    "state_state_hvp",
    "state_param_hvp",
    "param_state_hvp",
    "param_param_hvp",
)
_JACOBIAN_FIELDS = ("state_jacobian", "param_jacobian")
_FIELDS = _JACOBIAN_FIELDS + _HVP_FIELDS + (
    "apply_state_jacobian",
    "row_functional",
)


@dataclass(frozen=True)
class TransientFunctionalDerivatives(Generic[Array]):
    """Optional derivatives of a transient functional.

    Fields hold callables, typically bound methods of the functional.
    Bundles are frozen: a functional whose capability changes builds a
    new bundle.
    """

    state_jacobian: Optional[TransientJacobianFn[Array]] = None
    param_jacobian: Optional[TransientJacobianFn[Array]] = None
    state_state_hvp: Optional[TransientStepHVPFn[Array]] = None
    state_param_hvp: Optional[TransientStepHVPFn[Array]] = None
    param_state_hvp: Optional[TransientStepHVPFn[Array]] = None
    param_param_hvp: Optional[TransientParamHVPFn[Array]] = None
    apply_state_jacobian: Optional[TransientStateJacobianActionFn[Array]] = (
        None
    )
    row_functional: Optional[RowFunctionalFn[Array]] = None

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"TransientFunctionalDerivatives field '{name}' must be "
                    f"callable or None; got {type(value).__name__}"
                )
        jacobians = [
            getattr(self, name) is not None for name in _JACOBIAN_FIELDS
        ]
        if any(jacobians) and not all(jacobians):
            raise TypeError(
                "state_jacobian and param_jacobian must be given together"
            )
        if not all(jacobians) and any(
            getattr(self, name) is not None
            for name in _FIELDS
            if name not in _JACOBIAN_FIELDS
        ):
            raise TypeError(
                "the HVPs, apply_state_jacobian and row_functional require "
                "state_jacobian and param_jacobian"
            )
        present = [getattr(self, name) is not None for name in _HVP_FIELDS]
        if any(present) and not all(present):
            raise TypeError(
                "the four HVP fields must be given together or not at "
                "all; the second-order adjoint needs every one"
            )

    @staticmethod
    def none() -> "TransientFunctionalDerivatives[A]":
        """No derivatives: the functional is evaluated only."""
        return TransientFunctionalDerivatives()

    @staticmethod
    def first_order(
        state_jacobian: TransientJacobianFn[A],
        param_jacobian: TransientJacobianFn[A],
        *,
        apply_state_jacobian: Optional[
            TransientStateJacobianActionFn[A]
        ] = None,
        row_functional: Optional[RowFunctionalFn[A]] = None,
    ) -> "TransientFunctionalDerivatives[A]":
        """The Jacobians, and optionally the per-step Jacobian action and
        the per-QoI rows; no curvature."""
        return TransientFunctionalDerivatives(
            state_jacobian=state_jacobian,
            param_jacobian=param_jacobian,
            apply_state_jacobian=apply_state_jacobian,
            row_functional=row_functional,
        )

    @staticmethod
    def second_order(
        state_jacobian: TransientJacobianFn[A],
        param_jacobian: TransientJacobianFn[A],
        state_state_hvp: TransientStepHVPFn[A],
        state_param_hvp: TransientStepHVPFn[A],
        param_state_hvp: TransientStepHVPFn[A],
        param_param_hvp: TransientParamHVPFn[A],
        *,
        apply_state_jacobian: Optional[
            TransientStateJacobianActionFn[A]
        ] = None,
        row_functional: Optional[RowFunctionalFn[A]] = None,
    ) -> "TransientFunctionalDerivatives[A]":
        """The Jacobians and all four HVPs, and optionally the first-order
        extras."""
        return TransientFunctionalDerivatives(
            state_jacobian=state_jacobian,
            param_jacobian=param_jacobian,
            state_state_hvp=state_state_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            param_param_hvp=param_param_hvp,
            apply_state_jacobian=apply_state_jacobian,
            row_functional=row_functional,
        )


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
