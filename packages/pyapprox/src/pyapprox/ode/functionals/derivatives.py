"""Frozen bundle of a transient functional's optional derivatives.

A transient functional ``Q(y(t), p)`` always provides its value, its
state Jacobian ``(nstates, ntimes)`` and its parameter Jacobian. What it
may lack is listed here, so a consumer reads capability from data rather
than from which protocol the functional satisfies
(``docs/conventions/optional_methods.md``). Absence is ``None``.

Field signatures (``sol`` the trajectory, ``(nstates, ntimes)``;
``param`` ``(nparams, 1)``; ``time_idx`` a stored-column index):

- ``state_state_hvp``: ``(sol, param, time_idx, wvec (nstates, 1))
  -> (nstates, 1)`` — ``(d^2 Q/dy_n^2) w``.
- ``state_param_hvp``: ``(sol, param, time_idx, vvec (nparams, 1))
  -> (nstates, 1)`` — ``(d^2 Q/dy_n dp) v``.
- ``param_state_hvp``: ``(sol, param, time_idx, wvec (nstates, 1))
  -> (nparams, 1)`` — ``(d^2 Q/dp dy_n) w``.
- ``param_param_hvp``: ``(sol, param, vvec (nparams, 1))
  -> (nparams, 1)`` — ``(d^2 Q/dp^2) v``.
- ``apply_state_jacobian``: ``(sol, param, time_idx, wmat (nstates,
  ncols)) -> (nqoi, ncols)`` — ``(dQ/dy_n) W``, for tangent-linear
  sweeps that never form ``dQ/dy_n``.
- ``row_functional``: ``(qoi_idx) -> functional`` — QoI ``qoi_idx`` as a
  scalar functional, for the adjoint method's one-row sweeps.

The four HVPs come together or not at all: the second-order adjoint
needs every one, so a partial set is not a capability. Per-step HVPs
index stored columns, so they require a quadrature whose samples are the
columns themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar

from pyapprox.util.backends.protocols import Array, ArrayProtocol

if TYPE_CHECKING:
    from pyapprox.ode.functionals.protocols import (
        TransientFunctionalWithJacobianProtocol,
    )

A = TypeVar("A", bound=ArrayProtocol)

TransientStepHVPFn = Callable[[Array, Array, int, Array], Array]
TransientParamHVPFn = Callable[[Array, Array, Array], Array]
TransientStateJacobianActionFn = Callable[[Array, Array, int, Array], Array]
RowFunctionalFn = Callable[
    [int], "TransientFunctionalWithJacobianProtocol[Array]"
]

_HVP_FIELDS = (
    "state_state_hvp",
    "state_param_hvp",
    "param_state_hvp",
    "param_param_hvp",
)
_FIELDS = _HVP_FIELDS + ("apply_state_jacobian", "row_functional")


@dataclass(frozen=True)
class TransientFunctionalDerivatives(Generic[Array]):
    """Optional derivatives of a transient functional.

    Fields hold callables, typically bound methods of the functional.
    Bundles are frozen: a functional whose capability changes builds a
    new bundle.
    """

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
        present = [getattr(self, name) is not None for name in _HVP_FIELDS]
        if any(present) and not all(present):
            raise TypeError(
                "the four HVP fields must be given together or not at "
                "all; the second-order adjoint needs every one"
            )

    @staticmethod
    def none() -> "TransientFunctionalDerivatives[A]":
        """No optional derivatives."""
        return TransientFunctionalDerivatives()

    @staticmethod
    def first_order(
        *,
        apply_state_jacobian: Optional[
            TransientStateJacobianActionFn[A]
        ] = None,
        row_functional: Optional[RowFunctionalFn[A]] = None,
    ) -> "TransientFunctionalDerivatives[A]":
        """No curvature; optionally the per-step Jacobian action and the
        per-QoI rows."""
        return TransientFunctionalDerivatives(
            apply_state_jacobian=apply_state_jacobian,
            row_functional=row_functional,
        )

    @staticmethod
    def second_order(
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
        """All four HVPs, and optionally the first-order extras."""
        return TransientFunctionalDerivatives(
            state_state_hvp=state_state_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            param_param_hvp=param_param_hvp,
            apply_state_jacobian=apply_state_jacobian,
            row_functional=row_functional,
        )
