"""Adapter classes converting PyApprox protocols to pyrol interfaces.

All classes lazily import pyrol so the module can be imported even when
pyrol is not installed.  The actual pyrol dependency is only needed at
instantiation time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    LinearConstraintProtocol,
    NonlinearConstraintProtocol,
    NonlinearConstraintProtocolWithJacobian,
    NonlinearConstraintProtocolWithJacobianAndWHVP,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    import pyrol


def _require_pyrol() -> None:
    """Raise ImportError with a helpful message if pyrol is not installed."""
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency("pyrol", feature_name="ROL optimizer", extra_name="rol")


def _is_inexact_evaluable(obj: object) -> bool:
    """Check if obj supports tolerance-dependent evaluation."""
    from pyapprox.optimization.minimize.inexact.protocols import (
        InexactEvaluable,
    )

    return isinstance(obj, InexactEvaluable)


def _is_inexact_differentiable(obj: object) -> bool:
    """Check if obj supports tolerance-dependent jacobian."""
    from pyapprox.optimization.minimize.inexact.protocols import (
        InexactDifferentiable,
    )

    return isinstance(obj, InexactDifferentiable)


# ---------------------------------------------------------------------------
# Objective wrappers
# ---------------------------------------------------------------------------


@overload
def make_rol_objective(
    objective: FunctionWithJacobianAndHVPProtocol[Array],
    bkd: Backend[Array],
) -> object: ...


@overload
def make_rol_objective(
    objective: FunctionWithJacobianProtocol[Array],
    bkd: Backend[Array],
) -> object: ...


@overload
def make_rol_objective(
    objective: FunctionProtocol[Array],
    bkd: Backend[Array],
) -> object: ...


def make_rol_objective(
    objective: FunctionProtocol[Array],
    bkd: Backend[Array],
) -> object:
    """Create a pyrol.Objective wrapping the given objective.

    Dynamically selects whether to include gradient and hessVec methods
    based on what the objective provides. When the objective satisfies
    ``InexactEvaluable`` or ``InexactDifferentiable``, the adapter
    passes ROL's ``tol`` parameter through to ``inexact_value`` or
    ``inexact_jacobian``.
    """
    if not isinstance(objective, FunctionProtocol):
        raise TypeError(
            f"objective must satisfy FunctionProtocol, "
            f"got {type(objective).__name__}"
        )
    _require_pyrol()
    import pyrol

    has_jacobian = isinstance(objective, FunctionWithJacobianProtocol)
    has_hvp = isinstance(objective, FunctionWithJacobianAndHVPProtocol)
    inexact_eval = _is_inexact_evaluable(objective)
    inexact_diff = _is_inexact_differentiable(objective)

    class _Adapter(pyrol.Objective):
        def __init__(self) -> None:
            self._objective = objective
            self._bkd = bkd
            self._inexact_eval = inexact_eval
            self._inexact_diff = inexact_diff
            super().__init__()

        def value(
            self, x: pyrol.Vector, tol: float,
        ) -> float:
            x_col = self._bkd.asarray(x.array)[:, None]
            if self._inexact_eval:
                val = self._objective.inexact_value(  # type: ignore[attr-defined]
                    x_col,
                    float(tol),
                )
            else:
                val = self._objective(x_col)
            return float(self._bkd.to_numpy(val)[0, 0])

    if has_jacobian or inexact_diff:

        def _gradient(self, g, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            if self._inexact_diff:
                jac = self._objective.inexact_jacobian(
                    x_col,
                    float(tol),
                )
            else:
                jac = self._objective.jacobian(x_col)
            g[:] = self._bkd.to_numpy(jac[0, :])
            return g

        _Adapter.gradient = _gradient

    if has_hvp:

        def _hessVec(self, hv, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            v_col = self._bkd.asarray(v.array)[:, None]
            hvp = self._objective.hvp(x_col, v_col)
            hv[:] = self._bkd.to_numpy(hvp[:, 0])

        _Adapter.hessVec = _hessVec

    return _Adapter()


# ---------------------------------------------------------------------------
# Nonlinear constraint wrappers
# ---------------------------------------------------------------------------


@overload
def make_rol_nonlinear_constraint(
    constraint: NonlinearConstraintProtocolWithJacobianAndWHVP[Array],
    bkd: Backend[Array],
) -> object: ...


@overload
def make_rol_nonlinear_constraint(
    constraint: NonlinearConstraintProtocolWithJacobian[Array],
    bkd: Backend[Array],
) -> object: ...


@overload
def make_rol_nonlinear_constraint(
    constraint: NonlinearConstraintProtocol[Array],
    bkd: Backend[Array],
) -> object: ...


def make_rol_nonlinear_constraint(
    constraint: NonlinearConstraintProtocol[Array],
    bkd: Backend[Array],
) -> object:
    """Create a pyrol.Constraint wrapping the given nonlinear constraint.

    Dynamically selects methods based on what the constraint provides.
    When the constraint satisfies ``InexactEvaluable`` or
    ``InexactDifferentiable``, the adapter passes ROL's ``tol``
    parameter through to the inexact methods.
    """
    if not isinstance(constraint, NonlinearConstraintProtocol):
        raise TypeError(
            f"constraint must satisfy NonlinearConstraintProtocol, "
            f"got {type(constraint).__name__}"
        )
    _require_pyrol()
    import pyrol

    has_jacobian = isinstance(constraint, NonlinearConstraintProtocolWithJacobian)
    has_whvp = isinstance(constraint, NonlinearConstraintProtocolWithJacobianAndWHVP)
    inexact_eval = _is_inexact_evaluable(constraint)
    inexact_diff = _is_inexact_differentiable(constraint)

    class _Adapter(pyrol.Constraint):
        def __init__(self) -> None:
            self._constraint = constraint
            self._bkd = bkd
            self._inexact_eval = inexact_eval
            self._inexact_diff = inexact_diff
            super().__init__()

        def value(
            self, c: pyrol.Vector, x: pyrol.Vector, tol: float,
        ) -> None:
            x_col = self._bkd.asarray(x.array)[:, None]
            if self._inexact_eval:
                vals = self._constraint.inexact_value(  # type: ignore[attr-defined]
                    x_col,
                    float(tol),
                )
            else:
                vals = self._constraint(x_col)
            c[:] = self._bkd.to_numpy(vals[:, 0])

    if has_jacobian or inexact_diff:

        def _applyJacobian(self, jv, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            if self._inexact_diff:
                jac = self._bkd.to_numpy(
                    self._constraint.inexact_jacobian(x_col, float(tol))
                )
            else:
                jac = self._bkd.to_numpy(self._constraint.jacobian(x_col))
            jv[:] = jac @ v[:]

        _Adapter.applyJacobian = _applyJacobian

        def _applyAdjointJacobian(self, jv, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            if self._inexact_diff:
                jac = self._bkd.to_numpy(
                    self._constraint.inexact_jacobian(x_col, float(tol))
                )
            else:
                jac = self._bkd.to_numpy(self._constraint.jacobian(x_col))
            jv[:] = jac.T @ v[:]

        _Adapter.applyAdjointJacobian = _applyAdjointJacobian

    if has_whvp:

        def _applyAdjointHessian(self, hv, u, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            v_col = self._bkd.asarray(v.array)[:, None]
            u_col = self._bkd.asarray(u.array)[:, None]
            hvp = self._constraint.whvp(x_col, v_col, u_col)
            hv[:] = self._bkd.to_numpy(hvp[:, 0])

        _Adapter.applyAdjointHessian = _applyAdjointHessian

    return _Adapter()


# ---------------------------------------------------------------------------
# Linear operator / constraint wrappers
# ---------------------------------------------------------------------------


def make_rol_linear_operator(
    A: Array,
    bkd: Backend[Array],
) -> object:
    """Create a pyrol.LinearOperator from a coefficient matrix."""
    _require_pyrol()
    import pyrol

    A_np = bkd.to_numpy(A)

    class _Adapter(pyrol.LinearOperator):
        def __init__(self) -> None:
            self._A = A_np
            super().__init__()

        def apply(
            self, hv: pyrol.Vector, v: pyrol.Vector, tol: float,
        ) -> None:
            hv[:] = self._A @ v[:]

        def applyAdjoint(
            self, hv: pyrol.Vector, v: pyrol.Vector, tol: float,
        ) -> None:
            hv[:] = self._A.T @ v[:]

    return _Adapter()


def make_rol_linear_constraint(
    constraint: LinearConstraintProtocol[Array],
    bkd: Backend[Array],
) -> tuple[Any, ...]:
    """Create ROL linear constraint components from a PyApproxLinearConstraint.

    Returns
    -------
    tuple
        (rol_linear_constraint, emul, bounds_or_None, is_equality)
    """
    if not isinstance(constraint, LinearConstraintProtocol):
        raise TypeError(
            f"constraint must satisfy LinearConstraintProtocol, "
            f"got {type(constraint).__name__}"
        )
    _require_pyrol()
    import pyrol
    from pyrol.vectors import NumPyVector

    A = constraint.A()
    lb = bkd.to_numpy(constraint.lb())
    ub = bkd.to_numpy(constraint.ub())

    linop = make_rol_linear_operator(A, bkd)
    nrows = bkd.to_numpy(A).shape[0]
    is_equality = np.allclose(lb, ub)

    if is_equality:
        b = NumPyVector(np.full(nrows, -ub))
    else:
        b = NumPyVector(np.zeros(nrows))

    rol_con = pyrol.LinearConstraint(linop, b)
    emul = NumPyVector(np.zeros(nrows))

    if is_equality:
        return rol_con, emul, None, True
    else:
        bounds = pyrol.Bounds(NumPyVector(lb), NumPyVector(ub))
        return rol_con, emul, bounds, False
