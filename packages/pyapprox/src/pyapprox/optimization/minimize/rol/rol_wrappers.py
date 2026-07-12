"""Adapter classes converting PyApprox protocols to pyrol interfaces.

All classes lazily import pyrol so the module can be imported even when
pyrol is not installed.  The actual pyrol dependency is only needed at
instantiation time.

Capability is read from each object's Derivatives bundle (via the
migration shim ``as_derivatives``): ``gradient``/``hessVec`` and
``applyJacobian``/``applyAdjointJacobian``/``applyAdjointHessian`` are
attached only when the corresponding bundle field resolves to a callable,
and pyrol reacts to their absence with its internal secant/BFGS.
Tolerance-aware evaluation comes from the bundle's ``inexact`` suite.

Note the constraint side uses ``resolved_whvp`` with the constraint's OWN
nqoi: a scalar constraint exposing only plain ``hvp`` now gets
second-order treatment in ROL via the exact w[0]*hvp lift (previously
unavailable — a deliberate improvement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyapprox.interface.functions.legacy_adapter import (
    as_derivatives,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.optimization.minimize.constraints.protocols import (
    LinearConstraintProtocol,
    NonlinearConstraintProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    import pyrol


def _require_pyrol() -> None:
    """Raise ImportError with a helpful message if pyrol is not installed."""
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency("pyrol", feature_name="ROL optimizer", extra_name="rol")


# ---------------------------------------------------------------------------
# Objective wrappers
# ---------------------------------------------------------------------------


def make_rol_objective(
    objective: FunctionProtocol[Array],
    bkd: Backend[Array],
) -> "pyrol.Objective":
    """Create a pyrol.Objective wrapping the given objective.

    Attaches ``gradient``/``hessVec`` only when the objective's
    Derivatives bundle provides them. When the bundle carries an
    ``inexact`` suite, ROL's ``tol`` parameter is passed through to its
    tolerance-aware value/jacobian.
    """
    if not isinstance(objective, FunctionProtocol):
        raise TypeError(
            f"objective must satisfy FunctionProtocol, "
            f"got {type(objective).__name__}"
        )
    _require_pyrol()
    import pyrol

    derivs = as_derivatives(objective)
    bundle_jacobian = derivs.jacobian
    bundle_hvp = derivs.resolved_hvp(objective.nqoi(), bkd)
    suite = derivs.inexact
    inexact_value = None if suite is None else suite.value
    inexact_jacobian = None if suite is None else suite.jacobian

    class _Adapter(pyrol.Objective):
        def __init__(self) -> None:
            self._bkd = bkd
            super().__init__()

        def value(
            self, x: pyrol.Vector, tol: float,
        ) -> float:
            x_col = self._bkd.asarray(x.array)[:, None]
            if inexact_value is not None:
                val = inexact_value(x_col, float(tol))
            else:
                val = objective(x_col)
            return float(self._bkd.to_numpy(val)[0, 0])

    if bundle_jacobian is not None or inexact_jacobian is not None:

        def _gradient(
            self: Any, g: pyrol.Vector, x: pyrol.Vector, tol: float
        ) -> pyrol.Vector:
            x_col = bkd.asarray(x.array)[:, None]
            if inexact_jacobian is not None:
                jac = inexact_jacobian(x_col, float(tol))
            elif bundle_jacobian is not None:
                jac = bundle_jacobian(x_col)
            else:
                raise RuntimeError("gradient attached without capability")
            g[:] = bkd.to_numpy(jac[0, :])
            return g

        _Adapter.gradient = _gradient

    if bundle_hvp is not None:
        narrowed_hvp = bundle_hvp

        def _hessVec(
            self: Any,
            hv: pyrol.Vector,
            v: pyrol.Vector,
            x: pyrol.Vector,
            tol: float,
        ) -> None:
            x_col = bkd.asarray(x.array)[:, None]
            v_col = bkd.asarray(v.array)[:, None]
            hvp = narrowed_hvp(x_col, v_col)
            hv[:] = bkd.to_numpy(hvp[:, 0])

        _Adapter.hessVec = _hessVec

    return _Adapter()


# ---------------------------------------------------------------------------
# Nonlinear constraint wrappers
# ---------------------------------------------------------------------------


def make_rol_nonlinear_constraint(
    constraint: NonlinearConstraintProtocol[Array],
    bkd: Backend[Array],
) -> "pyrol.Constraint":
    """Create a pyrol.Constraint wrapping the given nonlinear constraint.

    Attaches Jacobian/adjoint-Hessian methods only when the constraint's
    Derivatives bundle provides them (``resolved_whvp`` with the
    constraint's own nqoi). When the bundle carries an ``inexact`` suite,
    ROL's ``tol`` is passed through to its tolerance-aware methods.
    """
    if not isinstance(constraint, NonlinearConstraintProtocol):
        raise TypeError(
            f"constraint must satisfy NonlinearConstraintProtocol, "
            f"got {type(constraint).__name__}"
        )
    _require_pyrol()
    import pyrol

    derivs = as_derivatives(constraint)
    bundle_jacobian = derivs.jacobian
    bundle_whvp = derivs.resolved_whvp(constraint.nqoi())
    suite = derivs.inexact
    inexact_value = None if suite is None else suite.value
    inexact_jacobian = None if suite is None else suite.jacobian

    class _Adapter(pyrol.Constraint):
        def __init__(self) -> None:
            self._bkd = bkd
            super().__init__()

        def value(
            self, c: pyrol.Vector, x: pyrol.Vector, tol: float,
        ) -> None:
            x_col = self._bkd.asarray(x.array)[:, None]
            if inexact_value is not None:
                vals = inexact_value(x_col, float(tol))
            else:
                vals = constraint(x_col)
            c[:] = self._bkd.to_numpy(vals[:, 0])

    if bundle_jacobian is not None or inexact_jacobian is not None:

        def _numpy_jacobian(x: pyrol.Vector, tol: float) -> Any:
            x_col = bkd.asarray(x.array)[:, None]
            if inexact_jacobian is not None:
                return bkd.to_numpy(inexact_jacobian(x_col, float(tol)))
            if bundle_jacobian is not None:
                return bkd.to_numpy(bundle_jacobian(x_col))
            raise RuntimeError("jacobian attached without capability")

        def _applyJacobian(
            self: Any,
            jv: pyrol.Vector,
            v: pyrol.Vector,
            x: pyrol.Vector,
            tol: float,
        ) -> None:
            jv[:] = _numpy_jacobian(x, tol) @ v[:]

        _Adapter.applyJacobian = _applyJacobian

        def _applyAdjointJacobian(
            self: Any,
            jv: pyrol.Vector,
            v: pyrol.Vector,
            x: pyrol.Vector,
            tol: float,
        ) -> None:
            jv[:] = _numpy_jacobian(x, tol).T @ v[:]

        _Adapter.applyAdjointJacobian = _applyAdjointJacobian

    if bundle_whvp is not None:
        narrowed_whvp = bundle_whvp

        def _applyAdjointHessian(
            self: Any,
            hv: pyrol.Vector,
            u: pyrol.Vector,
            v: pyrol.Vector,
            x: pyrol.Vector,
            tol: float,
        ) -> None:
            x_col = bkd.asarray(x.array)[:, None]
            v_col = bkd.asarray(v.array)[:, None]
            u_col = bkd.asarray(u.array)[:, None]
            hvp = narrowed_whvp(x_col, v_col, u_col)
            hv[:] = bkd.to_numpy(hvp[:, 0])

        _Adapter.applyAdjointHessian = _applyAdjointHessian

    return _Adapter()


# ---------------------------------------------------------------------------
# Linear operator / constraint wrappers
# ---------------------------------------------------------------------------


def make_rol_linear_operator(
    A: Array,
    bkd: Backend[Array],
) -> "pyrol.LinearOperator":
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
