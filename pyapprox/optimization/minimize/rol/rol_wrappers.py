"""Adapter classes converting PyApprox protocols to pyrol interfaces.

All classes lazily import pyrol so the module can be imported even when
pyrol is not installed.  The actual pyrol dependency is only needed at
instantiation time.
"""

from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _require_pyrol() -> None:
    """Raise ImportError with a helpful message if pyrol is not installed."""
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency("pyrol", feature_name="ROL optimizer", extra_name="rol")


class ROLObjectiveAdapter(Generic[Array]):
    """Wraps an ObjectiveProtocol as a pyrol.Objective."""

    def __init__(self, objective: object, bkd: Backend[Array]) -> None:
        _require_pyrol()
        import pyrol

        self._objective = objective
        self._bkd = bkd
        # Dynamically inherit from pyrol.Objective
        self.__class__ = type(
            "ROLObjectiveAdapter",
            (pyrol.Objective,),
            {
                "value": self._value,
                "gradient": self._gradient,
                "hessVec": self._hessVec,
            },
        )
        pyrol.Objective.__init__(self)

    def _value(self, x, tol):  # type: ignore[no-untyped-def]
        x_col = self._bkd.asarray(x.array)[:, None]
        val = self._objective(x_col)
        return self._bkd.to_numpy(val)[0, 0]

    def _gradient(self, g, x, tol):  # type: ignore[no-untyped-def]
        x_col = self._bkd.asarray(x.array)[:, None]
        jac = self._objective.jacobian(x_col)
        g[:] = self._bkd.to_numpy(jac[0, :])
        return g

    def _hessVec(self, hv, v, x, tol):  # type: ignore[no-untyped-def]
        x_col = self._bkd.asarray(x.array)[:, None]
        v_col = self._bkd.asarray(v.array)[:, None]
        hvp = self._objective.hvp(x_col, v_col)
        hv[:] = self._bkd.to_numpy(hvp[:, 0])


def make_rol_objective(
    objective: object, bkd: Backend[Array],
) -> object:
    """Create a pyrol.Objective wrapping the given objective.

    Dynamically selects whether to include gradient and hessVec methods
    based on what the objective provides.
    """
    _require_pyrol()
    import pyrol

    has_jacobian = hasattr(objective, "jacobian")
    has_hvp = hasattr(objective, "hvp")

    class _Adapter(pyrol.Objective):
        def __init__(self) -> None:
            self._objective = objective
            self._bkd = bkd
            super().__init__()

        def value(self, x, tol):  # type: ignore[no-untyped-def, override]
            x_col = self._bkd.asarray(x.array)[:, None]
            val = self._objective(x_col)
            return self._bkd.to_numpy(val)[0, 0]

    if has_jacobian:
        def _gradient(self, g, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            jac = self._objective.jacobian(x_col)
            g[:] = self._bkd.to_numpy(jac[0, :])
            return g
        _Adapter.gradient = _gradient  # type: ignore[attr-defined]

    if has_hvp:
        def _hessVec(self, hv, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            v_col = self._bkd.asarray(v.array)[:, None]
            hvp = self._objective.hvp(x_col, v_col)
            hv[:] = self._bkd.to_numpy(hvp[:, 0])
        _Adapter.hessVec = _hessVec  # type: ignore[attr-defined]

    return _Adapter()


class ROLNonlinearConstraintAdapter(Generic[Array]):
    """Wraps a NonlinearConstraintProtocol as a pyrol.Constraint."""

    pass


def make_rol_nonlinear_constraint(
    constraint: object, bkd: Backend[Array],
) -> object:
    """Create a pyrol.Constraint wrapping the given nonlinear constraint.

    Dynamically selects methods based on what the constraint provides.
    """
    _require_pyrol()
    import pyrol

    has_jacobian = hasattr(constraint, "jacobian")
    has_whvp = hasattr(constraint, "whvp")

    class _Adapter(pyrol.Constraint):
        def __init__(self) -> None:
            self._constraint = constraint
            self._bkd = bkd
            super().__init__()

        def value(self, c, x, tol):  # type: ignore[no-untyped-def, override]
            x_col = self._bkd.asarray(x.array)[:, None]
            vals = self._constraint(x_col)
            c[:] = self._bkd.to_numpy(vals[:, 0])

    if has_jacobian:
        def _applyJacobian(self, jv, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            jac = self._bkd.to_numpy(
                self._constraint.jacobian(x_col)
            )
            jv[:] = jac @ v[:]
        _Adapter.applyJacobian = _applyJacobian  # type: ignore[attr-defined]

        def _applyAdjointJacobian(self, jv, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            jac = self._bkd.to_numpy(
                self._constraint.jacobian(x_col)
            )
            jv[:] = jac.T @ v[:]
        _Adapter.applyAdjointJacobian = _applyAdjointJacobian  # type: ignore[attr-defined]

    if has_whvp:
        def _applyAdjointHessian(self, hv, u, v, x, tol):  # type: ignore[no-untyped-def]
            x_col = self._bkd.asarray(x.array)[:, None]
            v_col = self._bkd.asarray(v.array)[:, None]
            u_col = self._bkd.asarray(u.array)[:, None]
            hvp = self._constraint.whvp(x_col, v_col, u_col)
            hv[:] = self._bkd.to_numpy(hvp[:, 0])
        _Adapter.applyAdjointHessian = _applyAdjointHessian  # type: ignore[attr-defined]

    return _Adapter()


def make_rol_linear_operator(
    A: Array, bkd: Backend[Array],
) -> object:
    """Create a pyrol.LinearOperator from a coefficient matrix."""
    _require_pyrol()
    import pyrol

    A_np = bkd.to_numpy(A)

    class _Adapter(pyrol.LinearOperator):
        def __init__(self) -> None:
            self._A = A_np
            super().__init__()

        def apply(self, hv, v, tol):  # type: ignore[no-untyped-def, override]
            hv[:] = self._A @ v[:]

        def applyAdjoint(self, hv, v, tol):  # type: ignore[no-untyped-def, override]
            hv[:] = self._A.T @ v[:]

    return _Adapter()


def make_rol_linear_constraint(
    constraint: object, bkd: Backend[Array],
) -> tuple:
    """Create ROL linear constraint components from a PyApproxLinearConstraint.

    Returns
    -------
    tuple
        (rol_linear_constraint, emul, bounds_or_None, is_equality)
    """
    _require_pyrol()
    import pyrol
    from pyrol.vectors import NumPyVector

    A = constraint.A()  # type: ignore[union-attr]
    lb = bkd.to_numpy(constraint.lb())  # type: ignore[union-attr]
    ub = bkd.to_numpy(constraint.ub())  # type: ignore[union-attr]

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
