"""ROL optimizer implementing BindableOptimizerProtocol."""

from typing import Generic, Optional, Self, cast

import numpy as np

from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.constraints.validation import (
    validate_constraints,
)
from pyapprox.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.rol.rol_result import ROLOptimizerResult
from pyapprox.optimization.minimize.rol.rol_wrappers import (
    _require_pyrol,
    make_rol_linear_constraint,
    make_rol_nonlinear_constraint,
    make_rol_objective,
)
from pyapprox.util.backends.protocols import Array, Backend


class ROLOptimizer(Generic[Array]):
    """Optimizer using ROL (Rapid Optimization Library).

    Supports two usage patterns:

    1. Direct binding (constructor API):
    ```python
    optimizer = ROLOptimizer(
        objective=obj, bounds=bounds, verbosity=0
    )
    result = optimizer.minimize(init_guess)
    ```

    2. Deferred binding:
    ```python
    optimizer = ROLOptimizer(verbosity=0)
    optimizer.bind(objective, bounds)
    result = optimizer.minimize(init_guess)
    ```

    Requires the optional ``rol-python`` package. Install with::

        pip install rol-python
    """

    def __init__(
        self,
        objective: Optional[ObjectiveProtocol[Array]] = None,
        bounds: Optional[Array] = None,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
        verbosity: int = 0,
        parameters: Optional[object] = None,
        status_test: Optional[object] = None,
    ) -> None:
        _require_pyrol()
        self._verbosity = verbosity
        self._parameters = parameters
        self._status_test = status_test
        self._init_constraints = constraints

        self._objective: Optional[ObjectiveProtocol[Array]] = None
        self._bounds: Optional[Array] = None
        self._constraints: Optional[SequenceOfConstraintProtocols[Array]] = (
            None
        )
        self._is_bound = False

        if objective is not None:
            if bounds is None:
                raise ValueError(
                    "bounds must be provided when objective is provided"
                )
            self.bind(objective, bounds, constraints)

    def bind(
        self,
        objective: ObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
    ) -> Self:
        """Bind objective, bounds, and constraints. Returns self for chaining."""
        validate_objective(objective)
        self._objective = objective
        self._bounds = bounds
        if constraints:
            validate_constraints(constraints)
            self._constraints = list(constraints)
        else:
            self._constraints = None
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        return self._is_bound

    def copy(self) -> Self:
        return cast(
            Self,
            ROLOptimizer(
                objective=None,
                bounds=None,
                constraints=self._init_constraints,
                verbosity=self._verbosity,
                parameters=self._parameters,
                status_test=self._status_test,
            ),
        )

    def bkd(self) -> Backend[Array]:
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        return self._objective.bkd()

    def default_parameters(self) -> object:
        """Return a pyrol.ParameterList with sensible trust-region defaults."""
        import pyrol

        params = pyrol.ParameterList()
        params["General"] = pyrol.ParameterList()
        params["General"]["Output Level"] = 1
        params["Step"] = pyrol.ParameterList()
        params["Step"]["Trust Region"] = pyrol.ParameterList()
        params["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
        return params

    @staticmethod
    def inexact_gradient_parameters(
        tol_scaling: float = 0.1,
    ) -> object:
        """Return a pyrol.ParameterList configured for inexact gradients.

        Enables ROL's trust-region inexact gradient mode per Kouri et al.
        (2013). ROL will pass an adaptive tolerance ``tol`` to
        ``gradient(g, x, tol)`` and ``value(x, tol)`` callbacks,
        starting large and tightening as convergence progresses.

        Parameters
        ----------
        tol_scaling : float, optional
            Tolerance scaling factor (κ₁ in Kouri et al.).
            Controls how aggressively ROL tightens the tolerance.
            Default ``0.1``.
        """
        import pyrol

        params = pyrol.ParameterList()
        params["General"] = pyrol.ParameterList()
        params["General"]["Output Level"] = 1
        params["General"]["Inexact Gradient"] = True
        params["General"]["Inexact Objective Function"] = True
        params["Step"] = pyrol.ParameterList()
        params["Step"]["Trust Region"] = pyrol.ParameterList()
        params["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
        params["Step"]["Trust Region"]["Inexact"] = pyrol.ParameterList()
        params["Step"]["Trust Region"]["Inexact"]["Gradient"] = (
            pyrol.ParameterList()
        )
        params["Step"]["Trust Region"]["Inexact"]["Gradient"][
            "Tolerance Scaling"
        ] = tol_scaling
        return params

    def minimize(self, init_guess: Array) -> ROLOptimizerResult[Array]:
        """Run optimization starting from the initial guess.

        Parameters
        ----------
        init_guess : Array
            Initial guess, shape (nvars, 1).

        Returns
        -------
        ROLOptimizerResult[Array]
            Optimization result.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None

        import pyrol
        from pyrol.vectors import NumPyVector

        bkd = self.bkd()

        # Build initial point
        x0_np = bkd.to_numpy(init_guess[:, 0]).copy()
        x0 = NumPyVector(x0_np)

        # Wrap objective
        rol_objective = make_rol_objective(self._objective, bkd)

        # Create problem
        problem = pyrol.Problem(rol_objective, x0, x0.dual())

        # Add bound constraints
        np_bounds = bkd.to_numpy(self._bounds)
        has_finite_bounds = np.any(np_bounds[:, 0] != -np.inf) or np.any(
            np_bounds[:, 1] != np.inf
        )
        if has_finite_bounds:
            problem.addBoundConstraint(
                pyrol.Bounds(
                    NumPyVector(np_bounds[:, 0].copy()),
                    NumPyVector(np_bounds[:, 1].copy()),
                )
            )

        # Add constraints
        if self._constraints:
            self._add_constraints(problem, self._constraints, bkd)

        # Finalize problem
        stream = pyrol.getCout()
        problem.finalize(False, self._verbosity > 0, stream)

        # Solve
        params = (
            self._parameters
            if self._parameters is not None
            else self.default_parameters()
        )
        solver = pyrol.Solver(problem, params)
        if self._status_test is not None:
            if self._verbosity > 0:
                solver.solve(self._status_test, True, stream)
            else:
                solver.solve(self._status_test, True)
        elif self._verbosity > 0:
            solver.solve(stream)
        else:
            solver.solve()

        # Extract result
        fun_val = rol_objective.value(x0, 0.0)
        return ROLOptimizerResult(
            x_array=bkd.asarray(x0.array),
            fun_value=float(fun_val),
            success_flag=True,
            bkd=bkd,
            msg="ROL optimization completed",
        )

    def _add_constraints(
        self,
        problem: object,
        constraints: SequenceOfConstraintProtocols[Array],
        bkd: Backend[Array],
    ) -> None:
        """Add linear and nonlinear constraints to the ROL problem."""
        import pyrol
        from pyrol.vectors import NumPyVector

        neq_lin = 0
        nineq_lin = 0
        neq_nonlin = 0
        nineq_nonlin = 0

        for con in constraints:
            if isinstance(con, PyApproxLinearConstraint):
                rol_con, emul, bounds, is_eq = make_rol_linear_constraint(
                    con, bkd
                )
                if is_eq:
                    problem.addLinearConstraint(
                        f"EqLinearConstraint_{neq_lin}", rol_con, emul
                    )
                    neq_lin += 1
                else:
                    problem.addLinearConstraint(
                        f"IneqLinearConstraint_{nineq_lin}",
                        rol_con,
                        emul,
                        bounds,
                    )
                    nineq_lin += 1
            else:
                # Nonlinear constraint
                rol_con_nl = make_rol_nonlinear_constraint(con, bkd)
                nqoi = con.nqoi()  # type: ignore[union-attr]
                emul = NumPyVector(np.zeros(nqoi))
                lb = bkd.to_numpy(con.lb())
                ub = bkd.to_numpy(con.ub())
                is_eq = np.allclose(lb, ub)
                if is_eq:
                    problem.addConstraint(
                        f"EqNonLinearConstraint_{neq_nonlin}",
                        rol_con_nl,
                        emul,
                    )
                    neq_nonlin += 1
                else:
                    nl_bounds = pyrol.Bounds(
                        NumPyVector(lb.copy()), NumPyVector(ub.copy())
                    )
                    problem.addConstraint(
                        f"IneqNonLinearConstraint_{nineq_nonlin}",
                        rol_con_nl,
                        emul,
                        nl_bounds,
                    )
                    nineq_nonlin += 1
