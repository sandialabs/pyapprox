from typing import Generic, Optional, Self, Union, List, Dict, Any, cast

import numpy as np
from scipy.optimize import Bounds, minimize as scipy_minimize

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.constraints.validation import (
    validate_constraints,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
)
from pyapprox.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.interface.functions.numpy.wrappers import (
    NumpyFunctionWrapper,
    NumpyFunctionWithJacobianWrapper,
    NumpyFunctionWithJacobianAndHVPWrapper,
    NumpyFunctionWithJacobianAndWHVPWrapper,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocol,
)

# Type alias for the wrapped objective returned by numpy_function_wrapper_factory
_WrappedObjective = Union[
    NumpyFunctionWrapper[Array],
    NumpyFunctionWithJacobianWrapper[Array],
    NumpyFunctionWithJacobianAndHVPWrapper[Array],
    NumpyFunctionWithJacobianAndWHVPWrapper[Array],
]


def _convert_constraints_for_slsqp(
    constraints: SequenceOfConstraintProtocols[Array],
) -> List[Dict[str, Any]]:
    """Convert constraints to SLSQP dict format.

    SLSQP expects constraints as dicts with keys 'type', 'fun', and
    optionally 'jac'. Inequality constraints must satisfy fun(x) >= 0,
    equality constraints must satisfy fun(x) == 0.

    For a constraint lb <= f(x) <= ub:
    - If lb == ub (equality): one eq constraint f(x) - lb == 0
    - If lb is finite: ineq constraint f(x) - lb >= 0
    - If ub is finite: ineq constraint ub - f(x) >= 0
    """
    slsqp_constraints: List[Dict[str, Any]] = []

    for constraint in constraints:
        if isinstance(constraint, PyApproxLinearConstraint):
            bkd = constraint.bkd()
            A_np = bkd.to_numpy(constraint.A())
            lb_np = bkd.to_numpy(constraint.lb())
            ub_np = bkd.to_numpy(constraint.ub())

            is_eq = np.allclose(lb_np, ub_np)
            if is_eq:
                slsqp_constraints.append({
                    'type': 'eq',
                    'fun': lambda x, A=A_np, b=lb_np: A @ x - b,
                    'jac': lambda x, A=A_np: A,
                })
            else:
                if np.all(np.isfinite(lb_np)):
                    slsqp_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, A=A_np, b=lb_np: A @ x - b,
                        'jac': lambda x, A=A_np: A,
                    })
                if np.all(np.isfinite(ub_np)):
                    slsqp_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, A=A_np, b=ub_np: b - A @ x,
                        'jac': lambda x, A=A_np: -A,
                    })
        else:
            # Nonlinear constraint
            con = numpy_function_wrapper_factory(
                cast(NonlinearConstraintProtocol[Array], constraint),
            )
            con_bkd = constraint.bkd()
            lb_np = con_bkd.to_numpy(constraint.lb())
            ub_np = con_bkd.to_numpy(constraint.ub())

            has_jac = hasattr(con, "jacobian")
            is_eq = np.allclose(lb_np, ub_np)

            if is_eq:
                d: Dict[str, Any] = {
                    'type': 'eq',
                    'fun': lambda x, c=con, b=lb_np: (
                        c(x[:, None])[:, 0] - b
                    ),
                }
                if has_jac:
                    d['jac'] = lambda x, c=con: c.jacobian(x[:, None])
                slsqp_constraints.append(d)
            else:
                if np.all(np.isfinite(lb_np)):
                    d = {
                        'type': 'ineq',
                        'fun': lambda x, c=con, b=lb_np: (
                            c(x[:, None])[:, 0] - b
                        ),
                    }
                    if has_jac:
                        d['jac'] = lambda x, c=con: c.jacobian(x[:, None])
                    slsqp_constraints.append(d)
                if np.all(np.isfinite(ub_np)):
                    d = {
                        'type': 'ineq',
                        'fun': lambda x, c=con, b=ub_np: (
                            b - c(x[:, None])[:, 0]
                        ),
                    }
                    if has_jac:
                        d['jac'] = lambda x, c=con: -c.jacobian(x[:, None])
                    slsqp_constraints.append(d)

    return slsqp_constraints


class ScipySLSQPOptimizer(Generic[Array]):
    """Optimizer using SciPy's SLSQP method.

    This class wraps SciPy's Sequential Least Squares Programming (SLSQP)
    optimizer and integrates with PyApprox's function and constraint wrappers.

    SLSQP is a gradient-based optimizer that supports bounds, equality, and
    inequality constraints. Unlike trust-constr, SLSQP uses a projected
    gradient approach which can be more robust when the solution lies on
    active constraint boundaries.

    Supports two usage patterns:

    1. Direct binding (original API):
    ```python
    optimizer = ScipySLSQPOptimizer(
        objective=obj, bounds=bounds, maxiter=100
    )
    result = optimizer.minimize(init_guess)
    ```

    2. Deferred binding (new API):
    ```python
    optimizer = ScipySLSQPOptimizer(maxiter=100)
    optimizer.bind(objective, bounds)
    result = optimizer.minimize(init_guess)
    ```
    """

    def __init__(
        self,
        objective: Optional[ObjectiveProtocol[Array]] = None,
        bounds: Optional[Array] = None,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
        disp: bool = False,
        maxiter: Optional[int] = None,
        ftol: Optional[float] = None,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        objective : Optional[ObjectiveProtocol[Array]], optional
            Objective function for the optimization problem. If None, must
            call bind() before minimize(). Defaults to None.
        bounds : Optional[Array], optional
            Bounds for the optimization variables. Required if objective
            is provided. Defaults to None.
        constraints : Optional[SequenceOfConstraintProtocols[Array]], optional
            Constraints for the optimization problem. Defaults to None.
        disp : bool, optional
            Whether to display convergence messages. Defaults to False.
        maxiter : Optional[int], optional
            Maximum number of iterations. Defaults to SciPy's default (100).
        ftol : Optional[float], optional
            Precision goal for the objective function value in the stopping
            criterion. Defaults to SciPy's default.
        """
        # Store options for copy()
        self._disp = disp
        self._maxiter = maxiter
        self._ftol = ftol
        self._init_constraints = constraints

        # Build SciPy options dict
        self._opts: Dict[str, Any] = {
            "maxiter": maxiter,
            "ftol": ftol,
            "disp": disp,
        }
        # Remove None values to let SciPy use its defaults
        self._opts = {
            key: value for key, value in self._opts.items() if value is not None
        }

        # Initialize unbound state
        self._objective: Optional[_WrappedObjective[Array]] = None
        self._bounds: Optional[Bounds] = None
        self._constraints: Optional[List[Dict[str, Any]]] = None
        self._is_bound = False

        # Backward compatible: if objective/bounds provided, bind immediately
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
        """Bind objective, bounds, and constraints. Returns self for chaining.

        Parameters
        ----------
        objective : ObjectiveProtocol[Array]
            Objective function for the optimization problem.
        bounds : Array
            Bounds for the optimization variables, shape (nvars, 2).
        constraints : Optional[SequenceOfConstraintProtocols[Array]], optional
            Constraints for the optimization problem. Defaults to None.

        Returns
        -------
        Self
            Returns self to enable method chaining.
        """
        validate_objective(objective)
        self._objective = numpy_function_wrapper_factory(objective)
        self._bounds = self._convert_bounds(
            bounds, self._objective.nvars(), self._objective.bkd()
        )
        if constraints:
            validate_constraints(constraints)
            self._constraints = _convert_constraints_for_slsqp(constraints)
        else:
            self._constraints = None
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        """Return True if bound to an objective.

        Returns
        -------
        bool
            True if bind() has been called, False otherwise.
        """
        return self._is_bound

    def copy(self) -> Self:
        """Return an unbound copy with same options.

        Returns
        -------
        Self
            A new optimizer instance with the same options, unbound.
        """
        return cast(
            Self,
            ScipySLSQPOptimizer(
                objective=None,
                bounds=None,
                constraints=self._init_constraints,
                disp=self._disp,
                maxiter=self._maxiter,
                ftol=self._ftol,
            ),
        )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.

        Raises
        ------
        RuntimeError
            If the optimizer has not been bound.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        return self._objective.bkd()

    def _convert_bounds(
        self, bounds: Array, nvars: int, bkd: Backend[Array]
    ) -> Bounds:
        """Convert bounds to a SciPy-compatible Bounds object.

        Parameters
        ----------
        bounds : Array
            Bounds for the optimization variables.
        nvars : int
            Number of variables in the optimization problem.
        bkd : Backend[Array]
            Backend used for array conversions.

        Returns
        -------
        Bounds
            SciPy-compatible bounds object.
        """
        if bounds is None:
            return Bounds(
                np.full((nvars,), -np.inf),
                np.full((nvars,), np.inf),
            )
        np_bounds = bkd.to_numpy(bounds)
        return Bounds(np_bounds[:, 0], np_bounds[:, 1])

    def _objective_gradient_from_jacobian(self, sample: Array) -> Array:
        assert self._objective is not None
        return self._objective.jacobian(sample[:, None])[0]  # type: ignore[union-attr,no-any-return]

    def minimize(self, init_guess: Array) -> ScipyOptimizerResultWrapper[Array]:
        """Perform the optimization.

        Parameters
        ----------
        init_guess : Array
            Initial guess for the optimization variables.

        Returns
        -------
        ScipyOptimizerResultWrapper
            Wrapped optimization result.

        Raises
        ------
        RuntimeError
            If the optimizer has not been bound.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None

        jac = (
            self._objective_gradient_from_jacobian
            if hasattr(self._objective, "jacobian")
            else None
        )

        scipy_result = scipy_minimize(
            lambda x: self._objective(x[:, None])[:, 0],
            self.bkd().to_numpy(init_guess[:, 0]),
            method="SLSQP",
            jac=jac,
            bounds=self._bounds,
            constraints=self._constraints if self._constraints else (),
            options=self._opts,
        )
        # Wrap the SciPy result
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())
