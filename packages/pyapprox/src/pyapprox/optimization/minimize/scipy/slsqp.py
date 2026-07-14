from functools import partial
from typing import Any, Dict, Generic, List, Optional, Self

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

from pyapprox.interface.functions.numpy.adapter import (
    NumpyArray,
    NumpyDerivativesAdapter,
    NumpyFn,
)
from pyapprox.interface.functions.protocols.constraint import (
    NonlinearConstraintProtocol,
)
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.constraints.validation import (
    validate_constraints,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend


def _linear_fun_minus_bound(
    matrix: NumpyArray, bound: NumpyArray, x: NumpyArray
) -> NumpyArray:
    return np.asarray(matrix @ x - bound)


def _linear_bound_minus_fun(
    matrix: NumpyArray, bound: NumpyArray, x: NumpyArray
) -> NumpyArray:
    return np.asarray(bound - matrix @ x)


def _linear_jac(matrix: NumpyArray, x: NumpyArray) -> NumpyArray:
    return matrix


def _linear_neg_jac(matrix: NumpyArray, x: NumpyArray) -> NumpyArray:
    return np.asarray(-matrix)


def _nonlinear_fun_minus_bound(
    adapter: NumpyDerivativesAdapter[Array],
    bound: NumpyArray,
    x: NumpyArray,
) -> NumpyArray:
    return np.asarray(adapter(x[:, None])[:, 0] - bound)


def _nonlinear_bound_minus_fun(
    adapter: NumpyDerivativesAdapter[Array],
    bound: NumpyArray,
    x: NumpyArray,
) -> NumpyArray:
    return np.asarray(bound - adapter(x[:, None])[:, 0])


def _nonlinear_jac(np_jac: NumpyFn, x: NumpyArray) -> NumpyArray:
    return np_jac(x[:, None])


def _nonlinear_neg_jac(np_jac: NumpyFn, x: NumpyArray) -> NumpyArray:
    return np.asarray(-np_jac(x[:, None]))


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
            matrix_np = bkd.to_numpy(constraint.A())
            lb_np = bkd.to_numpy(constraint.lb())
            ub_np = bkd.to_numpy(constraint.ub())

            is_eq = np.allclose(lb_np, ub_np)
            if is_eq:
                slsqp_constraints.append(
                    {
                        "type": "eq",
                        "fun": partial(
                            _linear_fun_minus_bound, matrix_np, lb_np
                        ),
                        "jac": partial(_linear_jac, matrix_np),
                    }
                )
            else:
                if np.all(np.isfinite(lb_np)):
                    slsqp_constraints.append(
                        {
                            "type": "ineq",
                            "fun": partial(
                                _linear_fun_minus_bound, matrix_np, lb_np
                            ),
                            "jac": partial(_linear_jac, matrix_np),
                        }
                    )
                if np.all(np.isfinite(ub_np)):
                    slsqp_constraints.append(
                        {
                            "type": "ineq",
                            "fun": partial(
                                _linear_bound_minus_fun, matrix_np, ub_np
                            ),
                            "jac": partial(_linear_neg_jac, matrix_np),
                        }
                    )
            continue

        # Nonlinear constraint: capability read from its Derivatives bundle
        if not isinstance(constraint, NonlinearConstraintProtocol):
            raise TypeError(
                "constraint must satisfy NonlinearConstraintProtocol or be "
                f"a PyApproxLinearConstraint, got {type(constraint).__name__}"
            )
        adapter = NumpyDerivativesAdapter(
            constraint, constraint.derivatives()
        )
        np_jac = adapter.jacobian()
        con_bkd = constraint.bkd()
        lb_np = con_bkd.to_numpy(constraint.lb())
        ub_np = con_bkd.to_numpy(constraint.ub())

        is_eq = np.allclose(lb_np, ub_np)
        if is_eq:
            entry: Dict[str, Any] = {
                "type": "eq",
                "fun": partial(_nonlinear_fun_minus_bound, adapter, lb_np),
            }
            if np_jac is not None:
                entry["jac"] = partial(_nonlinear_jac, np_jac)
            slsqp_constraints.append(entry)
        else:
            if np.all(np.isfinite(lb_np)):
                entry = {
                    "type": "ineq",
                    "fun": partial(
                        _nonlinear_fun_minus_bound, adapter, lb_np
                    ),
                }
                if np_jac is not None:
                    entry["jac"] = partial(_nonlinear_jac, np_jac)
                slsqp_constraints.append(entry)
            if np.all(np.isfinite(ub_np)):
                entry = {
                    "type": "ineq",
                    "fun": partial(
                        _nonlinear_bound_minus_fun, adapter, ub_np
                    ),
                }
                if np_jac is not None:
                    entry["jac"] = partial(_nonlinear_neg_jac, np_jac)
                slsqp_constraints.append(entry)

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
        self._objective: Optional[NumpyDerivativesAdapter[Array]] = None
        # derivative capability captured once at bind(); value varies,
        # attribute shape never does
        self._np_jac: Optional[NumpyFn] = None
        self._bounds: Optional[Bounds] = None
        self._constraints: Optional[List[Dict[str, Any]]] = None
        self._is_bound = False

        # Backward compatible: if objective/bounds provided, bind immediately
        if objective is not None:
            if bounds is None:
                raise ValueError("bounds must be provided when objective is provided")
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
        adapter = NumpyDerivativesAdapter(
            objective, objective.derivatives()
        )
        self._objective = adapter
        self._np_jac = adapter.jacobian()
        self._bounds = self._convert_bounds(
            bounds, adapter.nvars(), adapter.bkd()
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
        return type(self)(
            objective=None,
            bounds=None,
            constraints=self._init_constraints,
            disp=self._disp,
            maxiter=self._maxiter,
            ftol=self._ftol,
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
        objective = self._objective
        if objective is None:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        return objective.bkd()

    def _convert_bounds(self, bounds: Array, nvars: int, bkd: Backend[Array]) -> Bounds:
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
        objective = self._objective
        if objective is None or self._bounds is None:
            raise RuntimeError("Optimizer not bound. Call bind() first.")

        np_jac = self._np_jac
        jac: Optional[NumpyFn] = None
        if np_jac is not None:

            def jac(x: NumpyArray) -> NumpyArray:
                # np.asarray: numpy stubs type __getitem__ as Any
                return np.asarray(np_jac(x[:, None])[0])

        scipy_result = scipy_minimize(
            lambda x: objective(x[:, None])[:, 0],
            self.bkd().to_numpy(init_guess[:, 0]),
            method="SLSQP",
            jac=jac,
            bounds=self._bounds,
            constraints=self._constraints if self._constraints else (),
            options=self._opts,
        )
        # Wrap the SciPy result
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())
