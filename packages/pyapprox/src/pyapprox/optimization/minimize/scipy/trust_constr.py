from typing import Any, Callable, Generic, Optional, Self

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

from pyapprox.interface.functions.legacy_adapter import (
    as_derivatives,
)
from pyapprox.interface.functions.numpy.adapter import (
    NumpyArray,
    NumpyDerivativesAdapter,
    NumpyFn,
    NumpyHVPFn,
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
from pyapprox.optimization.minimize.scipy.scipy_constraint_factory import (
    convert_constraints,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend


class ScipyTrustConstrOptimizer(Generic[Array]):
    """Optimizer using SciPy's trust-constr method.

    This class wraps SciPy's trust-constr optimizer and integrates with
    PyApprox's function and constraint wrappers.

    Supports two usage patterns:

    1. Direct binding (original API):
    ```python
    optimizer = ScipyTrustConstrOptimizer(
        objective=obj, bounds=bounds, maxiter=100
    )
    result = optimizer.minimize(init_guess)
    ```

    2. Deferred binding (new API):
    ```python
    optimizer = ScipyTrustConstrOptimizer(maxiter=100)
    optimizer.bind(objective, bounds)
    result = optimizer.minimize(init_guess)
    ```
    """

    def __init__(
        self,
        objective: Optional[ObjectiveProtocol[Array]] = None,
        bounds: Optional[Array] = None,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        gtol: Optional[float] = None,
        xtol: Optional[float] = None,
        barrier_tol: Optional[float] = None,
        callback: Optional[Callable[..., Any]] = None,
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
        verbosity : int, optional
            Verbosity level for the optimizer. Defaults to 0.
        maxiter : Optional[int], optional
            Maximum number of iterations. Defaults to SciPy's default.
        gtol : Optional[float], optional
            Gradient tolerance for termination. Defaults to SciPy's default.
        xtol : Optional[float], optional
            Step tolerance for termination. Defaults to SciPy's default.
        barrier_tol : Optional[float], optional
            Barrier tolerance for termination. Defaults to SciPy's default.
        callback : Optional[Callable], optional
            Callback function called after each iteration. For trust-constr,
            signature is ``callback(x, state) -> bool`` where returning
            True stops the optimizer. Defaults to None.
        """
        # Store options for copy()
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol
        self._xtol = xtol
        self._barrier_tol = barrier_tol
        self._callback = callback
        self._init_constraints = constraints

        # Build SciPy options dict
        self._opts = {
            "maxiter": maxiter,
            "gtol": gtol,
            "xtol": xtol,
            "barrier_tol": barrier_tol,
            "verbose": verbosity,
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
        self._np_hvp: Optional[NumpyHVPFn] = None
        self._bounds: Optional[Bounds] = None
        self._constraints: Optional[object] = None
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
            objective, as_derivatives(objective)
        )
        self._objective = adapter
        self._np_jac = adapter.jacobian()
        self._np_hvp = adapter.hvp()
        # Use objective's backend directly since we're not fully bound yet
        self._bounds = self._convert_bounds(
            bounds, adapter.nvars(), adapter.bkd()
        )
        if constraints:
            validate_constraints(constraints)
            self._constraints = convert_constraints(constraints)
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
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
            xtol=self._xtol,
            barrier_tol=self._barrier_tol,
            callback=self._callback,
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
                keep_feasible=True,
            )
        np_bounds = bkd.to_numpy(bounds)
        return Bounds(np_bounds[:, 0], np_bounds[:, 1], keep_feasible=True)

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

        np_hvp = self._np_hvp
        hessp: Optional[NumpyHVPFn] = None
        if np_hvp is not None:

            def hessp(x: NumpyArray, p: NumpyArray) -> NumpyArray:
                return np.asarray(np_hvp(x[:, None], p[:, None])[:, 0])

        scipy_result = scipy_minimize(
            lambda x: objective(x[:, None])[:, 0],
            self.bkd().to_numpy(init_guess[:, 0]),
            method="trust-constr",
            jac=jac,
            hessp=hessp,
            bounds=self._bounds,
            constraints=self._constraints,
            options=self._opts,
            callback=self._callback,
        )
        # Wrap the SciPy result
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())
