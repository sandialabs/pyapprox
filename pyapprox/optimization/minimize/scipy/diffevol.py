from typing import Generic, Literal, Optional, Self, Tuple, Union, cast

import numpy as np
from scipy.optimize import Bounds, differential_evolution

from pyapprox.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.interface.functions.numpy.wrappers import (
    NumpyFunctionWithJacobianAndHVPWrapper,
    NumpyFunctionWithJacobianAndWHVPWrapper,
    NumpyFunctionWithJacobianWrapper,
    NumpyFunctionWrapper,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
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
from pyapprox.optimization.minimize.scipy.scipy_constraint_factory import (
    convert_constraints,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend

StrategyType = Literal[
    "best1bin",
    "best1exp",
    "rand1exp",
    "randtobest1exp",
    "currenttobest1exp",
    "best2exp",
    "rand2exp",
    "randtobest1bin",
    "currenttobest1bin",
    "best2bin",
    "rand2bin",
    "rand1bin",
]

# Type alias for the wrapped objective returned by numpy_function_wrapper_factory
_WrappedObjective = Union[
    NumpyFunctionWrapper[Array],
    NumpyFunctionWithJacobianWrapper[Array],
    NumpyFunctionWithJacobianAndHVPWrapper[Array],
    NumpyFunctionWithJacobianAndWHVPWrapper[Array],
]


class ScipyDifferentialEvolutionOptimizer(Generic[Array]):
    """Optimizer using SciPy's differential evolution method.

    This class wraps SciPy's differential evolution optimizer and integrates
    with PyApprox's function wrappers.

    Supports two usage patterns:

    1. Direct binding (original API):
    ```python
    optimizer = ScipyDifferentialEvolutionOptimizer(
        objective=obj, bounds=bounds, maxiter=100
    )
    result = optimizer.minimize(init_guess)
    ```

    2. Deferred binding (new API):
    ```python
    optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=100)
    optimizer.bind(objective, bounds)
    result = optimizer.minimize(init_guess)
    ```
    """

    def __init__(
        self,
        objective: Optional[FunctionProtocol[Array]] = None,
        bounds: Optional[Array] = None,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
        strategy: StrategyType = "best1bin",
        maxiter: int = 1000,
        popsize: int = 15,
        tol: float = 0.01,
        mutation: Union[float, Tuple[float, float]] = (0.5, 1),
        recombination: float = 0.7,
        seed: Optional[int] = None,
        disp: bool = False,
        polish: bool = False,
        raise_on_failure: bool = True,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        objective : Optional[FunctionProtocol[Array]], optional
            Objective function for the optimization problem. If None, must
            call bind() before minimize(). Defaults to None.
        bounds : Optional[Array], optional
            Bounds for the optimization variables. Required if objective
            is provided. Defaults to None.
        constraints : Optional[SequenceOfConstraintProtocols[Array]], optional
            Constraints for the optimization problem. Defaults to None.
        strategy : str, optional
            The differential evolution strategy to use. Defaults to "best1bin".
        maxiter : int, optional
            Maximum number of iterations. Defaults to 1000.
        popsize : int, optional
            Population size multiplier. Defaults to 15.
        tol : float, optional
            Tolerance for convergence. Defaults to 0.01.
        mutation : Union[float, tuple], optional
            Mutation constant or tuple of mutation constants. Defaults to (0.5, 1).
        recombination : float, optional
            Recombination constant. Defaults to 0.7.
        seed : Optional[int], optional
            Seed for random number generator. Defaults to None.
        disp : bool, optional
            Whether to display convergence messages. Defaults to False.
        polish : bool, optional
            Whether to use L-BFGS-B polishing on the best result.
            Defaults to False.
        raise_on_failure : bool, optional
            Whether to raise an error if optimization fails to converge.
            If False, returns the best result found even without convergence.
            Useful when used as a global optimizer before local refinement.
            Defaults to True.
        """
        # Store options for copy()
        self._strategy = strategy
        self._maxiter = maxiter
        self._popsize = popsize
        self._tol = tol
        self._mutation = mutation
        self._recombination = recombination
        self._seed = seed
        self._disp = disp
        self._polish = polish
        self._raise_on_failure = raise_on_failure
        self._init_constraints = constraints

        # Initialize unbound state
        self._objective: Optional[_WrappedObjective[Array]] = None
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
        objective: FunctionProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
    ) -> Self:
        """Bind objective, bounds, and constraints. Returns self for chaining.

        Parameters
        ----------
        objective : FunctionProtocol[Array]
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
        # Use objective's backend directly since we're not fully bound yet
        self._bounds = self._convert_bounds(
            bounds, self._objective.nvars(), self._objective.bkd()
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
        return cast(
            Self,
            ScipyDifferentialEvolutionOptimizer(
                objective=None,
                bounds=None,
                constraints=self._init_constraints,
                strategy=self._strategy,
                maxiter=self._maxiter,
                popsize=self._popsize,
                tol=self._tol,
                mutation=self._mutation,
                recombination=self._recombination,
                seed=self._seed,
                disp=self._disp,
                polish=self._polish,
                raise_on_failure=self._raise_on_failure,
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

    def _convert_bounds(self, bounds: Array, nvars: int, bkd: Backend[Array]) -> Bounds:
        """Convert bounds to a SciPy-compatible format.

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
            SciPy-compatible Bounds object.
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
            If the optimizer has not been bound, or if optimization fails
            and raise_on_failure is True.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None

        scipy_result = differential_evolution(
            func=lambda x: self._objective(x[:, None])[:, 0],
            bounds=self._bounds,
            strategy=self._strategy,
            maxiter=self._maxiter,
            popsize=self._popsize,
            tol=self._tol,
            mutation=self._mutation,
            recombination=self._recombination,
            seed=self._seed,
            disp=self._disp,
            polish=self._polish,
            x0=init_guess[:, 0],
            constraints=self._constraints if self._constraints is not None else (),
        )

        # Check for failure if raise_on_failure is True
        if self._raise_on_failure and not scipy_result.success:
            raise RuntimeError(
                f"Differential evolution optimization failed: {scipy_result.message}"
            )

        # Wrap the SciPy result
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())
