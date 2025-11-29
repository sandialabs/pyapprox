from typing import Generic, Union, Optional, Literal

import numpy as np
from scipy.optimize import differential_evolution, Bounds

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.typing.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.typing.optimization.minimize.constraints.validation import (
    validate_constraints,
)
from pyapprox.typing.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.typing.optimization.minimize.scipy.scipy_constraint_factory import (
    convert_constraints,
)
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)


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


class ScipyDifferentialEvolutionOptimizer(Generic[Array]):
    """
    Optimizer using SciPy's differential evolution method.

    This class wraps SciPy's differential evolution optimizer and integrates
    with PyApprox's function wrappers.
    """

    def __init__(
        self,
        objective: FunctionProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
        strategy: StrategyType = "best1bin",
        maxiter: int = 1000,
        popsize: int = 15,
        tol: float = 0.01,
        mutation: Union[float, tuple] = (0.5, 1),
        recombination: float = 0.7,
        seed: Optional[int] = None,
        disp: bool = False,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        objective : FunctionProtocol[Array]
            Objective function for the optimization problem.
        bounds : Array
            Bounds for the optimization variables.
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
        """
        # Validate and wrap the objective using the factory
        validate_objective(objective)
        self._objective = numpy_function_wrapper_factory(objective)

        # Validate and wrap the constraints using the factory if provided
        if constraints:
            validate_constraints(constraints)
            self._constraints = convert_constraints(constraints)
        else:
            self._constraints = None

        # Wrap the bounds
        self._bounds = self._convert_bounds(bounds, self._objective.nvars())

        # Store optimizer options
        self._strategy = strategy
        self._maxiter = maxiter
        self._popsize = popsize
        self._tol = tol
        self._mutation = mutation
        self._recombination = recombination
        self._seed = seed
        self._disp = disp

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        return self._objective.bkd()

    def _convert_bounds(self, bounds: Array, nvars: int) -> Bounds:
        """
        Convert bounds to a SciPy-compatible format.

        Parameters
        ----------
        bounds : Array
            Bounds for the optimization variables.
        nvars : int
            Number of variables in the optimization problem.

        Returns
        -------
        list
            List of tuples representing bounds for each variable.
        """
        if bounds is None:
            return Bounds(
                np.full((nvars,), -np.inf),
                np.full((nvars,), np.inf),
                keep_feasible=True,
            )
        np_bounds = self.bkd().to_numpy(bounds)
        return Bounds(np_bounds[:, 0], np_bounds[:, 1], keep_feasible=True)

    def minimize(self, init_guess: Array) -> ScipyOptimizerResultWrapper:
        """
        Perform the optimization.

        Returns
        -------
        ScipyOptimizerResultWrapper
            Wrapped optimization result.
        """
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
            x0=init_guess[:, 0],
        )

        # Wrap the SciPy result
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())
