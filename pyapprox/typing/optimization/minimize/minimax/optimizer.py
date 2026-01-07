"""
Minimax optimizer.

Wraps a constrained optimizer to solve min_x max_i f_i(x).
"""

from typing import Generic, Optional, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.typing.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocolWithJacobian,
)

from .protocols import MultiQoIObjectiveProtocol
from .objective import MinimaxObjective
from .constraint import MinimaxConstraint


class MinimaxOptimizer(Generic[Array]):
    """
    Optimizer for minimax problems: min_x max_i f_i(x).

    Transforms the minimax problem into a constrained optimization:
        min_{t, x} t
        s.t. t >= f_i(x) for all i
             original constraints (adjusted for slack variable)

    Parameters
    ----------
    model : MultiQoIObjectiveProtocol[Array]
        Multi-output objective function to minimize the maximum of.
    bounds : Array
        Bounds for the original optimization variables. Shape: (nmodel_vars, 2)
    constraints : Optional[Sequence[ConstraintProtocol[Array]]]
        Additional constraints on the original variables.
    verbosity : int
        Verbosity level for the underlying optimizer.
    maxiter : Optional[int]
        Maximum number of iterations.
    gtol : Optional[float]
        Gradient tolerance.

    Notes
    -----
    The slack variable t has bounds [-inf, inf] by default.
    The original bounds are applied to variables x.
    """

    def __init__(
        self,
        model: MultiQoIObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[Sequence[NonlinearConstraintProtocolWithJacobian[Array]]] = None,
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        gtol: Optional[float] = None,
    ) -> None:
        self._model = model
        self._bkd = model.bkd()
        self._original_bounds = bounds
        self._additional_constraints = constraints or []
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol

        self._setup_transformed_problem()

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def _setup_transformed_problem(self) -> None:
        """Set up the transformed optimization problem."""
        nmodel_vars = self._model.nvars()

        # Create minimax objective: just returns t
        self._objective = MinimaxObjective(nmodel_vars, self._bkd)

        # Create minimax constraint: t >= f_i(x)
        self._minimax_constraint = MinimaxConstraint(self._model)

        # Extend bounds to include slack variable t
        # t has bounds [-inf, inf] (or could use [min_possible, inf])
        slack_bounds = self._bkd.reshape(
            self._bkd.asarray([-float("inf"), float("inf")]), (1, 2)
        )
        self._bounds = self._bkd.vstack([slack_bounds, self._original_bounds])

    def minimize(
        self, init_guess: Optional[Array] = None
    ) -> ScipyOptimizerResultWrapper[Array]:
        """
        Solve the minimax optimization problem.

        Parameters
        ----------
        init_guess : Optional[Array]
            Initial guess for original variables x. Shape: (nmodel_vars, 1)
            If None, uses uniform initialization within bounds.

        Returns
        -------
        ScipyOptimizerResultWrapper[Array]
            Optimization result. The optima includes [t, x] where t is the
            minimax value and x are the optimal variables.
        """
        nmodel_vars = self._model.nvars()

        if init_guess is None:
            # Initialize x at center of bounds
            lower = self._original_bounds[:, 0:1]
            upper = self._original_bounds[:, 1:2]
            init_x = (lower + upper) / 2
        else:
            init_x = init_guess

        # Evaluate model at initial x to get smart initial value for t
        f_vals = self._model(init_x)  # Shape: (nqoi, 1)

        # Initialize t to max of objectives (ensures feasibility)
        init_t = self._bkd.reshape(self._bkd.max(f_vals), (1, 1))

        init_full = self._bkd.vstack([init_t, init_x])

        # Collect all constraints
        all_constraints = [self._minimax_constraint] + list(
            self._additional_constraints
        )

        # Create optimizer
        optimizer = ScipyTrustConstrOptimizer(
            objective=self._objective,
            bounds=self._bounds,
            constraints=all_constraints,
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        # Run optimization
        result = optimizer.minimize(init_full)

        return result

    def extract_original_variables(self, optima: Array) -> Array:
        """
        Extract original variables from optimization result.

        Parameters
        ----------
        optima : Array
            Full optimization result [t, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Original variables x. Shape: (nmodel_vars, 1)
        """
        return optima[1:]

    def get_minimax_value(self, optima: Array) -> Array:
        """
        Get the minimax value from optimization result.

        Parameters
        ----------
        optima : Array
            Full optimization result [t, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Minimax value t. Shape: (1, 1)
        """
        return self._bkd.reshape(optima[0, 0], (1, 1))
