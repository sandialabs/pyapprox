"""
Minimax local OED solver.

Uses slack variable formulation to solve min_w max_i f_i(w) problems,
suitable for G-optimal designs.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.minimax import MinimaxOptimizer
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.expdesign.local.protocols.criterion import (
    LocalOEDCriterionProtocol,
)

from .base import LocalOEDSolverBase


class MinimaxLocalOEDSolver(LocalOEDSolverBase[Array], Generic[Array]):
    """
    Local OED solver for minimax problems (G-optimal designs).

    Transforms the problem min_w max_i f_i(w) into a constrained problem
    using a slack variable formulation:
        min_{t, w} t
        s.t. t >= f_i(w) for all i
             sum(w) = 1, w >= 0

    Parameters
    ----------
    criterion : LocalOEDCriterionProtocol[Array]
        Vector criterion returning multiple objectives (e.g., G-optimal).
        Should have nqoi() > 1 (number of prediction points).
    bkd : Backend[Array]
        Computational backend.
    verbosity : int
        Verbosity level for optimizer output.
    maxiter : int, optional
        Maximum number of iterations.
    gtol : float, optional
        Gradient tolerance for convergence.

    Examples
    --------
    >>> criterion = GOptimalCriterion(design_matrices, pred_factors, bkd)
    >>> solver = MinimaxLocalOEDSolver(criterion, bkd)
    >>> optimal_weights = solver.construct()
    """

    def __init__(
        self,
        criterion: LocalOEDCriterionProtocol[Array],
        bkd: Backend[Array],
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        gtol: Optional[float] = None,
    ) -> None:
        super().__init__(criterion, bkd)
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol

    def _create_extended_simplex_constraint(
        self, n_slack: int = 1
    ) -> PyApproxLinearConstraint[Array]:
        """
        Create simplex constraint extended for slack variables.

        The minimax optimizer prepends slack variable(s) to the optimization
        vector, so the constraint matrix needs leading zero column(s).

        Parameters
        ----------
        n_slack : int
            Number of slack variables prepended to the vector.

        Returns
        -------
        PyApproxLinearConstraint[Array]
            Extended simplex constraint: [0...0, 1...1] @ [t, w] = 1
        """
        nvars = self.nvars()
        zeros = self._bkd.zeros((1, n_slack))
        ones = self._bkd.ones((1, nvars))
        A = self._bkd.hstack([zeros, ones])
        lb = self._bkd.ones((1,))
        ub = self._bkd.ones((1,))
        return PyApproxLinearConstraint(A, lb, ub, self._bkd)

    def construct(self, init_weights: Optional[Array] = None) -> Array:
        """
        Find optimal design weights using minimax optimization.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nvars, 1)
            If None, uses uniform weights.

        Returns
        -------
        Array
            Optimal design weights. Shape: (nvars, 1)
        """
        if init_weights is None:
            init_weights = self._default_init_weights()

        bounds = self._create_bounds()
        # MinimaxOptimizer adds 1 slack variable (t), so extend the constraint
        simplex_constraint = self._create_extended_simplex_constraint(n_slack=1)

        # Use MinimaxOptimizer with simplex constraint
        optimizer = MinimaxOptimizer(
            model=self._criterion,
            bounds=bounds,
            constraints=[simplex_constraint],
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        result = optimizer.minimize(init_weights)
        self._result = result
        self._optimizer = optimizer

        # Extract original weights (without slack variable)
        return optimizer.extract_original_variables(result.optima())

    def get_minimax_value(self) -> Array:
        """
        Get the minimax value (maximum prediction variance).

        Returns
        -------
        Array
            Minimax value t. Shape: (1, 1)

        Raises
        ------
        AttributeError
            If construct() has not been called yet.
        """
        if not hasattr(self, "_result"):
            raise AttributeError(
                "No result available. Call construct() first."
            )
        return self._optimizer.get_minimax_value(self._result.optima())

    def get_result(self) -> ScipyOptimizerResultWrapper[Array]:
        """
        Get the full optimization result.

        Returns
        -------
        ScipyOptimizerResultWrapper
            Full optimization result including convergence info.

        Raises
        ------
        AttributeError
            If construct() has not been called yet.
        """
        if not hasattr(self, "_result"):
            raise AttributeError(
                "No result available. Call construct() first."
            )
        return self._result
