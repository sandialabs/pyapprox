"""
AVaR (Average Value at Risk) local OED solver.

Uses slack variable formulation to solve min_w AVaR_alpha[f_i(w)] problems,
suitable for R-optimal (risk-based) designs.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.avar import AVaROptimizer
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


class AVaRLocalOEDSolver(LocalOEDSolverBase[Array], Generic[Array]):
    """
    Local OED solver for risk-based problems (R-optimal designs).

    Minimizes the Average Value at Risk (AVaR/CVaR) of prediction variances,
    which is the expected value of the worst (1-alpha) fraction of outcomes.

    Uses slack variable formulation:
        min_{t, s, w} t + (1/(n*(1-alpha))) * sum(s_i)
        s.t. t + s_i >= f_i(w) for all i
             s_i >= 0
             sum(w) = 1, w >= 0

    Parameters
    ----------
    criterion : LocalOEDCriterionProtocol[Array]
        Vector criterion returning multiple objectives (e.g., R-optimal).
        Should have nqoi() > 1 (number of prediction points).
    alpha : float
        Risk level in [0, 1). AVaR_alpha averages the worst (1-alpha) outcomes.
        alpha=0 gives mean, alpha->1 approaches minimax.
    bkd : Backend[Array]
        Computational backend.
    verbosity : int
        Verbosity level for optimizer output.
    maxiter : int, optional
        Maximum number of iterations.
    gtol : float, optional
        Gradient tolerance for convergence.

    Notes
    -----
    AVaR (also known as CVaR or Expected Shortfall) is a coherent risk measure.
    For alpha close to 1, it approaches the minimax (G-optimal) criterion.
    For alpha = 0, it equals the mean (I-optimal-like behavior).

    Examples
    --------
    >>> criterion = ROptimalCriterion(design_matrices, pred_factors, bkd)
    >>> solver = AVaRLocalOEDSolver(criterion, alpha=0.9, bkd=bkd)
    >>> optimal_weights = solver.construct()
    """

    def __init__(
        self,
        criterion: LocalOEDCriterionProtocol[Array],
        alpha: float,
        bkd: Backend[Array],
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        gtol: Optional[float] = None,
    ) -> None:
        super().__init__(criterion, bkd)
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._alpha = alpha
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol

    def alpha(self) -> float:
        """Risk level in [0, 1)."""
        return self._alpha

    def _create_extended_simplex_constraint(
        self, n_slack: int
    ) -> PyApproxLinearConstraint[Array]:
        """
        Create simplex constraint extended for slack variables.

        The AVaR optimizer prepends slack variables to the optimization
        vector, so the constraint matrix needs leading zero columns.

        Parameters
        ----------
        n_slack : int
            Number of slack variables prepended to the vector.

        Returns
        -------
        PyApproxLinearConstraint[Array]
            Extended simplex constraint: [0...0, 1...1] @ [t, s, w] = 1
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
        Find optimal design weights using AVaR optimization.

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
        # AVaROptimizer adds 1 + nscenarios slack variables: [t, s_1, ..., s_n]
        nscenarios = self._criterion.nqoi()
        simplex_constraint = self._create_extended_simplex_constraint(
            n_slack=1 + nscenarios
        )

        # Use AVaROptimizer with simplex constraint
        optimizer = AVaROptimizer(
            model=self._criterion,
            bounds=bounds,
            alpha=self._alpha,
            constraints=[simplex_constraint],
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        result = optimizer.minimize(init_weights)
        self._result = result
        self._optimizer = optimizer

        # Extract original weights (without slack variables)
        return optimizer.extract_original_variables(result.optima())

    def get_var_value(self) -> Array:
        """
        Get the VaR (Value at Risk) estimate.

        Returns
        -------
        Array
            VaR value t. Shape: (1, 1)

        Raises
        ------
        AttributeError
            If construct() has not been called yet.
        """
        if not hasattr(self, "_result"):
            raise AttributeError(
                "No result available. Call construct() first."
            )
        return self._optimizer.get_var_value(self._result.optima())

    def get_avar_value(self) -> Array:
        """
        Get the AVaR (Average Value at Risk) value.

        This is the objective value: t + (1/(n*(1-alpha))) * sum(s).

        Returns
        -------
        Array
            AVaR value. Shape: (1, 1)

        Raises
        ------
        AttributeError
            If construct() has not been called yet.
        """
        if not hasattr(self, "_result"):
            raise AttributeError(
                "No result available. Call construct() first."
            )
        return self._optimizer.get_avar_value(self._result.optima())

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
