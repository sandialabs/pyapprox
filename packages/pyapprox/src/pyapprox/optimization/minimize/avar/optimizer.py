"""
AVaR (Average Value at Risk) optimizer.

Wraps a constrained optimizer to solve min_x AVaR_alpha[f_i(x)].
"""

from typing import Generic, Optional, Sequence

from pyapprox.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocolWithJacobian,
)
from pyapprox.optimization.minimize.minimax.protocols import (
    MultiQoIObjectiveProtocol,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.backends.protocols import Array, Backend

from .constraint import AVaRConstraint
from .objective import AVaRObjective


class AVaROptimizer(Generic[Array]):
    """
    Optimizer for AVaR problems: min_x AVaR_alpha[f_i(x)].

    Transforms the AVaR problem into a constrained optimization:
        min_{t, s, x} t + (1/(n*(1-alpha))) * sum(s_i)
        s.t. s_i + t >= f_i(x) for all i
             s_i >= 0 (via bounds)
             original constraints (adjusted for slack variables)

    Parameters
    ----------
    model : MultiQoIObjectiveProtocol[Array]
        Multi-output objective function to minimize the AVaR of.
    bounds : Array
        Bounds for the original optimization variables. Shape: (nmodel_vars, 2)
    alpha : float
        Risk level in [0, 1). AVaR_alpha averages the worst (1-alpha) outcomes.
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
    The slack variable t has bounds [-inf, inf].
    The excess slack variables s_i have bounds [0, inf].
    The original bounds are applied to variables x.

    Variable ordering: [t, s_1, ..., s_n, x_1, ..., x_m]
    """

    def __init__(
        self,
        model: MultiQoIObjectiveProtocol[Array],
        bounds: Array,
        alpha: float,
        constraints: Optional[
            Sequence[NonlinearConstraintProtocolWithJacobian[Array]]
        ] = None,
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        gtol: Optional[float] = None,
    ) -> None:
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._model = model
        self._bkd = model.bkd()
        self._original_bounds = bounds
        self._alpha = alpha
        self._additional_constraints = constraints or []
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol
        self._nscenarios = model.nqoi()

        self._setup_transformed_problem()

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def alpha(self) -> float:
        """Risk level in [0, 1)."""
        return self._alpha

    def _setup_transformed_problem(self) -> None:
        """Set up the transformed optimization problem."""
        nmodel_vars = self._model.nvars()

        # Create AVaR objective
        self._objective = AVaRObjective(
            nmodel_vars, self._nscenarios, self._alpha, self._bkd
        )

        # Create AVaR constraint
        self._avar_constraint = AVaRConstraint(self._model)

        # Extend bounds to include slack variables
        # t has bounds [-inf, inf]
        t_bounds = self._bkd.reshape(
            self._bkd.asarray([-float("inf"), float("inf")]), (1, 2)
        )
        # s_i have bounds [0, inf] (non-negative excess)
        s_bounds = self._bkd.full((self._nscenarios, 2), 0.0)
        s_bounds[:, 1] = float("inf")

        self._bounds = self._bkd.vstack([t_bounds, s_bounds, self._original_bounds])

    # TODO: we should pass in bindable optimizer satisfying protocol
    # that lets us swap in different optimization methods
    def minimize(
        self, init_guess: Optional[Array] = None
    ) -> ScipyOptimizerResultWrapper[Array]:
        """
        Solve the AVaR optimization problem.

        Parameters
        ----------
        init_guess : Optional[Array]
            Initial guess for original variables x. Shape: (nmodel_vars, 1)
            If None, uses uniform initialization within bounds.

        Returns
        -------
        ScipyOptimizerResultWrapper[Array]
            Optimization result. The optima includes [t, s, x] where:
            - t is the optimal VaR estimate
            - s are the excess slack values
            - x are the optimal variables
        """
        self._model.nvars()

        if init_guess is None:
            # Initialize x at center of bounds
            lower = self._original_bounds[:, 0:1]
            upper = self._original_bounds[:, 1:2]
            init_x = (lower + upper) / 2
        else:
            init_x = init_guess

        # Evaluate model at initial x to get smart initial values
        f_vals = self._model(init_x)  # Shape: (nscenarios, 1)

        # Initialize t to max of objectives (ensures feasibility)
        init_t = self._bkd.reshape(self._bkd.max(f_vals), (1, 1))

        # Initialize s to max(0, f_i - t) for feasibility
        zeros = self._bkd.zeros_like(f_vals)
        init_s = self._bkd.maximum(f_vals - init_t, zeros)

        init_full = self._bkd.vstack([init_t, init_s, init_x])

        # Collect all constraints
        all_constraints = [self._avar_constraint] + list(self._additional_constraints)

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
            Full optimization result [t, s, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Original variables x. Shape: (nmodel_vars, 1)
        """
        return optima[1 + self._nscenarios :]

    def get_var_value(self, optima: Array) -> Array:
        """
        Get the VaR (Value at Risk) estimate from optimization result.

        Parameters
        ----------
        optima : Array
            Full optimization result [t, s, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            VaR value t. Shape: (1, 1)
        """
        return self._bkd.reshape(optima[0, 0], (1, 1))

    def get_avar_value(self, optima: Array) -> Array:
        """
        Get the AVaR value from optimization result.

        This computes the objective value: t + (1/(n*(1-alpha))) * sum(s).

        Parameters
        ----------
        optima : Array
            Full optimization result [t, s, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            AVaR value. Shape: (1, 1)
        """
        return self._objective(optima)

    def get_excess_slacks(self, optima: Array) -> Array:
        """
        Get the excess slack variables from optimization result.

        Parameters
        ----------
        optima : Array
            Full optimization result [t, s, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Excess slack values s. Shape: (nscenarios, 1)
        """
        return optima[1 : 1 + self._nscenarios]
