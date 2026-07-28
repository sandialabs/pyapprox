"""
Relaxed (continuous) solver for OED problems.

The relaxed solver treats design weights as continuous variables in [0, 1]
with a sum-to-one constraint, using trust-region constrained optimization.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Tuple

from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.expdesign.protocols.objective import OEDObjectiveProtocol
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.backends.protocols import Array, Backend


@dataclass
class RelaxedOEDConfig:
    """Configuration for relaxed OED solver.

    Parameters
    ----------
    verbosity : int
        Optimizer verbosity level. Default 0.
    maxiter : int, optional
        Maximum optimizer iterations. None uses SciPy default.
    gtol : float, optional
        Gradient tolerance. None uses SciPy default.
    xtol : float, optional
        Step tolerance. None uses SciPy default.
    weight_floor : float
        Lower bound on every design weight. MC OED objectives divide
        by the weights (effective noise variance ``sigma^2 / w_i``),
        so the objective is singular at ``w_i = 0``; a positive floor
        enforces iterates stay in the domain of definition instead of
        relying on the optimizer keeping strictly interior iterates.
        The optimum shifts by at most ``nobs * weight_floor`` in
        probability mass (value change well below MC noise at the
        default). Set to 0.0 to recover the closed simplex.
    """

    verbosity: int = 0
    maxiter: Optional[int] = None
    gtol: Optional[float] = None
    xtol: Optional[float] = None
    weight_floor: float = 1e-6


class RelaxedOEDSolver(Generic[Array]):
    """Relaxed (continuous) solver for any OED objective.

    Solves the continuous relaxation of the OED problem:
        min objective(w)
        s.t. sum(w) = 1
             weight_floor <= w_i <= 1

    Parameters
    ----------
    objective : OEDObjectiveProtocol[Array]
        Any OED objective function satisfying the protocol.
    config : RelaxedOEDConfig, optional
        Solver configuration for the default optimizer. Uses defaults
        if None. Ignored when ``optimizer`` is provided.
    optimizer : BindableOptimizerProtocol, optional
        Configured unbound optimizer (e.g. ``ScipyTrustConstrOptimizer``).
        Cloned during ``solve()`` to avoid shared state. If None, a
        ``ScipyTrustConstrOptimizer`` is built from ``config``.
    """

    def __init__(
        self,
        objective: OEDObjectiveProtocol[Array],
        config: Optional[RelaxedOEDConfig] = None,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        self._objective = objective
        self._bkd = objective.bkd()
        self._config = config or RelaxedOEDConfig()
        self._optimizer = optimizer
        self._nobs = objective.nvars()

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def _create_bounds(self) -> Array:
        """Create bounds array for design weights.

        Returns
        -------
        Array
            Bounds array. Shape: (nobs, 2)
            Each row is [lower, upper] = [weight_floor, 1]
        """
        bounds = self._bkd.zeros((self._nobs, 2))
        bounds[:, 0] = self._config.weight_floor
        bounds[:, 1] = 1.0  # Upper bound
        return bounds

    def _create_sum_constraint(self) -> PyApproxLinearConstraint[Array]:
        """Create sum-to-one equality constraint.

        Returns
        -------
        PyApproxLinearConstraint[Array]
            Linear constraint: sum(w) = 1
        """
        # A @ w = 1, where A is row of ones
        A = self._bkd.ones((1, self._nobs))
        lb = self._bkd.asarray([1.0])
        ub = self._bkd.asarray([1.0])
        return PyApproxLinearConstraint(A, lb, ub, self._bkd)

    def solve(self, init_weights: Optional[Array] = None) -> Tuple[Array, float]:
        """Solve the relaxed OED problem.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nobs, 1)
            If None, uses uniform weights.

        Returns
        -------
        optimal_weights : Array
            Optimal design weights. Shape: (nobs, 1)
        optimal_value : float
            Objective value at optimal design.
        """
        # Default to uniform weights
        if init_weights is None:
            init_weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        # Create bounds and constraint
        bounds = self._create_bounds()
        sum_constraint = self._create_sum_constraint()

        optimizer: BindableOptimizerProtocol[Array]
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            # Explicit type application: the constructor takes no
            # Array-typed arguments, so inference cannot bind Array
            optimizer = ScipyTrustConstrOptimizer[Array](
                verbosity=self._config.verbosity,
                maxiter=self._config.maxiter,
                gtol=self._config.gtol,
                xtol=self._config.xtol,
            )
        optimizer.bind(self._objective, bounds, [sum_constraint])

        # Run optimization
        result = optimizer.minimize(init_weights)

        # Extract optimal weights (optima() returns (nvars, 1))
        optimal_weights = result.optima()

        # Compute objective value at optimal
        optimal_value = float(
            self._bkd.to_numpy(self._objective(optimal_weights))[0, 0]
        )

        return optimal_weights, optimal_value


class RelaxedKLOEDSolver(RelaxedOEDSolver[Array]):
    """Relaxed (continuous) solver for KL-OED.

    Specializes RelaxedOEDSolver for KL-OED objectives, adding
    expected_information_gain to the return values.

    Solves the continuous relaxation of the OED problem:
        min -EIG(w)
        s.t. sum(w) = 1
             weight_floor <= w_i <= 1

    Parameters
    ----------
    objective : KLOEDObjective[Array]
        The KL-OED objective function.
    config : RelaxedOEDConfig, optional
        Solver configuration for the default optimizer. Uses defaults
        if None. Ignored when ``optimizer`` is provided.
    optimizer : BindableOptimizerProtocol, optional
        Configured unbound optimizer, cloned during ``solve()``.
    """

    def __init__(
        self,
        objective: KLOEDObjective[Array],
        config: Optional[RelaxedOEDConfig] = None,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        super().__init__(objective, config, optimizer)
        self._kl_objective = objective

    def solve(self, init_weights: Optional[Array] = None) -> Tuple[Array, float]:
        """Solve the relaxed KL-OED problem.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nobs, 1)
            If None, uses uniform weights.

        Returns
        -------
        optimal_weights : Array
            Optimal design weights. Shape: (nobs, 1)
        optimal_eig : float
            Expected information gain at optimal design.
        """
        optimal_weights, _ = super().solve(init_weights)

        # Compute EIG at optimal
        optimal_eig = self._kl_objective.expected_information_gain(optimal_weights)

        return optimal_weights, optimal_eig
