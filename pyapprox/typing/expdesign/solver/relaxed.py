"""
Relaxed (continuous) solver for KL-OED.

The relaxed solver treats design weights as continuous variables in [0, 1]
with a sum-to-one constraint, using trust-region constrained optimization.
"""

from typing import Generic, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.objective import KLOEDObjective
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)


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
    """

    verbosity: int = 0
    maxiter: Optional[int] = None
    gtol: Optional[float] = None
    xtol: Optional[float] = None


class OEDObjectiveWrapper(Generic[Array]):
    """Wrapper adapting KLOEDObjective for optimization.

    The optimizer expects:
    - __call__(x) where x is (nvars, nsamples)
    - jacobian(x) where x is (nvars, 1)

    KLOEDObjective uses design_weights of shape (nobs, 1).
    This wrapper ensures shape compatibility.
    """

    def __init__(
        self, objective: KLOEDObjective[Array], bkd: Backend[Array]
    ) -> None:
        self._objective = objective
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of variables (= nobs)."""
        return self._objective.nvars()

    def nqoi(self) -> int:
        """Number of outputs (= 1)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate objective at samples.

        Parameters
        ----------
        samples : Array
            Design weights. Shape: (nobs, nsamples)

        Returns
        -------
        Array
            Objective values. Shape: (1, nsamples)
        """
        nsamples = samples.shape[1]
        results = []
        for j in range(nsamples):
            weights = samples[:, j : j + 1]  # (nobs, 1)
            val = self._objective(weights)  # (1, 1)
            results.append(val[0, 0])
        return self._bkd.reshape(
            self._bkd.asarray(results), (1, nsamples)
        )

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nobs)
        """
        return self._objective.jacobian(sample)


class RelaxedKLOEDSolver(Generic[Array]):
    """Relaxed (continuous) solver for KL-OED.

    Solves the continuous relaxation of the OED problem:
        min -EIG(w)
        s.t. sum(w) = 1
             0 <= w_i <= 1

    Parameters
    ----------
    objective : KLOEDObjective[Array]
        The KL-OED objective function.
    config : RelaxedOEDConfig, optional
        Solver configuration. Uses defaults if None.
    """

    def __init__(
        self,
        objective: KLOEDObjective[Array],
        config: Optional[RelaxedOEDConfig] = None,
    ) -> None:
        self._objective = objective
        self._bkd = objective.bkd()
        self._config = config or RelaxedOEDConfig()
        self._nobs = objective.nobs()

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
            Each row is [lower, upper] = [0, 1]
        """
        bounds = self._bkd.zeros((self._nobs, 2))
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

    def solve(
        self, init_weights: Optional[Array] = None
    ) -> Tuple[Array, float]:
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
        optimal_eig : float
            Expected information gain at optimal design.
        """
        # Default to uniform weights
        if init_weights is None:
            init_weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        # Create wrapped objective
        wrapped_objective = OEDObjectiveWrapper(self._objective, self._bkd)

        # Create bounds and constraint
        bounds = self._create_bounds()
        sum_constraint = self._create_sum_constraint()

        # Create optimizer
        optimizer = ScipyTrustConstrOptimizer(
            wrapped_objective,
            bounds,
            constraints=[sum_constraint],
            verbosity=self._config.verbosity,
            maxiter=self._config.maxiter,
            gtol=self._config.gtol,
            xtol=self._config.xtol,
        )

        # Run optimization
        result = optimizer.minimize(init_weights)

        # Extract optimal weights (optima() returns (nvars, 1))
        optimal_weights = result.optima()

        # Compute EIG at optimal
        optimal_eig = self._objective.expected_information_gain(optimal_weights)

        return optimal_weights, optimal_eig

    def solve_multistart(
        self,
        n_starts: int = 5,
        seed: Optional[int] = None,
    ) -> Tuple[Array, float]:
        """Solve with multiple random starting points.

        Parameters
        ----------
        n_starts : int
            Number of random starting points.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        optimal_weights : Array
            Best design weights found. Shape: (nobs, 1)
        optimal_eig : float
            Best expected information gain found.
        """
        rng = np.random.default_rng(seed)

        best_weights = None
        best_eig = -np.inf

        for _ in range(n_starts):
            # Generate random initial weights on simplex
            raw = rng.exponential(size=(self._nobs,))
            init_np = raw / raw.sum()
            init_weights = self._bkd.reshape(
                self._bkd.asarray(init_np), (self._nobs, 1)
            )

            try:
                weights, eig = self.solve(init_weights)
                if eig > best_eig:
                    best_eig = eig
                    best_weights = weights
            except Exception:
                # Skip failed optimizations
                continue

        if best_weights is None:
            # Fallback to uniform if all failed
            best_weights = self._bkd.ones((self._nobs, 1)) / self._nobs
            best_eig = self._objective.expected_information_gain(best_weights)

        return best_weights, best_eig
