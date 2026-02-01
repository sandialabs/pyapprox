"""MLBLUE-specific optimizers.

This module provides optimizers specialized for MLBLUE sample allocation,
including the semidefinite programming (SPD) optimizer that uses cvxpy.
"""

from typing import Generic, Optional, TYPE_CHECKING

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.optional_deps import import_optional_dependency

from pyapprox.typing.statest.groupacv.optimization import (
    MLBLUEObjective,
)

if TYPE_CHECKING:
    from pyapprox.typing.statest.groupacv.mlblue import MLBLUEEstimator


class MLBLUESPDOptimizer(Generic[Array]):
    """Semidefinite programming optimizer for MLBLUE sample allocation.

    Uses cvxpy to solve the MLBLUE sample allocation problem as a
    semidefinite program (SDP). This approach is particularly effective
    for single-output problems and can find globally optimal allocations.

    Parameters
    ----------
    solver_name : str, optional
        Name of the cvxpy solver to use. Default is "CLARABEL" which is
        bundled with cvxpy and works reliably across platforms.
        Other options include "CVXOPT", "SCS", "MOSEK" (requires license).

    Raises
    ------
    ImportError
        If cvxpy is not installed.

    Notes
    -----
    This optimizer requires cvxpy as an optional dependency.
    Install with: pip install pyapprox[cvxpy] or pip install cvxpy

    The SPD formulation only works for single-output statistics (nstats=1).

    **Platform-specific notes for CVXOPT solver:**

    On macOS, using CVXOPT may cause an OpenMP library conflict when PyTorch
    and other numerical libraries are also loaded. This manifests as a crash
    with the error "Initializing libomp.dylib, but found libomp.dylib already
    initialized."

    To use CVXOPT on macOS, set the environment variable before importing:

        export KMP_DUPLICATE_LIB_OK=TRUE

    Or in Python before importing torch/numpy:

        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    Note: This workaround may affect performance or produce incorrect results
    in rare cases. The default CLARABEL solver avoids this issue entirely.

    Examples
    --------
    >>> from pyapprox.typing.statest.groupacv import MLBLUEEstimator
    >>> from pyapprox.typing.statest.groupacv.mlblue_optimizer import (
    ...     MLBLUESPDOptimizer,
    ... )
    >>> # Create estimator
    >>> est = MLBLUEEstimator(stat, costs)
    >>> # Create SPD optimizer (uses CLARABEL by default)
    >>> optimizer = MLBLUESPDOptimizer()
    >>> # Allocate samples
    >>> optimizer.set_estimator(est)
    >>> optimizer.set_budget(target_cost=100, min_nhf_samples=10)
    >>> result = optimizer.minimize()
    """

    def __init__(self, solver_name: Optional[str] = None):
        # Import cvxpy with helpful error message if not installed
        self._cvxpy = import_optional_dependency(
            "cvxpy",
            feature_name="MLBLUESPDOptimizer",
            extra_name="cvxpy",
        )
        # Auto-detect solver if not specified
        if solver_name is None:
            solver_name = self._get_default_solver()
        self._solver_name = solver_name
        self._est: Optional["MLBLUEEstimator"] = None
        self._target_cost: Optional[float] = None
        self._min_nhf_samples: Optional[int] = None
        self._min_nlf_samples: Optional[Array] = None
        self._objective: Optional[MLBLUEObjective] = None
        self._bkd: Optional[Backend[Array]] = None

    def _get_default_solver(self) -> str:
        """Get the best available SDP solver.

        Returns CLARABEL (bundled with cvxpy) as the default since it's
        reliable across platforms. CVXOPT can be used explicitly if needed.
        """
        installed = self._cvxpy.installed_solvers()
        # CLARABEL is bundled with cvxpy and handles SDP reliably
        if "CLARABEL" in installed:
            return "CLARABEL"
        # CVXOPT is good but may have platform-specific issues
        if "CVXOPT" in installed:
            return "CVXOPT"
        # SCS can also handle SDP
        if "SCS" in installed:
            return "SCS"
        # Fallback
        return "CLARABEL"

    def set_estimator(
        self,
        est: "MLBLUEEstimator",
        objective: Optional[MLBLUEObjective] = None,
    ) -> None:
        """Set the estimator and optionally a custom objective.

        Parameters
        ----------
        est : MLBLUEEstimator
            The MLBLUE estimator to optimize.

        objective : MLBLUEObjective, optional
            Custom objective function. If None, uses default MLBLUEObjective.
        """
        self._est = est
        self._bkd = est._bkd
        if objective is None:
            objective = MLBLUEObjective(est._bkd)
        self._objective = objective
        self._objective.set_estimator(est)

    def set_budget(
        self,
        target_cost: float,
        min_nhf_samples: int = 1,
        min_nlf_samples: Optional[Array] = None,
    ) -> None:
        """Set the budget constraints.

        Parameters
        ----------
        target_cost : float
            Maximum total computational cost.

        min_nhf_samples : int, optional
            Minimum number of high-fidelity samples. Default is 1.

        min_nlf_samples : Array, optional
            Minimum number of samples for each low-fidelity model.
            Shape: (nmodels - 1,). If None, no minimum is enforced.
        """
        self._target_cost = target_cost
        self._min_nhf_samples = min_nhf_samples
        self._min_nlf_samples = min_nlf_samples

    def _cvxpy_psi(self, nsps_cvxpy):
        """Construct the psi matrix as a cvxpy expression."""
        Psi = self._est._psi_blocks_flat @ nsps_cvxpy
        Psi = self._cvxpy.reshape(
            Psi,
            (self._est.nmodels(), self._est.nmodels()),
            order="F",
        )
        return Psi

    def _cvxpy_spd_constraint(self, nsps_cvxpy, t_cvxpy):
        """Construct the SPD constraint matrix."""
        Psi = self._cvxpy_psi(nsps_cvxpy)
        mat = self._cvxpy.bmat(
            [
                [Psi, self._est._asketch.T],
                [
                    self._est._asketch,
                    self._cvxpy.reshape(t_cvxpy, (1, 1), order="F"),
                ],
            ]
        )
        return mat

    def minimize(self, iterate: Optional[Array] = None) -> dict:
        """Solve the MLBLUE sample allocation problem using SDP.

        Parameters
        ----------
        iterate : Array, optional
            Initial guess (ignored for SPD solver).

        Returns
        -------
        dict
            Optimization result with keys:
            - 'x': Optimal partition sample counts, shape (nsubsets, 1)
            - 'fun': Optimal objective value
            - 'success': Whether optimization succeeded

        Raises
        ------
        RuntimeError
            If the estimator has not been set, budget not set,
            statistics has multiple outputs, or solver fails.
        """
        if self._est is None:
            raise RuntimeError(
                "Must call set_estimator() before minimize()"
            )
        if self._target_cost is None:
            raise RuntimeError(
                "Must call set_budget() before minimize()"
            )
        if self._est._stat.nstats() != 1:
            raise RuntimeError(
                "SPD solver only works for single outputs (nstats=1)"
            )

        # Define cvxpy variables
        t_cvxpy = self._cvxpy.Variable(nonneg=True)
        nsps_cvxpy = self._cvxpy.Variable(self._est.nsubsets(), nonneg=True)

        # Objective: minimize t (which bounds the variance)
        obj = self._cvxpy.Minimize(t_cvxpy)

        # Get subset costs
        subset_costs = self._est._get_model_subset_costs(
            self._est._subsets, self._est._costs
        )

        # Build constraints
        constraints = [
            subset_costs @ nsps_cvxpy <= self._target_cost,
            self._est._partitions_per_model[0] @ nsps_cvxpy
            >= self._min_nhf_samples,
        ]

        # Optional low-fidelity sample constraints
        if self._min_nlf_samples is not None:
            for ii in range(self._est.nmodels() - 1):
                constraints.append(
                    self._est._partitions_per_model[ii + 1] @ nsps_cvxpy
                    >= self._min_nlf_samples[ii]
                )

        # SPD constraint: the Schur complement matrix must be positive semidefinite
        constraints.append(
            self._cvxpy_spd_constraint(nsps_cvxpy, t_cvxpy) >> 0
        )

        # Solve the problem
        prob = self._cvxpy.Problem(obj, constraints)
        prob.solve(solver=self._solver_name)

        if t_cvxpy.value is None:
            raise RuntimeError(
                f"SPD solver ({self._solver_name}) did not converge. "
                "Try a different solver or check problem formulation."
            )

        return {
            "x": self._bkd.array(nsps_cvxpy.value)[:, None],
            "fun": float(t_cvxpy.value),
            "success": True,
        }
