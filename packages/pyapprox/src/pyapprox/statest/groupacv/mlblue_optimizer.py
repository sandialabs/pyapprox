"""MLBLUE-specific allocation optimizers.

This module provides allocation optimizers specialized for MLBLUE,
including the semidefinite programming (SPD) optimizer that uses cvxpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional

from pyapprox.statest.groupacv.allocation import GroupACVAllocationResult
from pyapprox.statest.groupacv.optimization import (
    MLBLUEObjective,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.optional_deps import import_optional_dependency

if TYPE_CHECKING:
    from cvxpy.expressions.expression import Expression as CvxpyExpression

    from pyapprox.statest.groupacv.mlblue import MLBLUEEstimator


class MLBLUESPDAllocationOptimizer(Generic[Array]):
    """Semidefinite programming allocation optimizer for MLBLUE.

    Uses cvxpy to solve the MLBLUE sample allocation problem as a
    semidefinite program (SDP). This approach is particularly effective
    for single-output problems and can find globally optimal allocations.

    Parameters
    ----------
    estimator : MLBLUEEstimator
        The MLBLUE estimator to optimize allocation for.
    solver_name : str, optional
        Name of the cvxpy solver to use. Default is "CLARABEL" which is
        bundled with cvxpy and works reliably across platforms.
        Other options include "CVXOPT", "SCS", "MOSEK" (requires license).
    objective : MLBLUEObjective, optional
        Custom objective function. If None, uses default MLBLUEObjective.

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
    >>> from pyapprox.statest.groupacv import MLBLUEEstimator
    >>> from pyapprox.statest.groupacv import (
    ...     MLBLUESPDAllocationOptimizer,
    ... )
    >>> # Create estimator
    >>> est = MLBLUEEstimator(stat, costs)
    >>> # Create allocator and optimize
    >>> allocator = MLBLUESPDAllocationOptimizer(est)
    >>> result = allocator.optimize(target_cost=100, min_nhf_samples=10)
    >>> est.set_allocation(result)
    """

    def __init__(
        self,
        estimator: "MLBLUEEstimator[Array]",
        solver_name: Optional[str] = None,
        objective: Optional[MLBLUEObjective[Array]] = None,
    ):
        # Import cvxpy with helpful error message if not installed
        self._cvxpy = import_optional_dependency(
            "cvxpy",
            feature_name="MLBLUESPDAllocationOptimizer",
            extra_name="cvxpy",
        )

        self._est = estimator
        self._bkd = estimator._bkd

        # Auto-detect solver if not specified
        if solver_name is None:
            solver_name = self._get_default_solver()
        self._solver_name = solver_name

        # Set up objective
        if objective is None:
            objective = MLBLUEObjective(estimator._bkd)
        self._objective = objective
        self._objective.set_estimator(estimator)

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

    def _cvxpy_psi(self, nsps_cvxpy: CvxpyExpression) -> CvxpyExpression:
        """Construct the psi matrix as a cvxpy expression."""
        Psi = self._est._psi_blocks_flat @ nsps_cvxpy
        Psi = self._cvxpy.reshape(
            Psi,
            (self._est.nmodels(), self._est.nmodels()),
            order="F",
        )
        return Psi

    def _cvxpy_spd_constraint(
        self, nsps_cvxpy: CvxpyExpression, t_cvxpy: CvxpyExpression,
    ) -> CvxpyExpression:
        """Construct the SPD constraint matrix."""
        Psi = self._cvxpy_psi(nsps_cvxpy)
        result: CvxpyExpression = self._cvxpy.bmat(
            [
                [Psi, self._est._asketch.T],
                [
                    self._est._asketch,
                    self._cvxpy.reshape(t_cvxpy, (1, 1), order="F"),
                ],
            ]
        )
        return result

    def optimize(
        self,
        target_cost: float,
        min_nhf_samples: int = 1,
        min_nlf_samples: Optional[Array] = None,
        init_guess: Optional[Array] = None,
        round_nsamples: bool = True,
    ) -> GroupACVAllocationResult[Array]:
        """Find optimal sample allocation using SDP.

        Parameters
        ----------
        target_cost : float
            Maximum computational budget.
        min_nhf_samples : int, optional
            Minimum number of high-fidelity samples. Default is 1.
        min_nlf_samples : Array, optional
            Minimum number of samples for each low-fidelity model.
            Shape: (nmodels - 1,). If None, no minimum is enforced.
        init_guess : Array, optional
            Ignored for SDP solver (convex problem finds global optimum).
        round_nsamples : bool, optional
            Whether to round result to integers. Default is True.

        Returns
        -------
        GroupACVAllocationResult
            Optimization result with npartition_samples.

        Raises
        ------
        RuntimeError
            If statistics has multiple outputs or solver fails.
        """
        if self._est._stat.nstats() != 1:
            raise RuntimeError("SPD solver only works for single outputs (nstats=1)")

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
            subset_costs @ nsps_cvxpy <= target_cost,
            self._est._partitions_per_model[0] @ nsps_cvxpy >= min_nhf_samples,
        ]

        # Optional low-fidelity sample constraints
        if min_nlf_samples is not None:
            for ii in range(self._est.nmodels() - 1):
                constraints.append(
                    self._est._partitions_per_model[ii + 1] @ nsps_cvxpy
                    >= min_nlf_samples[ii]
                )

        # SPD constraint: the Schur complement matrix must be positive semidefinite
        constraints.append(self._cvxpy_spd_constraint(nsps_cvxpy, t_cvxpy) >> 0)

        # Solve the problem
        prob = self._cvxpy.Problem(obj, constraints)
        prob.solve(solver=self._solver_name)

        if t_cvxpy.value is None:
            return GroupACVAllocationResult(
                npartition_samples=self._bkd.zeros((self._est.nsubsets(),)),
                nsamples_per_model=self._bkd.zeros((self._est.nmodels(),)),
                actual_cost=0.0,
                objective_value=self._bkd.array([float("inf")]),
                success=False,
                message=(
                    f"SPD solver ({self._solver_name}) did not converge. "
                    "Try a different solver or check problem formulation."
                ),
            )

        npartition_samples = self._bkd.flatten(self._bkd.array(nsps_cvxpy.value))

        # Round if requested
        if round_nsamples:
            npartition_samples = self._bkd.floor(npartition_samples + 1e-4)

        # Compute derived quantities
        nsamples_per_model = self._est._compute_nsamples_per_model(npartition_samples)
        actual_cost = self._bkd.to_float(self._est._estimator_cost(npartition_samples))
        obj_value = self._bkd.array([float(t_cvxpy.value)])

        return GroupACVAllocationResult(
            npartition_samples=npartition_samples,
            nsamples_per_model=nsamples_per_model,
            actual_cost=actual_cost,
            objective_value=obj_value,
            success=True,
            message="",
        )
