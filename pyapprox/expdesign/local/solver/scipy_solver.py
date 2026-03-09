"""
SciPy-based local OED solver.

Uses trust-region constrained optimization to find optimal design weights
for scalar criteria (D, A, C, I-optimal).
"""

from typing import Generic, Optional

from pyapprox.expdesign.local.protocols.criterion import (
    LocalOEDCriterionProtocol,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.backends.protocols import Array, Backend

from .base import LocalOEDSolverBase

# TODO: We should generalize this to take any configured bindable optimizer
# not just scipy #see pyapprox.optimization.minimize protocol.
# This will allow us to remove the options from the __init__ method
class ScipyLocalOEDSolver(LocalOEDSolverBase[Array], Generic[Array]):
    """
    Local OED solver using SciPy's trust-region constrained optimizer.

    Suitable for scalar criteria (nqoi=1) such as D-optimal, A-optimal,
    C-optimal, and I-optimal.

    Parameters
    ----------
    criterion : LocalOEDCriterionProtocol[Array]
        Scalar criterion to minimize (must have nqoi() == 1).
    bkd : Backend[Array]
        Computational backend.
    verbosity : int
        Verbosity level for optimizer output (0=silent, 1=summary, 2=detailed).
    maxiter : int, optional
        Maximum number of iterations.
    gtol : float, optional
        Gradient tolerance for convergence.

    Raises
    ------
    ValueError
        If criterion has nqoi() > 1 (use MinimaxLocalOEDSolver instead).

    Examples
    --------
    >>> criterion = DOptimalCriterion(design_matrices, bkd)
    >>> solver = ScipyLocalOEDSolver(criterion, bkd)
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
        if criterion.nqoi() != 1:
            raise ValueError(
                f"ScipyLocalOEDSolver requires scalar criterion (nqoi=1), "
                f"got nqoi={criterion.nqoi()}. Use MinimaxLocalOEDSolver for "
                f"vector criteria like G-optimal."
            )
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol

    def construct(self, init_weights: Optional[Array] = None) -> Array:
        """
        Find optimal design weights using trust-region optimization.

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
        simplex_constraint = self._create_simplex_constraint()

        optimizer = ScipyTrustConstrOptimizer(
            objective=self._criterion,
            bounds=bounds,
            constraints=[simplex_constraint],
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        result = optimizer.minimize(init_weights)
        self._result = result

        return result.optima()

    def get_result(self) -> "ScipyOptimizerResultWrapper[Array]":
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
            raise AttributeError("No result available. Call construct() first.")
        return self._result
