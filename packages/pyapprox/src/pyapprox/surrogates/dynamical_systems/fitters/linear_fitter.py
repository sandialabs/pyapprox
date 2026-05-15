"""Linear-in-parameters fitter for dynamical systems.

Owns the linear solve directly: evaluates basis matrix, solves
the linear system, and returns a new fitted vector field.
Matches the expansion fitter pattern (LeastSquaresFitter, OMPFitter).
"""

from typing import Generic, Optional

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.optimization.linear.base import LinearSystemSolver
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.dynamical_systems.vector_fields import (
    BasisExpansionVectorField,
)
from pyapprox.util.backends.protocols import Array, Backend


class LinearInParamsFitter(Generic[Array]):
    """Fits BasisExpansionVectorField via linear regression.

    Evaluates the basis matrix at training states and solves
    the linear system Phi @ c = derivatives^T for coefficients.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    solver : LinearSystemSolver[Array], optional
        Solver for the linear system. If None, uses LeastSquares.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolver[Array]] = None,
    ):
        self._bkd = bkd
        self._solver = solver if solver is not None else LeastSquaresSolver(bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        vector_field: BasisExpansionVectorField[Array],
        dataset: SnapshotDataset[Array],
    ) -> DirectSolverResult[Array]:
        """Fit vector field coefficients to snapshot data.

        Parameters
        ----------
        vector_field : BasisExpansionVectorField[Array]
            Vector field to fit (not modified).
        dataset : SnapshotDataset[Array]
            Training data with states and derivatives.

        Returns
        -------
        DirectSolverResult[Array]
            Result with fitted vector field and coefficients.
        """
        expansion = vector_field.expansion()
        states = dataset.states()
        derivs = dataset.derivatives()

        Phi = expansion.basis_matrix(states)
        coef = self._solver.solve(Phi, derivs.T)

        fitted_vf = vector_field.with_params(coef)

        return DirectSolverResult(
            surrogate=fitted_vf,
            params=coef,
        )
