"""Sparse grid-based sensitivity analysis.

This module provides sensitivity analysis using sparse grid surrogates
by converting them to Polynomial Chaos Expansions.
"""

from typing import Generic, List, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate import OrthonormalPolynomial1D
from pyapprox.typing.surrogates.sparsegrids.combination import (
    CombinationSparseGrid,
)
from pyapprox.typing.surrogates.sparsegrids.converters.pce import (
    SparseGridToPCEConverter,
)
from pyapprox.typing.sensitivity.variance_based.pce import (
    PolynomialChaosSensitivityAnalysis,
)


class SparseGridSensitivityAnalysis(Generic[Array]):
    """Sensitivity analysis using sparse grid surrogates.

    Converts the sparse grid to a Polynomial Chaos Expansion and then
    computes Sobol indices analytically from the PCE coefficients.

    Parameters
    ----------
    sparse_grid : CombinationSparseGrid[Array]
        A fitted sparse grid surrogate with values set.
    orthonormal_bases_1d : Sequence[OrthonormalPolynomial1D[Array]]
        Univariate orthonormal polynomial bases for each dimension.
        These should match the input distribution for the sensitivity
        analysis to be valid.

    Examples
    --------
    >>> from pyapprox.typing.sensitivity.variance_based import (
    ...     SparseGridSensitivityAnalysis,
    ... )
    >>> # Assuming sparse_grid is a fitted sparse grid and bases_1d are
    >>> # orthonormal polynomials matching the input distribution
    >>> sa = SparseGridSensitivityAnalysis(sparse_grid, bases_1d)
    >>> main = sa.main_effects()
    >>> total = sa.total_effects()
    """

    def __init__(
        self,
        sparse_grid: CombinationSparseGrid[Array],
        orthonormal_bases_1d: Sequence[OrthonormalPolynomial1D[Array]],
    ) -> None:
        if not isinstance(sparse_grid, CombinationSparseGrid):
            raise TypeError(
                "sparse_grid must be a CombinationSparseGrid, "
                f"got {type(sparse_grid).__name__}"
            )
        self._bkd = sparse_grid.bkd()
        self._sparse_grid = sparse_grid
        self._orthonormal_bases_1d = list(orthonormal_bases_1d)

        # Convert sparse grid to PCE
        converter = SparseGridToPCEConverter(
            self._bkd, self._orthonormal_bases_1d
        )
        pce = converter.convert(sparse_grid)

        # Create PCE sensitivity analysis
        self._pce_sa = PolynomialChaosSensitivityAnalysis(pce)

    def bkd(self) -> Backend[Array]:
        """Return the backend used for array operations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._sparse_grid.nvars()

    def mean(self) -> Array:
        """Return the mean of the output.

        Returns
        -------
        Array
            Shape (nqoi,) - mean for each QoI.
        """
        return self._pce_sa.mean()

    def variance(self) -> Array:
        """Return the variance of the output.

        Returns
        -------
        Array
            Shape (nqoi,) - variance for each QoI.
        """
        return self._pce_sa.variance()

    def main_effects(self) -> Array:
        """Return first-order (main effect) Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - main effect index for each variable and QoI.
        """
        return self._pce_sa.main_effects()

    def total_effects(self) -> Array:
        """Return total-order Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - total effect index for each variable and QoI.
        """
        return self._pce_sa.total_effects()

    def sobol_indices(
        self, variable_sets: List[tuple[int, ...]] | None = None
    ) -> Array:
        """Return Sobol indices for specified interaction terms.

        Parameters
        ----------
        variable_sets : List[Tuple[int, ...]] | None
            Variable index sets to compute interactions for.

        Returns
        -------
        Array
            Shape (nterms, nqoi) - Sobol index for each interaction term.
        """
        return self._pce_sa.sobol_indices(variable_sets)

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._pce_sa.nqoi()
