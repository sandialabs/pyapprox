"""Marginalization for PCE FunctionTrain surrogates.

Provides tools to create lower-dimensional FunctionTrains by integrating out
(marginalizing) variables. The resulting FunctionTrain works seamlessly with
existing statistics modules like FunctionTrainMoments and FunctionTrainSensitivity.
"""

from typing import Any, Generic, List, Optional, Sequence, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.typing.surrogates.functiontrain import FunctionTrain
from pyapprox.typing.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
)
from pyapprox.typing.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)


class FunctionTrainMarginalization(Generic[Array]):
    """Marginalize variables from a PCE FunctionTrain.

    Supports creating lower-dimensional marginal FunctionTrains by
    integrating out specified variables. The marginalized FunctionTrain
    can be used with PCEFunctionTrain, FunctionTrainMoments, and
    FunctionTrainSensitivity without modification.

    Parameters
    ----------
    pce_ft : PCEFunctionTrain[Array]
        The PCE FunctionTrain to marginalize.

    Warning
    -------
    Assumes all basis expansions use orthonormal polynomials (e.g., Legendre,
    Hermite, Laguerre). Results are mathematically incorrect for non-orthonormal
    bases such as monomials.

    Notes
    -----
    Currently only supports nqoi=1. Multi-QoI support may be added later.

    Examples
    --------
    >>> # Create 1D marginal keeping only variable 0
    >>> marg = FunctionTrainMarginalization(pce_ft)
    >>> ft_1d = marg.marginal([0])
    >>> pce_1d = PCEFunctionTrain(ft_1d)
    >>> moments_1d = FunctionTrainMoments(pce_1d)
    """

    def __init__(self, pce_ft: PCEFunctionTrain[Array]) -> None:
        if not isinstance(pce_ft, PCEFunctionTrain):
            raise TypeError(
                f"Expected PCEFunctionTrain, got {type(pce_ft).__name__}"
            )
        self._pce_ft = pce_ft
        self._bkd = pce_ft.bkd()

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def pce_ft(self) -> PCEFunctionTrain[Array]:
        """Return underlying PCEFunctionTrain."""
        return self._pce_ft

    def marginalize(self, var_indices: Sequence[int]) -> FunctionTrain[Array]:
        """Return lower-dimensional FunctionTrain with variables integrated out.

        Parameters
        ----------
        var_indices : Sequence[int]
            Indices of variables to integrate out (marginalize).

        Returns
        -------
        FunctionTrain[Array]
            New FunctionTrain with len(var_indices) fewer variables.
            Works with PCEFunctionTrain, FunctionTrainMoments, etc.

        Raises
        ------
        ValueError
            If var_indices includes all variables.
            Use FunctionTrainMoments.mean() instead.
        IndexError
            If any index is out of bounds.
        """
        nvars = self._pce_ft.nvars()

        # Validate indices
        for idx in var_indices:
            if idx < 0 or idx >= nvars:
                raise IndexError(
                    f"Variable index {idx} out of bounds [0, {nvars})"
                )

        # Check not marginalizing all variables
        unique_indices = set(var_indices)
        if len(unique_indices) == nvars:
            raise ValueError(
                "Cannot marginalize all variables. "
                "Use FunctionTrainMoments.mean() to compute the mean instead."
            )

        # If nothing to marginalize, return a copy
        if len(unique_indices) == 0:
            return self._pce_ft.ft().with_cores(list(self._pce_ft.ft().cores()))

        # Pre-compute all expected cores BEFORE any modifications
        # This ensures we use the original coefficients
        pce_cores = self._pce_ft.pce_cores()
        expected_cores = {k: pce_cores[k].expected_core() for k in unique_indices}

        # Get working copies of cores
        working_cores = list(self._pce_ft.ft().cores())

        # Find contiguous groups of variables to marginalize
        # and identify the surviving neighbor for each group
        sorted_indices = sorted(unique_indices)
        keep_indices = sorted(set(range(nvars)) - unique_indices)

        # Process each variable to marginalize
        # Strategy: replace each marginalized core with None (placeholder),
        # then for each contiguous group, compute the product of expected
        # cores and absorb into the nearest surviving neighbor.

        # Mark marginalized positions with their expected cores
        # Use a mixed list: FunctionTrainCore for kept, Array for marginalized
        mixed_cores: List[Union[FunctionTrainCore[Array], Array]] = list(
            working_cores
        )
        for k in unique_indices:
            mixed_cores[k] = expected_cores[k]  # Replace with expected core matrix

        # Now build the result by processing left to right
        # When we encounter a sequence of matrices (marginalized cores),
        # multiply them together and absorb into the next surviving core
        result_cores: List[FunctionTrainCore[Array]] = []
        pending_matrix: Optional[Array] = None  # Accumulated expected cores

        for k in range(nvars):
            if k in unique_indices:
                # This is a marginalized variable - accumulate its expected core
                E_k = expected_cores[k]
                if pending_matrix is None:
                    pending_matrix = E_k
                else:
                    pending_matrix = self._bkd.dot(pending_matrix, E_k)
            else:
                # This is a kept variable
                core_k = working_cores[k]
                if pending_matrix is not None:
                    # Absorb accumulated expected cores into this core (left-multiply)
                    core_k = self._multiply_coefficients_left(pending_matrix, core_k)
                    pending_matrix = None
                result_cores.append(core_k)

        # If there's a pending matrix at the end, absorb into the last result core
        if pending_matrix is not None and len(result_cores) > 0:
            last_core = result_cores[-1]
            result_cores[-1] = self._multiply_coefficients_right(
                last_core, pending_matrix
            )

        return FunctionTrain(result_cores, self._bkd, nqoi=self._pce_ft.nqoi())

    def marginal(self, keep_indices: Sequence[int]) -> FunctionTrain[Array]:
        """Return FunctionTrain marginal keeping only specified variables.

        This is the complement of marginalize() - specify which variables
        to KEEP rather than which to integrate out.

        Parameters
        ----------
        keep_indices : Sequence[int]
            Indices of variables to keep in the marginal.

        Returns
        -------
        FunctionTrain[Array]
            New FunctionTrain with only the specified variables.

        Raises
        ------
        ValueError
            If keep_indices is empty (would marginalize all).

        Notes
        -----
        The returned FunctionTrain has variables reindexed 0, 1, ..., len(keep_indices)-1
        in the same order as keep_indices. Original indices are not preserved.

        Example: marginal([2, 0]) returns 2-variable FT where:
        - New variable 0 corresponds to original variable 2
        - New variable 1 corresponds to original variable 0

        Examples
        --------
        >>> # Create 1D marginal keeping variable 0
        >>> marginal_x0 = marg.marginal([0])
        >>> # Create 2D marginal keeping variables 1 and 3
        >>> marginal_x1x3 = marg.marginal([1, 3])
        """
        if len(keep_indices) == 0:
            raise ValueError(
                "Cannot keep zero variables. "
                "Use FunctionTrainMoments.mean() to compute the mean instead."
            )

        all_vars = set(range(self._pce_ft.nvars()))
        keep_set = set(keep_indices)
        marginalize_indices = list(all_vars - keep_set)
        return self.marginalize(marginalize_indices)

    def _extract_coef_tensor(self, core: FunctionTrainCore[Array]) -> Array:
        """Extract coefficients as tensor from core.

        Returns
        -------
        Array
            Shape: (r_left, r_right, nterms)
            coef[i, j, ℓ] = coefficient of basis ℓ in expansion (i,j)
        """
        r_left, r_right = core.ranks()
        nterms = core.get_nterms(0, 0)  # All expansions have same nterms

        rows = []
        for ii in range(r_left):
            cols = []
            for jj in range(r_right):
                # get_coefficients() returns (nterms, nqoi=1)
                coef = core.get_basisexp(ii, jj).get_coefficients()[:, 0]
                cols.append(coef)
            rows.append(self._bkd.stack(cols, axis=0))  # (r_right, nterms)
        return self._bkd.stack(rows, axis=0)  # (r_left, r_right, nterms)

    def _multiply_coefficients_right(
        self,
        core: FunctionTrainCore[Array],
        matrix: Array,  # Shape: (r_right_old, r_right_new)
    ) -> FunctionTrainCore[Array]:
        """Right-multiply all coefficient matrices by given matrix.

        coef_new[i, j', ℓ] = Σ_j coef[i, j, ℓ] * matrix[j, j']

        In einsum: new_coef = einsum('ijl,jk->ikl', coef, matrix)

        Parameters
        ----------
        core : FunctionTrainCore[Array]
            Core to transform.
        matrix : Array
            Transformation matrix. Shape: (r_right_old, r_right_new)

        Returns
        -------
        FunctionTrainCore[Array]
            New core with transformed coefficients.
        """
        coef = self._extract_coef_tensor(core)  # (r_left, r_right, nterms)
        new_coef = self._bkd.einsum('ijl,jk->ikl', coef, matrix)
        return self._build_core_from_coefficients(core, new_coef)

    def _multiply_coefficients_left(
        self,
        matrix: Array,  # Shape: (r_left_new, r_left_old)
        core: FunctionTrainCore[Array],
    ) -> FunctionTrainCore[Array]:
        """Left-multiply all coefficient matrices by given matrix.

        coef_new[i', j, ℓ] = Σ_i matrix[i', i] * coef[i, j, ℓ]

        In einsum: new_coef = einsum('ki,ijl->kjl', matrix, coef)

        Parameters
        ----------
        matrix : Array
            Transformation matrix. Shape: (r_left_new, r_left_old)
        core : FunctionTrainCore[Array]
            Core to transform.

        Returns
        -------
        FunctionTrainCore[Array]
            New core with transformed coefficients.
        """
        coef = self._extract_coef_tensor(core)  # (r_left, r_right, nterms)
        new_coef = self._bkd.einsum('ki,ijl->kjl', matrix, coef)
        return self._build_core_from_coefficients(core, new_coef)

    def _build_core_from_coefficients(
        self,
        template_core: FunctionTrainCore[Array],
        new_coef: Array,  # Shape: (new_r_left, new_r_right, nterms)
    ) -> FunctionTrainCore[Array]:
        """Create new core with given coefficients.

        Uses template_core's basis structure (via get_basisexp(0,0).with_params())
        but with new coefficients and potentially different ranks.

        Parameters
        ----------
        template_core : FunctionTrainCore[Array]
            Core to use as template for basis structure.
        new_coef : Array
            New coefficients. Shape: (new_r_left, new_r_right, nterms)

        Returns
        -------
        FunctionTrainCore[Array]
            New core with given coefficients.
        """
        new_r_left, new_r_right, nterms = new_coef.shape
        template_bexp = template_core.get_basisexp(0, 0)

        new_basisexps: List[List[BasisExpansionProtocol[Array]]] = []
        for ii in range(new_r_left):
            row: List[BasisExpansionProtocol[Array]] = []
            for jj in range(new_r_right):
                # Extract coefficients for this position: (nterms,) -> (nterms, 1)
                coef_ij = self._bkd.reshape(new_coef[ii, jj, :], (nterms, 1))
                new_bexp = template_bexp.with_params(coef_ij)
                row.append(new_bexp)
            new_basisexps.append(row)

        return FunctionTrainCore(new_basisexps, self._bkd)


# =============================================================================
# Convenience functions for common cases (extensible pattern)
# =============================================================================


def marginal_1d(
    pce_ft: PCEFunctionTrain[Array], var_idx: int
) -> FunctionTrain[Array]:
    """Create 1D marginal FunctionTrain for a single variable.

    Parameters
    ----------
    pce_ft : PCEFunctionTrain[Array]
        The PCE FunctionTrain to marginalize.
    var_idx : int
        Index of the variable to keep.

    Returns
    -------
    FunctionTrain[Array]
        1D FunctionTrain (single variable).

    Examples
    --------
    >>> ft_1d = marginal_1d(pce_ft, 0)
    >>> pce_1d = PCEFunctionTrain(ft_1d)
    >>> moments = FunctionTrainMoments(pce_1d)
    """
    return FunctionTrainMarginalization(pce_ft).marginal([var_idx])


def marginal_2d(
    pce_ft: PCEFunctionTrain[Array], var_idx1: int, var_idx2: int
) -> FunctionTrain[Array]:
    """Create 2D marginal FunctionTrain for two variables.

    Parameters
    ----------
    pce_ft : PCEFunctionTrain[Array]
        The PCE FunctionTrain to marginalize.
    var_idx1 : int
        Index of the first variable to keep.
    var_idx2 : int
        Index of the second variable to keep.

    Returns
    -------
    FunctionTrain[Array]
        2D FunctionTrain (two variables).

    Examples
    --------
    >>> ft_2d = marginal_2d(pce_ft, 0, 2)
    >>> pce_2d = PCEFunctionTrain(ft_2d)
    >>> moments = FunctionTrainMoments(pce_2d)
    """
    return FunctionTrainMarginalization(pce_ft).marginal([var_idx1, var_idx2])


def all_marginals_1d(
    pce_ft: PCEFunctionTrain[Array],
) -> List[FunctionTrain[Array]]:
    """Create all 1D marginals.

    Parameters
    ----------
    pce_ft : PCEFunctionTrain[Array]
        The PCE FunctionTrain to marginalize.

    Returns
    -------
    List[FunctionTrain[Array]]
        List of 1D FunctionTrains, one per variable.
        marginals[k] is the marginal keeping only variable k.

    Examples
    --------
    >>> all_1d = all_marginals_1d(pce_ft)
    >>> for k, ft_1d in enumerate(all_1d):
    ...     pce_1d = PCEFunctionTrain(ft_1d)
    ...     moments = FunctionTrainMoments(pce_1d)
    ...     print(f"Var[E[f|x_{k}]] = {moments.variance()}")
    """
    marg = FunctionTrainMarginalization(pce_ft)
    return [marg.marginal([k]) for k in range(pce_ft.nvars())]
