"""Sobol sensitivity indices for PCE FunctionTrain."""

from typing import Generic, List, Optional, Sequence, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.functiontrain.statistics.moments import (
    FunctionTrainMoments,
)


class FunctionTrainSensitivity(Generic[Array]):
    """Compute Sobol sensitivity indices from PCE FunctionTrain.

    Parameters
    ----------
    moments : FunctionTrainMoments[Array]
        Precomputed moments (provides mean, variance).

    Raises
    ------
    TypeError
        If moments is not a FunctionTrainMoments instance.

    Notes
    -----
    Complexity:
    - Main effects: O(d · r² · p)
    - Total effects: O(d · r⁶)
    - General S_u: O(d · r⁶) per subset

    For r > 10, Kronecker sweeps may become expensive.

    Warning
    -------
    Assumes all basis expansions use orthonormal polynomials. Results are
    mathematically incorrect for non-orthonormal bases.
    """

    def __init__(self, moments: FunctionTrainMoments[Array]) -> None:
        if not isinstance(moments, FunctionTrainMoments):
            raise TypeError(
                f"Expected FunctionTrainMoments, got {type(moments).__name__}"
            )
        self._moments = moments
        self._pce_ft = moments.pce_ft()
        self._bkd = self._pce_ft.bkd()
        # Caches for sweep products
        self._left_mean: Optional[List[Array]] = None
        self._right_mean: Optional[List[Array]] = None
        self._left_kron: Optional[List[Array]] = None
        self._right_kron: Optional[List[Array]] = None

    def moments(self) -> FunctionTrainMoments[Array]:
        """Access underlying FunctionTrainMoments."""
        return self._moments

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    # ==================== Mean Sweeps ====================

    def _compute_mean_sweeps(self) -> Tuple[List[Array], List[Array]]:
        """Compute and cache L̄_k and R̄_k for all k.

        Forward: L̄_0 = [1], L̄_{k} = L̄_{k-1} · Θ_{k-1}^{(0)}
        Backward: R̄_{d-1} = [1], R̄_k = Θ_{k+1}^{(0)} · R̄_{k+1}

        Returns
        -------
        left_mean : List[Array]
            L̄_k for k = 0, ..., d-1. left[k] is product of cores 0..k-1.
        right_mean : List[Array]
            R̄_k for k = 0, ..., d-1. right[k] is product of cores k+1..d-1.
        """
        if self._left_mean is not None and self._right_mean is not None:
            return self._left_mean, self._right_mean

        cores = self._pce_ft.pce_cores()
        d = len(cores)

        # Forward sweep: L̄_k (left[k] is product of cores 0..k-1)
        left: List[Array] = []
        L = self._bkd.array([[1.0]])  # Shape (1, 1) for k=0
        left.append(L)
        for k in range(d - 1):
            L = self._bkd.dot(L, cores[k].expected_core())
            left.append(L)

        # Backward sweep: R̄_k (right[k] is product of cores k+1..d-1)
        right: List[Array] = []
        R = self._bkd.array([[1.0]])  # Shape (1, 1) for k=d-1
        # Build in reverse, then reverse the list
        right_reversed = [R]
        for k in range(d - 2, -1, -1):
            R = self._bkd.dot(cores[k + 1].expected_core(), R)
            right_reversed.append(R)
        right = list(reversed(right_reversed))

        self._left_mean = left
        self._right_mean = right
        return left, right

    # ==================== Kronecker Sweeps ====================

    def _compute_kron_sweeps(self) -> Tuple[List[Array], List[Array]]:
        """Compute and cache L̃_k and R̃_k for all k.

        Forward: L̃_0 = [1], L̃_k = L̃_{k-1} · M_{k-1}
        Backward: R̃_{d-1} = [1], R̃_k = M_{k+1} · R̃_{k+1}

        Returns
        -------
        left_kron : List[Array]
            L̃_k for k = 0, ..., d-1. Shapes vary by position.
        right_kron : List[Array]
            R̃_k for k = 0, ..., d-1. Shapes vary by position.
        """
        if self._left_kron is not None and self._right_kron is not None:
            return self._left_kron, self._right_kron

        cores = self._pce_ft.pce_cores()
        d = len(cores)

        # Forward sweep: L̃_k
        left: List[Array] = []
        L = self._bkd.array([[1.0]])  # Shape (1, 1)
        left.append(L)
        for k in range(d - 1):
            L = self._bkd.dot(L, cores[k].expected_kron_core())
            left.append(L)

        # Backward sweep: R̃_k
        right_reversed = [self._bkd.array([[1.0]])]
        R = self._bkd.array([[1.0]])  # Shape (1, 1)
        for k in range(d - 2, -1, -1):
            R = self._bkd.dot(cores[k + 1].expected_kron_core(), R)
            right_reversed.append(R)
        right = list(reversed(right_reversed))

        self._left_kron = left
        self._right_kron = right
        return left, right

    # ==================== Main Effects ====================

    def main_effect_variance(self, var_idx: int) -> Array:
        """V_k = Σ_{ℓ≥1} (L̄_k · Θ_k^{(ℓ)} · R̄_k)².

        Parameters
        ----------
        var_idx : int
            Variable index (0-indexed).

        Returns
        -------
        Array
            Main effect variance. Shape: (1,).
        """
        self._validate_var_idx(var_idx)
        left, right = self._compute_mean_sweeps()
        core = self._pce_ft.pce_cores()[var_idx]

        V_k = self._bkd.array([0.0])
        for ell in range(1, core.nterms()):  # Skip constant (ell=0)
            theta_ell = core.coefficient_matrix(ell)
            # c_ell = L̄_k · Θ_k^{(ℓ)} · R̄_k (scalar)
            c_ell = self._bkd.dot(
                self._bkd.dot(left[var_idx], theta_ell),
                right[var_idx]
            )
            V_k = V_k + self._bkd.reshape(c_ell, (1,)) ** 2

        return V_k

    def main_effect_index(self, var_idx: int) -> Array:
        """S_k = V_k / Var[f].

        Parameters
        ----------
        var_idx : int
            Variable index (0-indexed).

        Returns
        -------
        Array
            Main effect Sobol index. Shape: (1,), value in [0, 1].
        """
        return self.main_effect_variance(var_idx) / self._moments.variance()

    def all_main_effects(self) -> Array:
        """Compute [S_0, ..., S_{d-1}].

        Returns
        -------
        Array
            Main effect indices. Shape: (nvars,)
        """
        nvars = self._pce_ft.nvars()
        indices = [self.main_effect_index(k) for k in range(nvars)]
        return self._bkd.concatenate(indices, axis=0)

    # ==================== Total Effects ====================

    def total_effect_variance(self, var_idx: int) -> Array:
        """V_k^T = L̃_k · ΔM_k · R̃_k.

        Parameters
        ----------
        var_idx : int
            Variable index (0-indexed).

        Returns
        -------
        Array
            Total effect variance. Shape: (1,).
        """
        self._validate_var_idx(var_idx)
        left, right = self._compute_kron_sweeps()
        core = self._pce_ft.pce_cores()[var_idx]

        delta_M = core.delta_kron_core()
        V_T = self._bkd.dot(
            self._bkd.dot(left[var_idx], delta_M),
            right[var_idx]
        )
        return self._bkd.reshape(V_T, (1,))

    def total_effect_index(self, var_idx: int) -> Array:
        """S_k^T = V_k^T / Var[f].

        Parameters
        ----------
        var_idx : int
            Variable index (0-indexed).

        Returns
        -------
        Array
            Total effect Sobol index. Shape: (1,), value in [0, 1].
        """
        return self.total_effect_variance(var_idx) / self._moments.variance()

    def all_total_effects(self) -> Array:
        """Compute [S_0^T, ..., S_{d-1}^T].

        Returns
        -------
        Array
            Total effect indices. Shape: (nvars,)
        """
        nvars = self._pce_ft.nvars()
        indices = [self.total_effect_index(k) for k in range(nvars)]
        return self._bkd.concatenate(indices, axis=0)

    # ==================== General Sobol Indices ====================

    def sobol_variance(self, index_set: Sequence[int]) -> Array:
        """V_u for arbitrary subset u ⊆ {0, ..., d-1}.

        V_u = Π_k N_k^{[u]} where:
        - N_k^{[u]} = ΔM_k if k ∈ u
        - N_k^{[u]} = M_k^{(0)} if k ∉ u

        Parameters
        ----------
        index_set : Sequence[int]
            Variable indices in the subset.

        Returns
        -------
        Array
            Sobol variance. Shape: (1,).
        """
        for idx in index_set:
            self._validate_var_idx(idx)

        index_set_s = set(index_set)
        cores = self._pce_ft.pce_cores()

        result = self._bkd.array([[1.0]])
        for k, core in enumerate(cores):
            if k in index_set_s:
                N_k = core.delta_kron_core()
            else:
                N_k = core.mean_kron_core()
            result = self._bkd.dot(result, N_k)

        return self._bkd.reshape(result, (1,))

    def sobol_index(self, index_set: Sequence[int]) -> Array:
        """S_u = V_u / Var[f].

        Parameters
        ----------
        index_set : Sequence[int]
            Variable indices in the subset.

        Returns
        -------
        Array
            Sobol index. Shape: (1,), value in [0, 1].
        """
        return self.sobol_variance(index_set) / self._moments.variance()

    # ==================== Validation ====================

    def _validate_var_idx(self, var_idx: int) -> None:
        """Validate variable index is in bounds."""
        nvars = self._pce_ft.nvars()
        if var_idx < 0 or var_idx >= nvars:
            raise IndexError(
                f"Variable index {var_idx} out of bounds [0, {nvars})"
            )
