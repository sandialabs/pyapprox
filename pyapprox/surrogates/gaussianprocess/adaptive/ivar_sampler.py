"""IVAR (Integrated Variance Reduction) adaptive sampler for adaptive GP."""

from typing import Generic

from pyapprox.surrogates.gaussianprocess.statistics.protocols import (
    KernelIntegralCalculatorProtocol,
)
from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.incremental_cholesky import (
    IncrementalCholeskyFactorization,
)


class IVARSampler(Generic[Array]):
    """Adaptive sampler using IVAR (Integrated Variance Reduction).

    Greedy selection minimizing integrated posterior variance. Uses
    incremental Cholesky factorization for efficiency. Operates entirely
    in scaled space.

    Parameters
    ----------
    candidates_scaled : Array
        Candidate locations of shape (nvars, ncandidates) in scaled space.
    bkd : Backend[Array]
        Backend for numerical computations.
    nugget : float
        Nugget added to candidate kernel matrix diagonal for numerical
        stability.
    """

    def __init__(
        self,
        candidates_scaled: Array,
        bkd: Backend[Array],
        nugget: float = 0.0,
    ) -> None:
        self._candidates = candidates_scaled
        self._bkd = bkd
        self._nugget = nugget
        self._ncandidates = candidates_scaled.shape[1]
        self._selected_indices: list[int] = []
        self._kernel: KernelProtocol[Array] | None = None
        self._K: Array | None = None
        self._P: Array | None = None
        self._cholesky: IncrementalCholeskyFactorization[Array] | None = None
        self._integral_calc: KernelIntegralCalculatorProtocol[Array] | None = None
        self._best_obj_vals: list[float] = []

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def set_integral_calculator(
        self, calc: KernelIntegralCalculatorProtocol[Array]
    ) -> None:
        """Set the integral calculator for P matrix computation.

        Parameters
        ----------
        calc : KernelIntegralCalculatorProtocol[Array]
            Calculator that provides P() method.
        """
        self._integral_calc = calc

    def set_P(self, P: Array) -> None:
        """Directly set the P matrix.

        Parameters
        ----------
        P : Array
            P matrix of shape (ncandidates, ncandidates).
        """
        self._P = P

    def set_kernel(self, kernel: KernelProtocol[Array]) -> None:
        """Update the kernel and rebuild internal state.

        Recomputes K matrix, optionally recomputes P matrix via integral
        calculator, resets incremental Cholesky, and re-adds existing
        pivots.

        Parameters
        ----------
        kernel : KernelProtocol[Array]
            New kernel (after hyperparameter optimization).
        """
        bkd = self._bkd
        self._kernel = kernel

        # Recompute K
        K = kernel(self._candidates, self._candidates)
        if self._nugget > 0:
            K = K + self._nugget * bkd.eye(self._ncandidates)
        self._K = K

        # Recompute P if integral calculator available
        if self._integral_calc is not None:
            self._P = self._integral_calc.P()

        # Reset incremental Cholesky and re-add existing pivots
        self._cholesky = IncrementalCholeskyFactorization(K, bkd)
        for idx in self._selected_indices:
            self._cholesky.add_pivot(idx)

    def select_samples(self, nsamples: int) -> Array:
        """Select new sample locations via greedy IVAR minimization.

        Parameters
        ----------
        nsamples : int
            Number of new samples to select.

        Returns
        -------
        samples : Array
            Selected samples of shape (nvars, nsamples) in scaled space.

        Raises
        ------
        ValueError
            If candidates exhausted, kernel not set, or P matrix not set.
        """
        if self._K is None or self._cholesky is None:
            raise ValueError("Must call set_kernel() before select_samples()")
        if self._P is None:
            raise ValueError(
                "P matrix not set. Call set_P() or set_integral_calculator() "
                "before set_kernel()."
            )

        n_available = self._ncandidates - len(self._selected_indices)
        if nsamples > n_available:
            raise ValueError(
                "All candidates exhausted; generate a new candidate set "
                "or increase its size"
            )

        bkd = self._bkd
        new_pivots: list[int] = []

        # If initial pivots were set but no objective has been tracked yet,
        # compute the objective for the current selected set by direct
        # matrix solve: -trace(K[sel]^{-1} P[sel]).
        if self._selected_indices and not self._best_obj_vals:
            idx = bkd.asarray(self._selected_indices)
            Kmat = self._K[idx, :][:, idx]
            Pmat = self._P[idx, :][:, idx]
            obj = -bkd.trace(bkd.solve(Kmat, Pmat))
            self._best_obj_vals.append(bkd.to_float(obj))

        for _ in range(nsamples):
            obj_vals = self._compute_objective()
            # Select candidate with minimum objective (best variance reduction)
            best = bkd.to_int(bkd.argmin(obj_vals))
            self._best_obj_vals.append(bkd.to_float(obj_vals[best]))
            new_pivots.append(best)
            self._selected_indices.append(best)
            self._cholesky.add_pivot(best)

        new_pivot_indices = bkd.asarray(new_pivots)
        return self._candidates[:, new_pivot_indices]

    def _compute_objective(self) -> Array:
        """Compute IVAR objective for all candidates.

        Returns the absolute integrated variance for each candidate if it
        were added to the selected set. Lower is better. Already-selected
        candidates get inf.

        When pivots exist, includes the running baseline
        (``_best_obj_vals[-1]``) so that the returned values represent
        absolute integrated variance, matching the legacy convention.
        """
        bkd = self._bkd
        assert self._K is not None
        assert self._P is not None
        assert self._cholesky is not None

        n = self._cholesky.npivots()
        vals = bkd.full((self._ncandidates,), float("inf"))

        # Mark selected candidates as unavailable
        selected_set = set(self._selected_indices)
        useful = []
        for i in range(self._ncandidates):
            if i not in selected_set:
                useful.append(i)
        if len(useful) == 0:
            return vals

        useful_arr = bkd.asarray(useful)

        if n == 0:
            # First point: objective = -P_ii / K_ii
            P_diag = bkd.diag(self._P)[useful_arr]
            K_diag = bkd.diag(self._K)[useful_arr]
            vals[useful_arr] = -P_diag / K_diag
            return vals

        self._cholesky.L()
        L_inv = self._cholesky.L_inv()
        pivots = self._selected_indices

        # A_12: K[pivots, useful_candidates], shape (n, n_useful)
        A_12 = self._K[pivots, :][:, useful_arr]

        # L_12 = L^{-1} @ A_12, shape (n, n_useful)
        L_12 = L_inv @ A_12

        # Remaining variance for each candidate
        K_diag_useful = bkd.diag(self._K)[useful_arr]
        residual_var = K_diag_useful - bkd.sum(L_12 * L_12, axis=0)

        # Mark candidates with non-positive residual variance as invalid
        valid_mask = residual_var > 0
        valid_indices_np = bkd.to_numpy(useful_arr)
        valid_mask_np = bkd.to_numpy(valid_mask)

        valid_candidates = []
        valid_local = []
        for ii, (idx, is_valid) in enumerate(zip(valid_indices_np, valid_mask_np)):
            if is_valid:
                valid_candidates.append(int(idx))
                valid_local.append(ii)

        if len(valid_candidates) == 0:
            return vals

        valid_arr = bkd.asarray(valid_candidates)
        valid_local_arr = bkd.asarray(valid_local)

        L_12_valid = L_12[:, valid_local_arr]
        L_22 = bkd.sqrt(residual_var[valid_local_arr])

        # P matrix blocks
        P_11 = self._P[pivots, :][:, pivots]  # (n, n)
        P_12 = self._P[pivots, :][:, valid_arr]  # (n, n_valid)
        P_22 = bkd.diag(self._P)[valid_arr]  # (n_valid,)

        # C = -(L_12 / L_22)^T @ L_inv, shape (n_valid, n)
        C = -((L_12_valid / L_22).T @ L_inv)

        # Incremental IVAR terms
        term1 = bkd.sum(C.T * (P_11 @ C.T), axis=0)
        term2 = 2.0 * bkd.sum(C.T / L_22 * P_12, axis=0)
        term3 = P_22 / L_22**2

        # Absolute objective: matches legacy formula
        # vals = -(-baseline + term1 + term2 + term3)
        #       = baseline - term1 - term2 - term3
        baseline = self._best_obj_vals[-1] if self._best_obj_vals else 0.0
        vals[valid_arr] = -(-baseline + term1 + term2 + term3)
        return vals

    def set_initial_pivots(self, pivot_indices: list[int]) -> None:
        """Pre-seed selected indices before greedy selection.

        Used to initialize the sampler with known starting points,
        e.g., the midpoint of the highest-fidelity output in
        multioutput IVAR.

        Parameters
        ----------
        pivot_indices : list[int]
            Indices into the candidate array to pre-select.
        """
        for idx in pivot_indices:
            self._selected_indices.append(idx)
            if self._cholesky is not None:
                self._cholesky.add_pivot(idx)

    def _brute_force_objective(self, new_idx: int) -> Array:
        """Compute objective by direct matrix solve for testing.

        Computes -trace(solve(K[sel+new, sel+new], P[sel+new, sel+new]))
        where sel are the currently selected indices and new is the
        candidate being evaluated.

        Parameters
        ----------
        new_idx : int
            Index of the candidate to evaluate.

        Returns
        -------
        obj : Array
            Scalar objective value.
        """
        assert self._K is not None
        assert self._P is not None
        bkd = self._bkd
        indices = self._selected_indices + [new_idx]
        idx = bkd.asarray(indices)
        Kmat = self._K[idx, :][:, idx]
        Pmat = self._P[idx, :][:, idx]
        return -bkd.trace(bkd.solve(Kmat, Pmat))

    def selected_indices(self) -> list[int]:
        """Return the list of selected candidate indices."""
        return list(self._selected_indices)

    def add_additional_training_samples(self, new_samples: Array) -> None:
        """No-op: tracking is handled internally by select_samples."""
