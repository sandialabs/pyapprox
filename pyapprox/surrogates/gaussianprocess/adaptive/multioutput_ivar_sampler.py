"""Multioutput IVAR sampler for multi-fidelity Gaussian Processes.

Extends the single-output IVAR sampler to handle multiple outputs with
per-output candidate sets, cost-weighted priority, and multi-fidelity
Monte Carlo P matrix computation.
"""

from typing import Generic, List

import numpy as np

from pyapprox.surrogates.gaussianprocess.adaptive.ivar_sampler import (
    IVARSampler,
)
from pyapprox.surrogates.kernels.multioutput.protocols import (
    MultiOutputKernelProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class MultiOutputIVARSampler(Generic[Array]):
    """Adaptive IVAR sampler for multioutput Gaussian Processes.

    Wraps IVARSampler to handle list-of-arrays candidates, cost-weighted
    priority selection, and per-output result mapping.

    For multioutput/multi-fidelity GPs, the candidate set is a list of
    arrays (one per output). Internally, candidates are concatenated into
    a single array and the IVAR objective is computed on the combined
    kernel matrix. Priority is cost-weighted: at each step the candidate
    producing the largest variance reduction per unit cost is selected.

    Parameters
    ----------
    candidates_list : List[Array]
        Candidate locations per output, each of shape (nvars, n_i).
    costs : Array
        Cost per output evaluation, shape (noutputs,).
    bkd : Backend[Array]
        Backend for numerical computations.
    nugget : float
        Nugget for numerical stability.
    nquad_samples : int
        Number of Monte Carlo quadrature samples for P matrix computation.
    """

    def __init__(
        self,
        candidates_list: List[Array],
        costs: Array,
        bkd: Backend[Array],
        nugget: float = 0.0,
        nquad_samples: int = 10000,
    ) -> None:
        self._bkd = bkd
        self._costs = costs
        self._nquad_samples = nquad_samples
        self._noutputs = len(candidates_list)
        self._candidates_list = candidates_list

        # Track partition indices for mapping global indices to per-output
        ncandidates_per_output = [c.shape[1] for c in candidates_list]
        self._ncandidates_per_output = bkd.asarray(ncandidates_per_output)
        partition = [0]
        for n in ncandidates_per_output:
            partition.append(partition[-1] + n)
        self._partition_indices = partition
        self._n_total = partition[-1]

        # Per-candidate output IDs
        output_ids = []
        for ii, n in enumerate(ncandidates_per_output):
            output_ids.extend([ii] * n)
        self._output_ids = output_ids

        # Concatenate candidates into single array
        candidates_concat = bkd.hstack(candidates_list)

        # Create internal single-output IVAR sampler
        self._ivar = IVARSampler(candidates_concat, bkd, nugget)

        # Note: objective values are tracked via self._ivar._best_obj_vals

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def set_kernel(self, kernel: MultiOutputKernelProtocol[Array]) -> None:
        """Set the multioutput kernel and compute K and P matrices.

        The kernel must accept List[Array] inputs (multioutput kernel
        protocol). K is computed as kernel(candidates_list) and P is
        computed via multi-fidelity Monte Carlo quadrature.

        Parameters
        ----------
        kernel : MultiOutputKernelProtocol[Array]
            Kernel that accepts List[Array] and returns stacked matrix.
        """
        bkd = self._bkd

        # Compute K matrix via multioutput kernel (stacked format)
        K_result = kernel(self._candidates_list)
        assert not isinstance(K_result, list)
        K: Array = K_result
        if self._ivar._nugget > 0:
            K = K + self._ivar._nugget * bkd.eye(self._n_total)
        self._ivar._K = K

        # Compute P matrix via multi-fidelity Monte Carlo
        P = self._compute_P_multifidelity_mc(kernel)
        self._ivar._P = P

        # Reset Cholesky and re-add existing pivots
        from pyapprox.util.linalg.incremental_cholesky import (
            IncrementalCholeskyFactorization,
        )

        self._ivar._cholesky = IncrementalCholeskyFactorization(K, bkd)
        for idx in self._ivar._selected_indices:
            self._ivar._cholesky.add_pivot(idx)
        self._ivar._kernel = kernel  # type: ignore[assignment]

    def set_P(self, P: Array) -> None:
        """Directly set the P matrix.

        Parameters
        ----------
        P : Array
            P matrix of shape (n_total, n_total).
        """
        self._ivar.set_P(P)

    def _compute_P_multifidelity_mc(
        self, kernel: MultiOutputKernelProtocol[Array]
    ) -> Array:
        """Compute P matrix via multi-fidelity Monte Carlo.

        Only evaluates quadrature at the highest-fidelity output
        (last output), following the multi-fidelity convention where
        we minimize the integrated variance of the highest-fidelity
        model.

        Parameters
        ----------
        kernel : MultiOutputKernelProtocol[Array]
            Kernel accepting List[Array] inputs.

        Returns
        -------
        P : Array
            P matrix of shape (n_total, n_total).
        """
        bkd = self._bkd
        nvars = self._candidates_list[0].shape[0]

        # Generate quadrature points for HF output only
        np.random.seed(42)
        hf_quad_pts = bkd.asarray(np.random.rand(nvars, self._nquad_samples).tolist())

        # Build quad list: empty arrays for LF outputs, quad pts for HF
        empty = bkd.zeros((nvars, 0))
        quad_list = [empty for _ in range(self._noutputs - 1)] + [hf_quad_pts]

        # K(quad, candidates): stacked format (not block_format)
        K_qc_result = kernel(quad_list, self._candidates_list)
        assert not isinstance(K_qc_result, list)
        K_qc: Array = K_qc_result

        # P_ij = (1/nquad) sum_m K(z_m, x_i) K(z_m, x_j)
        P: Array = (K_qc.T @ K_qc) / self._nquad_samples
        return P

    def set_initial_pivots(self, pivot_indices: list[int]) -> None:
        """Pre-seed selected indices.

        Parameters
        ----------
        pivot_indices : list[int]
            Global indices (into concatenated candidates) to pre-select.
        """
        self._ivar.set_initial_pivots(pivot_indices)

    def select_samples(self, nsamples: int) -> List[Array]:
        """Select new samples via cost-weighted greedy IVAR.

        At each step, computes the IVAR objective, converts to
        cost-weighted priority (variance reduction per unit cost),
        and selects the candidate with highest priority.

        Parameters
        ----------
        nsamples : int
            Number of new samples to select.

        Returns
        -------
        samples_list : List[Array]
            Selected samples per output, each of shape (nvars, n_selected_i).
            Outputs with no selected samples get shape (nvars, 0).
        """
        if self._ivar._K is None or self._ivar._cholesky is None:
            raise ValueError("Must call set_kernel() before select_samples()")
        if self._ivar._P is None:
            raise ValueError("P matrix not set.")

        bkd = self._bkd
        new_pivots: list[int] = []

        # Bootstrap baseline from initial pivots if needed
        if self._ivar._selected_indices and not self._ivar._best_obj_vals:
            idx = bkd.asarray(self._ivar._selected_indices)
            Kmat = self._ivar._K[idx, :][:, idx]
            Pmat = self._ivar._P[idx, :][:, idx]
            obj = -bkd.trace(bkd.solve(Kmat, Pmat))
            self._ivar._best_obj_vals.append(
                float(bkd.to_numpy(bkd.reshape(obj, (1,)))[0])
            )

        for _ in range(nsamples):
            # Get raw objective values from IVAR
            obj_vals = self._ivar._compute_objective()

            # Convert to cost-weighted priority
            priorities = self._objective_to_priority(obj_vals)

            # Select candidate with highest priority
            best = bkd.to_int(bkd.argmax(priorities))
            new_pivots.append(best)
            self._ivar._selected_indices.append(best)
            self._ivar._cholesky.add_pivot(best)

            # Track objective value on the base IVAR
            self._ivar._best_obj_vals.append(bkd.to_float(obj_vals[best]))

        # Map global pivot indices to per-output arrays
        return self._partition_samples(new_pivots)

    def _objective_to_priority(self, obj_vals: Array) -> Array:
        """Convert raw IVAR objective to cost-weighted priority.

        Matches legacy ``_priority_from_objective``:
        - First selection (no baseline): ``priority = vals`` (lower vals
          = more negative = better, select argmax which picks least
          negative, but the first-step case uses raw IVAR objective
          directly).
        - Subsequent: ``priority = (-vals + prev_best) / cost``.

        Parameters
        ----------
        obj_vals : Array
            IVAR objective values, shape (n_total,). Lower is better.

        Returns
        -------
        priorities : Array
            Cost-weighted priorities, shape (n_total,). Higher = better.
        """
        bkd = self._bkd

        if len(self._ivar._best_obj_vals) == 0:
            # First selection: match legacy which returns vals directly
            # (argmax picks the least-negative = best first candidate)
            return obj_vals

        prev_best = self._ivar._best_obj_vals[-1]
        priorities = bkd.full((self._n_total,), float("-inf"))

        # Cost-weighted variance reduction for active candidates
        obj_np = bkd.to_numpy(obj_vals)
        costs_np = bkd.to_numpy(self._costs)
        for ii in range(self._n_total):
            if not np.isinf(obj_np[ii]):
                cost = costs_np[self._output_ids[ii]]
                priorities[ii] = bkd.asarray((-obj_np[ii] + prev_best) / cost)

        return priorities

    def _partition_samples(self, pivot_indices: list[int]) -> List[Array]:
        """Map global pivot indices to per-output sample arrays.

        Parameters
        ----------
        pivot_indices : list[int]
            Global indices into concatenated candidates.

        Returns
        -------
        samples_list : List[Array]
            Per-output samples, each of shape (nvars, n_selected_i).
        """
        bkd = self._bkd
        nvars = self._candidates_list[0].shape[0]
        candidates_concat = bkd.hstack(self._candidates_list)

        per_output: list[list[int]] = [[] for _ in range(self._noutputs)]
        for idx in pivot_indices:
            output_id = self._output_ids[idx]
            per_output[output_id].append(idx)

        result = []
        for output_id in range(self._noutputs):
            if len(per_output[output_id]) == 0:
                result.append(bkd.zeros((nvars, 0)))
            else:
                idx_arr = bkd.asarray(per_output[output_id])
                result.append(candidates_concat[:, idx_arr])
        return result

    def best_obj_vals(self) -> list[float]:
        """Return tracked best objective values per step."""
        return list(self._ivar._best_obj_vals)

    def selected_indices(self) -> list[int]:
        """Return all selected global indices."""
        return self._ivar.selected_indices()
