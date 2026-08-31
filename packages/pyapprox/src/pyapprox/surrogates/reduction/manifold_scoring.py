"""Shared scoring for greedy and adaptive manifold construction.

Two operations are needed by any manifold whose basis is chosen from
snapshot data:

1. Score a *trial* configuration cheaply in SVD coordinates -- the
   objective :math:`J'` of Schwerdtner & Peherstorfer (2024), Eq. (12):
   project the centered data onto the active subspace, fit the optimal
   correction :math:`W` to the orthogonal residual, and return the
   corrected residual energy plus the regularization penalty. Because
   the data is expressed in the orthonormal left-singular basis,
   projection is row selection and the regression scales with the number
   of snapshots, not the ambient dimension.

2. Fit the final correction weight matrix :math:`W` in the ambient space
   (Eq. 6) once the basis is fixed.

:class:`ManifoldScorer` holds the precomputed SVD coordinates and the
regularization and exposes both, so a greedy and an adaptive encoder
share one implementation.

The scorer is told which monomials to use by a multi-index set rather
than by the parameters that generated one. A trial subspace of lower
dimension takes the restriction of that set to its leading variables
(:func:`restrict_indices_to_leading_vars`), which for a downward-closed
set is exactly the index set those variables support. Selection and the
final fit therefore use the same terms by construction, and an
anisotropic index set works wherever a degree band does.
"""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pyapprox.surrogates.affine.indices import (
    restrict_indices_to_leading_vars,
)
from pyapprox.surrogates.reduction.feature_maps import FeatureMap
from pyapprox.util.backends.protocols import Array, Backend


def center_and_decompose(
    snapshots: Array,
    bkd: Backend[Array],
    center: bool = True,
    precomputed_svd: Optional[
        Tuple[Array, Array, Array, Array]
    ] = None,
    mean: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Center the snapshot matrix and return its (cached or fresh) thin SVD.

    The thin SVD is the dominant cost of manifold construction and is
    shared by every method and reduced dimension at a given training set,
    so callers may compute it once and inject it via
    ``precomputed_svd = (phi, svals, psi_t, mean)`` to skip recomputation.
    The injected ``mean`` is used as-is, so the caller is responsible for
    centering consistently with ``center``.

    Snapshot selection is handled by the caller: pass a non-redundant
    column subset as ``snapshots`` and the basis is built from it like any
    other data. One subtlety then arises with centering -- a deliberately
    extremal subset (high-leverage corners rather than typical snapshots)
    has a mean that is a biased estimate of the true snapshot centroid,
    which shifts the affine manifold off the data center. To avoid this,
    compute the mean over the full pre-selection data and pass it via
    ``mean``; the subset is then centered by the unbiased mean before the
    SVD. Leave ``mean`` as ``None`` to center by the mean of ``snapshots``
    itself, the usual centered PCA.

    Parameters
    ----------
    snapshots : Array
        Shape: (nstates, nsnapshots). Columns are data points.
    bkd : Backend
        Computational backend.
    center : bool
        Subtract the mean before the SVD.
    precomputed_svd : tuple, optional
        ``(phi, svals, psi_t, mean)`` to reuse instead of recomputing.
    mean : Array, optional
        Mean used for centering. Shape: (nstates,) or (nstates, 1).

    Returns
    -------
    centered : Array
        ``snapshots - mean``. Shape: (nstates, nsnapshots).
    mean : Array
        Shape: (nstates, 1).
    phi, svals, psi_t : Array
        Thin SVD factors of ``centered``, so that
        ``centered = phi diag(svals) psi_t``.

    Notes
    -----
    TODO: this duplicates the centering and thin SVD that
    :class:`~pyapprox.surrogates.kle.DataDrivenKLE` performs. Collapsing
    the two is blocked on the right singular factor: the scorer needs
    ``psi_t`` to build snapshot coordinates, and the KLE exposes only the
    left factor and the spectrum. Retargeting onto it would also supply
    the metric this function lacks. Doing so requires checking the
    conventions that were free to drift while the right factor was
    discarded -- per-column sign flips break ``A = U S V^T`` unless
    applied to both factors, and the eigenvalue scaling and truncation
    must match across the two.
    """
    if precomputed_svd is not None:
        phi, svals, psi_t, mean = precomputed_svd
        return snapshots - mean, mean, phi, svals, psi_t

    if mean is not None:
        mean = bkd.reshape(mean, (mean.shape[0], 1))
    elif center:
        mean = bkd.reshape(
            bkd.mean(snapshots, axis=1), (snapshots.shape[0], 1)
        )
    else:
        mean = bkd.zeros((snapshots.shape[0], 1))
    centered = snapshots - mean
    phi, svals, psi_t = bkd.svd(centered, full_matrices=False)
    return centered, mean, phi, svals, psi_t


class ManifoldScorer(Generic[Array]):
    """Score manifold configurations in precomputed SVD coordinates.

    Parameters
    ----------
    coords : Array
        Shape: (rank, nsnapshots). Coordinates of every centered snapshot
        in the full left-singular basis:
        ``coords[a, n] = phi_a^T centered_n = svals_a * psi_t[a, n]``.
    gamma : float
        Tikhonov regularization for the correction least-squares.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, coords: Array, gamma: float, bkd: Backend[Array]
    ) -> None:
        self._coords = coords
        self._gamma = gamma
        self._bkd = bkd
        # Per-row energy ||coords[i, :]||^2, precomputed once. In SVD
        # coordinates a row of (Sigma Psi^T) has norm sigma_i, so this
        # equals svals**2; computing it directly keeps the scorer agnostic
        # to how coords was formed. Used to get ||R||^2 = total - sum over
        # active rows without ever forming the residual matrix.
        self._row_energy = bkd.sum(coords * coords, axis=1)
        self._total_energy = bkd.sum(self._row_energy)
        # Memo of candidate-column index patterns, keyed by the number of
        # selected dimensions; identical across candidates at a step.
        self._candidate_index_cache: Dict[int, npt.NDArray[np.int64]] = {}
        # Lazily-filled cache of element-wise powers of every coordinate
        # row, z_i^k for integer k. These depend only on coords, not on the
        # step or candidate, so computing them once avoids recomputing the
        # exponentiation per candidate per step. Power 1 is coords itself.
        self._power_cache: Dict[int, Array] = {1: coords}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def coords(self) -> Array:
        """Snapshot coordinates in the left-singular basis, (rank, N)."""
        return self._coords

    def gamma(self) -> float:
        """Tikhonov regularization used for the correction fit."""
        return self._gamma

    def _power(self, k: int) -> Array:
        """Element-wise ``coords ** k`` for every row, cached by ``k``."""
        if k not in self._power_cache:
            self._power_cache[k] = self._coords**k
        return self._power_cache[k]

    def objective(
        self, active_rows: Sequence[int], feature_map: FeatureMap[Array]
    ) -> float:
        """Corrected-residual energy for a trial subspace (Eq. 12).

        Parameters
        ----------
        active_rows : sequence of int
            Indices (into the singular basis) of the directions spanning
            the trial subspace V. The encoded coordinates are these rows
            of ``coords``; the residual target is all other rows.
        feature_map : FeatureMap
            Maps encoded coordinates of shape ``(len(active_rows), N)`` to
            features ``(p, N)``. If the map produces no terms the
            objective is just the residual energy, no correction being
            possible.

        Returns
        -------
        float
            ``||residual - W feats||_F^2 + gamma ||W||_F^2`` at the
            optimal W.

        Notes
        -----
        Evaluated by the exact closed form

        .. math::

            \\|R - WF\\|^2 + \\gamma\\|W\\|^2
            = \\|R\\|^2 - 2\\,\\mathrm{tr}(WG^T)
              + \\mathrm{tr}(W(FF^T)W^T) + \\gamma\\|W\\|^2,

        with :math:`W = GM^{-1}`, :math:`M = FF^T + \\gamma I`,
        :math:`G = RF^T`. No ``(c, N)`` corrected-residual matrix is
        materialized, and :math:`\\|R\\|^2` comes from the precomputed row
        energies, so the only N-sized work is the single contraction
        ``F coords^T``.
        """
        rows = list(active_rows)
        feats = feature_map(self._coords[rows, :])
        norm_r2 = self._residual_energy(rows)
        if feats.shape[0] == 0:
            return float(self._bkd.to_numpy(norm_r2))
        full_cross = self._bkd.dot(self._coords, feats.T)
        feature_gram = self._bkd.dot(feats, feats.T)
        return self._objective_from_cross(
            full_cross, feature_gram, rows, norm_r2
        )

    def _residual_energy(self, active_rows: Sequence[int]) -> Array:
        """||R||^2 = total energy minus the active rows' energy."""
        if len(active_rows) == 0:
            return self._total_energy
        active_energy = self._bkd.sum(self._row_energy[list(active_rows)])
        return self._total_energy - active_energy

    def _objective_from_cross(
        self,
        full_cross: Array,
        feature_gram: Array,
        active_rows: Sequence[int],
        norm_r2: Array,
    ) -> float:
        """Closed-form objective from ``coords F^T`` and ``FF^T``.

        Splitting the N-sized contraction and the Gram out lets the greedy
        sweep reuse the selected-set blocks across candidates.
        """
        bkd = self._bkd
        active = set(active_rows)
        keep = [a for a in range(full_cross.shape[0]) if a not in active]
        g = full_cross[keep, :]

        gram = feature_gram + self._gamma * bkd.eye(feature_gram.shape[0])
        coef = bkd.solve(gram, g.T)
        w = coef.T

        term_cross = bkd.sum(w * g)
        term_quad = bkd.sum(bkd.dot(w, feature_gram) * w)
        penalty = self._gamma * bkd.sum(coef * coef)
        return float(
            bkd.to_numpy(norm_r2 - 2.0 * term_cross + term_quad + penalty)
        )

    def greedy_scores(
        self,
        selected: Sequence[int],
        candidates: Sequence[int],
        indices: Array,
    ) -> List[float]:
        """Objective of ``selected + [j]`` for each candidate ``j``.

        Reuses the selected-set work across candidates: the static feature
        block (monomials among ``selected``), its N-sized contraction
        ``coords F_static^T``, and its Gram block are computed once; per
        candidate only the columns involving ``j`` are formed. Returns one
        objective per candidate, matching :meth:`objective` on
        ``selected + [j]``.

        Parameters
        ----------
        selected : sequence of int
            Currently selected singular-vector rows, the fixed subspace.
        candidates : sequence of int
            Candidate rows to score, each extending ``selected`` by one.
        indices : Array
            Shape: (nreduced, p). The multi-index set of the feature map
            the final fit will use. A trial of dimension ``a`` uses this
            set restricted to its first ``a`` variables.

        Returns
        -------
        list of float
            ``scores[i]`` is the objective for
            ``selected + [candidates[i]]``.
        """
        bkd = self._bkd
        selected = list(selected)
        candidates = list(candidates)
        nselected = len(selected)

        # Static block: monomials among the selected dims only. Candidate-
        # independent, so computed once and reused across every chunk. At
        # the first step there is no static block.
        f_static = None
        if nselected >= 1:
            static_indices = np.asarray(
                bkd.to_numpy(self._trial_indices(indices, nselected))
            ).astype(np.int64)
            if static_indices.shape[1] > 0:
                f_static = self._eval_monomials(selected, static_indices)

        if f_static is not None:
            cross_static = bkd.dot(self._coords, f_static.T)
            gram_static = bkd.dot(f_static, f_static.T)
        else:
            cross_static = None
            gram_static = None

        # The batched per-candidate work materializes an (ncands, p_c, N)
        # feature tensor. For a high-degree, high-dimension selection both
        # p_c and N are large, so scoring every candidate at once peaks at
        # many gigabytes. Chunking bounds peak memory; results are
        # identical since each candidate is scored independently, the
        # batched speedup is retained within a chunk, and the static block
        # above is computed only once.
        chunk = self._candidate_chunk_size(
            indices, nselected, len(candidates)
        )
        scores: List[float] = []
        for start in range(0, len(candidates), chunk):
            scores.extend(
                self._greedy_scores_chunk(
                    selected,
                    candidates[start : start + chunk],
                    indices,
                    f_static,
                    cross_static,
                    gram_static,
                )
            )
        return scores

    def _greedy_scores_chunk(
        self,
        selected: List[int],
        candidates: List[int],
        indices: Array,
        f_static: Optional[Array],
        cross_static: Optional[Array],
        gram_static: Optional[Array],
    ) -> List[float]:
        """Score one chunk of candidates, the batched core."""
        bkd = self._bkd
        f_cand_stack = self._candidate_feature_stack(
            selected, candidates, indices
        )
        if f_cand_stack.shape[1] == 0:
            # No correction terms at all; every score is residual energy.
            return [
                float(
                    bkd.to_numpy(self._residual_energy(selected + [j]))
                )
                for j in candidates
            ]

        # The three N-sized contractions, batched over candidates:
        #   cross_cand[n]       = coords @ f_cand[n]^T     (rank, p_c)
        #   gram_cand[n]        = f_cand[n] @ f_cand[n]^T  (p_c, p_c)
        #   gram_static_cand[n] = f_static @ f_cand[n]^T   (p_s, p_c)
        # Written as matmul against the transposed stack rather than as an
        # einsum: numpy's einsum does not reach batched BLAS for a
        # contraction of this shape, and is several times slower for it.
        f_cand_t = bkd.moveaxis(f_cand_stack, -1, -2)
        cross_cand = self._coords @ f_cand_t
        gram_cand = f_cand_stack @ f_cand_t
        # The three static blocks are produced together or not at all, so
        # bind them as one value: either every one is present or the step
        # has no selected dimensions to build them from.
        static: Optional[Tuple[Array, Array, Array]] = None
        if (
            f_static is not None
            and cross_static is not None
            and gram_static is not None
        ):
            static = (cross_static, gram_static, f_static @ f_cand_t)

        # Assemble each candidate's Gram and residual-cross from the
        # batched blocks (slicing and stacking only), then solve every
        # ridge system in one batched solve.
        grams, gs, norm_r2s = [], [], []
        for position, j in enumerate(candidates):
            trial = selected + [j]
            norm_r2s.append(self._residual_energy(trial))

            if static is not None:
                cross_sel, gram_sel, gram_sel_cand = static
                full_cross = bkd.hstack([cross_sel, cross_cand[position]])
                feature_gram = bkd.vstack(
                    [
                        bkd.hstack([gram_sel, gram_sel_cand[position]]),
                        bkd.hstack(
                            [
                                gram_sel_cand[position].T,
                                gram_cand[position],
                            ]
                        ),
                    ]
                )
            else:
                full_cross = cross_cand[position]
                feature_gram = gram_cand[position]

            active = set(trial)
            keep = [a for a in range(full_cross.shape[0]) if a not in active]
            gs.append(full_cross[keep, :])
            grams.append(feature_gram)

        return self._batched_objectives(grams, gs, norm_r2s)

    # Peak-memory budget (bytes) for the per-chunk (ncands, p_c, N)
    # feature tensor and its batched intermediates. Conservative so the
    # process stays well under typical RAM; the batched speedup is
    # essentially unaffected since each chunk still scores many candidates.
    _CHUNK_MEM_BYTES = 1_000_000_000

    def _candidate_chunk_size(
        self, indices: Array, nselected: int, ncandidates: int
    ) -> int:
        """Candidates per chunk so the feature tensor fits the budget.

        ``p_c`` is the number of new-dimension-involving monomials a
        candidate adds; it grows combinatorially with ``nselected`` and the
        degree, which is what makes a high-degree map at high dimension
        expensive. Returns at least 1.
        """
        nsnapshots = self._coords.shape[1]
        p_c = self._candidate_indices(indices, nselected).shape[1]
        if p_c == 0:
            return ncandidates
        bytes_per_candidate = p_c * nsnapshots * 8
        per_chunk = max(1, self._CHUNK_MEM_BYTES // bytes_per_candidate)
        return int(min(ncandidates, per_chunk))

    def _trial_indices(self, indices: Array, nreduced: int) -> Array:
        """The feature map's index set restricted to ``nreduced`` vars."""
        return restrict_indices_to_leading_vars(indices, nreduced, self._bkd)

    def _candidate_indices(
        self, indices: Array, nselected: int
    ) -> npt.NDArray[np.int64]:
        """Index columns of a trial of size ``nselected + 1`` that involve
        the new last dimension.

        Identical for every candidate at a step, so built once and memoized
        by ``nselected``.

        Returns
        -------
        numpy.ndarray
            Shape (nselected + 1, p_c). Integer indices of those columns.
        """
        if nselected not in self._candidate_index_cache:
            trial = self._trial_indices(indices, nselected + 1)
            trial_np = np.asarray(self._bkd.to_numpy(trial)).astype(np.int64)
            involves_new = [
                c
                for c in range(trial_np.shape[1])
                if trial_np[nselected, c] > 0
            ]
            self._candidate_index_cache[nselected] = trial_np[
                :, involves_new
            ]
        return self._candidate_index_cache[nselected]

    def _candidate_feature_stack(
        self, selected: List[int], candidates: List[int], indices: Array
    ) -> Array:
        """Every candidate's feature columns as (ncands, p_c, N).

        The index pattern is shared and computed once; only the
        per-candidate coordinate rows differ. Candidates are looped rather
        than broadcast into one (n, d, p_c, N) tensor, which would
        materialize a prohibitively large intermediate.
        """
        candidate_indices = self._candidate_indices(indices, len(selected))
        return self._bkd.stack(
            [
                self._eval_monomials(selected + [j], candidate_indices)
                for j in candidates
            ],
            axis=0,
        )

    def _eval_monomials(
        self, rows: Sequence[int], index_matrix: npt.NDArray[np.int64]
    ) -> Array:
        """Monomials ``prod_d z[rows[d]] ** index_matrix[d, c]`` -> (p, N).

        Uses the cached power tables (``z_i^k`` computed once across the
        whole sweep) instead of recomputing the exponentiation per call:
        each monomial is a product of looked-up power rows. The indices
        arrive already converted to numpy, since they are read once per
        element and a backend array would sync on every read.
        """
        bkd = self._bkd
        nsnapshots = self._coords.shape[1]
        terms: List[Array] = []
        for c in range(index_matrix.shape[1]):
            term = bkd.full((nsnapshots,), 1.0)
            for d in range(len(rows)):
                power = int(index_matrix[d, c])
                if power:
                    term = term * self._power(power)[rows[d], :]
            terms.append(term)
        if not terms:
            return bkd.zeros((0, nsnapshots))
        return bkd.stack(terms, axis=0)

    def _batched_objectives(
        self, grams: List[Array], gs: List[Array], norm_r2s: List[Array]
    ) -> List[float]:
        """Objectives for many candidates via one batched ridge solve.

        ``grams[i]`` is ``FF^T`` and ``gs[i]`` is ``RF^T`` for candidate
        ``i``; all share the same p and c. Stacks them, solves
        ``(M + gamma I) coef = G^T`` for every candidate at once, then
        forms each objective by the closed form.
        """
        bkd = self._bkd
        gram_stack = bkd.stack(grams, axis=0)
        gram_stack = gram_stack + self._gamma * bkd.eye(grams[0].shape[0])
        gt_stack = bkd.stack([g.T for g in gs], axis=0)
        coef_stack = bkd.solve(gram_stack, gt_stack)

        scores: List[float] = []
        for i in range(len(grams)):
            coef = coef_stack[i]
            w = coef.T
            term_cross = bkd.sum(w * gs[i])
            term_quad = bkd.sum(bkd.dot(w, grams[i]) * w)
            penalty = self._gamma * bkd.sum(coef * coef)
            scores.append(
                float(
                    bkd.to_numpy(
                        norm_r2s[i]
                        - 2.0 * term_cross
                        + term_quad
                        + penalty
                    )
                )
            )
        return scores

    def objective_bruteforce(
        self, active_rows: Sequence[int], feature_map: FeatureMap[Array]
    ) -> float:
        """Reference objective, materializing the corrected residual.

        The direct statement of the definition that :meth:`objective`
        computes in closed form. It forms the full ``(c, N)``
        corrected-residual matrix and is therefore slower; it exists so
        the fast path can be checked against an obviously-correct one.
        """
        bkd = self._bkd
        rows = list(active_rows)
        z = self._coords[rows, :]
        active = set(rows)
        keep = [a for a in range(self._coords.shape[0]) if a not in active]
        residual = self._coords[keep, :]

        feats = feature_map(z)
        if feats.shape[0] == 0:
            return float(bkd.to_numpy(bkd.sum(residual * residual)))

        gram = bkd.dot(feats, feats.T) + self._gamma * bkd.eye(
            feats.shape[0]
        )
        coef = bkd.solve(gram, bkd.dot(feats, residual.T))
        corrected = residual - bkd.dot(coef.T, feats)
        err = bkd.sum(corrected * corrected)
        penalty = self._gamma * bkd.sum(coef * coef)
        return float(bkd.to_numpy(err + penalty))

    def fit_weights(
        self,
        centered: Array,
        basis: Array,
        feature_map: FeatureMap[Array],
        gamma: Optional[float] = None,
    ) -> Array:
        """Fit the ambient correction weight matrix W (Eq. 6).

        Minimizes ``||P_V S + W h(V^T S) - S||_F^2 + gamma ||W||_F^2``.
        The linear part ``P_V S`` cancels the in-subspace component,
        leaving the orthogonal residual ``S - P_V S`` as the target.

        Parameters
        ----------
        centered : Array
            Shape: (nstates, nsnapshots). Centered snapshots.
        basis : Array
            Shape: (nstates, nreduced). The selected orthonormal basis V.
        feature_map : FeatureMap
            Maps encoded coordinates ``(nreduced, N)`` to features
            ``(p, N)``.
        gamma : float, optional
            Overrides the scorer's gamma, as validation selection needs.

        Returns
        -------
        Array
            Shape: (nstates, p). The weight matrix W, empty when the
            feature map has no terms.
        """
        bkd = self._bkd
        g = self._gamma if gamma is None else gamma

        z = bkd.dot(basis.T, centered)
        feats = feature_map(z)
        if feats.shape[0] == 0:
            return bkd.zeros((centered.shape[0], 0))

        residual = centered - bkd.dot(basis, z)
        gram = bkd.dot(feats, feats.T) + g * bkd.eye(feats.shape[0])
        coef = bkd.solve(gram, bkd.dot(feats, residual.T))
        return coef.T

    def fit_weights_multi(
        self,
        centered: Array,
        basis: Array,
        feature_map: FeatureMap[Array],
        gammas: Sequence[float],
    ) -> List[Array]:
        """Fit W for several gammas, sharing the gamma-independent work.

        The feature evaluation, the projection residual, the Gram, and the
        cross do not depend on gamma -- only the ``+ gamma I`` and the
        solve do. Computing the shared pieces once and looping only the
        small solve avoids redoing the expensive contractions per gamma.

        Returns
        -------
        list of Array
            ``W[i]``, shape (nstates, p), for ``gammas[i]``.
        """
        bkd = self._bkd
        z = bkd.dot(basis.T, centered)
        feats = feature_map(z)
        if feats.shape[0] == 0:
            return [bkd.zeros((centered.shape[0], 0)) for _ in gammas]

        residual = centered - bkd.dot(basis, z)
        gram_base = bkd.dot(feats, feats.T)
        cross = bkd.dot(feats, residual.T)
        eye = bkd.eye(feats.shape[0])
        return [
            bkd.solve(gram_base + g * eye, cross).T for g in gammas
        ]

    def gram_condition(
        self,
        centered: Array,
        basis: Array,
        feature_map: FeatureMap[Array],
        gamma: float,
    ) -> float:
        """Condition number of the regularized Gram ``h h^T + gamma I``.

        This is the matrix actually inverted in the W fit, so its
        condition number measures the ill-conditioning of the
        least-squares problem directly: a near-collinear (dense,
        high-degree) feature map gives a huge value at small gamma, which
        is what makes an unregularized fit blow up on held-out data.
        Returns 1.0 when there are no correction terms.
        """
        bkd = self._bkd
        feats = feature_map(bkd.dot(basis.T, centered))
        if feats.shape[0] == 0:
            return 1.0
        gram = bkd.dot(feats, feats.T) + gamma * bkd.eye(feats.shape[0])
        return float(bkd.to_numpy(bkd.cond(gram)))

    def select_gamma(
        self,
        centered: Array,
        basis: Array,
        feature_map: FeatureMap[Array],
        gamma_grid: Sequence[float],
        val_centered: Array,
        return_diagnostics: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """Pick the gamma minimizing held-out reconstruction error.

        Fits all candidate gammas via :meth:`fit_weights_multi`, so the
        gamma-independent contractions are computed once, reconstructs the
        validation snapshots with each, and returns the gamma with the
        lowest error.

        Parameters
        ----------
        return_diagnostics : bool
            If True, also return a per-gamma dict with keys ``"gammas"``,
            ``"val_err"`` (validation Frobenius error) and ``"gram_cond"``
            (condition number of the regularized Gram). The condition
            number tests the ill-conditioning directly: when a dense
            high-degree feature map is near-collinear it is enormous at
            small gamma and the unregularized fit blows up out of sample.

        Returns
        -------
        best_gamma : float
            The selected gamma.
        diagnostics : dict, optional
            Returned only if ``return_diagnostics`` is True.
        """
        bkd = self._bkd
        z_val = bkd.dot(basis.T, val_centered)
        feats_val = feature_map(z_val)
        proj_val = bkd.dot(basis, z_val)

        weights_per_gamma = self.fit_weights_multi(
            centered, basis, feature_map, gamma_grid
        )

        gram_base = None
        if return_diagnostics:
            feats = feature_map(bkd.dot(basis.T, centered))
            if feats.shape[0] > 0:
                gram_base = bkd.dot(feats, feats.T)

        best_gamma = None
        best_err = None
        val_errs: List[float] = []
        gram_conds: List[float] = []
        for g, weights in zip(gamma_grid, weights_per_gamma):
            diff = proj_val + bkd.dot(weights, feats_val) - val_centered
            err = float(bkd.to_numpy(bkd.sum(diff * diff)))
            if best_err is None or err < best_err:
                best_err = err
                best_gamma = g
            if return_diagnostics:
                val_errs.append(err)
                if gram_base is not None:
                    reg = gram_base + g * bkd.eye(gram_base.shape[0])
                    gram_conds.append(float(bkd.to_numpy(bkd.cond(reg))))
                else:
                    gram_conds.append(1.0)

        if best_gamma is None:
            raise ValueError("gamma_grid must not be empty")
        if return_diagnostics:
            return best_gamma, {
                "gammas": list(gamma_grid),
                "val_err": val_errs,
                "gram_cond": gram_conds,
            }
        return best_gamma
