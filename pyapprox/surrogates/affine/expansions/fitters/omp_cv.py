"""OMP term selection via cross-validation.

Runs OMP once to obtain the greedy selection path, then evaluates
leave-one-out (LOO) or leave-many-out (LMO) cross-validation on each
nested least-squares system to select the best truncation.
"""

from typing import Generic, List, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.surrogates.affine.expansions.fitters.omp import OMPFitter
from pyapprox.surrogates.affine.expansions.crossvalidation import (
    leave_one_out_lsq_cross_validation,
    leave_many_out_lsq_cross_validation,
    get_random_k_fold_sample_indices,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)


class OMPCVFitter(Generic[Array]):
    """Select best OMP truncation via LOO/LMO cross-validation.

    Runs OMP once to obtain the full greedy selection path (ordered list
    of basis terms). Then for each candidate truncation k = 1, 2, ...,
    n_selected:

    1. Forms the (nsamples, k) sub-basis matrix from the first k columns
       selected by OMP.
    2. Applies the fast LOO or LMO formula to the resulting least-squares
       system.
    3. Records the CV score.

    Returns the truncation with smallest CV score.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_nonzeros : int
        Maximum terms for OMP (upper bound on search).
    rtol : float
        OMP residual tolerance. Default: 0.0 (run full path).
    alpha : float
        Ridge regularization for CV formula. Default: 0.0.
    nfolds : int, optional
        Number of folds for LMO CV. If None, uses LOO CV.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_nonzeros: int,
        rtol: float = 0.0,
        alpha: float = 0.0,
        nfolds: Optional[int] = None,
    ):
        if max_nonzeros < 1:
            raise ValueError(f"max_nonzeros must be >= 1, got {max_nonzeros}")
        self._bkd = bkd
        self._max_nonzeros = max_nonzeros
        self._rtol = rtol
        self._alpha = alpha
        self._nfolds = nfolds

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def max_nonzeros(self) -> int:
        """Return maximum number of non-zero coefficients."""
        return self._max_nonzeros

    def rtol(self) -> float:
        """Return OMP residual tolerance."""
        return self._rtol

    def alpha(self) -> float:
        """Return ridge regularization parameter."""
        return self._alpha

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> CVSelectionResult[Array, BasisExpansionProtocol[Array]]:
        """Run OMP then select best truncation via CV.

        Parameters
        ----------
        expansion : BasisExpansionProtocol[Array]
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
            Only nqoi=1 supported.

        Returns
        -------
        CVSelectionResult
            Result with best expansion and CV diagnostics.

        Raises
        ------
        ValueError
            If nqoi > 1.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        if values.shape[0] != 1:
            raise ValueError(
                f"OMPCVFitter only supports nqoi=1, got {values.shape[0]}"
            )

        nsamples = samples.shape[1]

        # Step 1: Run OMP to get the full selection path
        omp_fitter = OMPFitter(
            bkd,
            max_nonzeros=self._max_nonzeros,
            rtol=self._rtol,
        )
        omp_result = omp_fitter.fit(expansion, samples, values)
        selection_order = omp_result.selection_order()
        n_selected = omp_result.n_nonzero()

        # Get the full basis matrix
        full_basis_mat = expansion.basis_matrix(samples)  # (nsamples, nterms)
        nterms = full_basis_mat.shape[1]
        values_T = values.T  # (nsamples, 1)

        # Pre-generate fold indices for LMO
        fold_indices = None
        if self._nfolds is not None:
            fold_indices = get_random_k_fold_sample_indices(
                nsamples, self._nfolds, random=True, bkd=bkd
            )

        # Step 2: For each truncation k, evaluate CV score
        cv_scores_list: List[float] = []
        all_params: List[Array] = []
        candidate_labels: List[int] = list(range(1, n_selected + 1))

        for k in range(1, n_selected + 1):
            # Sub-basis matrix from first k selected columns
            active_indices = [int(selection_order[i]) for i in range(k)]
            sub_basis_mat = full_basis_mat[:, active_indices]

            # Check if system is overdetermined enough for CV
            if nsamples <= k + 2:
                cv_scores_list.append(float("inf"))
                all_params.append(bkd.zeros((nterms, 1)))
                continue

            # Compute CV score using fast formula
            if self._nfolds is None:
                _, cv_score, sub_coef = leave_one_out_lsq_cross_validation(
                    sub_basis_mat, values_T, self._alpha, bkd=bkd
                )
            else:
                _, cv_score, sub_coef = leave_many_out_lsq_cross_validation(
                    sub_basis_mat, values_T, fold_indices, self._alpha,
                    bkd=bkd
                )

            cv_scores_list.append(float(cv_score[0]))

            # Build full coefficient array from active coefficients
            full_coef = bkd.zeros((nterms, 1))
            for ii, idx in enumerate(active_indices):
                full_coef[idx, 0] = sub_coef[ii, 0]
            all_params.append(full_coef)

        # Step 3: Find best truncation
        cv_scores_array = bkd.asarray(cv_scores_list)
        best_index = int(bkd.argmin(cv_scores_array))
        best_k = candidate_labels[best_index]
        best_params = all_params[best_index]

        # Create fitted expansion
        fitted_expansion = expansion.with_params(best_params)

        return CVSelectionResult(
            surrogate=fitted_expansion,
            params=best_params,
            cv_scores=cv_scores_array,
            candidate_labels=candidate_labels,
            best_index=best_index,
            best_label=best_k,
            all_params=all_params,
        )
