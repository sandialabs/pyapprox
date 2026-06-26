"""Ridge regression with LOO/LMO cross-validation for alpha selection.

Sweeps over candidate regularization strengths and selects the best
via leave-one-out (LOO) or leave-many-out (LMO) cross-validation
using the fast hat matrix formula (no refitting per fold).
"""

from typing import Generic, List, Optional, Sequence

from pyapprox.surrogates.affine.expansions.crossvalidation import (
    get_random_k_fold_sample_indices,
    leave_many_out_lsq_cross_validation,
    leave_one_out_lsq_cross_validation,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class RidgeCVFitter(Generic[Array]):
    """Select best ridge regularization strength via LOO/LMO CV.

    For each candidate alpha, the fast hat matrix formula computes
    LOO or LMO cross-validation errors without refitting. Returns
    the alpha with the smallest CV score.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    alphas : sequence of float
        Candidate regularization strengths to evaluate.
    nfolds : int, optional
        Number of folds for LMO CV. If None, uses LOO CV.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        alphas: Sequence[float],
        nfolds: Optional[int] = None,
    ):
        if len(alphas) < 1:
            raise ValueError("alphas must contain at least one value")
        self._bkd = bkd
        self._alphas = list(alphas)
        self._nfolds = nfolds

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def alphas(self) -> List[float]:
        return list(self._alphas)

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> CVSelectionResult[Array, BasisExpansionProtocol[Array]]:
        """Sweep alphas and select best via CV.

        Parameters
        ----------
        expansion : BasisExpansionProtocol[Array]
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (nqoi, nsamples).

        Returns
        -------
        CVSelectionResult
            Result with best expansion and CV diagnostics.
        """
        bkd = self._bkd

        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        nqoi = values.shape[0]
        nsamples = samples.shape[1]

        basis_mat = expansion.basis_matrix(samples)
        values_T = values.T

        fold_indices: List[Array] = []
        if self._nfolds is not None:
            fold_indices = get_random_k_fold_sample_indices(
                nsamples, self._nfolds, random=True, bkd=bkd
            )

        cv_scores_list: List[float] = []
        all_params: List[Array] = []

        for alpha in self._alphas:
            if self._nfolds is None:
                _, cv_score, coef = leave_one_out_lsq_cross_validation(
                    basis_mat, values_T, alpha, bkd=bkd
                )
            else:
                if not fold_indices:
                    raise RuntimeError("fold_indices empty with nfolds set")
                _, cv_score, coef = leave_many_out_lsq_cross_validation(
                    basis_mat, values_T, fold_indices, alpha, bkd=bkd
                )

            cv_scores_list.append(
                float(bkd.sum(cv_score)) / nqoi
            )
            all_params.append(coef)

        cv_scores_array = bkd.asarray(cv_scores_list)
        best_index = int(bkd.to_numpy(bkd.argmin(cv_scores_array)))
        best_alpha = self._alphas[best_index]
        best_params = all_params[best_index]

        fitted_expansion = expansion.with_params(best_params)

        return CVSelectionResult(
            surrogate=fitted_expansion,
            params=best_params,
            cv_scores=cv_scores_array,
            candidate_labels=self._alphas,
            best_index=best_index,
            best_label=best_alpha,
            all_params=all_params,
        )
