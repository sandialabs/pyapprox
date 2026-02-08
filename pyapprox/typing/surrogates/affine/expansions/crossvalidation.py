"""Cross-validation functions for least-squares basis expansions.

Provides fast leave-one-out (LOO) and leave-many-out (LMO) cross-validation
using the hat matrix formula, avoiding repeated refitting.
"""

from typing import Generic, List, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


def get_random_k_fold_sample_indices(
    nsamples: int,
    nfolds: int,
    random: bool = True,
    bkd: Backend[Array] = None,  # type: ignore[assignment]
) -> List[Array]:
    """Generate fold indices for k-fold cross-validation.

    Partitions sample indices into nfolds groups. Each group contains
    approximately nsamples // nfolds samples.

    Parameters
    ----------
    nsamples : int
        Total number of samples.
    nfolds : int
        Number of folds.
    random : bool
        If True, randomly permute indices before splitting.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    list[Array]
        List of nfolds arrays, each containing sample indices for one fold.
    """
    sample_indices = bkd.arange(nsamples, dtype=int)
    if random:
        sample_indices = bkd.asarray(
            np.random.permutation(sample_indices), dtype=int
        )

    fold_sample_indices: List[Array] = [
        bkd.empty(0, dtype=int) for _ in range(nfolds)
    ]
    nn = 0
    while nn < nsamples:
        for jj in range(nfolds):
            fold_sample_indices[jj] = bkd.hstack(
                [fold_sample_indices[jj], sample_indices[nn]]
            )
            nn += 1
            if nn >= nsamples:
                break
    if bkd.unique(bkd.hstack(fold_sample_indices)).shape[0] != nsamples:
        raise RuntimeError("Fold indices do not cover all samples.")
    return fold_sample_indices


def leave_one_out_lsq_cross_validation(
    basis_mat: Array,
    values: Array,
    alpha: float = 0,
    coef: Optional[Array] = None,
    bkd: Backend[Array] = None,  # type: ignore[assignment]
) -> Tuple[Array, Array, Array]:
    r"""Leave-one-out cross-validation using the hat matrix formula.

    For least-squares regression with optional ridge regularization
    (alpha > 0), the LOO CV errors can be computed without refitting:

    .. math::

        e_i = \frac{r_i}{1 - h_i}

    where :math:`r_i` are the training residuals and
    :math:`h_i = x_i^T (X^T X + \alpha I)^{-1} x_i` are the leverage
    values (diagonal of the hat matrix).

    Parameters
    ----------
    basis_mat : Array
        Basis (design) matrix. Shape: (nsamples, nterms).
        Requires nsamples > nterms + 2.
    values : Array
        Target values. Shape: (nsamples, nqoi). Must be 2D.
    alpha : float
        Ridge regularization parameter. Default: 0 (ordinary LS).
    coef : Array, optional
        Precomputed coefficients. Shape: (nterms, nqoi).
        If None, computed from the data.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    cv_errors : Array
        LOO errors. Shape: (nsamples, nqoi)
    cv_score : Array
        RMS of LOO errors. Shape: (nqoi,)
    coef : Array
        Fitted coefficients. Shape: (nterms, nqoi)
    """
    assert values.ndim == 2
    assert basis_mat.shape[0] > basis_mat.shape[1] + 2
    gram_mat = basis_mat.T @ basis_mat
    gram_mat = gram_mat + alpha * bkd.eye(gram_mat.shape[0])
    H_mat = basis_mat @ (bkd.inv(gram_mat) @ basis_mat.T)
    H_diag = bkd.diag(H_mat)
    if coef is None:
        coef = bkd.lstsq(gram_mat, basis_mat.T @ values)
    assert coef.ndim == 2
    residuals = basis_mat @ coef - values
    cv_errors = residuals / (1 - H_diag[:, None])
    cv_score = bkd.sqrt(bkd.sum(cv_errors**2, axis=0) / basis_mat.shape[0])
    return cv_errors, cv_score, coef


def leave_many_out_lsq_cross_validation(
    basis_mat: Array,
    values: Array,
    fold_sample_indices: List[Array],
    alpha: float = 0,
    coef: Optional[Array] = None,
    bkd: Backend[Array] = None,  # type: ignore[assignment]
) -> Tuple[List[Array], Array, Array]:
    r"""Leave-many-out cross-validation using block hat matrix formula.

    For each fold k, the CV errors are computed without refitting using:

    .. math::

        e_k = (I - X_k (X^T X + \alpha I)^{-1} X_k^T)^{-1} r_k

    where :math:`X_k` is the sub-matrix of rows in fold k and :math:`r_k`
    are the corresponding training residuals.

    Parameters
    ----------
    basis_mat : Array
        Basis (design) matrix. Shape: (nsamples, nterms).
    values : Array
        Target values. Shape: (nsamples, nqoi). Must be 2D.
    fold_sample_indices : list[Array]
        Indices for each fold, as returned by get_random_k_fold_sample_indices.
    alpha : float
        Ridge regularization parameter. Default: 0.
    coef : Array, optional
        Precomputed coefficients. Shape: (nterms, nqoi).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    cv_errors : list[Array]
        Per-fold CV errors. Each element has shape (nfold_samples, nqoi).
    cv_score : Array
        RMS of all CV errors. Shape: scalar or (nqoi,).
    coef : Array
        Fitted coefficients. Shape: (nterms, nqoi)
    """
    nfolds = len(fold_sample_indices)
    nsamples = basis_mat.shape[0]
    gram_mat = basis_mat.T @ basis_mat
    gram_mat = gram_mat + alpha * bkd.eye(gram_mat.shape[0])
    if coef is None:
        coef = bkd.lstsq(gram_mat, basis_mat.T @ values)
    residuals = basis_mat @ coef - values
    gram_mat_inv = bkd.inv(gram_mat)

    cv_errors: List[Array] = []
    cv_score = bkd.zeros(values.shape[1])
    for kk in range(nfolds):
        indices_kk = fold_sample_indices[kk]
        nvalidation_samples_kk = indices_kk.shape[0]
        assert nsamples - nvalidation_samples_kk >= basis_mat.shape[1]
        basis_mat_kk = basis_mat[indices_kk, :]
        residuals_kk = residuals[indices_kk, :]

        H_mat = bkd.eye(nvalidation_samples_kk) - basis_mat_kk @ (
            gram_mat_inv @ basis_mat_kk.T
        )
        H_mat_inv = bkd.inv(H_mat)
        cv_errors_kk = H_mat_inv @ residuals_kk
        cv_errors.append(cv_errors_kk)
        cv_score = cv_score + bkd.sum(cv_errors_kk**2, axis=0)

    cv_score = bkd.sqrt(cv_score / nsamples)
    return cv_errors, cv_score, coef


def get_cross_validation_rsquared(
    cv_score: Array,
    train_vals: Array,
    bkd: Backend[Array] = None,  # type: ignore[assignment]
) -> Array:
    r"""Compute R-squared coefficient of variation from CV score.

    .. math::

        R^2 = 1 - \frac{CV^2}{\text{Var}[Y]}

    where CV is the cross-validation RMSE and Var[Y] is the variance
    of the training values.

    Parameters
    ----------
    cv_score : Array
        Cross-validation RMSE score.
    train_vals : Array
        Training values (for computing variance).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        R-squared value.
    """
    denom = bkd.std(train_vals)
    rsq = 1 - (cv_score / denom) ** 2
    return rsq
