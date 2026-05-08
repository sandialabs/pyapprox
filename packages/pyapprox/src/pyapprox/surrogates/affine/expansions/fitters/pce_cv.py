"""PCE level selection via cross-validation.

Selects the best index set level for a PCE by evaluating leave-one-out (LOO)
or leave-many-out (LMO) cross-validation scores for each candidate level.
The index sets are produced by an IndexSequenceProtocol object that maps
integer levels to multi-index arrays.
"""

from typing import Generic, List, Optional

from pyapprox.surrogates.affine.expansions.crossvalidation import (
    get_random_k_fold_sample_indices,
    leave_many_out_lsq_cross_validation,
    leave_one_out_lsq_cross_validation,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)
from pyapprox.surrogates.affine.protocols import (
    BasisExpansionProtocol,
    IndexSequenceProtocol,
    MultiIndexBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class PCEDegreeSelectionFitter(Generic[Array]):
    """Select best PCE index set via LOO/LMO cross-validation.

    For each candidate level, uses an ``IndexSequenceProtocol`` to produce
    a multi-index set, computes the basis matrix, fits via least-squares
    (with optional ridge regularization), and evaluates the CV score using
    the fast LOO or LMO formula. Returns the level with smallest CV score.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    levels : list[int]
        Candidate levels to evaluate.
    index_sequence : IndexSequenceProtocol[Array]
        Object mapping integer level to multi-index set of shape
        ``(nvars, nterms)``.
    alpha : float
        Ridge regularization parameter. Default: 0.0 (ordinary LS).
    nfolds : int, optional
        Number of folds for LMO CV. If None, uses LOO CV.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        levels: List[int],
        index_sequence: IndexSequenceProtocol[Array],
        alpha: float = 0.0,
        nfolds: Optional[int] = None,
    ):
        if len(levels) == 0:
            raise ValueError("levels must be non-empty")
        if not isinstance(index_sequence, IndexSequenceProtocol):
            raise TypeError(
                "index_sequence must satisfy IndexSequenceProtocol, "
                f"got {type(index_sequence).__name__}"
            )
        self._bkd = bkd
        self._levels = list(levels)
        self._index_sequence = index_sequence
        self._alpha = alpha
        self._nfolds = nfolds

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def levels(self) -> List[int]:
        """Return candidate levels."""
        return self._levels

    def index_sequence(self) -> IndexSequenceProtocol[Array]:
        """Return the index sequence object."""
        return self._index_sequence

    def alpha(self) -> float:
        """Return ridge regularization parameter."""
        return self._alpha

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> CVSelectionResult[Array, BasisExpansionProtocol[Array]]:
        """Select best level and fit expansion.

        For each candidate level:
        1. Compute indices via index_sequence(level)
        2. Set indices on the expansion's basis
        3. Evaluate basis matrix
        4. Compute CV score via fast LOO/LMO formula
        5. Record coefficients

        The level with smallest CV score is selected and the expansion
        is returned with those coefficients.

        Parameters
        ----------
        expansion : BasisExpansionProtocol[Array]
            Must have basis with set_indices() (MultiIndexBasisProtocol).
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (nqoi, nsamples) or (nsamples,) for nqoi=1.

        Returns
        -------
        CVSelectionResult
            Result with best expansion and CV diagnostics.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        nqoi = values.shape[0]
        nsamples = samples.shape[1]

        # Values transposed for CV functions: (nsamples, nqoi)
        values_T = values.T

        # Get basis (must support set_indices)
        basis = expansion.get_basis()
        if not isinstance(basis, MultiIndexBasisProtocol):
            raise TypeError(
                "expansion.get_basis() must satisfy MultiIndexBasisProtocol"
            )

        # Pre-generate fold indices for LMO (same folds for all levels)
        fold_indices = None
        if self._nfolds is not None:
            fold_indices = get_random_k_fold_sample_indices(
                nsamples, self._nfolds, random=True, bkd=bkd
            )

        cv_scores_list = []
        all_params = []

        for level in self._levels:
            # Compute indices for this level
            indices = self._index_sequence(level)
            nterms = indices.shape[1]

            # Check if system is overdetermined enough for CV
            if nsamples <= nterms + 2:
                cv_scores_list.append(float("inf"))
                all_params.append(bkd.zeros((nterms, nqoi)))
                continue

            # Set indices on basis and compute basis matrix
            basis.set_indices(indices)
            basis_mat = expansion.basis_matrix(samples)  # (nsamples, nterms)

            # Compute CV score
            if self._nfolds is None:
                # LOO
                _, cv_score, coef = leave_one_out_lsq_cross_validation(
                    basis_mat, values_T, self._alpha, bkd=bkd
                )
            else:
                # LMO
                _, cv_score, coef = leave_many_out_lsq_cross_validation(
                    basis_mat, values_T, fold_indices, self._alpha, bkd=bkd
                )

            # cv_score shape: (nqoi,) - take mean across QoIs
            cv_scores_list.append(bkd.to_float(bkd.mean(cv_score)))
            all_params.append(coef)

        # Find best level
        cv_scores_array = bkd.asarray(cv_scores_list)
        best_index = bkd.to_int(bkd.argmin(cv_scores_array))
        best_level = self._levels[best_index]
        best_params = all_params[best_index]

        # Set the best indices and create fitted expansion
        best_indices = self._index_sequence(best_level)
        basis.set_indices(best_indices)
        fitted_expansion = expansion.with_params(best_params)

        return CVSelectionResult(
            surrogate=fitted_expansion,
            params=best_params,
            cv_scores=cv_scores_array,
            candidate_labels=self._levels,
            best_index=best_index,
            best_label=best_level,
            all_params=all_params,
        )
