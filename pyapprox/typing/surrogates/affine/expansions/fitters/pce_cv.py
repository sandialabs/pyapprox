"""PCE degree selection via cross-validation.

Selects the best polynomial degree for a PCE with hyperbolic index sets
by evaluating leave-one-out (LOO) or leave-many-out (LMO) cross-validation
scores for each candidate degree.
"""

from typing import Generic, List, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    BasisExpansionProtocol,
    MultiIndexBasisProtocol,
)
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.expansions.crossvalidation import (
    leave_one_out_lsq_cross_validation,
    leave_many_out_lsq_cross_validation,
    get_random_k_fold_sample_indices,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)


class PCEDegreeSelectionFitter(Generic[Array]):
    """Select best PCE degree via LOO/LMO cross-validation.

    For each candidate degree, builds a hyperbolic index set with the given
    pnorm, computes the basis matrix, fits via least-squares (with optional
    ridge regularization), and evaluates the CV score using the fast LOO or
    LMO formula. Returns the degree with smallest CV score.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    degrees : list[int]
        Candidate polynomial degrees to evaluate.
    pnorm : float
        p-norm for hyperbolic index sets. Default: 1.0 (total degree).
    alpha : float
        Ridge regularization parameter. Default: 0.0 (ordinary LS).
    nfolds : int, optional
        Number of folds for LMO CV. If None, uses LOO CV.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        degrees: List[int],
        pnorm: float = 1.0,
        alpha: float = 0.0,
        nfolds: Optional[int] = None,
    ):
        if len(degrees) == 0:
            raise ValueError("degrees must be non-empty")
        self._bkd = bkd
        self._degrees = list(degrees)
        self._pnorm = pnorm
        self._alpha = alpha
        self._nfolds = nfolds

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def degrees(self) -> List[int]:
        """Return candidate degrees."""
        return self._degrees

    def pnorm(self) -> float:
        """Return p-norm for hyperbolic indices."""
        return self._pnorm

    def alpha(self) -> float:
        """Return ridge regularization parameter."""
        return self._alpha

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> CVSelectionResult[Array, BasisExpansionProtocol[Array]]:
        """Select best degree and fit expansion.

        For each candidate degree:
        1. Compute hyperbolic indices
        2. Set indices on the expansion's basis
        3. Evaluate basis matrix
        4. Compute CV score via fast LOO/LMO formula
        5. Record coefficients

        The degree with smallest CV score is selected and the expansion
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
        nvars = samples.shape[0]

        # Values transposed for CV functions: (nsamples, nqoi)
        values_T = values.T

        # Get basis (must support set_indices)
        basis = expansion.get_basis()
        if not isinstance(basis, MultiIndexBasisProtocol):
            raise TypeError(
                "expansion.get_basis() must satisfy MultiIndexBasisProtocol"
            )

        # Pre-generate fold indices for LMO (same folds for all degrees)
        fold_indices = None
        if self._nfolds is not None:
            fold_indices = get_random_k_fold_sample_indices(
                nsamples, self._nfolds, random=True, bkd=bkd
            )

        cv_scores_list = []
        all_params = []

        for degree in self._degrees:
            # Compute hyperbolic indices for this degree
            indices = compute_hyperbolic_indices(
                nvars, degree, self._pnorm, bkd
            )
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
            cv_scores_list.append(float(bkd.mean(cv_score)))
            all_params.append(coef)

        # Find best degree
        cv_scores_array = bkd.asarray(cv_scores_list)
        best_index = int(bkd.argmin(cv_scores_array))
        best_degree = self._degrees[best_index]
        best_params = all_params[best_index]

        # Set the best indices and create fitted expansion
        best_indices = compute_hyperbolic_indices(
            nvars, best_degree, self._pnorm, bkd
        )
        basis.set_indices(best_indices)
        fitted_expansion = expansion.with_params(best_params)

        return CVSelectionResult(
            surrogate=fitted_expansion,
            params=best_params,
            cv_scores=cv_scores_array,
            candidate_labels=self._degrees,
            best_index=best_index,
            best_label=best_degree,
            all_params=all_params,
        )
