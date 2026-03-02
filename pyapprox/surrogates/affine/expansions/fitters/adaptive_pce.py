"""Adaptive PCE basis selection fitter.

Implements the iterative basis selection algorithm from:

  Jakeman, Eldred, Sargsyan (2015). "Enhancing l1-minimization estimates
  of polynomial chaos expansions using basis selection." Journal of
  Computational Physics, 289, 18-34.

Each outer iteration:
1. Restrict: shrink basis to only terms with nonzero OMP coefficients
2. Expand: recursively add admissible forward neighbors up to T times
3. Identify: run OMP+CV on each expanded set
4. Select: pick the expansion level with lowest CV error
5. Stop: if no improvement over previous best
"""

from typing import Generic, List, Optional

from pyapprox.surrogates.affine.expansions.fitters.omp_cv import (
    OMPCVFitter,
)
from pyapprox.surrogates.affine.indices.utils import (
    compute_hyperbolic_indices,
    hash_index,
    sort_indices_lexiographically,
)
from pyapprox.surrogates.affine.protocols import (
    BasisExpansionProtocol,
    MultiIndexBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdaptivePCEResult(Generic[Array]):
    """Result from adaptive PCE basis selection.

    All attributes accessed via methods per project convention.

    Parameters
    ----------
    surrogate : BasisExpansionProtocol[Array]
        Fitted expansion with best coefficients and index set.
    params : Array
        Fitted coefficients for the best iteration. Shape: (nterms, nqoi).
    cv_scores_history : list[float]
        Best CV score at each outer iteration.
    nterms_history : list[int]
        Number of basis terms at each outer iteration (the selected set).
    indices_history : list[Array]
        Index set used at each outer iteration (the selected set).
    best_iteration : int
        Outer iteration index with lowest CV score.
    final_indices : Array
        Multi-index set for the best iteration. Shape: (nvars, nterms).
    """

    def __init__(
        self,
        surrogate: BasisExpansionProtocol[Array],
        params: Array,
        cv_scores_history: List[float],
        nterms_history: List[int],
        indices_history: List[Array],
        best_iteration: int,
        final_indices: Array,
    ):
        self._surrogate = surrogate
        self._params = params
        self._cv_scores_history = cv_scores_history
        self._nterms_history = nterms_history
        self._indices_history = indices_history
        self._best_iteration = best_iteration
        self._final_indices = final_indices

    def surrogate(self) -> BasisExpansionProtocol[Array]:
        """Return the fitted surrogate."""
        return self._surrogate

    def params(self) -> Array:
        """Return fitted parameters. Shape: (nterms, nqoi)."""
        return self._params

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def cv_scores_history(self) -> List[float]:
        """Return best CV score at each outer iteration."""
        return self._cv_scores_history

    def nterms_history(self) -> List[int]:
        """Return number of basis terms at each outer iteration."""
        return self._nterms_history

    def indices_history(self) -> List[Array]:
        """Return index set at each outer iteration."""
        return self._indices_history

    def best_iteration(self) -> int:
        """Return outer iteration with lowest CV score."""
        return self._best_iteration

    def final_indices(self) -> Array:
        """Return multi-index set for the best iteration. Shape: (nvars, nterms)."""
        return self._final_indices

    def __call__(self, samples: Array) -> Array:
        """Evaluate fitted surrogate at samples.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nqoi, nsamples)
        """
        result: Array = self._surrogate(samples)
        return result

    def __repr__(self) -> str:
        return (
            f"AdaptivePCEResult(best_iteration={self._best_iteration}, "
            f"nterms={self._final_indices.shape[1]}, "
            f"n_iterations={len(self._cv_scores_history)})"
        )


def _expand_indices(
    indices: Array, max_level: int, pnorm: float, bkd: Backend[Array]
) -> Array:
    """Add all admissible forward neighbors to an index set.

    A forward neighbor lambda + e_k is admissible if all its backward
    neighbors exist in the current set (downward closure criterion) and
    its p-norm does not exceed max_level.

    Parameters
    ----------
    indices : Array
        Current index set. Shape: (nvars, nindices). Must be downward closed.
    max_level : int
        Maximum admissible p-norm level.
    pnorm : float
        p-norm exponent.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Expanded index set. Shape: (nvars, nindices_expanded).
        Sorted lexicographically.
    """
    nvars = indices.shape[0]
    nindices = indices.shape[1]

    # Build hash set of current indices
    current_set: set[int] = set()
    for jj in range(nindices):
        current_set.add(hash_index(indices[:, jj], bkd))

    # Find admissible forward neighbors
    new_indices = []
    new_hashes: set[int] = set()
    for jj in range(nindices):
        idx = indices[:, jj]
        for dim in range(nvars):
            # Forward neighbor in dimension dim
            neighbor = bkd.copy(idx)
            neighbor[dim] += 1

            # Check p-norm constraint
            neighbor_np = bkd.to_numpy(neighbor)
            level = float(sum(int(v) ** pnorm for v in neighbor_np)) ** (1.0 / pnorm)
            if level > max_level + 1e-10:
                continue

            nh = hash_index(neighbor, bkd)
            if nh in current_set or nh in new_hashes:
                continue

            # Check admissibility: all backward neighbors must be in
            # current set
            admissible = True
            for back_dim in range(nvars):
                if int(bkd.to_numpy(neighbor[back_dim])) > 0:
                    back_neighbor = bkd.copy(neighbor)
                    back_neighbor[back_dim] -= 1
                    if hash_index(back_neighbor, bkd) not in current_set:
                        admissible = False
                        break
            if admissible:
                new_indices.append(bkd.reshape(neighbor, (nvars, 1)))
                new_hashes.add(nh)

    if len(new_indices) == 0:
        return bkd.copy(indices)

    expanded = bkd.hstack([indices] + new_indices)
    return sort_indices_lexiographically(expanded, bkd)


class AdaptivePCEFitter(Generic[Array]):
    """Adaptive PCE basis selection fitter.

    Implements the iterative basis selection algorithm of Jakeman et al.
    (2015). At each outer iteration the algorithm:

    1. Restricts the basis to terms with nonzero OMP coefficients
    2. Expands the restricted set up to T times by adding all admissible
       forward neighbors
    3. Runs OMP+CV on each expanded set to identify the best expansion level
    4. Selects the expansion level with lowest CV error
    5. Terminates when CV error stops improving

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    initial_level : int
        Starting hyperbolic level for the index set. Default: 1.
    pnorm : float
        p-norm for hyperbolic index sets. Default: 1.0 (total degree).
    num_expansions : int
        Number of recursive expansion steps per outer iteration (T in the
        paper). Default: 3.
    max_level : int
        Maximum admissible level ceiling. Default: 20.
    max_iterations : int
        Maximum number of outer iterations. Default: 50.
    omp_max_nonzeros : int, optional
        Maximum nonzeros for OMP. None means auto (min of nterms, nsamples-3).
    omp_alpha : float
        Ridge regularization for OMP CV. Default: 0.0.
    omp_nfolds : int, optional
        Number of folds for LMO CV. None means LOO.
    restrict_tol : float
        Absolute threshold below which OMP coefficients are treated as zero
        in the restrict step. Default: 1e-10.
    verbosity : int
        Verbosity level. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        initial_level: int = 1,
        pnorm: float = 1.0,
        num_expansions: int = 3,
        max_level: int = 20,
        max_iterations: int = 50,
        omp_max_nonzeros: Optional[int] = None,
        omp_alpha: float = 0.0,
        omp_nfolds: Optional[int] = None,
        restrict_tol: float = 1e-10,
        verbosity: int = 0,
    ):
        self._bkd = bkd
        self._initial_level = initial_level
        self._pnorm = pnorm
        self._num_expansions = num_expansions
        self._max_level = max_level
        self._max_iterations = max_iterations
        self._omp_max_nonzeros = omp_max_nonzeros
        self._omp_alpha = omp_alpha
        self._omp_nfolds = omp_nfolds
        self._restrict_tol = restrict_tol
        self._verbosity = verbosity

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def initial_level(self) -> int:
        """Return initial hyperbolic level."""
        return self._initial_level

    def pnorm(self) -> float:
        """Return p-norm for hyperbolic indices."""
        return self._pnorm

    def num_expansions(self) -> int:
        """Return number of expansion steps per outer iteration (T)."""
        return self._num_expansions

    def max_level(self) -> int:
        """Return maximum admissible level."""
        return self._max_level

    def max_iterations(self) -> int:
        """Return maximum outer iterations."""
        return self._max_iterations

    def restrict_tol(self) -> float:
        """Return restrict-step zero threshold."""
        return self._restrict_tol

    def _run_omp_cv(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ):
        """Run OMP + CV on the current expansion basis."""
        bkd = self._bkd
        basis = expansion.get_basis()
        nterms = basis.nterms()
        nsamples = samples.shape[1]

        max_nz = nterms
        if self._omp_max_nonzeros is not None:
            max_nz = min(max_nz, self._omp_max_nonzeros)
        max_nz = min(max_nz, nsamples - 3)
        if max_nz < 1:
            return None

        omp_cv = OMPCVFitter(
            bkd,
            max_nonzeros=max_nz,
            alpha=self._omp_alpha,
            nfolds=self._omp_nfolds,
        )
        return omp_cv.fit(expansion, samples, values)

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> AdaptivePCEResult[Array]:
        """Run adaptive basis selection.

        Implements Algorithm 1 of Jakeman et al. (2015).

        Parameters
        ----------
        expansion : BasisExpansionProtocol[Array]
            Must have basis with set_indices() (MultiIndexBasisProtocol).
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
            Only nqoi=1 supported.

        Returns
        -------
        AdaptivePCEResult
            Result with best expansion and adaptation history.

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
                f"AdaptivePCEFitter only supports nqoi=1, got {values.shape[0]}"
            )

        nvars = samples.shape[0]

        # Validate basis
        basis = expansion.get_basis()
        if not isinstance(basis, MultiIndexBasisProtocol):
            raise TypeError(
                "expansion.get_basis() must satisfy MultiIndexBasisProtocol"
            )

        # Initialize: Λ^(0) = hyperbolic index set at initial_level
        initial_indices = compute_hyperbolic_indices(
            nvars, self._initial_level, self._pnorm, bkd
        )
        basis.set_indices(initial_indices)

        # Run initial OMP+CV on Λ^(0)
        cv_result = self._run_omp_cv(expansion, samples, values)
        if cv_result is None:
            return self._make_fallback_result(expansion, basis, bkd, initial_indices)

        star_cv = float(cv_result.cv_scores()[cv_result.best_index()])
        star_params = bkd.copy(cv_result.params())
        star_indices = bkd.copy(initial_indices)

        cv_scores_history: List[float] = [star_cv]
        nterms_history: List[int] = [initial_indices.shape[1]]
        indices_history: List[Array] = [bkd.copy(initial_indices)]
        best_iteration_idx = 0

        if self._verbosity > 0:
            print(f"Init: nterms={initial_indices.shape[1]}, CV={star_cv:.6e}")

        # Current coefficients from the initial fit
        current_params = cv_result.params()

        for iteration in range(1, self._max_iterations + 1):
            # ---- RESTRICT ----
            # Λ^(k,0) = {λ : α_λ ≠ 0} from previous iteration
            current_indices = bkd.copy(basis.get_indices())
            nterms = current_indices.shape[1]
            nonzero_idx = []
            for ii in range(nterms):
                if abs(float(bkd.to_numpy(current_params[ii, 0]))) > self._restrict_tol:
                    nonzero_idx.append(ii)

            if len(nonzero_idx) == 0:
                break

            restricted = current_indices[
                :, bkd.asarray(nonzero_idx, dtype=bkd.int64_dtype())
            ]
            restricted = sort_indices_lexiographically(restricted, bkd)

            if self._verbosity > 0:
                print(
                    f"Iter {iteration}: restricted to "
                    f"{restricted.shape[1]} nonzero terms"
                )

            # ---- EXPAND + IDENTIFY + SELECT ----
            # Inner loop: expand T times, run OMP+CV on each
            iter_best_cv = float("inf")
            iter_best_params: Optional[Array] = None
            iter_best_indices: Optional[Array] = None

            expanded = restricted
            for t in range(1, self._num_expansions + 1):
                expanded = _expand_indices(expanded, self._max_level, self._pnorm, bkd)

                if expanded.shape[1] == restricted.shape[1] and t > 1:
                    # No new indices added; further expansions won't help
                    break

                # Set indices and run OMP+CV
                basis.set_indices(expanded)
                cv_result_t = self._run_omp_cv(expansion, samples, values)
                if cv_result_t is None:
                    continue

                cv_t = float(cv_result_t.cv_scores()[cv_result_t.best_index()])

                if self._verbosity > 0:
                    print(f"  t={t}: nterms={expanded.shape[1]}, CV={cv_t:.6e}")

                if cv_t < iter_best_cv:
                    iter_best_cv = cv_t
                    iter_best_params = bkd.copy(cv_result_t.params())
                    iter_best_indices = bkd.copy(expanded)

            if iter_best_indices is None:
                break

            # Record this outer iteration's best
            cv_scores_history.append(iter_best_cv)
            nterms_history.append(iter_best_indices.shape[1])
            indices_history.append(bkd.copy(iter_best_indices))

            # ---- STOPPING CHECK ----
            if iter_best_cv >= star_cv:
                break

            # Update global best
            star_cv = iter_best_cv
            star_params = iter_best_params
            star_indices = bkd.copy(iter_best_indices)
            best_iteration_idx = iteration
            current_params = iter_best_params

            # Set basis to best of this iteration for next restrict step
            basis.set_indices(iter_best_indices)

        # FINALIZE
        basis.set_indices(star_indices)
        fitted_expansion = expansion.with_params(star_params)

        return AdaptivePCEResult(
            surrogate=fitted_expansion,
            params=star_params,
            cv_scores_history=cv_scores_history,
            nterms_history=nterms_history,
            indices_history=indices_history,
            best_iteration=best_iteration_idx,
            final_indices=star_indices,
        )

    def _make_fallback_result(
        self,
        expansion: BasisExpansionProtocol[Array],
        basis,
        bkd: Backend[Array],
        indices: Array,
    ) -> AdaptivePCEResult[Array]:
        """Create result when OMP cannot run (too few samples)."""
        params = bkd.zeros((indices.shape[1], 1))
        fitted = expansion.with_params(params)
        return AdaptivePCEResult(
            surrogate=fitted,
            params=params,
            cv_scores_history=[],
            nterms_history=[],
            indices_history=[],
            best_iteration=0,
            final_indices=indices,
        )
