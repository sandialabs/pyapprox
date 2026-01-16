"""Multivariate Leja and Fekete sampling.

This module provides linear algebra-based multivariate Leja and Fekete
sampling. These methods select points from a candidate set using
pivoted matrix factorization.

Key classes:
- LejaSampler: Incremental Leja sampling using pivoted LU factorization
- FeketeSampler: One-shot Fekete sampling using pivoted QR factorization
"""

from typing import Callable, Generic, Optional, Tuple, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisProtocol
from pyapprox.typing.util.linalg import PivotedLUFactorizer, PivotedQRFactorizer

from .protocols import LejaWeightingProtocol


class LejaSampler(Generic[Array]):
    """Multivariate Leja sampler using pivoted LU factorization.

    Leja sampling selects points incrementally from a candidate set.
    Each new point is chosen to maximize the determinant of the
    interpolation matrix, which is done efficiently via pivoted LU.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis : BasisProtocol[Array]
        Multivariate basis for interpolation.
    candidate_samples : Array
        Candidate sample locations. Shape: (nvars, ncandidates)
    weighting : LejaWeightingProtocol[Array], optional
        Weighting strategy. If provided, weights are applied to the
        basis matrix before factorization.
    tol : float
        Tolerance for detecting singular pivots.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> # Assume basis is defined
    >>> candidates = bkd.asarray(np.random.randn(2, 100))
    >>> # sampler = LejaSampler(bkd, basis, candidates)
    >>> # selected = sampler.sample(20)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis: BasisProtocol[Array],
        candidate_samples: Array,
        weighting: Optional[LejaWeightingProtocol[Array]] = None,
        tol: float = 1e-14,
    ):
        self._bkd = bkd
        self._basis = basis
        self._candidates = candidate_samples
        self._weighting = weighting
        self._tol = tol
        self._init_pivots: Optional[Array] = None

        # Compute basis matrix at candidates
        self._basis_mat = basis(candidate_samples)  # (ncandidates, nterms)

        # Apply weighting if provided
        if weighting is not None:
            weights = weighting(candidate_samples, self._basis_mat)
            self._precond_weights = self._bkd.sqrt(weights)
            self._precond_mat = self._precond_weights * self._basis_mat
        else:
            self._precond_weights = None
            self._precond_mat = self._basis_mat

        # Initialize LU factorizer (deferred until sample() is called)
        self._factorizer: Optional[PivotedLUFactorizer[Array]] = None
        self._nselected = 0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nsamples(self) -> int:
        """Return current number of selected samples."""
        return self._nselected

    def ncandidates(self) -> int:
        """Return total number of candidate samples."""
        return self._candidates.shape[1]

    def set_initial_pivots(self, init_pivots: Array) -> None:
        """Set initial pivot indices to use before automatic selection.

        These pivots will be used first (in order) before the algorithm
        selects additional points based on maximum determinant.

        Parameters
        ----------
        init_pivots : Array
            Indices into the candidate set to use as initial pivots.
            Must be set before calling sample().
        """
        if self._nselected > 0:
            raise RuntimeError(
                "Cannot set initial pivots after sampling has begun"
            )
        self._init_pivots = init_pivots

    def _ensure_factorizer(self) -> None:
        """Create the factorizer if not yet created."""
        if self._factorizer is None:
            self._factorizer = PivotedLUFactorizer(
                self._bkd,
                self._precond_mat.T,
                tol=self._tol,
                init_pivots=self._init_pivots,
            )

    def sample(self, nsamples: int) -> Array:
        """Generate Leja samples by selecting from candidates.

        Parameters
        ----------
        nsamples : int
            Number of samples to select.

        Returns
        -------
        Array
            Selected samples. Shape: (nvars, nsamples)
        """
        if nsamples > self.ncandidates():
            raise ValueError(
                f"Cannot select {nsamples} samples from "
                f"{self.ncandidates()} candidates"
            )

        self._ensure_factorizer()

        if nsamples > self._nselected:
            # Need to select more samples
            self._factorizer.update(nsamples)
            self._nselected = self._factorizer.npivots()

        # Get selected indices
        pivots = self._factorizer.pivots()[:nsamples]
        return self._candidates[:, pivots]

    def sample_incremental(self, n_new_samples: int) -> Array:
        """Add new samples incrementally.

        Parameters
        ----------
        n_new_samples : int
            Number of new samples to add.

        Returns
        -------
        Array
            The new samples only. Shape: (nvars, n_new_samples)
        """
        self._ensure_factorizer()
        old_count = self._nselected
        new_total = old_count + n_new_samples
        self._factorizer.update(new_total)
        self._nselected = self._factorizer.npivots()

        # Return only the new samples
        pivots = self._factorizer.pivots()[old_count:new_total]
        return self._candidates[:, pivots]

    def get_selected_indices(self) -> Array:
        """Return indices of selected samples from candidate set."""
        self._ensure_factorizer()
        return self._factorizer.pivots()

    def get_all_selected_samples(self) -> Array:
        """Return all currently selected samples."""
        self._ensure_factorizer()
        pivots = self._factorizer.pivots()
        return self._candidates[:, pivots]

    def success(self) -> bool:
        """Return True if factorization was successful."""
        if self._factorizer is None:
            return True  # No factorization attempted yet
        return self._factorizer.success()

    def termination_message(self) -> str:
        """Return termination status message."""
        if self._factorizer is None:
            return "Factorization not yet started"
        return self._factorizer.termination_message()

    def __repr__(self) -> str:
        return (
            f"LejaSampler(ncandidates={self.ncandidates()}, "
            f"nselected={self.nsamples()})"
        )


class FeketeSampler(Generic[Array]):
    """Fekete point sampler using pivoted QR factorization.

    Fekete sampling selects all points at once (not incrementally)
    using pivoted QR factorization. This finds points that approximately
    maximize the determinant of the interpolation matrix.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis : BasisProtocol[Array]
        Multivariate basis for interpolation.
    candidate_samples : Array
        Candidate sample locations. Shape: (nvars, ncandidates)
    weighting : LejaWeightingProtocol[Array], optional
        Weighting strategy.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> # Assume basis is defined
    >>> candidates = bkd.asarray(np.random.randn(2, 100))
    >>> # sampler = FeketeSampler(bkd, basis, candidates)
    >>> # selected = sampler.sample(20)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis: BasisProtocol[Array],
        candidate_samples: Array,
        weighting: Optional[LejaWeightingProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._basis = basis
        self._candidates = candidate_samples
        self._weighting = weighting

        # Compute basis matrix at candidates
        self._basis_mat = basis(candidate_samples)  # (ncandidates, nterms)

        # Apply weighting if provided
        if weighting is not None:
            weights = weighting(candidate_samples, self._basis_mat)
            self._precond_weights = self._bkd.sqrt(weights)
            self._precond_mat = self._precond_weights * self._basis_mat
        else:
            self._precond_weights = None
            self._precond_mat = self._basis_mat

        # Initialize QR factorizer
        self._factorizer = PivotedQRFactorizer(bkd)
        self._selected_indices: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ncandidates(self) -> int:
        """Return total number of candidate samples."""
        return self._candidates.shape[1]

    def sample(self, nsamples: int) -> Array:
        """Select Fekete points from candidates.

        Parameters
        ----------
        nsamples : int
            Number of points to select.

        Returns
        -------
        Array
            Selected samples. Shape: (nvars, nsamples)
        """
        if nsamples > self.ncandidates():
            raise ValueError(
                f"Cannot select {nsamples} samples from "
                f"{self.ncandidates()} candidates"
            )

        # Use QR with column pivoting on transposed preconditioned matrix
        self._selected_indices = self._factorizer.select_points(
            self._precond_mat, nsamples
        )
        return self._candidates[:, self._selected_indices]

    def get_selected_indices(self) -> Array:
        """Return indices of selected samples from candidate set."""
        if self._selected_indices is None:
            raise RuntimeError("Must call sample() first")
        return self._selected_indices

    def __repr__(self) -> str:
        n_selected = (
            len(self._selected_indices) if self._selected_indices is not None else 0
        )
        return (
            f"FeketeSampler(ncandidates={self.ncandidates()}, "
            f"nselected={n_selected})"
        )


class WeightedLejaSampler(LejaSampler[Array]):
    """Leja sampler with preconditioning weight updates.

    This variant allows updating the preconditioning weights after
    selection has begun, which is useful for adaptive sampling where
    weights depend on function evaluations.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis : BasisProtocol[Array]
        Multivariate basis for interpolation.
    candidate_samples : Array
        Candidate sample locations. Shape: (nvars, ncandidates)
    initial_weighting : LejaWeightingProtocol[Array], optional
        Initial weighting strategy.
    tol : float
        Tolerance for detecting singular pivots.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis: BasisProtocol[Array],
        candidate_samples: Array,
        initial_weighting: Optional[LejaWeightingProtocol[Array]] = None,
        tol: float = 1e-14,
    ):
        super().__init__(bkd, basis, candidate_samples, initial_weighting, tol)
        self._current_weights = self._precond_weights

    def update_weights(self, new_weighting: LejaWeightingProtocol[Array]) -> None:
        """Update preconditioning weights.

        This updates the factorization to use new weights without
        recomputing from scratch.

        Parameters
        ----------
        new_weighting : LejaWeightingProtocol[Array]
            New weighting strategy.
        """
        new_weights = new_weighting(self._candidates, self._basis_mat)
        new_sqrt_weights = self._bkd.sqrt(new_weights)

        if self._current_weights is not None:
            # Update preconditioning in factorizer
            self._factorizer.update_preconditioning(
                self._current_weights, new_sqrt_weights, self._nselected
            )

        self._current_weights = new_sqrt_weights
        self._weighting = new_weighting

    def __repr__(self) -> str:
        return (
            f"WeightedLejaSampler(ncandidates={self.ncandidates()}, "
            f"nselected={self.nsamples()})"
        )
