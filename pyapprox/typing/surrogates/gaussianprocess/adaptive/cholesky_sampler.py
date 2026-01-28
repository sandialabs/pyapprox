"""Cholesky adaptive sampler for adaptive GP."""

from typing import Generic

from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.linalg.pivoted_cholesky import (
    PivotedCholeskyFactorizer,
)


class CholeskySampler(Generic[Array]):
    """Adaptive sampler using pivoted Cholesky factorization.

    Selects samples by pivoted Cholesky on the candidate kernel matrix.
    Operates entirely in scaled space.

    Parameters
    ----------
    candidates_scaled : Array
        Candidate locations of shape (nvars, ncandidates) in scaled space.
    bkd : Backend[Array]
        Backend for numerical computations.
    nugget : float
        Nugget added to candidate kernel matrix diagonal for numerical
        stability. Separate from the GP's observation noise.
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
        self._factorizer: PivotedCholeskyFactorizer[Array] | None = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def set_kernel(self, kernel: KernelProtocol[Array]) -> None:
        """Update the kernel and rebuild factorizer with warm-start.

        Recomputes K = kernel(candidates, candidates) + nugget*I and
        restarts pivoted Cholesky with existing training indices as
        init_pivots.

        Parameters
        ----------
        kernel : KernelProtocol[Array]
            New kernel (after hyperparameter optimization).
        """
        bkd = self._bkd
        self._kernel = kernel
        K = kernel(self._candidates, self._candidates)
        if self._nugget > 0:
            K = K + self._nugget * bkd.eye(self._ncandidates)
        self._factorizer = PivotedCholeskyFactorizer(K, bkd)

        # Warm-start: use existing selected indices as init_pivots
        if len(self._selected_indices) > 0:
            init_pivots = bkd.asarray(self._selected_indices)
            self._factorizer.factorize(
                len(self._selected_indices), init_pivots=init_pivots
            )
        else:
            # Initialize internal state without selecting any pivots yet
            self._factorizer.factorize(0)

    def select_samples(self, nsamples: int) -> Array:
        """Select new sample locations via pivoted Cholesky.

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
            If candidates are exhausted or kernel not set.
        """
        if self._factorizer is None:
            raise ValueError("Must call set_kernel() before select_samples()")

        n_available = self._ncandidates - len(self._selected_indices)
        if nsamples > n_available:
            raise ValueError(
                "All candidates exhausted; generate a new candidate set "
                "or increase its size"
            )

        n_total = len(self._selected_indices) + nsamples
        if len(self._selected_indices) == 0:
            self._factorizer.factorize(n_total)
        else:
            self._factorizer.update(n_total)

        # Extract new pivot indices
        all_pivots = self._factorizer.pivots()
        new_pivot_indices = all_pivots[len(self._selected_indices):]

        # Convert to Python ints for tracking
        new_pivots_np = self._bkd.to_numpy(new_pivot_indices)
        for idx in new_pivots_np:
            self._selected_indices.append(int(idx))

        return self._candidates[:, new_pivot_indices]

    def add_additional_training_samples(
        self, new_samples: Array
    ) -> None:
        """No-op: tracking is handled internally by select_samples."""
