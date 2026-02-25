"""Cholesky adaptive sampler for adaptive GP."""

from typing import Generic

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.pivoted_cholesky import (
    PivotedCholeskyFactorizer,
)


class CholeskySampler(Generic[Array]):
    """Adaptive sampler using pivoted Cholesky factorization.

    Selects samples by pivoted Cholesky on the candidate kernel matrix.
    Operates entirely in scaled space.

    An optional weight function biases pivot selection toward
    high-weight regions. With weights, the pivot criterion becomes
    ``argmax(w[i] * K[i,i])`` instead of ``argmax(K[i,i])``.
    Typical weight functions are probability density functions that
    concentrate samples in high-probability regions.

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
        self._K: Array | None = None
        self._factorizer: PivotedCholeskyFactorizer[Array] | None = None
        self._weight_function: FunctionProtocol[Array] | None = None
        self._pivot_weights: Array | None = None
        self._weight_function_changed: bool = False

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def set_weight_function(
        self,
        weight_function: FunctionProtocol[Array] | None,
    ) -> None:
        """Set a weight function for biasing pivot selection.

        The weight function must satisfy ``FunctionProtocol`` with
        ``nqoi() == 1``. It is evaluated on the candidates (in scaled
        space) to produce weights of shape ``(ncandidates,)``. Pivots
        are then selected by ``argmax(weight[i] * diag[i])`` instead
        of ``argmax(diag[i])``.

        Setting a new weight function triggers re-factorization with
        existing pivots as init_pivots on the next ``set_kernel`` or
        ``select_samples`` call.

        Parameters
        ----------
        weight_function : FunctionProtocol[Array] | None
            Weight function accepting ``(nvars, ncandidates)`` and
            returning ``(1, ncandidates)``. Pass ``None`` to remove
            weighting (revert to uniform selection).
        """
        if weight_function is not None:
            self._weight_function = weight_function
            self._pivot_weights = self._compute_pivot_weights()
        else:
            self._weight_function = None
            self._pivot_weights = None
        self._weight_function_changed = True

    def _compute_pivot_weights(self) -> Array:
        """Evaluate weight function on candidates, return 1D weights."""
        assert self._weight_function is not None
        weights = self._weight_function(self._candidates)
        if weights.ndim != 2 or weights.shape[0] != 1:
            raise ValueError(
                "weight_function must return shape (1, ncandidates), "
                f"got {weights.shape}"
            )
        if weights.shape[1] != self._ncandidates:
            raise ValueError(
                f"weight_function returned {weights.shape[1]} weights, "
                f"expected {self._ncandidates}"
            )
        result: Array = weights[0]
        return result

    def set_kernel(self, kernel: KernelProtocol[Array]) -> None:
        """Update the kernel and rebuild factorizer with warm-start.

        Recomputes K = kernel(candidates, candidates) + nugget*I and
        restarts pivoted Cholesky with existing training indices as
        init_pivots. Uses current pivot weights if a weight function
        has been set.

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
        self._K = K
        self._weight_function_changed = False
        self._rebuild_factorizer()

    def _rebuild_factorizer(self) -> None:
        """Rebuild factorizer from current K and weights, warm-starting."""
        bkd = self._bkd
        if self._K is None:
            return
        self._factorizer = PivotedCholeskyFactorizer(self._K, bkd)
        if len(self._selected_indices) > 0:
            init_pivots = bkd.asarray(self._selected_indices)
            self._factorizer.factorize(
                len(self._selected_indices),
                init_pivots=init_pivots,
                pivot_weights=self._pivot_weights,
            )
        else:
            self._factorizer.factorize(
                0,
                pivot_weights=self._pivot_weights,
            )

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

        if self._weight_function_changed:
            self._rebuild_factorizer()
            self._weight_function_changed = False

        n_available = self._ncandidates - len(self._selected_indices)
        if nsamples > n_available:
            raise ValueError(
                "All candidates exhausted; generate a new candidate set "
                "or increase its size"
            )

        n_total = len(self._selected_indices) + nsamples
        if len(self._selected_indices) == 0:
            self._factorizer.factorize(
                n_total,
                pivot_weights=self._pivot_weights,
            )
        else:
            self._factorizer.update(n_total)

        # Extract new pivot indices
        all_pivots = self._factorizer.pivots()
        new_pivot_indices = all_pivots[len(self._selected_indices) :]

        # Convert to Python ints for tracking
        new_pivots_np = self._bkd.to_numpy(new_pivot_indices)
        for idx in new_pivots_np:
            self._selected_indices.append(int(idx))

        return self._candidates[:, new_pivot_indices]

    def add_additional_training_samples(self, new_samples: Array) -> None:
        """No-op: tracking is handled internally by select_samples."""
