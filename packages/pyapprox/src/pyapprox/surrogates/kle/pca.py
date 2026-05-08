"""Principal Component Analysis for dimensionality reduction."""

from typing import Generic, Optional

from pyapprox.surrogates.kle.data_driven_kle import DataDrivenKLE
from pyapprox.util.backends.protocols import Array, Backend


class PrincipalComponentAnalysis(DataDrivenKLE[Array], Generic[Array]):
    """Principal Component Analysis (PCA) for dimensionality reduction.

    Normalizes snapshots (subtract mean, divide by max per coordinate),
    then computes a reduced basis using SVD.

    Parameters
    ----------
    snapshots : Array, shape (ncoords, nsamples)
        Snapshot data to reduce.
    nterms : int
        Number of basis vectors to keep.
    bkd : Backend[Array]
        Computational backend.
    quad_weights : Array or None, shape (ncoords,)
        Optional quadrature weights for weighted SVD.
    """

    def __init__(
        self,
        snapshots: Array,
        nterms: int,
        bkd: Backend[Array],
        quad_weights: Optional[Array] = None,
    ):
        normalized_snapshots = (
            snapshots - bkd.mean(snapshots, axis=1)[:, None]
        ) / bkd.max(snapshots, axis=1)[:, None]
        super().__init__(
            normalized_snapshots,
            0.0,
            False,
            nterms,
            quad_weights,
            bkd=bkd,
        )

    def reduce_state(self, state: Array) -> Array:
        """Project a state onto the reduced basis.

        Parameters
        ----------
        state : Array
            Full-order state to be reduced.

        Returns
        -------
        Array
            Reduced-order state.
        """
        return self.eigenvectors().T @ state

    def expand_reduced_state(self, reduced_state: Array) -> Array:
        """Expand a reduced-order state back to the full-order state.

        Parameters
        ----------
        reduced_state : Array
            Reduced-order state to be expanded.

        Returns
        -------
        Array
            Full-order state.
        """
        return self.eigenvectors() @ reduced_state

    def snapshots(self) -> Array:
        """Return the normalized field samples."""
        return self._field_samples
