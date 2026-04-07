"""Protocols for adaptive GP sampling."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SamplingScheduleProtocol(Protocol):
    """Protocol for controlling how many samples to add at each step."""

    def nnew_samples(self) -> int:
        """Return number of new samples for the next step.

        Advances internal state so subsequent calls return the next value.
        """
        ...

    def is_exhausted(self) -> bool:
        """Return True if no more samples should be added."""
        ...


@runtime_checkable
class CandidateGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for generating candidate sample locations in scaled space."""

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def generate(self, ncandidates: int) -> Array:
        """Generate candidate samples in scaled space.

        Parameters
        ----------
        ncandidates : int
            Number of candidates to generate.

        Returns
        -------
        candidates : Array
            Candidate locations of shape (nvars, ncandidates) in scaled space.
        """
        ...


@runtime_checkable
class AdaptiveSamplerProtocol(Protocol, Generic[Array]):
    """Protocol for adaptive sample selection in scaled space."""

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def select_samples(self, nsamples: int) -> Array:
        """Select new sample locations from candidates.

        Parameters
        ----------
        nsamples : int
            Number of samples to select.

        Returns
        -------
        samples : Array
            Selected sample locations of shape (nvars, nsamples) in scaled
            space.
        """
        ...

    def set_kernel(self, kernel: KernelProtocol[Array]) -> None:
        """Update the kernel used for sample selection.

        Parameters
        ----------
        kernel : KernelProtocol[Array]
            New kernel (after hyperparameter optimization).
        """
        ...

    def add_additional_training_samples(self, new_samples: Array) -> None:
        """Update internal state after new training samples are added.

        Parameters
        ----------
        new_samples : Array
            New training samples of shape (nvars, nsamples) in scaled space.
            For Cholesky/IVAR: marks candidates as selected, updates
            factorization. For Sobol: no-op.
        """
        ...
