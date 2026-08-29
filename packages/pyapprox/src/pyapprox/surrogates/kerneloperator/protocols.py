"""Protocols for the kernel operator learning module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )


@runtime_checkable
class FunctionEncoderProtocol(Protocol, Generic[Array]):
    """Bidirectional map between a function's samples and a latent space.

    ``full_dim`` and ``latent_dim`` name the two dimensions, matching
    the reduction vocabulary used elsewhere. They are the same
    quantities a KLE calls ``ncoords`` and ``nterms``; the neutral names
    are used here because an encoder need not be defined on a mesh.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def full_dim(self) -> int:
        ...

    def latent_dim(self) -> int:
        ...

    def encode(self, samples: Array) -> Array:
        """Full to latent. (full_dim, N) -> (latent_dim, N)."""
        ...

    def decode(self, latents: Array) -> Array:
        """Latent to full. (latent_dim, N) -> (full_dim, N)."""
        ...


@runtime_checkable
class StdDecodingEncoderProtocol(FunctionEncoderProtocol[Array], Protocol):
    """An encoder that can also propagate a standard deviation.

    Separate from :class:`FunctionEncoderProtocol` because not every
    encoder can honestly provide it. The linear propagation
    ``sqrt(P^2 sigma^2)`` assumes the decoder is linear *and* that the
    latent coordinates are uncorrelated; a decoder with a nonlinear
    correction term makes the first assumption false rather than
    approximate, so such an encoder should decline to implement this
    rather than return a number that looks like a standard deviation
    and is not one.

    Only ``predict_std`` needs it. Ordinary prediction does not, so an
    encoder without it remains usable for everything else.
    """

    def decode_std(self, std_latents: Array) -> Array:
        """Propagate latent std to full space, without the mean shift.

        ``(latent_dim, N) -> (full_dim, N)``.
        """
        ...


@runtime_checkable
class LatentRegressorProtocol(Protocol, Generic[Array]):
    """Latent-space regressor mapping input codes to output codes."""

    def bkd(self) -> Backend[Array]:
        ...

    def ncodes_in(self) -> int:
        ...

    def ncodes_out(self) -> int:
        ...

    def hyp_list(self) -> HyperParameterList[Array]:
        ...

    def is_fitted(self) -> bool:
        ...

    def fit_internal(self, U: Array, V: Array) -> None:
        """Fit regressor. U: (ncodes_in, N), V: (ncodes_out, N)."""
        ...

    def predict(self, U_test: Array) -> Array:
        """Predict output codes. (ncodes_in, N_test) -> (ncodes_out, N_test)."""
        ...

    def predict_std(self, U_test: Array) -> Array:
        """Predict std of output codes. (ncodes_in, N_test) -> (ncodes_out, N_test)."""
        ...

    def neg_log_marginal_likelihood(self) -> Array:
        ...

    def clone_unfitted(self) -> LatentRegressorProtocol[Array]:
        ...

    def fit_with_optimizer(
        self,
        U: Array,
        V: Array,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> Tuple[Array, Array, Optional[object]]:
        """Fit with hyperparameter optimization.

        Returns (initial_hyps, optimized_hyps, opt_result).
        Each regressor delegates to its own GP fitter internally.
        """
        ...
