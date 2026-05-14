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
    """Bidirectional map between function grid samples and code space."""

    def bkd(self) -> Backend[Array]:
        ...

    def ncodes(self) -> int:
        ...

    def ngrid(self) -> int:
        ...

    def encode(self, f_grid: Array) -> Array:
        """Encode grid values to codes. (ngrid, N) -> (ncodes, N)."""
        ...

    def decode(self, codes: Array) -> Array:
        """Decode codes to grid values. (ncodes, N) -> (ngrid, N)."""
        ...

    def decode_std(self, std_codes: Array) -> Array:
        """Decode std codes without mean shift. (ncodes, N) -> (ngrid, N)."""
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
