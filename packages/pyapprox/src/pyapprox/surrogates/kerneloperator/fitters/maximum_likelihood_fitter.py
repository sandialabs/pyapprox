"""Maximum likelihood fitter for kernel operator surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, List, Optional

from pyapprox.surrogates.kerneloperator.fitters.results import (
    KernelOperatorOptimizedFitResult,
)
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)
from pyapprox.surrogates.kerneloperator.regressors.factory import (
    make_latent_regressor,
)
from pyapprox.surrogates.kerneloperator.surrogate import (
    KernelOperatorSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )

class KernelOperatorMaximumLikelihoodFitter(Generic[Array]):
    """Fit a kernel operator surrogate with hyperparameter optimization.

    Builds encoders from training data via factories, encodes data,
    creates a regressor via make_latent_regressor, and optimizes kernel
    hyperparameters by minimizing negative log marginal likelihood.

    To fit with fixed hyperparameters, set all hyperparameters inactive
    on the kernel before passing it here.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    input_encoder_factories : List[Callable[[Array, Backend[Array]], FunctionEncoderProtocol[Array]]]
        Factories for input encoders. Each called as factory(data, bkd).
    output_encoder_factories : List[Callable[[Array, Backend[Array]], FunctionEncoderProtocol[Array]]]
        Factories for output encoders. Each called as factory(data, bkd).
    kernel : object
        Scalar kernel or MultiOutputKernelProtocol.
    nugget : float
        Nugget for numerical stability.
    optimizer : BindableOptimizerProtocol or None
        Optimizer for hyperparameter optimization. If None, uses default.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        input_encoder_factories: List[Callable[[Array, Backend[Array]], FunctionEncoderProtocol[Array]]],
        output_encoder_factories: List[Callable[[Array, Backend[Array]], FunctionEncoderProtocol[Array]]],
        kernel: object,
        nugget: float = 1e-6,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._input_encoder_factories = input_encoder_factories
        self._output_encoder_factories = output_encoder_factories
        self._kernel = kernel
        self._nugget = nugget
        self._optimizer = optimizer

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def fit(
        self,
        u_train_grids: List[Array],
        v_train_grids: List[Array],
    ) -> KernelOperatorOptimizedFitResult[Array]:
        """Fit the surrogate from training data.

        Parameters
        ----------
        u_train_grids : List[Array]
            Input function values. u_train_grids[i] has shape
            (ngrid_in_i, N).
        v_train_grids : List[Array]
            Output function values. v_train_grids[j] has shape
            (ngrid_out_j, N).

        Returns
        -------
        KernelOperatorOptimizedFitResult[Array]
        """
        input_encoders: List[FunctionEncoderProtocol[Array]] = [
            factory(data, self._bkd)
            for factory, data in zip(
                self._input_encoder_factories, u_train_grids
            )
        ]
        output_encoders: List[FunctionEncoderProtocol[Array]] = [
            factory(data, self._bkd)
            for factory, data in zip(
                self._output_encoder_factories, v_train_grids
            )
        ]

        U_codes = self._bkd.concatenate(
            [enc.encode(u) for enc, u in zip(input_encoders, u_train_grids)],
            axis=0,
        )
        V_codes = self._bkd.concatenate(
            [enc.encode(v) for enc, v in zip(output_encoders, v_train_grids)],
            axis=0,
        )

        ncodes_in = sum(enc.ncodes() for enc in input_encoders)
        ncodes_out = sum(enc.ncodes() for enc in output_encoders)
        regressor = make_latent_regressor(
            self._kernel, ncodes_in, ncodes_out, self._bkd, self._nugget
        )

        initial_hyps, optimized_hyps, opt_result = (
            regressor.fit_with_optimizer(U_codes, V_codes, self._optimizer)
        )

        surrogate: KernelOperatorSurrogate[Array] = KernelOperatorSurrogate(
            input_encoders, output_encoders, regressor
        )
        nll = regressor.neg_log_marginal_likelihood()
        return KernelOperatorOptimizedFitResult(
            surrogate, nll, initial_hyps, optimized_hyps, opt_result
        )
