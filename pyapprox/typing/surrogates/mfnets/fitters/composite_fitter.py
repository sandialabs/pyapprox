"""Composite fitter: ALS initialization followed by gradient polish.

Runs alternating least squares first to get a good initial guess for
the polynomial coefficients, then optionally refines with a gradient-based
optimizer.
"""

from typing import Generic, List, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.typing.surrogates.mfnets.network import MFNet
from pyapprox.typing.surrogates.mfnets.fitters.als_fitter import (
    MFNetALSFitter,
)
from pyapprox.typing.surrogates.mfnets.fitters.gradient_fitter import (
    MFNetGradientFitter,
)
from pyapprox.typing.surrogates.mfnets.fitters.results import (
    MFNetALSFitResult,
    MFNetGradientFitResult,
)


class MFNetCompositeFitResult(Generic[Array]):
    """Result from composite ALS + gradient fitting.

    Parameters
    ----------
    surrogate : MFNet[Array]
        The fitted network.
    als_result : MFNetALSFitResult[Array]
        Result from the ALS phase.
    gradient_result : MFNetGradientFitResult[Array], optional
        Result from the gradient phase (None if skipped).
    """

    def __init__(
        self,
        surrogate: MFNet[Array],
        als_result: MFNetALSFitResult[Array],
        gradient_result: Optional[MFNetGradientFitResult[Array]] = None,
    ) -> None:
        self._surrogate = surrogate
        self._als_result = als_result
        self._gradient_result = gradient_result

    def surrogate(self) -> MFNet[Array]:
        return self._surrogate

    def als_result(self) -> MFNetALSFitResult[Array]:
        return self._als_result

    def gradient_result(self) -> Optional[MFNetGradientFitResult[Array]]:
        return self._gradient_result

    def loss_value(self) -> float:
        """Return final loss from gradient phase, or ALS final MSE."""
        if self._gradient_result is not None:
            return self._gradient_result.loss_value()
        return self._als_result.loss_history()[-1]

    def __call__(self, samples: Array) -> Array:
        return self._surrogate(samples)


class MFNetCompositeFitter(Generic[Array]):
    """Composite fitter: ALS initialization followed by gradient polish.

    Runs ALS first to get a good initial guess for polynomial coefficients,
    then optionally runs a gradient-based optimizer to refine all active
    parameters (including noise_std if unfixed).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    als_max_sweeps : int
        Maximum ALS sweeps. Default: 10.
    als_tol : float
        ALS convergence tolerance. Default: 1e-8.
    gradient_optimizer : BindableOptimizerProtocol, optional
        Optimizer for the gradient phase. If None, uses default
        ScipyTrustConstrOptimizer.
    skip_gradient : bool
        If True, skip the gradient refinement phase. Default: False.
    verbosity : int
        0=silent, 1=summary, 2=detailed. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        als_max_sweeps: int = 10,
        als_tol: float = 1e-8,
        gradient_optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        skip_gradient: bool = False,
        verbosity: int = 0,
    ) -> None:
        self._bkd = bkd
        self._als_max_sweeps = als_max_sweeps
        self._als_tol = als_tol
        self._gradient_optimizer = gradient_optimizer
        self._skip_gradient = skip_gradient
        self._verbosity = verbosity

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        network: MFNet[Array],
        train_samples_per_node: List[Array],
        train_values_per_node: List[Array],
    ) -> MFNetCompositeFitResult[Array]:
        """Fit the MFNet using ALS then gradient polish.

        Parameters
        ----------
        network : MFNet[Array]
            A validated MFNet network.
        train_samples_per_node : list of Array
            Per-node training samples indexed by node id.
        train_values_per_node : list of Array
            Per-node training values indexed by node id.

        Returns
        -------
        MFNetCompositeFitResult[Array]
        """
        # Phase 1: ALS
        als_fitter = MFNetALSFitter(
            self._bkd,
            max_sweeps=self._als_max_sweeps,
            tol=self._als_tol,
            verbosity=self._verbosity,
        )
        als_result = als_fitter.fit(
            network, train_samples_per_node, train_values_per_node
        )

        if self._skip_gradient:
            return MFNetCompositeFitResult(
                surrogate=network,
                als_result=als_result,
            )

        # Phase 2: Gradient polish
        grad_fitter = MFNetGradientFitter(
            self._bkd,
            optimizer=self._gradient_optimizer,
        )
        grad_result = grad_fitter.fit(
            network, train_samples_per_node, train_values_per_node
        )

        return MFNetCompositeFitResult(
            surrogate=network,
            als_result=als_result,
            gradient_result=grad_result,
        )
