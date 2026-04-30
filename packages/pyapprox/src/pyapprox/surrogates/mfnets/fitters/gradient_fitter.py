"""Gradient-based fitter for MFNet.

Uses BindableOptimizerProtocol to minimize the negative log-likelihood
loss over all active hyperparameters.
"""

from typing import Generic, List, Optional

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.mfnets.fitters.results import (
    MFNetGradientFitResult,
)
from pyapprox.surrogates.mfnets.losses import (
    MFNetNegLogLikelihoodLoss,
)
from pyapprox.surrogates.mfnets.network import MFNet
from pyapprox.util.backends.protocols import Array, Backend


class MFNetGradientFitter(Generic[Array]):
    """Gradient-based fitter for MFNet surrogates.

    Minimizes the negative log-likelihood loss using a configurable
    optimizer (default: ScipyTrustConstrOptimizer with finite differences).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : BindableOptimizerProtocol, optional
        Optimizer to use. Cloned during fit() to avoid shared state.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._optimizer = optimizer

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_optimizer(self, optimizer: BindableOptimizerProtocol[Array]) -> None:
        self._optimizer = optimizer

    def fit(
        self,
        network: MFNet[Array],
        train_samples_per_node: List[Array],
        train_values_per_node: List[Array],
    ) -> MFNetGradientFitResult[Array]:
        """Fit the MFNet by minimizing negative log-likelihood.

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
        MFNetGradientFitResult[Array]
            Result containing the fitted network and diagnostics.
        """
        bkd = self._bkd

        # Create loss
        loss = MFNetNegLogLikelihoodLoss(
            network, train_samples_per_node, train_values_per_node
        )

        nactive = loss.nvars()
        if nactive == 0:
            final_loss = float(bkd.to_numpy(loss(bkd.zeros((0, 1)))[0, 0]))
            return MFNetGradientFitResult(surrogate=network, loss_value=final_loss)

        # Get bounds from hyp_list
        bounds = network.hyp_list().get_active_bounds()

        # Clone optimizer (or use default)
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        # Bind and minimize
        optimizer.bind(loss, bounds)
        init_guess = network.hyp_list().get_active_values()
        init_guess = bkd.reshape(init_guess, (-1, 1))

        result = optimizer.minimize(init_guess)

        # Update network with optimal params
        optimal = result.optima()
        if optimal.ndim == 2:
            optimal = optimal[:, 0]
        network.hyp_list().set_active_values(optimal)
        network.sync_params()

        # Compute final loss
        final_loss = float(bkd.to_numpy(loss(optimal)[0, 0]))

        return MFNetGradientFitResult(
            surrogate=network,
            loss_value=final_loss,
            optimizer_result=result,
        )
