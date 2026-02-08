"""Loss functions for MFNet fitting.

Provides a Gaussian negative log-likelihood loss that sums over all
nodes in the network.
"""

import math
from typing import Dict, Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.mfnets.network import MFNet


class MFNetNegLogLikelihoodLoss(Generic[Array]):
    """Gaussian negative log-likelihood loss for MFNet.

    For each node N with training data (X_N, y_N) and noise std sigma_N:

        NLL_N = 0.5*n_N*log(2*pi) + n_N*log(sigma_N)
                + 0.5 * ||y_N - f_N(X_N)||^2 / sigma_N^2

    Total loss = sum_N NLL_N.

    Conforms to ObjectiveProtocol for use with BindableOptimizerProtocol.

    Parameters
    ----------
    network : MFNet[Array]
        The validated MFNet network.
    train_samples_per_node : list of Array
        Training samples for each node, indexed by node id.
    train_values_per_node : list of Array
        Training values for each node, indexed by node id.
    """

    def __init__(
        self,
        network: MFNet[Array],
        train_samples_per_node: List[Array],
        train_values_per_node: List[Array],
    ) -> None:
        self._network = network
        self._train_samples = train_samples_per_node
        self._train_values = train_values_per_node
        self._bkd = network.bkd()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        """Number of active hyperparameters."""
        return self._network.hyp_list().nactive_params()

    def nqoi(self) -> int:
        """Loss is scalar."""
        return 1

    def __call__(self, params: Array) -> Array:
        """Compute the total negative log-likelihood.

        Parameters
        ----------
        params : Array
            Active hyperparameter values. Shape: ``(nvars, 1)`` or ``(nvars,)``

        Returns
        -------
        Array
            Loss value. Shape: ``(1, 1)``
        """
        bkd = self._bkd
        p = bkd.flatten(params)

        # Update network parameters
        self._network.hyp_list().set_active_values(p)
        self._network._sync_from_hyp_list()

        total_nll = bkd.asarray([0.0])

        for node_id in self._network.topo_order():
            node = self._network.node(node_id)
            samples_n = self._train_samples[node_id]
            values_n = self._train_values[node_id]
            nsamples_n = samples_n.shape[1]

            # Evaluate subgraph at this node's training samples
            cache: Dict[int, Array] = {}
            pred = self._network.subgraph_values(
                samples_n, node_id, cache
            )  # (nqoi_node, nsamples_n)

            # Noise std (in user space)
            sigma = node.noise_std()  # (1,)

            # NLL contribution
            const = 0.5 * nsamples_n * math.log(2.0 * math.pi)
            log_sigma = bkd.log(sigma[0])
            residual = values_n - pred  # (nqoi_node, nsamples_n)
            sse = bkd.sum(residual * residual)
            nll = const + nsamples_n * log_sigma + 0.5 * sse / (sigma[0] ** 2)
            total_nll = total_nll + nll

        return bkd.reshape(total_nll, (1, 1))
