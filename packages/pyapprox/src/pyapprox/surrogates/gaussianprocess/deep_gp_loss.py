"""ELBO loss for Deep Gaussian Process optimization.

Implements the doubly-stochastic variational inference (DSVI) ELBO:

    L = sum_layers KL[q(u_l) || p(u_l)]
        - sum_observed_nodes (N/B) sum_batch E_prop[ E_{q(f|s)} [log p(y|f)] ]

where the outer expectation is over propagation samples (MC or quadrature)
and the inner expectation is closed-form for Gaussian likelihoods.
"""

import math
from typing import Dict, Generic, Hashable, Tuple

import torch

from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.hyperparameter import HyperParameterList


class DGPELBOLoss(Generic[Array]):
    """Negative ELBO loss for Deep GP hyperparameter optimization.

    Parameters
    ----------
    dgp : DeepGaussianProcess[Array]
        The deep GP model.
    data : Dict[Hashable, Tuple[Array, Array]]
        Training data for each observed node: {node_id: (X, y)}.
        X has shape (nvars, N), y has shape (1, N).
    n_propagation : int
        Number of propagation samples S for the outer expectation.
    """

    def __init__(
        self,
        dgp: DeepGaussianProcess[Array],
        data: Dict[Hashable, Tuple[Array, Array]],
        n_propagation: int = 10,
    ) -> None:
        layers = dgp.layers()
        for node_id, layer in layers.items():
            if layer.likelihood() is not None and node_id not in data:
                raise ValueError(
                    f"Node {node_id} has a likelihood but no training "
                    f"data was provided. Supply data for all observed nodes."
                )
        self._dgp = dgp
        self._data = data
        self._n_propagation = n_propagation
        self._bkd = dgp.bkd()
        self._hyp_list = dgp.hyp_list()

    def nvars(self) -> int:
        return self._hyp_list.nactive_params()

    def nqoi(self) -> int:
        return 1

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def __call__(self, params: Array) -> Array:
        """Compute negative ELBO.

        Parameters
        ----------
        params : Array
            Active hyperparameters, shape (nactive,) or (nactive, 1).

        Returns
        -------
        Array
            Negative ELBO, shape (1, 1).
        """
        bkd = self._bkd
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        self._hyp_list.set_active_values(params)

        neg_elbo = self._dgp.kl_total()

        dag = self._dgp.dag()
        layers = self._dgp.layers()
        propagator = self._dgp.propagator()

        for node, (X, y) in self._data.items():
            layer = layers[node]
            lik = layer.likelihood()
            if lik is None:
                raise ValueError(
                    f"Node {node} has training data but no likelihood"
                )

            means, variances, weights = propagator.predict_at(
                dag, layers, X, node, n_samples=self._n_propagation,
            )

            sigma2 = lik.noise_var()[0]
            N = y.shape[-1]
            # means: (S,1,N), variances: (S,1,N) -> squeeze to (S,N)
            mu = means[:, 0, :]
            fvar = variances[:, 0, :]
            residual = y[0, :] - mu  # (S, N)
            # weighted sum of data-dependent term: einsum avoids Python loop
            mse_plus_var = residual * residual + fvar  # (S, N)
            weighted_sum = bkd.einsum(
                "s,sn->", weights, mse_plus_var,
            )
            const = math.log(2.0 * math.pi) + bkd.log(sigma2)
            w_total = bkd.sum(weights)
            neg_elbo = neg_elbo + (
                0.5 * (w_total * N * const + weighted_sum / sigma2)
            )

        return bkd.reshape(neg_elbo, (1, 1))

    def __repr__(self) -> str:
        return (
            f"DGPELBOLoss(nvars={self.nvars()}, "
            f"n_propagation={self._n_propagation})"
        )


class TorchDGPELBOLoss(DGPELBOLoss[torch.Tensor]):
    """DGP ELBO loss with autograd-based jacobian for torch optimizers.

    Parameters are identical to DGPELBOLoss. The jacobian method uses
    bkd.jacobian (torch.autograd.functional.jacobian) to compute
    gradients of the negative ELBO w.r.t. active hyperparameters.
    """

    def __init__(
        self,
        dgp: DeepGaussianProcess[torch.Tensor],
        data: Dict[Hashable, Tuple[torch.Tensor, torch.Tensor]],
        n_propagation: int = 10,
    ) -> None:
        torch_bkd = dgp.bkd()
        if not isinstance(torch_bkd, TorchBkd):
            raise TypeError(
                "TorchDGPELBOLoss requires a TorchBkd backend, "
                f"got {type(torch_bkd).__name__}"
            )
        super().__init__(dgp, data, n_propagation)
        self._torch_bkd = torch_bkd

    def jacobian(self, params: torch.Tensor) -> torch.Tensor:
        """Gradient of negative ELBO via torch autograd.

        Parameters
        ----------
        params : torch.Tensor
            Active hyperparameters, shape (nactive,) or (nactive, 1).

        Returns
        -------
        torch.Tensor
            Jacobian, shape (1, nactive).
        """
        bkd = self._torch_bkd
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        def loss_scalar(p: torch.Tensor) -> torch.Tensor:
            return self(p)[0, 0]

        jac = bkd.jacobian(loss_scalar, params)
        return bkd.reshape(jac, (1, len(params)))
