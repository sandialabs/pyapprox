"""Single-layer ELBO loss for per-node DGP optimization.

Computes the negative ELBO for a single DGPLayer with deterministic input,
used by DGPSequentialFitter for layer-by-layer pre-fitting.

ELBO derivation for Gaussian likelihood with noise variance sigma^2:

    ELBO = E_q(f)[log p(y|f)] - KL[q(u) || p(u)]

    E_q(f)[log p(y|f)] = -N/2 log(2*pi*sigma^2)
                         - 1/(2*sigma^2) sum_n [(y_n - mu_n)^2 + v_n]

    -ELBO = KL[q(u)||p(u)]
            + N/2 log(2*pi*sigma^2)
            + 1/(2*sigma^2) sum_n [(y_n - mu_n)^2 + v_n]

where (mu, v) = layer.predict_marginal(h).

This is mathematically identical to the per-node contribution in
DGPELBOLoss.__call__ but without stochastic propagation: the input h
is deterministic, there is a single sample (S=1), and the weight is 1.
"""

import math
from typing import Generic

import torch

from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.hyperparameter import HyperParameterList


class SingleLayerELBOLoss(Generic[Array]):
    """Negative ELBO for a single DGPLayer with deterministic input.

    Parameters
    ----------
    layer : DGPLayer[Array]
        The DGP layer to evaluate.
    h : Array
        Deterministic layer input, shape (nvars_layer, N).
    y : Array
        Target values, shape (1, N).
    """

    def __init__(
        self,
        layer: DGPLayer[Array],
        h: Array,
        y: Array,
    ) -> None:
        if layer.likelihood() is None:
            raise ValueError(
                "SingleLayerELBOLoss requires a layer with a likelihood"
            )
        self._layer = layer
        self._h = h
        self._y = y
        self._bkd = layer.bkd()
        self._hyp_list = layer.hyp_list()

    def nvars(self) -> int:
        return self._hyp_list.nactive_params()

    def nqoi(self) -> int:
        return 1

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def __call__(self, samples: Array) -> Array:
        """Compute negative ELBO for this layer.

        Parameters
        ----------
        samples : Array
            Active hyperparameters, shape (nactive,) or (nactive, 1).

        Returns
        -------
        Array
            Negative ELBO, shape (1, 1).
        """
        bkd = self._bkd
        params = samples
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        self._hyp_list.set_active_values(params)

        neg_elbo = self._layer.kl_to_prior()

        mean, var = self._layer.predict_marginal(self._h)

        lik = self._layer.likelihood()
        assert lik is not None
        sigma2 = lik.noise_var()[0]
        N = self._y.shape[-1]

        residual = self._y[0, :] - mean[0, :]
        mse_plus_var = residual * residual + var[0, :]

        const = math.log(2.0 * math.pi) + bkd.log(sigma2)
        neg_elbo = neg_elbo + 0.5 * (
            N * const + bkd.sum(mse_plus_var) / sigma2
        )

        return bkd.reshape(neg_elbo, (1, 1))

    def __repr__(self) -> str:
        return f"SingleLayerELBOLoss(nvars={self.nvars()})"


class TorchSingleLayerELBOLoss(SingleLayerELBOLoss[torch.Tensor]):
    """Single-layer ELBO loss with autograd jacobian for torch optimizers."""

    def __init__(
        self,
        layer: DGPLayer[torch.Tensor],
        h: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        torch_bkd = layer.bkd()
        if not isinstance(torch_bkd, TorchBkd):
            raise TypeError(
                "TorchSingleLayerELBOLoss requires a TorchBkd backend, "
                f"got {type(torch_bkd).__name__}"
            )
        super().__init__(layer, h, y)
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
