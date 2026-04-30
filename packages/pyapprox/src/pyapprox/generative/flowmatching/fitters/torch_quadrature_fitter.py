"""Torch-based quadrature fitter for flow matching.

Trains a torch nn.Module velocity field using pre-computed quadrature
data with full-batch or mini-batch gradient descent.
"""

import copy
from typing import Optional

import numpy as np
import torch

from pyapprox.generative.flowmatching.fitters._torch_loss import (
    weighted_mse,
)
from pyapprox.generative.flowmatching.fitters.results import (
    FlowMatchingFitResult,
)
from pyapprox.generative.flowmatching.protocols import (
    ProbabilityPathProtocol,
    TimeWeightProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.generative.flowmatching.time_weight import UniformWeight
from pyapprox.util.backends.torch import TorchBkd


def _to_tensor(
    arr: np.ndarray | torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Convert array to torch tensor with specified dtype."""
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.as_tensor(np.asarray(arr), dtype=dtype)


class TorchQuadratureFitter:
    """Train a torch nn.Module VF using pre-computed quadrature data.

    Accepts quad_data from any backend — arrays are converted to torch
    tensors at the boundary.

    Parameters
    ----------
    lr : float
        Learning rate for Adam optimizer.
    n_epochs : int
        Number of training epochs.
    batch_size : int, optional
        Mini-batch size. None for full-batch.
    verbose : int
        Print loss every ``verbose`` epochs. 0 for silent.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        n_epochs: int = 1000,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self._lr = lr
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._verbose = verbose
        self._seed = seed

    def bkd(self) -> TorchBkd:
        """Return the computational backend."""
        return TorchBkd()

    def fit(
        self,
        vf: torch.nn.Module,
        path: ProbabilityPathProtocol[torch.Tensor],
        quad_data: FlowMatchingQuadData[torch.Tensor],
        time_weight: Optional[TimeWeightProtocol[torch.Tensor]] = None,
    ) -> FlowMatchingFitResult[torch.Tensor]:
        """Fit a torch VF using quadrature data.

        Parameters
        ----------
        vf : torch.nn.Module
            Velocity field to train. Deep-cloned internally.
        path : ProbabilityPathProtocol[torch.Tensor]
            Probability path (e.g. ``LinearPath(TorchBkd())``).
        quad_data : FlowMatchingQuadData
            Pre-assembled quadrature data (any backend).
        time_weight : TimeWeightProtocol[torch.Tensor], optional
            Time-dependent weight. Defaults to uniform.

        Returns
        -------
        FlowMatchingFitResult[torch.Tensor]
        """
        path_bkd = path.bkd()
        if not isinstance(path_bkd, TorchBkd):
            raise TypeError(
                f"path must use TorchBkd, got {type(path_bkd).__name__}"
            )

        if self._seed is not None:
            np.random.seed(self._seed)

        dtype = path_bkd.default_dtype()
        fitted_vf = copy.deepcopy(vf)

        tw: TimeWeightProtocol[torch.Tensor] = (
            time_weight if time_weight is not None else UniformWeight(path_bkd)
        )

        t = _to_tensor(quad_data.t(), dtype)
        x0 = _to_tensor(quad_data.x0(), dtype)
        x1 = _to_tensor(quad_data.x1(), dtype)
        weights = _to_tensor(quad_data.weights(), dtype)

        x_t = path.interpolate(t, x0, x1)
        u_t = path.target_field(t, x0, x1)

        c_raw = quad_data.c()
        if c_raw is not None:
            c = _to_tensor(c_raw, dtype)
            vf_input = torch.vstack([t, x_t, c])
        else:
            vf_input = torch.vstack([t, x_t])

        w_t = tw(t)  # (1, n_quad)
        combined_w = weights * w_t[0, :]

        optimizer = torch.optim.Adam(fitted_vf.parameters(), lr=self._lr)
        n_quad = vf_input.shape[1]

        for epoch in range(self._n_epochs):
            if self._batch_size is None or self._batch_size >= n_quad:
                v_pred = fitted_vf.forward_torch(vf_input)
                loss = weighted_mse(v_pred, u_t, combined_w)
            else:
                idx = np.random.permutation(n_quad)[: self._batch_size]
                v_pred = fitted_vf.forward_torch(vf_input[:, idx])
                loss = weighted_mse(v_pred, u_t[:, idx], combined_w[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self._verbose > 0 and (epoch + 1) % self._verbose == 0:
                print(
                    f"Epoch {epoch + 1}/{self._n_epochs}: "
                    f"loss={loss.item():.6e}"
                )

        with torch.no_grad():
            v_final = fitted_vf.forward_torch(vf_input)
            final_loss = weighted_mse(v_final, u_t, combined_w).item()

        return FlowMatchingFitResult(
            surrogate=fitted_vf,
            training_loss=final_loss,
        )
