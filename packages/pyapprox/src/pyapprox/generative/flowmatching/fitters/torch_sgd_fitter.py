"""Torch SGD fitter for stochastic flow matching.

Trains a torch nn.Module velocity field using fresh Monte Carlo samples
each step — the standard stochastic CFM training loop.
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
    TorchVFProtocol,
)
from pyapprox.generative.flowmatching.samplers import (
    SourceSamplerProtocol,
    TargetSamplerProtocol,
)
from pyapprox.generative.flowmatching.time_weight import UniformWeight
from pyapprox.util.backends.torch import TorchBkd


class TorchSGDFitter:
    """Train a torch nn.Module VF using stochastic CFM.

    Each training step draws fresh source and target samples,
    samples time uniformly, and performs one gradient step.

    Parameters
    ----------
    lr : float
        Learning rate for Adam optimizer.
    n_steps : int
        Number of training steps.
    batch_size : int
        Samples per training step.
    verbose : int
        Print loss every ``verbose`` steps. 0 for silent.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        n_steps: int = 10000,
        batch_size: int = 256,
        verbose: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self._lr = lr
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._verbose = verbose
        self._seed = seed

    def bkd(self) -> TorchBkd:
        """Return the computational backend."""
        return TorchBkd()

    def fit(
        self,
        vf: TorchVFProtocol,
        path: ProbabilityPathProtocol[torch.Tensor],
        source_sampler: SourceSamplerProtocol,
        target_sampler: TargetSamplerProtocol,
        time_weight: Optional[TimeWeightProtocol[torch.Tensor]] = None,
    ) -> FlowMatchingFitResult[torch.Tensor]:
        """Fit a torch VF using stochastic CFM training.

        Parameters
        ----------
        vf : TorchVFProtocol
            Velocity field to train. Deep-cloned internally.
        path : ProbabilityPathProtocol[torch.Tensor]
            Probability path (e.g. ``LinearPath(TorchBkd())``).
        source_sampler : SourceSamplerProtocol
            Draws source samples x0.
        target_sampler : TargetSamplerProtocol
            Draws target samples (x1, weights).
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

        rng = np.random.RandomState(self._seed)
        optimizer = torch.optim.Adam(fitted_vf.parameters(), lr=self._lr)
        last_loss = float("inf")

        for step in range(self._n_steps):
            x0 = source_sampler.sample_x0(self._batch_size)
            x1, w1 = target_sampler.sample_x1(self._batch_size)

            t_vals = torch.as_tensor(
                rng.uniform(0.0, 1.0, size=(1, self._batch_size)),
                dtype=dtype,
            )

            x_t = path.interpolate(t_vals, x0, x1)
            u_t = path.target_field(t_vals, x0, x1)

            vf_input = torch.vstack([t_vals, x_t])
            w_t = tw(t_vals)
            combined_w = w1 * w_t[0, :]

            v_pred = fitted_vf.forward_torch(vf_input)
            loss = weighted_mse(v_pred, u_t, combined_w)

            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()

            last_loss = loss.item()
            if self._verbose > 0 and (step + 1) % self._verbose == 0:
                print(
                    f"Step {step + 1}/{self._n_steps}: "
                    f"loss={last_loss:.6e}"
                )

        return FlowMatchingFitResult(
            surrogate=fitted_vf,
            training_loss=last_loss,
        )
