r"""Fitting a latent map that no linear solve can reach.

:class:`WeightedLeastSquaresOperatorFitter` solves one linear system,
which is possible only when the map is linear in its parameters. A
network is not, so it needs gradient descent -- and that is the whole
difference. The objective is the same one:

.. math:: \min_\theta \sum_i w_i \|g_\theta(\hat f^i) - \hat g^i\|_2^2

a weighted residual in *coefficients*, minimized by Adam rather than in
closed form. Because the objective is the same, the two fitters must
agree wherever both apply: a latent map that happens to be affine,
fitted here, reaches the least-squares solution. That is the property
worth testing, since it checks the optimizer against an exact reference
rather than against a tolerance someone chose.

**Coefficients, not fields.** The fit target is computed by the output
encoder once, before training, so the projection is constant data
thereafter. That matters beyond efficiency: an encoder over an assembled
mass matrix would otherwise put a numpy round trip inside the
differentiated path, where a metric's contribution to a gradient is easy
to lose without any error being raised.

**The optimizer is injected, not chosen here.** Which one to run, how it
is configured and whether to chain several are all decided by the
caller through a spec; this class only drives it. That is what lets a
polish exist without the fitter knowing what a polish is, and what lets
a different ``torch.optim`` class be used without a new fitter -- their
arguments are disjoint, so a class per optimizer or a union of every
optimizer's knobs would be the alternative.

**Lives beside the latent maps rather than in `fitters.py`** because it
imports torch, and that module is reachable from every numpy-only caller
of this package.
"""

from __future__ import annotations

import copy
from typing import Generic, List, Optional, Union, cast

import numpy as np
import torch

from pyapprox.surrogates.operatorlearning.latent_maps.torch_optimizers import (
    ChainedTorchOptimizer,
    TorchOptimizerSpecProtocol,
    torch_adam_then_lbfgs,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
    LatentMapProtocol,
    require_coefficient_error_is_field_error,
)
from pyapprox.surrogates.operatorlearning.surrogate import OperatorSurrogate
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


def _weighted_squared_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    r"""Return :math:`\sum_i w_i \|p_i - t_i\|_2^2`, a scalar.

    Summed rather than averaged, so the value *is* the objective as
    written above and is directly comparable with the residual the
    closed-form solver minimizes.
    """
    difference = predicted - target
    per_sample = torch.sum(difference * difference, dim=0)
    if weights is None:
        return per_sample.sum()
    return (weights * per_sample).sum()


class TorchGradientLatentMapFitter(Generic[Array]):
    r"""Fit a latent map by gradient descent on a coefficient residual.

    Parameters
    ----------
    input_encoder : FieldEncoderProtocol[Array]
        Maps input fields to the coefficients the latent map consumes.
    output_encoder : FieldEncoderProtocol[Array]
        Maps output fields to the coefficients it predicts. Must be an
        isometry unless ``allow_proxy``, since the residual minimized
        here measures a field error only then.
    bkd : Backend[Array]
        Computational backend. Must be a ``TorchBkd``: under any other
        the graph is severed and the fit would change nothing.
    optimizer : TorchOptimizerSpecProtocol or ChainedTorchOptimizer
        Which optimizer to run, and how it is configured. Defaults to
        Adam followed by an L-BFGS polish, which is better than either
        alone: Adam is cheap and tolerant of a poor start but has no
        line search, so near an exact fit -- where the gradient is
        numerical noise -- it keeps taking full-sized steps and drifts
        back away from the optimum. Measured on a problem with an exact
        solution, Adam alone reaches a loss of 5e-03 while the polish
        then takes it to 1e-18.

        A spec rather than a built optimizer because the parameters it
        must attach to belong to a copy this fitter makes, which the
        caller cannot reach. Pass ``torch_adam(...)`` alone, or a
        ``TorchOptimizerSpec`` over any ``torch.optim`` class, to choose
        differently.
    batch_size : int, optional
        Samples per step. None uses every sample, which is what an exact
        comparison against a closed-form solve needs -- a stochastic
        gradient reaches a neighbourhood of the minimizer rather than
        the minimizer itself. Ignored by stages that declare
        ``needs_full_sample``.
    allow_proxy : bool
        Fit against a non-isometric output encoder anyway, accepting the
        coefficient residual as an approximation of the field error.
    seed : int, optional
        Seeds the batch permutation, for a reproducible fit.
    """

    def __init__(
        self,
        input_encoder: FieldEncoderProtocol[Array],
        output_encoder: FieldEncoderProtocol[Array],
        bkd: Backend[Array],
        optimizer: Optional[
            Union[TorchOptimizerSpecProtocol, ChainedTorchOptimizer]
        ] = None,
        batch_size: Optional[int] = None,
        allow_proxy: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        # Assigned before the narrowing check, so Array is not fixed to
        # Tensor for every accessor annotated Backend[Array].
        self._bkd = bkd
        if not isinstance(bkd, TorchBkd):
            raise TypeError(
                f"{type(self).__name__} trains by autograd, which needs "
                f"a TorchBkd; under {type(bkd).__name__} the graph is "
                f"severed and the fit would change nothing silently."
            )
        if batch_size is not None and batch_size < 1:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )
        if optimizer is None:
            optimizer = torch_adam_then_lbfgs()
        if isinstance(optimizer, ChainedTorchOptimizer):
            self._stages = optimizer.stages()
        elif isinstance(optimizer, TorchOptimizerSpecProtocol):
            self._stages = [optimizer]
        else:
            raise TypeError(
                f"optimizer must satisfy TorchOptimizerSpecProtocol or "
                f"be a ChainedTorchOptimizer, got "
                f"{type(optimizer).__name__}"
            )
        self._input_encoder = input_encoder
        self._output_encoder = output_encoder
        self._batch_size = batch_size
        self._allow_proxy = allow_proxy
        self._seed = seed

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def fit(
        self,
        latent_map: LatentMapProtocol[Array],
        input_fields: Array,
        output_fields: Array,
        weights: Optional[Array] = None,
    ) -> OperatorSurrogate[Array]:
        """Fit from realizations of the input and output fields.

        Encodes both, then delegates to :meth:`fit_encoded`.
        """
        return self.fit_encoded(
            latent_map,
            self._input_encoder.encode(input_fields),
            self._output_encoder.encode(output_fields),
            weights=weights,
        )

    def fit_encoded(
        self,
        latent_map: LatentMapProtocol[Array],
        coefs_in: Array,
        coefs_out: Array,
        weights: Optional[Array] = None,
    ) -> OperatorSurrogate[Array]:
        """Fit from coefficients that are already encoded.

        Parameters
        ----------
        latent_map : LatentMapProtocol[Array]
            The map to fit. Not modified: a deep copy is trained and
            returned inside the surrogate, so the argument stays usable
            as an untrained template.
        coefs_in : Array
            Encoded input realizations. Shape: (ncodes_in, nsamples)
        coefs_out : Array
            Encoded output realizations. Shape: (ncodes_out, nsamples)
        weights : Array, optional
            Per-sample weights. Shape: (nsamples,)

        Returns
        -------
        OperatorSurrogate[Array]
            Wrapping the trained copy.
        """
        require_coefficient_error_is_field_error(
            self._output_encoder,
            f"{type(self).__name__} minimizes a coefficient residual, "
            f"which",
            self._allow_proxy,
        )
        if not isinstance(latent_map, torch.nn.Module):
            raise TypeError(
                f"{type(self).__name__} optimizes torch parameters, so "
                f"the latent map must be an nn.Module; "
                f"{type(latent_map).__name__} is not. A map linear in "
                f"its parameters wants "
                f"WeightedLeastSquaresOperatorFitter instead, which "
                f"solves for it directly."
            )
        nsamples = int(coefs_in.shape[1])
        if int(coefs_out.shape[1]) != nsamples:
            raise ValueError(
                f"coefs_out has {int(coefs_out.shape[1])} realizations "
                f"but coefs_in has {nsamples}"
            )
        if weights is not None and weights.shape != (nsamples,):
            raise ValueError(
                f"weights has wrong shape {weights.shape}, expected "
                f"({nsamples},)"
            )

        if self._seed is not None:
            np.random.seed(self._seed)

        fitted = copy.deepcopy(latent_map)
        parameters = list(fitted.parameters())
        if not parameters:
            raise ValueError(
                f"{type(latent_map).__name__} exposes no trainable "
                f"parameters, so there is nothing to optimize."
            )
        # The constructor established that the backend is TorchBkd, so
        # Array is Tensor for this instance -- but the class is generic
        # over Array and mypy cannot relate a runtime isinstance check
        # to a type parameter. The internals below are tensor
        # arithmetic, so narrow once, here, instead of threading a free
        # Array through them.
        torch_map = cast(LatentMapProtocol[torch.Tensor], fitted)
        tensor_in = torch.as_tensor(coefs_in)
        tensor_out = torch.as_tensor(coefs_out)
        tensor_weights = (
            None if weights is None else torch.as_tensor(weights)
        )
        for stage in self._stages:
            self._run_stage(
                stage,
                torch_map,
                parameters,
                tensor_in,
                tensor_out,
                tensor_weights,
            )

        return OperatorSurrogate(
            self._input_encoder, self._output_encoder, fitted, self._bkd
        )

    def _run_stage(
        self,
        stage: TorchOptimizerSpecProtocol,
        fitted: LatentMapProtocol[torch.Tensor],
        parameters: List[torch.nn.Parameter],
        coefs_in: torch.Tensor,
        coefs_out: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> None:
        """Run one optimizer to completion, in place.

        Every torch optimizer takes a closure, so the same loop drives a
        first-order method and a quasi-Newton one; what differs is how
        many times ``step`` is called and whether it may see a batch.
        L-BFGS asks for the full sample because it builds curvature from
        successive gradients, and a gradient that changed with the batch
        would corrupt that estimate.
        """
        nsamples = int(coefs_in.shape[1])
        optimizer = stage.build(parameters)
        full_sample = (
            stage.needs_full_sample()
            or self._batch_size is None
            or self._batch_size >= nsamples
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            if full_sample:
                batch_in, batch_out = coefs_in, coefs_out
                batch_weights = weights
            else:
                index = np.random.permutation(nsamples)[
                    : self._batch_size
                ]
                batch_in = coefs_in[:, index]
                batch_out = coefs_out[:, index]
                batch_weights = (
                    None if weights is None else weights[index]
                )
            predicted = torch.as_tensor(fitted(batch_in))
            loss = _weighted_squared_error(
                predicted, batch_out, batch_weights
            )
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        for _ in range(stage.nsteps()):
            # torch types step's closure as returning float while every
            # optimizer in fact takes the loss tensor it backpropagated.
            optimizer.step(closure)  # type: ignore[arg-type]
