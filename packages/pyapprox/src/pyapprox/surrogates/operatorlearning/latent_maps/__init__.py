"""Latent maps an operator surrogate can place between its encoders."""

from pyapprox.surrogates.operatorlearning.latent_maps.gradient_fitter import (
    TorchGradientLatentMapFitter,
)
from pyapprox.surrogates.operatorlearning.latent_maps.mlp import (
    MLPLatentMap,
)
from pyapprox.surrogates.operatorlearning.latent_maps.torch_optimizers import (
    ChainedTorchOptimizer,
    TorchOptimizerSpec,
    TorchOptimizerSpecProtocol,
    torch_adam,
    torch_adam_then_lbfgs,
    torch_lbfgs,
)

__all__ = [
    "ChainedTorchOptimizer",
    "MLPLatentMap",
    "TorchGradientLatentMapFitter",
    "TorchOptimizerSpec",
    "TorchOptimizerSpecProtocol",
    "torch_adam",
    "torch_adam_then_lbfgs",
    "torch_lbfgs",
]
