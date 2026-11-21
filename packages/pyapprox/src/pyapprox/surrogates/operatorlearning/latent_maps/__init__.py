"""Latent maps an operator surrogate can place between its encoders.

Every module here is bound to torch (``MLPLatentMap`` is an ``nn.Module``,
and the fitter drives ``torch.optim``), so this ``__init__`` re-exports
nothing and importing the package stays free of torch. Import from the
modules: ``latent_maps.mlp`` (``MLPLatentMap``),
``latent_maps.gradient_fitter`` (``TorchGradientLatentMapFitter``) and
``latent_maps.torch_optimizers`` (``TorchOptimizerSpec``,
``ChainedTorchOptimizer``, ``torch_adam``, ``torch_lbfgs``,
``torch_adam_then_lbfgs``).
"""
