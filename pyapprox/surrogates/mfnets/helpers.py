"""Utility helpers for MFNet workflows.

Provides common operations like synthetic data generation and random
coefficient initialization.
"""

from typing import List, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.mfnets.network import MFNet


def generate_synthetic_data(
    network: MFNet[Array],
    bkd: Backend[Array],
    nsamples_per_node: List[int],
    domain: Tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Array], List[Array]]:
    """Generate per-node training data from a network.

    For each node, generates uniform random samples and evaluates the
    network's subgraph to produce target values. Optionally adds
    Gaussian noise.

    Parameters
    ----------
    network : MFNet[Array]
        A validated MFNet to generate data from.
    bkd : Backend[Array]
        Computational backend.
    nsamples_per_node : list of int
        Number of training samples per node, indexed by node ID.
    domain : tuple of float
        Uniform sampling domain ``(low, high)`` per dimension.
        Default: ``(-1, 1)``.
    noise_std : float
        Standard deviation of additive Gaussian noise. Default: 0.0
        (noise-free).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_samples : list of Array
        Per-node training samples. Each has shape ``(nvars, nsamples_node)``.
    train_values : list of Array
        Per-node training values. Each has shape ``(nqoi_node, nsamples_node)``.
    """
    np.random.seed(seed)
    nvars = network.nvars()
    train_samples: List[Array] = []
    train_values: List[Array] = []

    for node_id in network.topo_order():
        n = nsamples_per_node[node_id]
        s = bkd.asarray(
            np.random.uniform(domain[0], domain[1], (nvars, n))
        )
        v = network.subgraph_values(s, node_id)
        if noise_std > 0:
            v = v + bkd.asarray(np.random.randn(*v.shape) * noise_std)
        train_samples.append(s)
        train_values.append(v)

    return train_samples, train_values


def randomize_coefficients(
    network: MFNet[Array],
    bkd: Backend[Array],
    seed: int = 0,
    scale: float = 1.0,
) -> None:
    """Set random coefficients on all models in the network.

    For each node, if the model supports ``set_coefficients``, sets them
    to random values drawn from N(0, scale^2). For
    ``MultiplicativeAdditiveDiscrepancy`` models, iterates over sub-models
    (scaling models and delta model).

    Parameters
    ----------
    network : MFNet[Array]
        A validated MFNet.
    bkd : Backend[Array]
        Computational backend.
    seed : int
        Random seed.
    scale : float
        Standard deviation of the random coefficients.
    """
    np.random.seed(seed)

    for node_id in network.topo_order():
        model = network.node(node_id).model()

        # MultiplicativeAdditiveDiscrepancy: set sub-model coefficients
        if hasattr(model, 'scaling_models') and hasattr(model, 'delta_model'):
            for sm in model.scaling_models():
                if hasattr(sm, 'set_coefficients') and hasattr(sm, 'nterms'):
                    coef = bkd.asarray(
                        np.random.randn(sm.nterms(), sm.nqoi()) * scale
                    )
                    sm.set_coefficients(coef)
            dm = model.delta_model()
            if hasattr(dm, 'set_coefficients') and hasattr(dm, 'nterms'):
                coef = bkd.asarray(
                    np.random.randn(dm.nterms(), dm.nqoi()) * scale
                )
                dm.set_coefficients(coef)
        # Simple model with set_coefficients (e.g., BasisExpansion)
        elif hasattr(model, 'set_coefficients') and hasattr(model, 'nterms'):
            coef = bkd.asarray(
                np.random.randn(model.nterms(), model.nqoi()) * scale
            )
            model.set_coefficients(coef)
