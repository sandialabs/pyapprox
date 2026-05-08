"""Test fixtures for Deep GP tests."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.inducing import InducingPoints
from pyapprox.surrogates.gaussianprocess.likelihoods import GaussianLikelihood
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.util.backends.protocols import Array, Backend


def make_synthetic_chain_data(
    n_layers: int,
    nvars: int,
    n_per_layer: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic data for a chain DGP.

    Returns list of (X, y) tuples, one per layer, where each layer's
    input is the previous layer's output concatenated with the original X.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(nvars, n_per_layer)
    data = []
    f_prev = None
    for ll in range(n_layers):
        if f_prev is not None:
            X_aug = np.vstack([X[:, :n_per_layer], f_prev[:, :n_per_layer]])
        else:
            X_aug = X[:, :n_per_layer]
        y = np.sin(X_aug[0:1, :]) + 0.1 * rng.randn(1, n_per_layer)
        data.append((X_aug, y))
        f_prev = y
    return data


def make_kernel(
    nvars: int, bkd: Backend[Array], fixed: bool = True
) -> Matern52Kernel:
    return Matern52Kernel(
        lenscale=[1.0],
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
        fixed=fixed,
    )


def make_inducing_and_likelihood(
    nvars: int,
    num_inducing: int,
    bkd: Backend[Array],
    noise_std: float = 0.1,
    fixed: bool = True,
    seed: int = 0,
) -> Tuple[InducingPoints, GaussianLikelihood]:
    rng = np.random.RandomState(seed)
    locs = bkd.array(rng.randn(nvars, num_inducing))
    ip = InducingPoints(
        nvars=nvars,
        num_inducing=num_inducing,
        bkd=bkd,
        inducing_locations=locs,
        inducing_bounds=(-5.0, 5.0),
    )
    lik = GaussianLikelihood(noise_std, (1e-6, 1.0), bkd)
    if fixed:
        ip.hyp_list().set_all_inactive()
        lik.hyp_list().set_all_inactive()
    return ip, lik
