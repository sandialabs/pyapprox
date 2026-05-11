"""Builder functions for common Deep GP architectures."""

from typing import Callable, Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np

from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.initializers import (
    InducingInitializer,
    RandomUniformInitializer,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ParentPassthroughMean,
    ZeroMean,
)
from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array, Backend

KernelFactory = Callable[[int, Backend[Array]], Kernel[Array]]


def build_single_fidelity_dgp(
    n_layers: int,
    nvars: int,
    num_inducing: int,
    kernel_factory: KernelFactory[Array],
    bkd: Backend[Array],
    noise_std: float = 0.1,
    noise_bounds: Tuple[float, float] = (1e-6, 1.0),
    inducing_bounds: Tuple[float, float] = (-5.0, 5.0),
    nugget: float = 1e-6,
    n_propagation: int = 10,
    seed: int = 0,
    initializer: Optional[InducingInitializer[Array]] = None,
) -> DeepGaussianProcess[Array]:
    """Build a single-fidelity Deep GP.

    Creates n_layers layers connected in sequence: 0 -> 1 -> ... -> n-1.
    Hidden layers have no likelihood; the final layer has a
    GaussianLikelihood. Hidden layers use ParentPassthroughMean so
    the prior mean passes through the parent's output.

    Parameters
    ----------
    n_layers : int
        Number of layers (must be >= 1).
    nvars : int
        Input dimensionality.
    num_inducing : int
        Number of inducing points per layer.
    kernel_factory : Callable[[int, Backend], Kernel]
        Factory that takes (nvars_for_layer, bkd) and returns a kernel.
    bkd : Backend[Array]
        Backend for numerical operations.
    noise_std : float
        Observation noise standard deviation for the output layer.
    noise_bounds : Tuple[float, float]
        Bounds for noise parameter optimization.
    inducing_bounds : Tuple[float, float]
        Bounds for inducing point locations.
    nugget : float
        Cholesky jitter for each layer.
    n_propagation : int
        Default number of propagation samples.
    seed : int
        Random seed for initial inducing point locations.
    initializer : Optional[InducingInitializer[Array]]
        Strategy for placing inducing points. Defaults to
        RandomUniformInitializer with inducing_bounds.

    Returns
    -------
    DeepGaussianProcess[Array]
        Unfitted deep GP with chain topology.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")

    if initializer is None:
        initializer = RandomUniformInitializer(inducing_bounds)

    rng = np.random.RandomState(seed)
    dag = nx.DiGraph()
    for i in range(n_layers):
        dag.add_node(i)
    for i in range(n_layers - 1):
        dag.add_edge(i, i + 1)

    layers: Dict[Hashable, DGPLayer[Array]] = {}
    for i in range(n_layers):
        mean: MeanFunction[Array]
        if i == 0:
            layer_nvars = nvars
            mean = ZeroMean(bkd)
        else:
            layer_nvars = nvars + 1
            mean = ParentPassthroughMean(parent_start=nvars, bkd=bkd)

        kernel = kernel_factory(layer_nvars, bkd)
        locs = initializer.initialize(num_inducing, layer_nvars, bkd, rng)
        ip = InducingPoints(
            nvars=layer_nvars,
            num_inducing=num_inducing,
            bkd=bkd,
            inducing_locations=locs,
            inducing_bounds=inducing_bounds,
        )
        vd = GaussianVariationalDistribution(num_inducing, bkd)

        lik: Optional[GaussianLikelihood[Array]] = None
        if i == n_layers - 1:
            lik = GaussianLikelihood(noise_std, noise_bounds, bkd)

        layers[i] = DGPLayer(
            kernel, mean, ip, vd, bkd,
            likelihood=lik, nugget=nugget,
        )

    propagator = LayerPropagator(bkd)
    return DeepGaussianProcess(
        dag, layers, propagator, bkd, n_propagation=n_propagation,
    )


def build_multilevel_dgp(
    level_nvars: List[int],
    num_inducing: int,
    kernel_factory: KernelFactory[Array],
    bkd: Backend[Array],
    noise_std: float = 0.1,
    noise_bounds: Tuple[float, float] = (1e-6, 1.0),
    inducing_bounds: Tuple[float, float] = (-5.0, 5.0),
    nugget: float = 1e-6,
    n_propagation: int = 10,
    seed: int = 0,
    initializer: Optional[InducingInitializer[Array]] = None,
) -> DeepGaussianProcess[Array]:
    """Build a multilevel Deep GP with parent-passthrough means.

    Creates a hierarchical chain where each level is a layer:
    0 -> 1 -> ... -> n-1. Level 0 is the coarsest (input: raw X).
    Subsequent levels take augmented input [X, f_parent] with a
    ParentPassthroughMean so the prior mean passes through the
    parent level's output.

    Every level gets a GaussianLikelihood since each level
    typically has its own observed data. Data for each level is
    supplied separately when fitting via the data dict.

    Parameters
    ----------
    level_nvars : List[int]
        Input dimensionality for each level. Typically all the same
        (the physical input dimension).
    num_inducing : int
        Number of inducing points per layer.
    kernel_factory : Callable[[int, Backend], Kernel]
        Factory that takes (nvars_for_layer, bkd) and returns a kernel.
    bkd : Backend[Array]
        Backend for numerical operations.
    noise_std : float
        Observation noise standard deviation for each layer.
    noise_bounds : Tuple[float, float]
        Bounds for noise parameter optimization.
    inducing_bounds : Tuple[float, float]
        Bounds for inducing point locations.
    nugget : float
        Cholesky jitter for each layer.
    n_propagation : int
        Default number of propagation samples.
    seed : int
        Random seed for initial inducing point locations.
    initializer : Optional[InducingInitializer[Array]]
        Strategy for placing inducing points. Defaults to
        RandomUniformInitializer with inducing_bounds.

    Returns
    -------
    DeepGaussianProcess[Array]
        Unfitted multilevel deep GP.
    """
    n_levels = len(level_nvars)
    if n_levels < 1:
        raise ValueError(
            f"Need at least 1 level, got {n_levels}"
        )

    if initializer is None:
        initializer = RandomUniformInitializer(inducing_bounds)

    rng = np.random.RandomState(seed)
    dag = nx.DiGraph()
    for i in range(n_levels):
        dag.add_node(i)
    for i in range(n_levels - 1):
        dag.add_edge(i, i + 1)

    layers: Dict[Hashable, DGPLayer[Array]] = {}
    for i in range(n_levels):
        base_nvars = level_nvars[i]
        mean: MeanFunction[Array]
        if i == 0:
            layer_nvars = base_nvars
            mean = ZeroMean(bkd)
        else:
            layer_nvars = base_nvars + 1
            mean = ParentPassthroughMean(
                parent_start=base_nvars, bkd=bkd,
            )

        kernel = kernel_factory(layer_nvars, bkd)
        locs = initializer.initialize(num_inducing, layer_nvars, bkd, rng)
        ip = InducingPoints(
            nvars=layer_nvars,
            num_inducing=num_inducing,
            bkd=bkd,
            inducing_locations=locs,
            inducing_bounds=inducing_bounds,
        )
        vd = GaussianVariationalDistribution(num_inducing, bkd)
        lik = GaussianLikelihood(noise_std, noise_bounds, bkd)

        layers[i] = DGPLayer(
            kernel, mean, ip, vd, bkd,
            likelihood=lik, nugget=nugget,
        )

    propagator = LayerPropagator(bkd)
    return DeepGaussianProcess(
        dag, layers, propagator, bkd, n_propagation=n_propagation,
    )
