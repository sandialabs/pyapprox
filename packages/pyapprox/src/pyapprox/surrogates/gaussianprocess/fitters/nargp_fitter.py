"""Exact NARGP fitter and fit result.

Fits a chain of exact GPs layer-by-layer in topological order
(Perdikaris 2017). Separated from exact_nargp.py (which holds the
model) to break the fitters ↔ exact_nargp import cycle.
"""

from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Tuple,
)

import networkx as nx

from pyapprox.optimization.minimize.protocols import BindableOptimizerProtocol
from pyapprox.surrogates.gaussianprocess.deep.input_builder import (
    InputBuilder,
    RootBuilder,
    SkipConnectedBuilder,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.exact_nargp import ExactNARGPModel
from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import (
    MeanFunction,
    ParentPassthroughMean,
    ZeroMean,
)
from pyapprox.surrogates.kernels.base import Kernel
from pyapprox.util.backends.protocols import Array, Backend


class ExactNARGPFitResult(Generic[Array]):
    """Result from fitting an exact NARGP model.

    Parameters
    ----------
    model : ExactNARGPModel[Array]
        The fitted NARGP model.
    per_layer_nll : Dict[Hashable, Array]
        Negative log marginal likelihood at each layer.
    """

    def __init__(
        self,
        model: ExactNARGPModel[Array],
        per_layer_nll: Dict[Hashable, Array],
    ) -> None:
        self._model = model
        self._per_layer_nll = per_layer_nll

    def surrogate(self) -> ExactNARGPModel[Array]:
        return self._model

    def per_layer_nll(self) -> Dict[Hashable, Array]:
        return self._per_layer_nll

    def neg_log_marginal_likelihood(self) -> Array:
        """Sum of per-layer NLLs."""
        bkd = self._model.bkd()
        vals = list(self._per_layer_nll.values())
        total = vals[0]
        for v in vals[1:]:
            total = total + v
        return bkd.reshape(total, (1, 1))


class ExactNARGPFitter(Generic[Array]):
    """Fit a chain of exact GPs in topological order (NARGP).

    Each layer is an independent exact GP fitted via maximum likelihood.
    Non-root layers receive augmented input [x, f_parent(x)] where
    f_parent is the parent GP's posterior mean evaluated at the
    training inputs. Parent uncertainty is intentionally discarded
    during training — this is the NARGP approximation of
    Perdikaris 2017.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    kernel_factory : Callable[[int, Backend[Array]], Kernel[Array]]
        Creates a kernel given (nvars, bkd). Called once per layer
        with the appropriate input dimensionality.
    nvars : int
        Number of physical input variables (before augmentation).
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for per-layer hyperparameter tuning. A fresh copy
        is created for each layer. If None, uses the default from
        GPMaximumLikelihoodFitter (ScipyTrustConstr).
    nugget : float
        Numerical stability parameter for each GP. Default 1e-6.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        kernel_factory: Callable[[int, Backend[Array]], Kernel[Array]],
        nvars: int,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        nugget: float = 1e-6,
    ) -> None:
        self._bkd = bkd
        self._kernel_factory = kernel_factory
        self._nvars = nvars
        self._optimizer = optimizer
        self._nugget = nugget

    def fit(
        self,
        dag: nx.DiGraph,
        data: Dict[Hashable, Tuple[Array, Array]],
    ) -> ExactNARGPFitResult[Array]:
        """Fit exact GPs layer-by-layer in topological order.

        Parameters
        ----------
        dag : nx.DiGraph
            Fidelity DAG. Edges point low → high fidelity.
        data : Dict[Hashable, Tuple[Array, Array]]
            Training data per node: {node: (X, y)} where X has shape
            (nvars, n_samples) and y has shape (1, n_samples).

        Returns
        -------
        ExactNARGPFitResult[Array]
            Fitted model and per-layer diagnostics.
        """
        fitted_gps: Dict[Hashable, ExactGaussianProcess[Array]] = {}
        input_builders: Dict[Hashable, InputBuilder[Array]] = {}
        per_layer_nll: Dict[Hashable, Array] = {}

        def _predict_at_node(node: Hashable, X: Array) -> Array:
            parents_of = list(dag.predecessors(node))
            bldr = input_builders[node]
            if not parents_of:
                return fitted_gps[node].predict(X)
            pmeans: List[Array] = [
                _predict_at_node(p, X) for p in parents_of
            ]
            h_aug = bldr.build(X, pmeans, self._bkd)
            return fitted_gps[node].predict(h_aug)

        for node in nx.topological_sort(dag):
            if node not in data:
                continue

            X_node, y_node = data[node]
            parents = list(dag.predecessors(node))

            if not parents:
                h = X_node
                input_dim = self._nvars
                builder: InputBuilder[Array] = RootBuilder()
                mean_fn: MeanFunction[Array] = ZeroMean(self._bkd)
            else:
                parent_means: List[Array] = []
                for p in parents:
                    parent_means.append(_predict_at_node(p, X_node))
                builder = SkipConnectedBuilder()
                h = builder.build(X_node, parent_means, self._bkd)
                input_dim = self._nvars + len(parents)
                mean_fn = ParentPassthroughMean(
                    parent_start=self._nvars, bkd=self._bkd,
                )

            kernel = self._kernel_factory(input_dim, self._bkd)
            gp = ExactGaussianProcess(
                kernel, input_dim, self._bkd, mean_fn, self._nugget,
            )

            fitter = GPMaximumLikelihoodFitter(
                self._bkd, optimizer=self._optimizer,
            )
            result = fitter.fit(gp, h, y_node)

            fitted_gps[node] = result.surrogate()
            input_builders[node] = builder
            per_layer_nll[node] = result.neg_log_marginal_likelihood()

        model = ExactNARGPModel(dag, fitted_gps, input_builders, self._bkd)
        return ExactNARGPFitResult(model, per_layer_nll)
