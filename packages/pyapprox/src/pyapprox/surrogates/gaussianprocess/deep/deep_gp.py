"""Deep Gaussian Process: DAG of DGPLayers with forward propagation.

Composes multiple sparse variational GP layers into a deep GP using
doubly-stochastic variational inference (Salimbeni & Deisenroth, 2017).
The DAG topology is defined by a networkx DiGraph; forward propagation
is delegated to a LayerPropagator strategy object.
"""

import copy
from typing import Dict, Generic, Hashable, List, Optional, Tuple

import networkx as nx

from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class DeepGaussianProcess(Generic[Array]):
    """DAG of DGPLayer nodes with forward propagation.

    Parameters
    ----------
    dag : nx.DiGraph
        Directed acyclic graph defining layer connectivity.
        Nodes are hashable identifiers; edges go from parent to child.
    layers : Dict[Hashable, DGPLayer[Array]]
        Map from node id to DGPLayer. Must have an entry for every
        node in the DAG.
    propagator : LayerPropagator[Array]
        Strategy for forward propagation through the DAG.
    bkd : Backend[Array]
        Backend for numerical operations.
    n_propagation : int
        Default number of propagation samples for predict/predict_std.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        layers: Dict[Hashable, DGPLayer[Array]],
        propagator: LayerPropagator[Array],
        bkd: Backend[Array],
        n_propagation: int = 10,
    ) -> None:
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("dag must be a directed acyclic graph")
        dag_nodes = set(dag.nodes())
        layer_keys = set(layers.keys())
        if dag_nodes != layer_keys:
            missing = dag_nodes - layer_keys
            extra = layer_keys - dag_nodes
            raise ValueError(
                f"layers keys must match dag nodes. "
                f"Missing: {missing}, Extra: {extra}"
            )

        self._dag = dag
        self._layers = layers
        self._propagator = propagator
        self._bkd = bkd
        self._n_propagation = n_propagation
        self._fitted = False

        self._leaf_nodes = [
            n for n in dag.nodes() if dag.out_degree(n) == 0
        ]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def dag(self) -> nx.DiGraph:
        return self._dag

    def layers(self) -> Dict[Hashable, DGPLayer[Array]]:
        return self._layers

    def propagator(self) -> LayerPropagator[Array]:
        return self._propagator

    def set_propagator(self, propagator: LayerPropagator[Array]) -> None:
        self._propagator = propagator

    def n_propagation(self) -> int:
        return self._n_propagation

    def leaf_nodes(self) -> List[Hashable]:
        return list(self._leaf_nodes)

    def _resolve_target(self, target: Optional[Hashable]) -> Hashable:
        if target is not None:
            return target
        if len(self._leaf_nodes) == 1:
            return self._leaf_nodes[0]
        raise ValueError(
            "target must be specified when DAG has multiple leaf nodes: "
            f"{self._leaf_nodes}"
        )

    def __call__(self, X: Array) -> Array:
        """Predict posterior mean at X, shape (1, n_test)."""
        return self.predict(X)

    def predict(
        self,
        X: Array,
        target: Optional[Hashable] = None,
        n_propagation: Optional[int] = None,
    ) -> Array:
        """Predictive mean, shape (1, n_test).

        Aggregated via law of total expectation over propagation samples.
        """
        target = self._resolve_target(target)
        S = n_propagation if n_propagation is not None else self._n_propagation
        mean, _ = self._propagator.predict_mean_and_std(
            self._dag, self._layers, X, target, n_samples=S,
        )
        return mean

    def predict_std(
        self,
        X: Array,
        target: Optional[Hashable] = None,
        n_propagation: Optional[int] = None,
    ) -> Array:
        """Predictive std, shape (1, n_test).

        Law of total variance over propagation samples.
        """
        target = self._resolve_target(target)
        S = n_propagation if n_propagation is not None else self._n_propagation
        _, std = self._propagator.predict_mean_and_std(
            self._dag, self._layers, X, target, n_samples=S,
        )
        return std

    def predictive_samples(
        self,
        X: Array,
        n_samples: int,
        target: Optional[Hashable] = None,
    ) -> Array:
        """Correlated samples through the DAG, shape (n_samples, 1, n_test)."""
        target = self._resolve_target(target)
        sample_cache = self._propagator.sample_forward(
            self._dag, self._layers, X, n_samples=n_samples,
        )
        return sample_cache[target]

    def hyp_list(self) -> HyperParameterList[Array]:
        """Aggregated hyperparameters from all layers in topological order."""
        hyps: List = []
        for node in nx.topological_sort(self._dag):
            hyps += self._layers[node].hyp_list().hyperparameters()
        return HyperParameterList(hyps)

    def kl_total(self) -> Array:
        """Sum of KL divergences across all layers."""
        bkd = self._bkd
        total = bkd.zeros((1,))[0]
        for layer in self._layers.values():
            total = total + layer.kl_to_prior()
        return total

    def is_fitted(self) -> bool:
        return self._fitted

    def set_fitted(self, fitted: bool = True) -> None:
        self._fitted = fitted

    def _clone_unfitted(self) -> "DeepGaussianProcess[Array]":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        n_nodes = self._dag.number_of_nodes()
        n_edges = self._dag.number_of_edges()
        return (
            f"DeepGaussianProcess(nodes={n_nodes}, edges={n_edges}, "
            f"n_propagation={self._n_propagation})"
        )
