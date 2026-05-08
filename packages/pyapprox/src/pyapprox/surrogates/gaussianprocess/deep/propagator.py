"""Forward propagation through a DAG of DGPLayers.

Propagates inputs through a directed acyclic graph of sparse variational
GP layers, maintaining correlation through shared ancestors by threading
the same sample index through the entire DAG.
"""

from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Optional, Tuple

import networkx as nx

from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class LayerOutputDist(Generic[Array]):
    """Per-sample conditional Gaussians at a DAG node.

    Attributes
    ----------
    means : Array, shape (S, d_out, n_points)
        Conditional means for each propagation sample.
    variances : Array, shape (S, d_out, n_points)
        Conditional variances for each propagation sample.
    weights : Array, shape (S,)
        Quadrature weights from the propagation rule.
    samples : Optional[Array], shape (S, d_out, n_points)
        Drawn samples for downstream layers. None for leaf nodes
        with no children.
    """

    means: Array
    variances: Array
    weights: Array
    samples: Optional[Array]


class LayerPropagator(Generic[Array]):
    """Forward propagation through a DAG of DGPLayers.

    For root layers (no parents), input is deterministic: S=1, weight=1.
    For non-root layers, each propagation sample s uses the same
    parent sample index s, preserving correlation through shared ancestors.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def forward(
        self,
        dag: nx.DiGraph,
        layers: Dict[Hashable, DGPLayer[Array]],
        X: Array,
        n_samples: int = 1,
    ) -> Dict[Hashable, LayerOutputDist[Array]]:
        """Propagate X through all DAG nodes in topological order.

        Parameters
        ----------
        dag : nx.DiGraph
            DAG defining layer connectivity.
        layers : Dict[Hashable, DGPLayer[Array]]
            Map from node id to DGPLayer.
        X : Array, shape (nvars, n_points)
            Original input features.
        n_samples : int
            Number of propagation samples S.

        Returns
        -------
        Dict[Hashable, LayerOutputDist[Array]]
            Cached output distributions for each node.
        """
        bkd = self._bkd
        cache: Dict[Hashable, LayerOutputDist[Array]] = {}

        for node in nx.topological_sort(dag):
            layer = layers[node]
            parents = list(dag.predecessors(node))
            has_children = dag.out_degree(node) > 0

            if not parents:
                mean, var = layer.predict_marginal(X)
                means = bkd.reshape(mean, (1,) + mean.shape)
                variances = bkd.reshape(var, (1,) + var.shape)
                weights = bkd.ones((1,))
                samples = None
                if has_children:
                    samples = layer.sample(X, n_samples=n_samples)
                cache[node] = LayerOutputDist(
                    means, variances, weights, samples,
                )
            else:
                S = n_samples
                parent_dists = [cache[p] for p in parents]

                S_parent = parent_dists[0].samples.shape[0]
                S = S_parent

                n_pts = X.shape[1]
                all_means = []
                all_vars = []

                for s in range(S):
                    parent_samples_s = []
                    for pdist in parent_dists:
                        parent_samples_s.append(pdist.samples[s])

                    h = bkd.vstack([X] + parent_samples_s)
                    mean_s, var_s = layer.predict_marginal(h)
                    all_means.append(mean_s)
                    all_vars.append(var_s)

                means = bkd.stack(all_means, axis=0)
                variances = bkd.stack(all_vars, axis=0)
                weights = bkd.full((S,), 1.0 / S)

                samples = None
                if has_children:
                    sample_list = []
                    for s in range(S):
                        parent_samples_s = []
                        for pdist in parent_dists:
                            parent_samples_s.append(pdist.samples[s])
                        h = bkd.vstack([X] + parent_samples_s)
                        samp = layer.sample(h, n_samples=1)
                        sample_list.append(samp[0])
                    samples = bkd.stack(sample_list, axis=0)

                cache[node] = LayerOutputDist(
                    means, variances, weights, samples,
                )

        return cache

    def predict_at(
        self,
        dag: nx.DiGraph,
        layers: Dict[Hashable, DGPLayer[Array]],
        X: Array,
        target_node: Hashable,
        n_samples: int = 1,
    ) -> Tuple[Array, Array, Array]:
        """Get conditional Gaussians at a specific node.

        Parameters
        ----------
        dag : nx.DiGraph
            DAG defining layer connectivity.
        layers : Dict[Hashable, DGPLayer[Array]]
            Map from node id to DGPLayer.
        X : Array, shape (nvars, n_points)
            Original input features.
        target_node : Hashable
            Node to get predictions for.
        n_samples : int
            Number of propagation samples S.

        Returns
        -------
        means : Array, shape (S, 1, N)
        vars : Array, shape (S, 1, N)
        weights : Array, shape (S,)
        """
        cache = self.forward(dag, layers, X, n_samples=n_samples)
        dist = cache[target_node]
        return dist.means, dist.variances, dist.weights

    def predict_mean_and_std(
        self,
        dag: nx.DiGraph,
        layers: Dict[Hashable, DGPLayer[Array]],
        X: Array,
        target_node: Hashable,
        n_samples: int = 1,
    ) -> Tuple[Array, Array]:
        """Aggregated predictive mean and std via law of total variance.

        mean = E_s[mu_s] = sum_s w_s * mu_s
        std  = sqrt(Var_s[mu_s] + E_s[sigma^2_s])

        Parameters
        ----------
        dag : nx.DiGraph
            DAG defining layer connectivity.
        layers : Dict[Hashable, DGPLayer[Array]]
            Map from node id to DGPLayer.
        X : Array, shape (nvars, n_points)
            Original input features.
        target_node : Hashable
            Node to get predictions for.
        n_samples : int
            Number of propagation samples S.

        Returns
        -------
        mean : Array, shape (1, N)
        std : Array, shape (1, N)
        """
        bkd = self._bkd
        means, variances, weights = self.predict_at(
            dag, layers, X, target_node, n_samples,
        )

        w = bkd.reshape(weights, (-1, 1, 1))
        mean = bkd.sum(w * means, axis=0)
        e_var = bkd.sum(w * variances, axis=0)
        e_mu_sq = bkd.sum(w * means * means, axis=0)
        var_mu = e_mu_sq - mean * mean
        total_var = var_mu + e_var
        total_var = total_var * (total_var >= 0.0)
        std = bkd.sqrt(total_var)

        return mean, std

    def sample_forward(
        self,
        dag: nx.DiGraph,
        layers: Dict[Hashable, DGPLayer[Array]],
        X: Array,
        n_samples: int,
    ) -> Dict[Hashable, Array]:
        """Draw correlated samples through the entire DAG.

        Parameters
        ----------
        dag : nx.DiGraph
            DAG defining layer connectivity.
        layers : Dict[Hashable, DGPLayer[Array]]
            Map from node id to DGPLayer.
        X : Array, shape (nvars, n_points)
            Original input features.
        n_samples : int
            Number of samples S.

        Returns
        -------
        Dict[Hashable, Array]
            Samples at each node, shape (n_samples, 1, n_points).
        """
        bkd = self._bkd
        sample_cache: Dict[Hashable, Array] = {}

        for node in nx.topological_sort(dag):
            layer = layers[node]
            parents = list(dag.predecessors(node))

            if not parents:
                sample_cache[node] = layer.sample(X, n_samples=n_samples)
            else:
                sample_list = []
                for s in range(n_samples):
                    parent_samples_s = []
                    for p in parents:
                        parent_samples_s.append(sample_cache[p][s])
                    h = bkd.vstack([X] + parent_samples_s)
                    samp = layer.sample(h, n_samples=1)
                    sample_list.append(samp[0])
                sample_cache[node] = bkd.stack(sample_list, axis=0)

        return sample_cache
