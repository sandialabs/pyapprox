"""Forward propagation through a DAG of DGPLayers.

Propagates inputs through a directed acyclic graph of sparse variational
GP layers, maintaining correlation through shared ancestors by threading
the same sample index through the entire DAG.

The propagation rule generates joint L-dimensional quadrature nodes
for the reparameterization noise across all stochastic layers.
"""

from dataclasses import dataclass
from typing import Dict, Generic, Hashable, List, Optional, Tuple

import networkx as nx

from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
    MonteCarloRule,
    PropagationRule,
)
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


def _count_node_dimensions(dag: "nx.DiGraph") -> int:
    """Number of reparameterization-noise dimensions for the DAG.

    Allocates one dimension per node. For Gaussian-likelihood ELBO
    computation the leaf node's dimension is unused, but allocating it
    keeps the rule's view of the DAG consistent across all propagator
    methods (forward, sample_forward).
    """
    return int(dag.number_of_nodes())


def _build_node_index(
    dag: "nx.DiGraph",
) -> Dict[Hashable, int]:
    """Map every node to its column index in the rule's nodes array."""
    return {
        node: idx
        for idx, node in enumerate(nx.topological_sort(dag))
    }


class LayerPropagator(Generic[Array]):
    """Forward propagation through a DAG of DGPLayers.

    For root layers (no parents), input is deterministic: S=1, weight=1
    for the mean/variance cache, but S reparameterized samples are
    drawn for children using the propagation rule's nodes.

    For non-root layers, each propagation sample s uses the same
    parent sample index s, preserving correlation through shared ancestors.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for numerical operations.
    rule : Optional[PropagationRule]
        Quadrature rule for propagation noise. Defaults to MonteCarloRule.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        rule: Optional[PropagationRule[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._rule = rule or MonteCarloRule()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def rule(self) -> PropagationRule[Array]:
        return self._rule

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

        n_dims = _count_node_dimensions(dag)
        node_idx = _build_node_index(dag)

        nodes: Optional[Array] = None
        weights: Optional[Array] = None
        if n_dims > 0:
            nodes, weights = self._rule(n_samples, n_dims, bkd)

        n_pts = X.shape[1]

        for node in nx.topological_sort(dag):
            layer = layers[node]
            parents = list(dag.predecessors(node))
            has_children = dag.out_degree(node) > 0

            L_uu = layer.compute_L_uu()

            if not parents:
                mean, var = layer.predict_marginal(X, L_uu=L_uu)
                means = bkd.reshape(mean, (1,) + mean.shape)
                variances = bkd.reshape(var, (1,) + var.shape)
                w = bkd.ones((1,))
                samples = None
                if has_children:
                    if nodes is None:
                        raise RuntimeError(
                            "nodes is None but has_children is True"
                        )
                    col = node_idx[node]
                    eps_col = nodes[:, col]
                    eps = bkd.reshape(eps_col, (n_samples, 1, 1))
                    eps = eps * bkd.full((n_samples, 1, n_pts), 1.0)
                    samples = layer.sample(
                        X, n_samples, eps=eps, L_uu=L_uu,
                    )
                cache[node] = LayerOutputDist(means, variances, w, samples)
            else:
                if nodes is None or weights is None:
                    raise RuntimeError(
                        "nodes/weights are None for non-root node"
                    )
                parent_dists = [cache[p] for p in parents]
                parent_samples_0 = parent_dists[0].samples
                if parent_samples_0 is None:
                    raise RuntimeError(
                        "Parent node has no samples for child propagation"
                    )
                S = parent_samples_0.shape[0]

                h_list: List[Array] = []
                for s in range(S):
                    parent_samples_s: List[Array] = []
                    for pdist in parent_dists:
                        if pdist.samples is None:
                            raise RuntimeError(
                                "Parent has no samples for child propagation"
                            )
                        parent_samples_s.append(pdist.samples[s])
                    h_list.append(layer.input_builder().build(
                        X, parent_samples_s, bkd,
                    ))
                h_all = bkd.concatenate(h_list, axis=1)

                mean_all, var_all = layer.predict_marginal(
                    h_all, L_uu=L_uu,
                )
                means_stacked = bkd.reshape(
                    mean_all, (S, 1, n_pts),
                )
                variances_stacked = bkd.reshape(
                    var_all, (S, 1, n_pts),
                )

                samples = None
                if has_children:
                    col = node_idx[node]
                    sigma_all = bkd.sqrt(var_all)
                    # (S,) -> (S, 1) broadcast to (S, n_pts) -> (1, S*n_pts)
                    eps_col = nodes[:, col]
                    eps_all = bkd.reshape(
                        bkd.reshape(eps_col, (S, 1)) * bkd.ones((1, n_pts)),
                        (1, S * n_pts),
                    )
                    samples_flat = mean_all + sigma_all * eps_all
                    samples = bkd.reshape(
                        samples_flat, (S, 1, n_pts),
                    )

                cache[node] = LayerOutputDist(
                    means_stacked, variances_stacked, weights, samples,
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

        n_dims = _count_node_dimensions(dag)
        node_idx = _build_node_index(dag)
        nodes, _ = self._rule(n_samples, n_dims, bkd)

        n_pts = X.shape[1]

        for node in nx.topological_sort(dag):
            layer = layers[node]
            parents = list(dag.predecessors(node))

            col = node_idx[node]

            L_uu = layer.compute_L_uu()

            if not parents:
                eps_col = nodes[:, col]
                eps = bkd.reshape(eps_col, (n_samples, 1, 1))
                eps = eps * bkd.full((n_samples, 1, n_pts), 1.0)
                sample_cache[node] = layer.sample(
                    X, n_samples=n_samples, eps=eps, L_uu=L_uu,
                )
            else:
                h_list: List[Array] = []
                for s in range(n_samples):
                    parent_samples_s = [
                        sample_cache[p][s] for p in parents
                    ]
                    h_list.append(layer.input_builder().build(
                        X, parent_samples_s, bkd,
                    ))
                h_all = bkd.concatenate(h_list, axis=1)

                mean_all, var_all = layer.predict_marginal(
                    h_all, L_uu=L_uu,
                )
                sigma_all = bkd.sqrt(var_all)

                eps_col = nodes[:, col]
                eps_all = bkd.reshape(
                    bkd.reshape(eps_col, (n_samples, 1))
                    * bkd.ones((1, n_pts)),
                    (1, n_samples * n_pts),
                )
                samples_flat = mean_all + sigma_all * eps_all
                sample_cache[node] = bkd.reshape(
                    samples_flat, (n_samples, 1, n_pts),
                )

        return sample_cache
