"""MFNet network class.

The MFNet is a DAG-based multi-fidelity surrogate. It owns a NetworkX
DiGraph where each node holds a local model and edges specify which
child outputs feed into parent nodes. Forward evaluation recurses
through the DAG in topological order (leaves first).
"""

from typing import Any, Dict, Generic, List, Optional

import networkx as nx

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.surrogates.mfnets.nodes import MFNetNode
from pyapprox.typing.surrogates.mfnets.edges import MFNetEdge


class MFNet(Generic[Array]):
    """A multi-fidelity network surrogate.

    The network is a DAG where leaf nodes are low-fidelity models and
    root nodes are high-fidelity models. Non-leaf nodes combine their
    children's outputs with their own model (e.g., a discrepancy model).

    Graph convention: edges go from child (low-fidelity) to parent
    (high-fidelity). In NetworkX terms, ``predecessors(node_id)`` gives
    children and ``successors(node_id)`` gives parents.

    Parameters
    ----------
    nvars : int
        Number of global input variables.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nvars: int, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._graph = nx.DiGraph()
        self._validated = False
        self._topo_order: List[int] = []
        self._root_ids: List[int] = []
        self._leaf_ids: List[int] = []
        self._hyp_list: Optional[HyperParameterList] = None
        self._train_samples: Optional[List[Array]] = None
        self._train_values: Optional[List[Array]] = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        """Total QoI: sum of root nodes' nqoi."""
        self._check_validated()
        return sum(
            self._graph.nodes[nid]["node"].model().nqoi()
            for nid in self._root_ids
        )

    def graph(self) -> nx.DiGraph:
        return self._graph

    def add_node(self, node: MFNetNode[Array]) -> None:
        """Add a node to the network.

        Parameters
        ----------
        node : MFNetNode[Array]
            The node to add.
        """
        if not isinstance(node, MFNetNode):
            raise TypeError(
                f"node must be MFNetNode, got {type(node).__name__}"
            )
        self._graph.add_node(node.node_id(), node=node)
        self._validated = False

    def add_edge(self, edge: MFNetEdge[Array]) -> None:
        """Add an edge connecting child to parent.

        Parameters
        ----------
        edge : MFNetEdge[Array]
            The edge to add. Direction: child -> parent.
        """
        if not isinstance(edge, MFNetEdge):
            raise TypeError(
                f"edge must be MFNetEdge, got {type(edge).__name__}"
            )
        self._graph.add_edge(
            edge.child_node_id(), edge.parent_node_id(), edge=edge
        )
        self._validated = False

    def validate(self) -> None:
        """Validate the network structure and prepare for evaluation.

        This must be called after all nodes and edges are added and before
        any evaluation. It:
        1. Verifies the graph is a DAG.
        2. Computes topological order (leaves first).
        3. Sets children/parent ids on each node.
        4. Validates edge output_ids against child nqoi.
        5. Validates each node's type constraints.
        6. Aggregates hyp_list from all nodes in topo order.
        """
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("MFNet graph must be a DAG")

        # Topological order: leaves first (reversed from nx default)
        self._topo_order = list(nx.topological_sort(self._graph))

        # Set children/parent ids on each node
        for node_id in self._graph.nodes:
            node: MFNetNode[Array] = self._graph.nodes[node_id]["node"]
            children_ids = list(self._graph.predecessors(node_id))
            parent_ids = list(self._graph.successors(node_id))
            node.set_children_ids(children_ids)
            node.set_parent_ids(parent_ids)

        # Classify nodes
        self._leaf_ids = [
            nid for nid in self._topo_order
            if self._graph.nodes[nid]["node"].is_leaf()
        ]
        self._root_ids = [
            nid for nid in self._topo_order
            if self._graph.nodes[nid]["node"].is_root()
        ]

        # Validate edges
        for u, v, data in self._graph.edges(data=True):
            edge: MFNetEdge[Array] = data["edge"]
            child_node: MFNetNode[Array] = self._graph.nodes[u]["node"]
            edge.validate(child_node.model().nqoi())

        # Validate each node
        for node_id in self._graph.nodes:
            node = self._graph.nodes[node_id]["node"]
            node.validate(self._nvars)

        # Aggregate hyp_list from all nodes in topological order
        all_hyps: List[Any] = []
        for node_id in self._topo_order:
            node = self._graph.nodes[node_id]["node"]
            all_hyps.extend(node.hyp_list().hyperparameters())
        self._hyp_list = HyperParameterList(all_hyps, self._bkd)

        self._validated = True

    def hyp_list(self) -> HyperParameterList:
        """Return aggregated hyperparameter list (all nodes, topo order)."""
        self._check_validated()
        assert self._hyp_list is not None
        return self._hyp_list

    def topo_order(self) -> List[int]:
        """Return topological order (leaves first)."""
        self._check_validated()
        return self._topo_order

    def nodes(self) -> List[MFNetNode[Array]]:
        """Return all nodes in topological order."""
        self._check_validated()
        return [
            self._graph.nodes[nid]["node"] for nid in self._topo_order
        ]

    def node(self, node_id: int) -> MFNetNode[Array]:
        """Return a specific node by id."""
        return self._graph.nodes[node_id]["node"]

    def root_nodes(self) -> List[MFNetNode[Array]]:
        self._check_validated()
        return [self._graph.nodes[nid]["node"] for nid in self._root_ids]

    def leaf_nodes(self) -> List[MFNetNode[Array]]:
        self._check_validated()
        return [self._graph.nodes[nid]["node"] for nid in self._leaf_ids]

    def set_training_data(
        self,
        train_samples_per_node: List[Array],
        train_values_per_node: List[Array],
    ) -> None:
        """Store per-node training data.

        Parameters
        ----------
        train_samples_per_node : list of Array
            Training samples for each node (indexed by node id).
            Each has shape ``(nvars, nsamples_node)``.
        train_values_per_node : list of Array
            Training values for each node (indexed by node id).
            Each has shape ``(nqoi_node, nsamples_node)``.
        """
        self._train_samples = train_samples_per_node
        self._train_values = train_values_per_node

    def train_samples(self) -> Optional[List[Array]]:
        return self._train_samples

    def train_values(self) -> Optional[List[Array]]:
        return self._train_values

    def subgraph_values(
        self,
        samples: Array,
        node_id: int,
        cache: Optional[Dict[int, Array]] = None,
    ) -> Array:
        """Evaluate the subgraph rooted at ``node_id``.

        Uses memoization to avoid recomputing shared child values.

        Parameters
        ----------
        samples : Array
            Global input samples. Shape: ``(nvars, nsamples)``
        node_id : int
            The node to evaluate.
        cache : dict, optional
            Memoization cache mapping node_id to output values.

        Returns
        -------
        Array
            Output values. Shape: ``(nqoi_node, nsamples)``
        """
        if cache is None:
            cache = {}
        if node_id in cache:
            return cache[node_id]

        node: MFNetNode[Array] = self._graph.nodes[node_id]["node"]
        active_vars = node.active_sample_vars()

        if node.is_leaf():
            values = node.model()(samples[active_vars])
            cache[node_id] = values
            return values

        # Collect child outputs
        child_val_parts: List[Array] = []
        for child_id in node.children_ids():
            child_vals = self.subgraph_values(samples, child_id, cache)
            # child_vals: (nqoi_child, nsamples)
            edge_data = self._graph.edges[child_id, node_id]["edge"]
            output_ids = edge_data.output_ids()
            # Select the specified outputs
            selected = child_vals[output_ids]  # (n_selected, nsamples)
            child_val_parts.append(selected)

        # Stack child outputs: (total_child_qoi, nsamples)
        stacked_child = self._bkd.vstack(child_val_parts)

        # Augmented input: [x_active; child_outputs]
        augmented = self._bkd.vstack(
            [samples[active_vars], stacked_child]
        )

        values = node.model()(augmented)
        cache[node_id] = values
        return values

    def __call__(self, samples: Array) -> Array:
        """Evaluate the network at the root nodes.

        Parameters
        ----------
        samples : Array
            Global input samples. Shape: ``(nvars, nsamples)``

        Returns
        -------
        Array
            Concatenated root outputs. Shape: ``(nqoi, nsamples)``
        """
        self._check_validated()
        cache: Dict[int, Array] = {}
        root_vals = [
            self.subgraph_values(samples, nid, cache)
            for nid in self._root_ids
        ]
        return self._bkd.vstack(root_vals)

    def _check_validated(self) -> None:
        if not self._validated:
            raise RuntimeError(
                "MFNet has not been validated. Call validate() first."
            )

    def _sync_from_hyp_list(self) -> None:
        """Sync all node models from the aggregated hyp_list.

        Called after the optimizer updates hyp_list values to propagate
        new parameter values into each node's model.
        """
        self._check_validated()
        assert self._hyp_list is not None
        # The hyp_list is built by concatenating each node's hyp_list
        # hyperparameters. Since HyperParameter objects are shared by
        # reference, set_active_values on the aggregate list automatically
        # updates each node's HyperParameter objects. We just need to
        # tell each node's model to sync from its hyp_list.
        for node_id in self._topo_order:
            node = self._graph.nodes[node_id]["node"]
            model = node.model()
            if hasattr(model, '_sync_from_hyp_list'):
                model._sync_from_hyp_list()
