"""GroupACV estimator variants.

This module provides concrete implementations of BaseGroupACVEstimator:
    - GroupACVEstimatorIS: Independent sampling estimator
    - GroupACVEstimatorNested: Nested sampling estimator
    - GroupACVEstimatorTree: Tree-structured nested sampling estimator
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import networkx as nx

from pyapprox.statest.groupacv.base import BaseGroupACVEstimator
from pyapprox.statest.groupacv.utils import (
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _get_allocation_matrix_tree,
    _nest_subsets,
)
from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    pass


class GroupACVEstimatorIS(BaseGroupACVEstimator[Array]):
    """GroupACV estimator with Independent Sampling.

    This estimator uses independent partitions where each subset has its own
    separate sample partition. The allocation matrix is identity-like.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def _get_allocation_matrix(self, subsets: List[Array]) -> Array:
        """Get independent sampling allocation matrix."""
        return _get_allocation_matrix_is(subsets, self._bkd)

    def _psi_matrix(self, npartition_samples: Array) -> Array:
        bkd = self._bkd
        psi = bkd.eye(self._nT_stats) * self._reg_blue
        for k in range(len(self._subsets)):
            psi = psi + self._block_precision_contribution(
                k, npartition_samples[k]
            )
        return psi

    def _grouped_acv_beta(self, npartition_samples: Array) -> Array:
        psi = self._psi_matrix(npartition_samples)
        psi_inv = self._inv(psi)
        bkd = self._bkd
        beta_rows = []
        for asketch_row in self._asketch:
            blocks = []
            for k in range(len(self._subsets)):
                w_k = self._block_weight_contribution(
                    k, npartition_samples[k], psi_inv
                )
                blocks.append(w_k @ asketch_row)
            beta_rows.append(bkd.concatenate(blocks))
        return bkd.stack(beta_rows, axis=0)


class GroupACVEstimatorNested(BaseGroupACVEstimator[Array]):
    """GroupACV estimator with Nested Sampling.

    This estimator uses nested partitions where samples are shared across
    subsets in a hierarchical manner. The allocation matrix is lower
    triangular.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def _preprocess_model_subsets(self, model_subsets: List[Array]) -> List[Array]:
        """Preprocess model subsets for nested sampling.

        Filters out the zero subset (if present) and nests the remaining
        subsets for hierarchical sampling.
        """
        zero = self._bkd.zeros((1,), dtype=int)
        filtered: List[Array] = []
        for subset in model_subsets:
            if not isinstance(subset, self._bkd.array_type()):
                raise ValueError(
                    f"subset must be an instance of {self._bkd.array_type()}"
                )
            typed_subset = cast(Array, subset)
            if not self._bkd.allclose(typed_subset, zero):
                filtered.append(typed_subset)
        nested, _ = _nest_subsets(filtered, self.nmodels(), self._bkd)
        return nested

    def _get_allocation_matrix(self, subsets: List[Array]) -> Array:
        """Get nested sampling allocation matrix."""
        return _get_allocation_matrix_nested(subsets, self._bkd)


class GroupACVEstimatorTree(BaseGroupACVEstimator[Array]):
    """GroupACV estimator with tree-structured nested sampling.

    Generalises :class:`GroupACVEstimatorNested` from a chain of sample sets
    to a rooted forest. Group ``k`` uses the sample sets on the path from its
    root down to ``k``, so two groups share exactly the partitions above their
    lowest common ancestor and draw independent samples below it. Writing
    :math:`a(k,k')` for that ancestor, the covariance between groups is

    .. math::
        \\mathrm{Cov}(\\hat{Q}^k, \\hat{Q}^{k'})
        = \\frac{\\hat{m}^{a(k,k')}}{\\hat{m}^k \\hat{m}^{k'}} \\hat{C}^{kk'},

    which recovers the chain's :math:`1/\\max(\\hat{m}^k, \\hat{m}^{k'})` when
    the tree degenerates to a path, and independent sampling when every group
    is its own root. A branching topology is therefore useful when only some
    groups should share samples: a chain forces one total order of reuse on
    every group, which over-couples groups that are only weakly correlated.

    Unlike the nested variant this estimator does not reorder the subsets it
    is given, because ``parents`` refers to them by position and reordering
    would silently repoint the edges.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    parents : List[int]
        ``parents[k]`` is the parent group of group ``k``, or -1 if ``k`` is
        a root. Parents must precede their children, and there must be at
        least one root. ``[-1, 0, 1, ..., K-2]`` gives the chain and
        ``[-1, -1, ..., -1]`` gives independent sampling.

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated. Must have
        the same length as ``parents``.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def __init__(
        self,
        stat: Any,
        costs: Array,
        parents: Optional[List[int]] = None,
        reg_blue: float = 0,
        model_subsets: Optional[List[Array]] = None,
        asketch: Optional[Array] = None,
        use_pseudo_inv: bool = True,
        known_quantities: Optional[Any] = None,
    ) -> None:
        if parents is None:
            raise ValueError(
                "parents must be given; it is what distinguishes this "
                "estimator from GroupACVEstimatorNested, which always uses "
                "the chain"
            )
        if model_subsets is not None and len(model_subsets) != len(parents):
            raise ValueError(
                f"len(model_subsets)={len(model_subsets)} != "
                f"len(parents)={len(parents)}; parents refers to subsets by "
                "position, so the two must correspond one to one"
            )
        self._parents = list(parents)
        super().__init__(
            stat,
            costs,
            reg_blue=reg_blue,
            model_subsets=model_subsets,
            asketch=asketch,
            use_pseudo_inv=use_pseudo_inv,
            known_quantities=known_quantities,
        )

    def parents(self) -> List[int]:
        """Return the parent index of each group, -1 for a root."""
        return list(self._parents)

    def _preprocess_model_subsets(self, model_subsets: List[Array]) -> List[Array]:
        """Return the subsets unchanged, in the order given.

        The nested variant sorts by size so that a chain is well defined.
        Here the topology is supplied explicitly, so sorting would break the
        correspondence between ``parents`` and the subsets.
        """
        if len(model_subsets) != len(self._parents):
            raise ValueError(
                f"len(model_subsets)={len(model_subsets)} != "
                f"len(parents)={len(self._parents)}"
            )
        return list(model_subsets)

    def _get_allocation_matrix(self, subsets: List[Array]) -> Array:
        """Get tree-structured allocation matrix."""
        return _get_allocation_matrix_tree(self._parents, self._bkd)

    def graph(self) -> Any:
        """Return the sampling topology as a networkx directed graph.

        Nodes are partition/group indices carrying a ``models`` attribute
        holding that group's model indices; an edge runs from each parent to
        its child, i.e. in the direction of decreasing sample reuse.

        Returns
        -------
        networkx.DiGraph
        """
        graph = nx.DiGraph()
        for kk, subset in enumerate(self._model_subsets):
            graph.add_node(
                kk, models=tuple(int(m) for m in self._bkd.to_numpy(subset))
            )
        for kk, parent in enumerate(self._parents):
            if parent != -1:
                graph.add_edge(parent, kk)
        return graph

    def plot(
        self,
        ax: Any = None,
        npartition_samples: Optional[Array] = None,
    ) -> Any:
        """Draw the sampling tree.

        Each node is labelled with its group's models and, when an allocation
        is supplied, the number of samples in that partition.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on. A new figure is created if None.

        npartition_samples : Array, optional
            Per-partition sample counts to annotate the nodes with.

        Returns
        -------
        matplotlib Axes
        """
        graph = self.graph()
        if ax is None:
            _, ax = plt.subplots(figsize=(1.8 * max(len(self._parents), 3), 4))

        # Depth from the root gives the vertical position, so that sample
        # reuse reads top to bottom; siblings are spread horizontally.
        depth: Dict[int, int] = {}
        for kk, parent in enumerate(self._parents):
            depth[kk] = 0 if parent == -1 else depth[parent] + 1
        by_depth: Dict[int, List[int]] = {}
        for kk, dd in depth.items():
            by_depth.setdefault(dd, []).append(kk)
        pos = {}
        for dd, nodes in by_depth.items():
            for jj, kk in enumerate(nodes):
                pos[kk] = (jj - 0.5 * (len(nodes) - 1), -dd)

        labels = {}
        for kk in graph.nodes:
            models = ",".join(str(m) for m in graph.nodes[kk]["models"])
            label = f"$\\mathcal{{S}}^{{{kk}}}$\n{{{models}}}"
            if npartition_samples is not None:
                nsamples = self._bkd.to_numpy(npartition_samples)[kk]
                label += f"\n$n$={nsamples:g}"
            labels[kk] = label

        nx.draw_networkx_edges(
            graph, pos, ax=ax, arrows=True, arrowsize=16,
            node_size=2600, edge_color="0.4",
        )
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_size=2600, node_color="white",
            edgecolors="0.2",
        )
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8)
        ax.set_axis_off()
        ax.margins(0.15)
        return ax
