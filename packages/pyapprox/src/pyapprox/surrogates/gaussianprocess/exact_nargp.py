"""Exact NARGP model (Nonlinear AutoRegressive GP, Perdikaris 2017).

Fitted chain of exact GPs for multi-fidelity prediction. Each non-root
GP takes augmented input [x, f_parent(x)]. Parent uncertainty is not
propagated — this is the NARGP approximation.

The fitter and fit-result classes live in fitters/nargp_fitter.py to
avoid an import cycle between this module and the fitters package.
"""

from __future__ import annotations

from typing import (
    Dict,
    Generic,
    Hashable,
    List,
)

import networkx as nx

from pyapprox.surrogates.gaussianprocess.deep.input_builder import (
    InputBuilder,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.util.backends.protocols import Array, Backend


class ExactNARGPModel(Generic[Array]):
    """Fitted chain of exact GPs for multi-fidelity prediction.

    Predictions propagate deterministically through the chain: each
    layer's posterior mean is fed as input to its children. Parent
    uncertainty is not propagated — this is the NARGP approximation.

    Parameters
    ----------
    dag : nx.DiGraph
        Directed acyclic graph encoding fidelity ordering.
        Edges point from lower to higher fidelity.
    gps : Dict[Hashable, ExactGaussianProcess[Array]]
        Fitted GP at each node.
    input_builders : Dict[Hashable, InputBuilder[Array]]
        Input assembly strategy at each node.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        gps: Dict[Hashable, ExactGaussianProcess[Array]],
        input_builders: Dict[Hashable, InputBuilder[Array]],
        bkd: Backend[Array],
    ) -> None:
        self._dag = dag
        self._gps = gps
        self._input_builders = input_builders
        self._bkd = bkd
        leaves = [n for n in dag.nodes() if dag.out_degree(n) == 0]
        if len(leaves) != 1:
            raise ValueError(
                f"Expected exactly one leaf node, got {len(leaves)}: {leaves}"
            )
        self._leaf = leaves[0]

    def dag(self) -> nx.DiGraph:
        return self._dag

    def gps(self) -> Dict[Hashable, ExactGaussianProcess[Array]]:
        return self._gps

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def _evaluate_at_node(self, node: Hashable, X: Array) -> Array:
        """Posterior mean at a node, recursing through parents."""
        parents = list(self._dag.predecessors(node))
        builder = self._input_builders[node]

        if not parents:
            return self._gps[node].predict(X)

        parent_means: List[Array] = [
            self._evaluate_at_node(p, X) for p in parents
        ]
        h = builder.build(X, parent_means, self._bkd)
        return self._gps[node].predict(h)

    def _build_leaf_input(self, X: Array) -> Array:
        """Build the leaf node's augmented input from X."""
        parents = list(self._dag.predecessors(self._leaf))
        builder = self._input_builders[self._leaf]

        if not parents:
            return X

        parent_means: List[Array] = [
            self._evaluate_at_node(p, X) for p in parents
        ]
        return builder.build(X, parent_means, self._bkd)

    def predict(self, X: Array) -> Array:
        """Predict posterior mean at the leaf (highest fidelity) node.

        Parameters
        ----------
        X : Array
            Physical input, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior mean, shape (1, n_test).
        """
        return self._evaluate_at_node(self._leaf, X)

    def predict_std(self, X: Array) -> Array:
        """Predict posterior std at the leaf node.

        Parent uncertainty is not propagated — only the leaf GP's
        conditional uncertainty given deterministic parent means.

        Parameters
        ----------
        X : Array
            Physical input, shape (nvars, n_test).

        Returns
        -------
        Array
            Posterior std, shape (1, n_test).
        """
        h = self._build_leaf_input(X)
        return self._gps[self._leaf].predict_std(h)
