"""
DAG-based autoregressive multi-output kernel.

This module provides a general autoregressive multi-output kernel that supports
arbitrary directed acyclic graph (DAG) structures for output dependencies.
Uses NetworkX DiGraph directly for graph operations.
"""

from typing import Dict, Generic, List, Optional, Tuple, Union

import networkx as nx

from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.surrogates.kernels.scalings import (
    PolynomialScaling,
    ScalingFunctionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class DAGMultiOutputKernel(Generic[Array]):
    """
    Autoregressive multi-output kernel with DAG structure.

    This kernel generalizes multi-level/autoregressive GPs to arbitrary directed
    acyclic graph (DAG) structures. Each node in the DAG represents an output,
    and edges represent dependencies.

    Mathematical Model
    ------------------
    For each output i with parents P(i):
        f_i(x) = sum_{j in P(i)} ρ_{j->i}(x) * f_j(x) + δ_i(x)

    where:
    - ρ_{j->i}(x) is the scaling function for edge j->i
    - δ_i(x) ~ GP(0, k_i(x, x')) is the discrepancy at node i

    The covariance between outputs i and j is:
        K[i,j](x, x') = sum_{k in CommonAncestors(i,j)}
            w_{k->i}(x) k_k(x,x') w_{k->j}(x')

    where w_{k->i}(x) is the product of scalings along any path from k to i.

    Examples of DAG Structures
    ---------------------------
    - Sequential: 0 -> 1 -> 2 (standard multi-level)
    - Tree: 0 -> 1, 0 -> 2 (single root, multiple branches)
    - Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 (multiple paths)

    Parameters
    ----------
    dag : nx.DiGraph
        NetworkX directed graph specifying output dependencies.
        Must be a DAG with nodes 0, 1, ..., noutputs-1.
    discrepancy_kernels : List[Kernel]
        Discrepancy kernel for each node, length = noutputs.
    edge_scalings : Dict[Tuple[int, int], ScalingFunctionProtocol], optional
        Scaling functions for edges. Keys are (parent, child) tuples.
        If not provided for an edge, uses constant scaling = 1.0.

    Attributes
    ----------
    _dag : nx.DiGraph
        The NetworkX DAG structure.
    _discrepancy_kernels : List[Kernel]
        Discrepancy kernels for each node.
    _edge_scalings : Dict[Tuple[int, int], ScalingFunctionProtocol]
        Scaling functions for each edge.
    _bkd : Backend
        Backend for computations.
    _noutputs : int
        Number of outputs.
    _nvars : int
        Number of input variables.

    Examples
    --------
    >>> import networkx as nx
    >>> from pyapprox.surrogates.kernels import MaternKernel, PolynomialScaling
    >>> from pyapprox.surrogates.kernels.multioutput import DAGMultiOutputKernel
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>>
    >>> # Create sequential 3-level structure: 0 -> 1 -> 2
    >>> dag = nx.DiGraph()
    >>> dag.add_nodes_from([0, 1, 2])
    >>> dag.add_edges_from([(0, 1), (1, 2)])
    >>>
    >>> # Discrepancy kernels for each level
    >>> k0 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    >>> k1 = MaternKernel(2.5, [0.8], (0.1, 10.0), 1, bkd)
    >>> k2 = MaternKernel(2.5, [0.5], (0.1, 10.0), 1, bkd)
    >>>
    >>> # Scaling functions for edges (linear: c0 + c1*x)
    >>> rho_01 = PolynomialScaling([0.9, 0.1], (0.5, 1.5), bkd)  # 0 -> 1
    >>> rho_12 = PolynomialScaling([0.85, 0.05], (0.5, 1.5), bkd)  # 1 -> 2
    >>> edge_scalings = {(0, 1): rho_01, (1, 2): rho_12}
    >>>
    >>> # Create kernel
    >>> kernel = DAGMultiOutputKernel(dag, [k0, k1, k2], edge_scalings)
    >>>
    >>> # Use with MultiOutputGP
    >>> X_list = [X0, X1, X2]  # Training data for each output
    >>> K = kernel(X_list)

    Notes
    -----
    The ``jacobian_wrt_params`` method requires all discrepancy kernels and
    edge scalings to implement ``jacobian_wrt_params``. This is an inherent
    requirement (not optional) since the DAG kernel's Jacobian is computed
    from the component Jacobians.

    ``hvp_wrt_params`` is not currently implemented for DAG kernels.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        discrepancy_kernels: List[Kernel[Array]],
        edge_scalings: Optional[Dict[Tuple[int, int], ScalingFunctionProtocol[Array]]] = None,
    ):
        """
        Initialize the DAGMultiOutputKernel.

        Parameters
        ----------
        dag : nx.DiGraph
            NetworkX directed acyclic graph.
        discrepancy_kernels : List[Kernel]
            Discrepancy kernels, one per node.
        edge_scalings : Dict[Tuple[int, int], ScalingFunctionProtocol], optional
            Scaling functions for edges.

        Raises
        ------
        ValueError
            If dag is not a DAG.
            If number of kernels doesn't match number of nodes.
            If kernels have different backends or nvars.
        """
        # Validate DAG
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Graph must be a directed acyclic graph (DAG)")

        self._dag = dag
        self._noutputs = dag.number_of_nodes()

        # Validate nodes are 0, 1, ..., noutputs-1
        expected_nodes = set(range(self._noutputs))
        if set(dag.nodes()) != expected_nodes:
            raise ValueError(
                f"DAG nodes must be {{0, 1, ..., {self._noutputs-1}}}, "
                f"got {set(dag.nodes())}"
            )

        if len(discrepancy_kernels) != self._noutputs:
            raise ValueError(
                f"Number of discrepancy_kernels ({len(discrepancy_kernels)}) "
                f"must equal noutputs ({self._noutputs})"
            )

        # Validate all kernels have same backend and nvars
        if discrepancy_kernels:
            bkd_class = discrepancy_kernels[0].bkd().__class__
            nvars = discrepancy_kernels[0].nvars()
            for i, kernel in enumerate(discrepancy_kernels[1:], 1):
                if kernel.bkd().__class__ != bkd_class:
                    raise ValueError(
                        f"All kernels must have same backend. "
                        f"Kernel 0 has {bkd_class.__name__}, "
                        f"kernel {i} has {kernel.bkd().__class__.__name__}"
                    )
                if kernel.nvars() != nvars:
                    raise ValueError(
                        f"All kernels must have same nvars. "
                        f"Kernel 0 has {nvars}, kernel {i} has {kernel.nvars()}"
                    )

            self._bkd = discrepancy_kernels[0].bkd()
            self._nvars = nvars
        else:
            raise ValueError("discrepancy_kernels cannot be empty")

        self._discrepancy_kernels = discrepancy_kernels

        # Set up edge scalings (default to constant 1.0 if not provided)
        self._edge_scalings: Dict[Tuple[int, int], ScalingFunctionProtocol[Array]] = {}
        if edge_scalings is None:
            edge_scalings = {}

        for parent, child in dag.edges():
            if (parent, child) in edge_scalings:
                self._edge_scalings[(parent, child)] = edge_scalings[(parent, child)]
            else:
                # Default to constant scaling = 1.0 (degree 0 polynomial)
                self._edge_scalings[(parent, child)] = PolynomialScaling(
                    coefficients=[1.0],
                    bounds=(0.1, 2.0),
                    bkd=self._bkd,
                    nvars=self._nvars,
                    fixed=True,  # Fixed constant scaling
                )

        # Combine all hyperparameters
        self._hyp_list = HyperParameterList([], bkd=self._bkd)
        for kernel in self._discrepancy_kernels:
            self._hyp_list = self._hyp_list + kernel.hyp_list()
        for scaling in self._edge_scalings.values():
            self._hyp_list = self._hyp_list + scaling.hyp_list()

        # Precompute path information for efficiency
        self._precompute_paths()

    def _precompute_paths(self) -> None:
        """
        Precompute paths from each ancestor to each descendant.

        Stores in self._paths: Dict[(ancestor, descendant), List[List[int]]]
        Each list of lists contains all simple paths from ancestor to descendant.
        """
        self._paths: Dict[Tuple[int, int], List[List[int]]] = {}

        for node in range(self._noutputs):
            # Find all paths from ancestors to this node
            ancestors = nx.ancestors(self._dag, node)
            ancestors.add(node)  # Include self

            for ancestor in ancestors:
                if ancestor == node:
                    self._paths[(ancestor, node)] = [[node]]
                else:
                    # Use NetworkX to find all simple paths
                    self._paths[(ancestor, node)] = list(
                        nx.all_simple_paths(self._dag, ancestor, node)
                    )

    def _compute_scaling_product(
        self, X: Array, path: List[int]
    ) -> Array:
        """
        Compute the product of scaling functions along a path.

        For path [k, ..., i], computes product of ρ_{k->...} * ... * ρ_{...->i}.

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).
        path : List[int]
            Path from ancestor to descendant (list of node indices).

        Returns
        -------
        product : Array
            Scaling product, shape (nsamples, 1).
        """
        if len(path) <= 1:
            # No edges to traverse
            return self._bkd.ones((X.shape[1], 1))

        product = self._bkd.ones((X.shape[1], 1))
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            scaling = self._edge_scalings[(parent, child)]
            # Use eval_scaling to get scalar values, not kernel matrix
            product = product * scaling.eval_scaling(X)

        return product

    def _compute_block(
        self,
        X1: Array,
        output1: int,
        X2: Array,
        output2: int,
    ) -> Optional[Array]:
        """
        Compute the covariance block K[output1, output2].

        Parameters
        ----------
        X1 : Array
            Input points for output1, shape (nvars, n1).
        output1 : int
            First output index.
        X2 : Array
            Input points for output2, shape (nvars, n2).
        output2 : int
            Second output index.

        Returns
        -------
        block : Array
            Covariance block, shape (n1, n2).
        """
        # Find common ancestors (including outputs themselves)
        ancestors1 = nx.ancestors(self._dag, output1)
        ancestors1.add(output1)
        ancestors2 = nx.ancestors(self._dag, output2)
        ancestors2.add(output2)
        common = ancestors1.intersection(ancestors2)

        if not common:
            # No common ancestors - outputs are independent
            return self._bkd.zeros((X1.shape[1], X2.shape[1]))

        block = None

        for ancestor in common:
            # Get discrepancy kernel for this ancestor
            k_ancestor = self._discrepancy_kernels[ancestor]

            # Compute kernel matrix
            K_ancestor = k_ancestor(X1, X2)

            # Get paths from ancestor to output1 and output2
            paths1 = self._paths.get((ancestor, output1), [[ancestor]])
            paths2 = self._paths.get((ancestor, output2), [[ancestor]])

            # For each combination of paths, compute contribution
            # In autoregressive structure, there's typically only one path
            # But with DAGs, there could be multiple paths
            # We sum over all path combinations
            for path1 in paths1:
                scaling1 = self._compute_scaling_product(X1, path1)

                for path2 in paths2:
                    scaling2 = self._compute_scaling_product(X2, path2)

                    # Contribution: w_{anc->out1}(X1)
                    # * K_anc * w_{anc->out2}(X2)^T
                    contribution = scaling1 * K_ancestor * scaling2.T

                    if block is None:
                        block = contribution
                    else:
                        block = block + contribution

        if block is not None:
            return block
        return self._bkd.zeros((X1.shape[1], X2.shape[1]))

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def noutputs(self) -> int:
        """Return the number of outputs."""
        return self._noutputs

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def dag(self) -> nx.DiGraph:
        """Return the underlying NetworkX DAG."""
        return self._dag

    def __call__(
        self,
        X1_list: List[Array],
        X2_list: Optional[List[Array]] = None,
        block_format: bool = False,
    ) -> Union[Array, List[List[Optional[Array]]]]:
        """
        Compute the multi-output kernel matrix.

        Parameters
        ----------
        X1_list : List[Array]
            List of input arrays, one per output.
            Each array has shape (nvars, n_samples_output_i).
        X2_list : List[Array], optional
            List of input arrays for cross-covariance.
            If None, computes self-covariance.
        block_format : bool, optional
            If True, return list of lists of blocks.
            If False, return stacked matrix (default).

        Returns
        -------
        K : Array or List[List[Array]]
            Kernel matrix.
        """
        if len(X1_list) != self._noutputs:
            raise ValueError(
                f"X1_list length ({len(X1_list)}) must match "
                f"noutputs ({self._noutputs})"
            )

        if X2_list is None:
            X2_list = X1_list
        else:
            if len(X2_list) != self._noutputs:
                raise ValueError(
                    f"X2_list length ({len(X2_list)}) must match "
                    f"noutputs ({self._noutputs})"
                )

        # Compute all blocks
        blocks = []
        for i in range(self._noutputs):
            row = []
            for j in range(self._noutputs):
                block = self._compute_block(X1_list[i], i, X2_list[j], j)
                row.append(block)
            blocks.append(row)

        if block_format:
            return blocks

        # Stack into single matrix
        rows = [self._bkd.hstack(blocks[i]) for i in range(self._noutputs)]
        return self._bkd.vstack(rows)

    def jacobian_wrt_params(
        self,
        X1_list: List[Array],
        X2_list: Optional[List[Array]] = None
    ) -> Array:
        """
        Compute Jacobian of kernel matrix w.r.t. hyperparameters.

        Parameters
        ----------
        X1_list : List[Array]
            Input points for each output.
        X2_list : Optional[List[Array]]
            Second set of input points. If None, uses X1_list.

        Returns
        -------
        jac : Array
            Jacobian, shape (n_total, n_total, nparams) where n_total = sum(n_i).
        """
        if X2_list is None:
            X2_list = X1_list

        # Get dimensions
        n_list = [X.shape[1] for X in X1_list]
        n_total = sum(n_list)
        nparams = self._hyp_list.nparams()

        # Initialize Jacobian tensor
        jac_full = self._bkd.zeros((n_total, n_total, nparams))

        # Track parameter offset
        param_offset = 0

        # 1. Discrepancy kernel parameter gradients
        for kernel_idx, kernel in enumerate(self._discrepancy_kernels):
            kernel_nparams = kernel.hyp_list().nparams()

            if kernel_nparams == 0:
                continue

            # For each pair of outputs (i, j), check if kernel_idx is a common ancestor
            row_offset = 0
            for i in range(self._noutputs):
                n_i = n_list[i]
                col_offset = 0

                for j in range(self._noutputs):
                    n_j = [X.shape[1] for X in X2_list][j]

                    # Check if kernel_idx is a common ancestor of i and j
                    ancestors_i = nx.ancestors(self._dag, i)
                    ancestors_i.add(i)
                    ancestors_j = nx.ancestors(self._dag, j)
                    ancestors_j.add(j)

                    if kernel_idx in ancestors_i.intersection(ancestors_j):
                        # Get paths from kernel_idx to i and j
                        paths_i = self._paths.get((kernel_idx, i), [[kernel_idx]])
                        paths_j = self._paths.get((kernel_idx, j), [[kernel_idx]])

                        # Compute kernel gradient at X1_list[i], X2_list[j]
                        # For cross-covariance (X1 != X2), gradients not yet supported
                        if X1_list[i] is not X2_list[j]:
                            # Check if arrays are the same
                            if (
                                X1_list[i].shape
                                != X2_list[j].shape
                                or not self._bkd.allclose(
                                    X1_list[i], X2_list[j]
                                )
                            ):
                                raise NotImplementedError(
                                    "Cross-covariance Jacobians "
                                    "not yet supported for "
                                    "DAG kernels"
                                )

                        kernel_jac_ij = kernel.jacobian_wrt_params(
                            X1_list[i]
                        )  # Shape: (n_i, n_i, kernel_nparams)

                        # Compute scaling products for all path combinations
                        for path_i in paths_i:
                            scaling_i = self._compute_scaling_product(
                                X1_list[i], path_i
                            )

                            for path_j in paths_j:
                                scaling_j = self._compute_scaling_product(
                                    X2_list[j], path_j
                                )

                                # Gradient contribution: scaling_i * ∂k/∂θ * scaling_j^T
                                for p_idx in range(kernel_nparams):
                                    dk_dtheta = kernel_jac_ij[:, :, p_idx]
                                    contrib = scaling_i * dk_dtheta * scaling_j.T

                                    jac_full[
                                        row_offset:row_offset+n_i,
                                        col_offset:col_offset+n_j,
                                        param_offset + p_idx
                                    ] = jac_full[
                                        row_offset:row_offset+n_i,
                                        col_offset:col_offset+n_j,
                                        param_offset + p_idx
                                    ] + contrib

                    col_offset += n_j
                row_offset += n_i

            param_offset += kernel_nparams

        # 2. Edge scaling parameter gradients
        for edge, scaling in self._edge_scalings.items():
            scaling_nparams = scaling.hyp_list().nparams()

            if scaling_nparams == 0:
                continue

            parent, child = edge

            # For each pair of outputs, check if this edge is used in any path
            row_offset = 0
            for i in range(self._noutputs):
                n_i = n_list[i]
                col_offset = 0

                for j in range(self._noutputs):
                    n_j = [X.shape[1] for X in X2_list][j]

                    # Find common ancestors
                    ancestors_i = nx.ancestors(self._dag, i)
                    ancestors_i.add(i)
                    ancestors_j = nx.ancestors(self._dag, j)
                    ancestors_j.add(j)
                    common = ancestors_i.intersection(ancestors_j)

                    # For each common ancestor, check if edge is in paths
                    for ancestor in common:
                        paths_i = self._paths.get((ancestor, i), [[ancestor]])
                        paths_j = self._paths.get((ancestor, j), [[ancestor]])

                        k_ancestor = self._discrepancy_kernels[ancestor]
                        K_ancestor = k_ancestor(X1_list[i], X2_list[j])

                        for path_i in paths_i:
                            for path_j in paths_j:
                                # Check if edge is in path_i or path_j
                                edge_in_path_i = any(
                                    path_i[k] == parent and path_i[k+1] == child
                                    for k in range(len(path_i) - 1)
                                )
                                edge_in_path_j = any(
                                    path_j[k] == parent and path_j[k+1] == child
                                    for k in range(len(path_j) - 1)
                                )

                                if not (edge_in_path_i or edge_in_path_j):
                                    continue

                                # Compute scaling products and their derivatives
                                scaling_i = self._compute_scaling_product(
                                    X1_list[i], path_i
                                )
                                scaling_j = self._compute_scaling_product(
                                    X2_list[j], path_j
                                )

                                # Get scaling Jacobians
                                X_for_scaling_i = X1_list[i]
                                X_for_scaling_j = X2_list[j]
                                scaling_jac_i = scaling.jacobian_wrt_params(
                                    X_for_scaling_i
                                )  # (n_i, scaling_nparams)
                                scaling_jac_j = scaling.jacobian_wrt_params(
                                    X_for_scaling_j
                                )  # (n_j, scaling_nparams)

                                for p_idx in range(scaling_nparams):
                                    contrib = self._bkd.zeros((n_i, n_j))

                                    if edge_in_path_i:
                                        # ∂scaling_i/∂θ * K * scaling_j^T
                                        dscaling_i = self._bkd.reshape(
                                            scaling_jac_i[:, p_idx],
                                            (n_i, 1),
                                        )
                                        # Recompute scaling_i_other
                                        scaling_i_other = self._bkd.ones(
                                            (n_i, 1)
                                        )
                                        for k in range(len(path_i) - 1):
                                            p, c = path_i[k], path_i[k+1]
                                            if (p, c) != edge:
                                                s = self._edge_scalings[
                                                    (p, c)
                                                ].eval_scaling(
                                                    X_for_scaling_i
                                                )
                                                scaling_i_other = (
                                                    scaling_i_other * s
                                                )

                                        contrib = (
                                            contrib
                                            + (dscaling_i * scaling_i_other)
                                            * K_ancestor * scaling_j.T
                                        )

                                    if edge_in_path_j:
                                        # scaling_i * K * ∂scaling_j/∂θ^T
                                        dscaling_j = self._bkd.reshape(
                                            scaling_jac_j[:, p_idx],
                                            (n_j, 1),
                                        )
                                        scaling_j_other = self._bkd.ones(
                                            (n_j, 1)
                                        )
                                        for k in range(len(path_j) - 1):
                                            p, c = path_j[k], path_j[k+1]
                                            if (p, c) != edge:
                                                s = self._edge_scalings[
                                                    (p, c)
                                                ].eval_scaling(
                                                    X_for_scaling_j
                                                )
                                                scaling_j_other = (
                                                    scaling_j_other * s
                                                )

                                        contrib = (
                                            contrib
                                            + scaling_i * K_ancestor
                                            * (dscaling_j * scaling_j_other).T
                                        )

                                    jac_full[
                                        row_offset:row_offset+n_i,
                                        col_offset:col_offset+n_j,
                                        param_offset + p_idx
                                    ] = jac_full[
                                        row_offset:row_offset+n_i,
                                        col_offset:col_offset+n_j,
                                        param_offset + p_idx
                                    ] + contrib

                    col_offset += n_j
                row_offset += n_i

            param_offset += scaling_nparams

        return jac_full

    def __repr__(self) -> str:
        """String representation."""
        edges = list(self._dag.edges())
        roots = [n for n in self._dag.nodes() if self._dag.in_degree(n) == 0]
        leaves = [n for n in self._dag.nodes() if self._dag.out_degree(n) == 0]

        return (
            f"DAGMultiOutputKernel(\n"
            f"  noutputs={self._noutputs},\n"
            f"  nvars={self._nvars},\n"
            f"  edges={edges},\n"
            f"  roots={roots},\n"
            f"  leaves={leaves}\n"
            f")"
        )
