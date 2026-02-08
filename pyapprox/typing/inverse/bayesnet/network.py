"""
Gaussian Bayesian Network using networkx.

A Bayesian network of linear-Gaussian CPDs where each node represents
a random variable and edges represent conditional dependencies.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple

import networkx as nx
import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from .factor import GaussianFactor
from .conversions import convert_cpd_to_canonical, convert_prior_to_factor


class GaussianNetwork(Generic[Array]):
    """
    Bayesian network of linear-Gaussian models using networkx.

    Each node has:
    - A unique integer ID (the networkx node identifier)
    - Number of dimensions (nvars)
    - Either a prior (if no parents) or a linear-Gaussian CPD

    The CPD for a node with parents is:
        x_child = sum_j(A_j @ x_parent_j) + b + noise
        noise ~ N(0, noise_cov)

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> network = GaussianNetwork(bkd)
    >>> # Add root node with prior
    >>> network.add_node(0, nvars=1, prior_mean=np.array([0.0]),
    ...                  prior_cov=np.array([[1.0]]))
    >>> # Add child node with CPD
    >>> network.add_node(1, nvars=1, parents=[0],
    ...                  cpd_coefficients=[np.array([[1.0]])],
    ...                  cpd_offset=np.array([0.0]),
    ...                  cpd_noise_cov=np.array([[0.5]]))
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._graph: nx.DiGraph = nx.DiGraph()
        self._node_data: Dict[int, Dict[str, Any]] = {}

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def graph(self) -> nx.DiGraph:
        """Get the underlying networkx graph."""
        return self._graph

    def add_node(
        self,
        node_id: int,
        nvars: int,
        prior_mean: Optional[Array] = None,
        prior_cov: Optional[Array] = None,
        parents: Optional[List[int]] = None,
        cpd_coefficients: Optional[List[Array]] = None,
        cpd_offset: Optional[Array] = None,
        cpd_noise_cov: Optional[Array] = None,
    ) -> None:
        """
        Add a node to the network.

        For root nodes (no parents), provide prior_mean and prior_cov.
        For non-root nodes, provide parents and CPD parameters.

        Parameters
        ----------
        node_id : int
            Unique identifier for this node.
        nvars : int
            Number of dimensions for this variable.
        prior_mean : Array, optional
            Prior mean for root nodes. Shape: (nvars,)
        prior_cov : Array, optional
            Prior covariance for root nodes. Shape: (nvars, nvars)
        parents : List[int], optional
            Parent node IDs for non-root nodes.
        cpd_coefficients : List[Array], optional
            Coefficient matrices for CPD. One per parent.
            cpd_coefficients[i] has shape (nvars, parent_nvars[i])
        cpd_offset : Array, optional
            Offset/intercept for CPD. Shape: (nvars,)
        cpd_noise_cov : Array, optional
            Noise covariance for CPD. Shape: (nvars, nvars)
        """
        if node_id in self._graph:
            raise ValueError(f"Node {node_id} already exists")

        self._graph.add_node(node_id)
        self._node_data[node_id] = {"nvars": nvars}

        if parents is None or len(parents) == 0:
            # Root node - must have prior
            if prior_mean is None or prior_cov is None:
                raise ValueError(
                    f"Root node {node_id} requires prior_mean and prior_cov"
                )
            if prior_mean.ndim == 2:
                prior_mean = prior_mean.flatten()
            self._node_data[node_id]["prior_mean"] = prior_mean
            self._node_data[node_id]["prior_cov"] = prior_cov
            self._node_data[node_id]["is_root"] = True
        else:
            # Non-root node - must have CPD
            if cpd_coefficients is None or cpd_noise_cov is None:
                raise ValueError(
                    f"Non-root node {node_id} requires cpd_coefficients "
                    "and cpd_noise_cov"
                )
            if len(cpd_coefficients) != len(parents):
                raise ValueError(
                    f"cpd_coefficients length {len(cpd_coefficients)} "
                    f"doesn't match parents length {len(parents)}"
                )

            # Add edges from parents
            for parent_id in parents:
                if parent_id not in self._graph:
                    raise ValueError(f"Parent node {parent_id} not found")
                self._graph.add_edge(parent_id, node_id)

            if cpd_offset is None:
                cpd_offset = self._bkd.zeros((nvars,))
            elif cpd_offset.ndim == 2:
                cpd_offset = cpd_offset.flatten()

            self._node_data[node_id]["parents"] = list(parents)
            self._node_data[node_id]["cpd_coefficients"] = cpd_coefficients
            self._node_data[node_id]["cpd_offset"] = cpd_offset
            self._node_data[node_id]["cpd_noise_cov"] = cpd_noise_cov
            self._node_data[node_id]["is_root"] = False

    def get_node_nvars(self, node_id: int) -> int:
        """Get the number of dimensions for a node."""
        return self._node_data[node_id]["nvars"]

    def get_parents(self, node_id: int) -> List[int]:
        """Get parent node IDs."""
        return list(self._graph.predecessors(node_id))

    def get_children(self, node_id: int) -> List[int]:
        """Get child node IDs."""
        return list(self._graph.successors(node_id))

    def is_root(self, node_id: int) -> bool:
        """Check if node is a root (no parents)."""
        return self._node_data[node_id]["is_root"]

    def get_prior_cov(self, node_id: int) -> Array:
        """Get the prior covariance for a root node.

        Parameters
        ----------
        node_id : int
            A root node ID.

        Returns
        -------
        Array
            Prior covariance matrix. Shape: (nvars, nvars)
        """
        data = self._node_data[node_id]
        if not data["is_root"]:
            raise ValueError(
                f"Node {node_id} is not a root node"
            )
        return data["prior_cov"]

    def get_cpd_coefficients(self, node_id: int) -> List[Array]:
        """Get the CPD coefficient matrices for a non-root node.

        Parameters
        ----------
        node_id : int
            A non-root node ID.

        Returns
        -------
        List[Array]
            Coefficient matrices, one per parent.
            cpd_coefficients[i] has shape (nvars_child, nvars_parent_i)
        """
        data = self._node_data[node_id]
        if data["is_root"]:
            raise ValueError(
                f"Node {node_id} is a root node (no CPD)"
            )
        return data["cpd_coefficients"]

    def get_cpd_noise_cov(self, node_id: int) -> Array:
        """Get the CPD noise covariance for a non-root node.

        Parameters
        ----------
        node_id : int
            A non-root node ID.

        Returns
        -------
        Array
            Noise covariance matrix. Shape: (nvars, nvars)
        """
        data = self._node_data[node_id]
        if data["is_root"]:
            raise ValueError(
                f"Node {node_id} is a root node (no CPD)"
            )
        return data["cpd_noise_cov"]

    def nodes(self) -> List[int]:
        """Get all node IDs."""
        return list(self._graph.nodes())

    def convert_to_factors(self) -> List[GaussianFactor[Array]]:
        """
        Convert network to list of Gaussian factors.

        Each node becomes one factor:
        - Root nodes become prior factors
        - Non-root nodes become CPD factors

        Returns
        -------
        List[GaussianFactor]
            List of factors for variable elimination.
        """
        factors: List[GaussianFactor[Array]] = []

        for node_id in self._graph.nodes():
            data = self._node_data[node_id]
            nvars = data["nvars"]

            if data["is_root"]:
                # Prior factor
                factor = convert_prior_to_factor(
                    data["prior_mean"],
                    data["prior_cov"],
                    node_id,
                    self._bkd,
                )
            else:
                # CPD factor
                parents = data["parents"]
                cpd_coeffs = data["cpd_coefficients"]

                # Build combined coefficient matrix A
                parent_nvars_per_var = [
                    self._node_data[p]["nvars"] for p in parents
                ]
                total_parent_dims = sum(parent_nvars_per_var)
                A = self._bkd.zeros((nvars, total_parent_dims))
                A_np = self._bkd.to_numpy(A)

                offset = 0
                for i, (parent_id, coeff) in enumerate(zip(parents, cpd_coeffs)):
                    coeff_np = self._bkd.to_numpy(coeff)
                    parent_nvars = self._node_data[parent_id]["nvars"]
                    A_np[:, offset:offset + parent_nvars] = coeff_np
                    offset += parent_nvars

                A = self._bkd.asarray(A_np)

                factor = convert_cpd_to_canonical(
                    A,
                    data["cpd_offset"],
                    data["cpd_noise_cov"],
                    parents,
                    parent_nvars_per_var,
                    node_id,
                    self._bkd,
                )

            factors.append(factor)

        return factors

    def topological_order(self) -> List[int]:
        """
        Get nodes in topological order (parents before children).

        Returns
        -------
        List[int]
            Node IDs in topological order.
        """
        return list(nx.topological_sort(self._graph))

    def reverse_topological_order(self) -> List[int]:
        """
        Get nodes in reverse topological order (children before parents).

        This is the typical elimination order for querying root nodes.

        Returns
        -------
        List[int]
            Node IDs in reverse topological order.
        """
        return list(reversed(self.topological_order()))

    def sample(self, nsamples: int) -> Dict[int, Array]:
        """
        Sample from the joint distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples.

        Returns
        -------
        Dict[int, Array]
            Dictionary mapping node_id to samples.
            Each array has shape (nvars, nsamples).
        """
        samples: Dict[int, Array] = {}

        for node_id in self.topological_order():
            data = self._node_data[node_id]
            nvars = data["nvars"]

            if data["is_root"]:
                # Sample from prior
                mean = data["prior_mean"]
                cov = data["prior_cov"]
                L = self._bkd.cholesky(cov)
                std_normal = self._bkd.asarray(
                    np.random.randn(nvars, nsamples)
                )
                if mean.ndim == 1:
                    mean = mean[:, None]
                node_samples = L @ std_normal + mean
            else:
                # Sample from CPD
                parents = data["parents"]
                cpd_coeffs = data["cpd_coefficients"]
                offset = data["cpd_offset"]
                noise_cov = data["cpd_noise_cov"]

                # Compute mean = sum A_i @ parent_samples + offset
                mean = self._bkd.zeros((nvars, nsamples))
                mean_np = self._bkd.to_numpy(mean)

                for parent_id, coeff in zip(parents, cpd_coeffs):
                    parent_samples = self._bkd.to_numpy(samples[parent_id])
                    coeff_np = self._bkd.to_numpy(coeff)
                    mean_np += coeff_np @ parent_samples

                offset_np = self._bkd.to_numpy(offset)
                if offset_np.ndim == 1:
                    offset_np = offset_np[:, None]
                mean_np += offset_np

                mean = self._bkd.asarray(mean_np)

                # Add noise
                L = self._bkd.cholesky(noise_cov)
                std_normal = self._bkd.asarray(
                    np.random.randn(nvars, nsamples)
                )
                node_samples = mean + L @ std_normal

            samples[node_id] = node_samples

        return samples

    def __repr__(self) -> str:
        """Return string representation."""
        n_nodes = len(self._graph.nodes())
        n_edges = len(self._graph.edges())
        return f"GaussianNetwork(nodes={n_nodes}, edges={n_edges})"
