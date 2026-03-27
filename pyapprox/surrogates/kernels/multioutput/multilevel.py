"""
Multi-level autoregressive kernel for multi-fidelity Gaussian Processes.

This module provides the MultiLevelKernel, which is a convenience wrapper around
DAGMultiOutputKernel for the special case of sequential (hierarchical) structures.
"""

from typing import Dict, Generic, List, Optional, Tuple, Union

import networkx as nx

from pyapprox.surrogates.kernels.multioutput.dag_kernel import (
    DAGMultiOutputKernel,
)
from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.surrogates.kernels.scalings import (
    ScalingFunctionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class MultiLevelKernel(Generic[Array]):
    """
    Multi-level autoregressive kernel with sequential structure.

    This is a convenience wrapper around DAGMultiOutputKernel for the special
    case of sequential (hierarchical) multi-level structures: 0 -> 1 -> 2 -> ...

    The kernel implements an autoregressive multi-fidelity GP structure where
    each level builds upon the previous level with spatially varying correlation:

        f_0(x) ~ GP(0, k_0(x, x'))                                 # Level 0
        f_1(x) = ρ_0(x) f_0(x) + δ_1(x)                           # Level 1
        f_2(x) = ρ_1(x) f_1(x) + δ_2(x)                           # Level 2
        ...

    where:
    - f_l(x) is the l-th fidelity level
    - ρ_l(x) is the spatially varying scaling function from level l to l+1
    - δ_l(x) ~ GP(0, k_l(x, x')) is the discrepancy at level l
    - k_l is the discrepancy kernel for level l

    The covariance structure is computed by DAGMultiOutputKernel:
        Cov[f_i(x), f_j(x')] = sum_{k=0}^{min(i,j)} w_ik(x) k_k(x, x') w_jk(x')

    where w_ik(x) is the product of scalings from level k to level i.

    Parameters
    ----------
    kernels : List[Kernel]
        List of discrepancy kernels, one per level. kernels[l] is the
        discrepancy kernel for level l. All kernels must have the same
        nvars and backend.
    scalings : List[ScalingFunctionProtocol]
        List of spatially varying scaling functions. scalings[l] is ρ_l(x),
        the scaling from level l to level l+1. Length must be len(kernels) - 1.

    Attributes
    ----------
    _dag_kernel : DAGMultiOutputKernel
        Underlying DAG kernel with sequential structure.
    _nlevels : int
        Number of fidelity levels.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels import MaternKernel
    >>> from pyapprox.surrogates.kernels.multioutput import (
    ...     MultiLevelKernel, LinearScaling
    ... )
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> nvars = 1
    >>> # Create kernels for two levels
    >>> k0 = MaternKernel(2.5, [1.0], (0.1, 10.0), nvars, bkd)
    >>> k1 = MaternKernel(2.5, [0.5], (0.1, 10.0), nvars, bkd)
    >>> # Create spatially varying scaling: ρ(x) = 0.9 + 0.1*x
    >>> rho_0 = LinearScaling(0.9, [0.1], (0.5, 1.5), bkd)
    >>> # Create multi-level kernel
    >>> ml_kernel = MultiLevelKernel([k0, k1], [rho_0])
    >>> # Use with MultiOutputGP
    >>> X_list = [bkd.array(np.random.randn(1, 10)), bkd.array(np.random.randn(1, 5))]
    >>> K = ml_kernel(X_list)  # Cross-level covariance

    Notes
    -----
    This class is equivalent to creating a DAGMultiOutputKernel with a
    sequential DAG structure (0 -> 1 -> 2 -> ...). For more complex
    autoregressive structures (tree, diamond, etc.), use DAGMultiOutputKernel
    directly.

    The ``jacobian_wrt_params`` method delegates to the underlying
    DAGMultiOutputKernel and requires all discrepancy kernels and
    scalings to implement ``jacobian_wrt_params``.

    ``hvp_wrt_params`` is not implemented (inherited limitation from
    DAGMultiOutputKernel).
    """

    def __init__(
        self,
        kernels: List[Kernel[Array]],
        scalings: List[ScalingFunctionProtocol[Array]],
    ):
        """
        Initialize the MultiLevelKernel.

        Parameters
        ----------
        kernels : List[Kernel]
            Discrepancy kernels for each level.
        scalings : List[ScalingFunctionProtocol]
            Scaling functions between levels (length = len(kernels) - 1).

        Raises
        ------
        ValueError
            If kernels list is empty.
            If number of scalings doesn't match.
        """
        if not kernels:
            raise ValueError("kernels list cannot be empty")

        if len(scalings) != len(kernels) - 1:
            raise ValueError(
                f"Number of scalings ({len(scalings)}) must be "
                f"len(kernels) - 1 = {len(kernels) - 1}"
            )

        self._nlevels = len(kernels)

        # Create sequential DAG: 0 -> 1 -> 2 -> ... -> n-1
        dag = nx.DiGraph()
        dag.add_nodes_from(range(self._nlevels))
        edges = [(i, i + 1) for i in range(self._nlevels - 1)]
        dag.add_edges_from(edges)

        # Map scalings to edges
        edge_scalings: Dict[Tuple[int, int], ScalingFunctionProtocol[Array]] = {}
        for i, scaling in enumerate(scalings):
            edge_scalings[(i, i + 1)] = scaling

        # Create underlying DAG kernel
        self._dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        return self._dag_kernel.bkd()

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the combined list of hyperparameters.

        Returns
        -------
        hyp_list : HyperParameterList[Array]
            Combined hyperparameters from all kernels and scalings.
        """
        return self._dag_kernel.hyp_list()

    def noutputs(self) -> int:
        """
        Return the number of outputs (levels) modeled by this kernel.

        Returns
        -------
        noutputs : int
            Number of levels.
        """
        return self._nlevels

    def nvars(self) -> int:
        """
        Return the number of input variables.

        Returns
        -------
        nvars : int
            Number of input dimensions.
        """
        return self._dag_kernel.nvars()

    def __call__(
        self,
        X1_list: List[Array],
        X2_list: Optional[List[Array]] = None,
        block_format: bool = False,
    ) -> Union[Array, List[List[Optional[Array]]]]:
        """
        Compute the multi-level kernel matrix.

        Parameters
        ----------
        X1_list : List[Array]
            List of input data arrays, one per level.
            Each array has shape (nvars, n_samples_level_i).
        X2_list : List[Array], optional
            List of input data arrays for cross-covariance.
            If None, computes self-covariance using X1_list.
        block_format : bool, optional
            If True, return kernel as list of lists of blocks.
            If False, return stacked kernel matrix (default).

        Returns
        -------
        K : Array or List[List[Optional[Array]]]
            Multi-level kernel matrix.

            If block_format=False:
                Single array of shape (n_total_1, n_total_2) where
                n_total = sum(n_samples per level in X_list)

            If block_format=True:
                List of lists K[i][j] representing kernel blocks.

        Raises
        ------
        ValueError
            If X1_list length doesn't match nlevels.
            If X2_list is provided but length doesn't match nlevels.
        """
        return self._dag_kernel(X1_list, X2_list, block_format)

    def jacobian_wrt_params(
        self, X1_list: List[Array], X2_list: Optional[List[Array]] = None
    ) -> Array:
        """
        Compute Jacobian of kernel matrix w.r.t. hyperparameters.

        Delegates to the underlying DAGMultiOutputKernel.

        Parameters
        ----------
        X1_list : List[Array]
            Input points for each level.
        X2_list : Optional[List[Array]]
            Second set of input points. If None, uses X1_list.

        Returns
        -------
        jac : Array
            Jacobian, shape (n_total, n_total, nparams) where n_total = sum(n_i).
        """
        return self._dag_kernel.jacobian_wrt_params(X1_list, X2_list)

    def __repr__(self) -> str:
        """
        String representation.

        Returns
        -------
        repr : str
            String representation.
        """
        return (
            f"MultiLevelKernel(\n"
            f"  nlevels={self._nlevels},\n"
            f"  nvars={self.nvars()},\n"
            f"  sequential_structure=0->1->...->{self._nlevels - 1}\n"
            f")"
        )
