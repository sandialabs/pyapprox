"""MFNet node classes.

Nodes encapsulate a local model, noise standard deviation, and metadata
about which global input variables the node uses.
"""

from typing import Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameterList,
    LogHyperParameter,
)
from pyapprox.typing.surrogates.mfnets.protocols import NodeModelProtocol


class MFNetNode(Generic[Array]):
    """A node in an MFNet graph.

    Each node owns a local model (satisfying ``NodeModelProtocol``), a noise
    standard deviation hyperparameter, and an optional specification of which
    global input variables feed into the node's model.

    Terminology (for a graph ``leaf -> mid -> root``):
      - **leaf**: lowest-fidelity, no children
      - **root**: highest-fidelity, no parents
      - **children** of a node: its predecessors (lower fidelity)
      - **parents** of a node: its successors (higher fidelity)

    Parameters
    ----------
    node_id : int
        Non-negative integer identifier for this node.
    model : NodeModelProtocol[Array]
        The local model for this node.
    noise_std : float
        Gaussian noise standard deviation for this node's likelihood.
    bkd : Backend[Array]
        Computational backend.
    noise_std_bounds : tuple, optional
        Bounds for the noise_std parameter. Default: ``(1e-8, 1e6)``.
    fixed_noise_std : bool, optional
        Whether to fix noise_std during optimization. Default: ``True``.
    active_sample_vars : Array, optional
        Indices of global input variables used by this node. If ``None``,
        all variables are used (set during ``validate``).
    """

    def __init__(
        self,
        node_id: int,
        model: NodeModelProtocol[Array],
        noise_std: float,
        bkd: Backend[Array],
        noise_std_bounds: Tuple[float, float] = (1e-8, 1e6),
        fixed_noise_std: bool = True,
        active_sample_vars: Optional[Array] = None,
    ) -> None:
        if node_id < 0:
            raise ValueError("node_id must be non-negative")
        if not isinstance(model, NodeModelProtocol):
            raise TypeError(
                f"model must satisfy NodeModelProtocol, "
                f"got {type(model).__name__}"
            )
        self._bkd = bkd
        self._node_id = node_id
        self._model = model
        self._noise_std_hyp = LogHyperParameter(
            name=f"node_{node_id}_noise_std",
            nparams=1,
            user_values=noise_std,
            user_bounds=noise_std_bounds,
            bkd=bkd,
            fixed=fixed_noise_std,
        )
        self._active_sample_vars = active_sample_vars
        self._children_ids: List[int] = []
        self._parent_ids: List[int] = []

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def node_id(self) -> int:
        return self._node_id

    def model(self) -> NodeModelProtocol[Array]:
        return self._model

    def noise_std(self) -> Array:
        """Return noise_std in user space (exponentiated)."""
        return self._noise_std_hyp.exp_values()

    def noise_std_hyp(self) -> LogHyperParameter[Array]:
        return self._noise_std_hyp

    def active_sample_vars(self) -> Optional[Array]:
        return self._active_sample_vars

    def set_active_sample_vars(self, indices: Array) -> None:
        self._active_sample_vars = indices

    def children_ids(self) -> List[int]:
        return self._children_ids

    def set_children_ids(self, ids: List[int]) -> None:
        self._children_ids = ids

    def parent_ids(self) -> List[int]:
        return self._parent_ids

    def set_parent_ids(self, ids: List[int]) -> None:
        self._parent_ids = ids

    def is_leaf(self) -> bool:
        return len(self._children_ids) == 0

    def is_root(self) -> bool:
        return len(self._parent_ids) == 0

    def hyp_list(self) -> HyperParameterList:
        """Aggregate of model hyp_list + noise_std hyperparameter."""
        model_hyps = self._model.hyp_list().hyperparameters()
        return HyperParameterList(
            model_hyps + [self._noise_std_hyp], self._bkd
        )

    def validate(self, nvars_global: int) -> None:
        """Validate this node's configuration.

        Called by ``MFNet.validate()``. Sets ``active_sample_vars`` to all
        variables if not specified. Checks that the node type is consistent
        with the graph structure.

        Parameters
        ----------
        nvars_global : int
            Number of global input variables in the network.
        """
        if self._active_sample_vars is None:
            self._active_sample_vars = self._bkd.asarray(
                list(range(nvars_global)), dtype=int
            )
        self._check_node_type()

    def _check_node_type(self) -> None:
        """Check that this node has both children and parents.

        Overridden by ``RootMFNetNode`` and ``LeafMFNetNode``.
        """
        if self.is_root() or self.is_leaf():
            raise ValueError(
                "MFNetNode (interior) must have both children and parents. "
                f"Node {self._node_id} has {len(self._children_ids)} children "
                f"and {len(self._parent_ids)} parents."
            )


class RootMFNetNode(MFNetNode[Array]):
    """A root (high-fidelity) node in an MFNet. Has children but no parents."""

    def _check_node_type(self) -> None:
        if not self.is_root():
            raise ValueError(
                f"RootMFNetNode {self._node_id} cannot have parents, "
                f"but has parent_ids={self._parent_ids}"
            )


class LeafMFNetNode(MFNetNode[Array]):
    """A leaf (low-fidelity) node in an MFNet. Has parents but no children."""

    def _check_node_type(self) -> None:
        if not self.is_leaf():
            raise ValueError(
                f"LeafMFNetNode {self._node_id} cannot have children, "
                f"but has children_ids={self._children_ids}"
            )
