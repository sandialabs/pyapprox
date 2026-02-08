"""MFNet edge class.

An edge connects a child node (lower fidelity) to a parent node (higher
fidelity) and specifies which of the child's outputs are passed to the parent.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


class MFNetEdge(Generic[Array]):
    """An edge connecting a child node to a parent node.

    Parameters
    ----------
    child_node_id : int
        ID of the child (lower-fidelity) node.
    parent_node_id : int
        ID of the parent (higher-fidelity) node.
    child_output_ids : Array, optional
        Indices of the child's outputs that feed the parent. If ``None``,
        all outputs are used (set during validation based on child nqoi).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        child_node_id: int,
        parent_node_id: int,
        bkd: Backend[Array],
        child_output_ids: Optional[Array] = None,
    ) -> None:
        self._bkd = bkd
        self._child_node_id = child_node_id
        self._parent_node_id = parent_node_id
        self._child_output_ids = child_output_ids

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def child_node_id(self) -> int:
        return self._child_node_id

    def parent_node_id(self) -> int:
        return self._parent_node_id

    def output_ids(self) -> Optional[Array]:
        """Return indices of child outputs passed to parent.

        May be ``None`` before validation (defaults to all child outputs).
        """
        return self._child_output_ids

    def set_output_ids(self, ids: Array) -> None:
        self._child_output_ids = ids

    def validate(self, child_nqoi: int) -> None:
        """Validate and finalize output_ids.

        Parameters
        ----------
        child_nqoi : int
            Number of QoI of the child node.
        """
        if self._child_output_ids is None:
            self._child_output_ids = self._bkd.asarray(
                list(range(child_nqoi)), dtype=int
            )
        max_id = int(self._bkd.to_numpy(self._bkd.max(self._child_output_ids)))
        if max_id >= child_nqoi:
            raise ValueError(
                f"child_output_ids max ({max_id}) >= child nqoi ({child_nqoi})"
            )
