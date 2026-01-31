"""Tree enumeration for ACV recursion indices.

Provides utilities for generating all valid recursion indices (connected trees)
for Approximate Control Variate (ACV) estimators.

This module ports the tree enumeration logic from the legacy module
`pyapprox/multifidelity/_optim.py` with proper type hints and backend support.
"""

from typing import Generic, Iterator, List, Optional, Union
from itertools import product
from functools import reduce

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd


class ModelTree(Generic[Array]):
    """Tree structure representing recursion relationships for ACV estimators.

    A tree node represents a model, and children represent models that use
    this model as their recursion parent. The tree structure encodes which
    low-fidelity model samples are used as control variates for each model.

    Parameters
    ----------
    root : int
        The model index for this node (0 = high-fidelity).
    children : List[ModelTree] or List[int], optional
        Child nodes (models coupled with this one). Can be ModelTree instances
        or integer model indices (which will be converted to ModelTree).
    bkd : Backend[Array], optional
        Computational backend. If None, uses NumpyBkd.

    Examples
    --------
    Star topology (all LF coupled with HF):
    >>> tree = ModelTree(0, [ModelTree(1), ModelTree(2)])
    >>> tree.to_index()
    array([0, 0])

    Chain topology (MLMC-style):
    >>> tree = ModelTree(0, [ModelTree(1, [ModelTree(2)])])
    >>> tree.to_index()
    array([0, 1])
    """

    def __init__(
        self,
        root: int,
        children: Optional[List[Union["ModelTree[Array]", int]]] = None,
        bkd: Optional[Backend[Array]] = None,
    ) -> None:
        if children is None:
            children = []
        if isinstance(children, np.ndarray):
            children = list(children)

        self._children: List[ModelTree[Array]] = []
        for child in children:
            if isinstance(child, ModelTree):
                self._children.append(child)
            else:
                # Convert integer to ModelTree
                self._children.append(ModelTree(int(child), [], bkd))

        self._root = root
        self._bkd = bkd if bkd is not None else NumpyBkd()

    def root(self) -> int:
        """Return the root model index."""
        return self._root

    def children(self) -> List["ModelTree[Array]"]:
        """Return the list of child trees."""
        return self._children

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def num_nodes(self) -> int:
        """Return total number of nodes in the tree.

        Returns
        -------
        int
            Total number of nodes (models) in the tree.
        """
        nnodes = 1
        for child in self._children:
            nnodes += child.num_nodes()
        return nnodes

    def to_index(self) -> Array:
        """Convert tree structure to recursion index array.

        The recursion index specifies parent-child relationships:
        index[m-1] = parent model index for model m (m >= 1).

        Returns
        -------
        Array
            Recursion index array. Shape: (nmodels-1,)
            where nmodels = num_nodes().

        Examples
        --------
        >>> tree = ModelTree(0, [ModelTree(1), ModelTree(2)])
        >>> tree.to_index()
        array([0, 0])  # Both models 1 and 2 have parent 0 (HF)

        >>> tree = ModelTree(0, [ModelTree(1, [ModelTree(2)])])
        >>> tree.to_index()
        array([0, 1])  # Model 1 has parent 0, model 2 has parent 1
        """
        nnodes = self.num_nodes()
        index: List[Optional[int]] = [None for _ in range(nnodes)]
        index[0] = self._root
        self._to_index_recursive(index, self)
        # Return all entries except the first (root)
        return self._bkd.asarray(index[1:])

    def _to_index_recursive(
        self, index: List[Optional[int]], root: "ModelTree[Array]"
    ) -> None:
        """Recursively fill the index array."""
        for child in root._children:
            index[child._root] = root._root
            self._to_index_recursive(index, child)

    def __repr__(self) -> str:
        index = self.to_index()
        index_np = self._bkd.to_numpy(index).astype(int).tolist()
        return f"{self.__class__.__name__}(root={self._root}, index={index_np})"


def _update_list_for_reduce(
    mylist: tuple, indices: tuple
) -> tuple:
    """Helper function for building sub-children lists.

    Used with functools.reduce to partition children into groups.
    """
    mylist[indices[0]].append(indices[1])
    return mylist


def generate_all_trees(
    children: List[int],
    root: int,
    tree_depth: int,
    bkd: Optional[Backend[Array]] = None,
) -> Iterator[ModelTree[Array]]:
    """Generate all connected trees with given depth constraint.

    This is the core algorithm for enumerating all valid recursion
    structures. Uses the product of (0,1) to decide which children
    become sub-roots vs stay as leaves.

    Parameters
    ----------
    children : List[int]
        Available child model indices.
    root : int
        Root model index (typically 0 for HF).
    tree_depth : int
        Maximum tree depth. depth=1 means all children are direct children
        of root, depth=2 allows one level of nesting, etc.
    bkd : Backend[Array], optional
        Computational backend. If None, uses NumpyBkd.

    Yields
    ------
    ModelTree[Array]
        Each valid tree structure.

    Notes
    -----
    Algorithm:
    1. If tree_depth < 2 or no children, yield single tree with all
       children as direct children of root.
    2. Otherwise, for each product of (0,1) for children:
       - 0: child stays as leaf (no sub-children)
       - 1: child becomes sub-root (can have sub-children)
    3. For sub-roots, distribute remaining children among them.
    4. Recursively generate sub-trees for each sub-root.
    5. Combine via product of all sub-tree combinations.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> trees = list(generate_all_trees([1, 2], 0, 2, bkd))
    >>> len(trees)
    2
    >>> [t.to_index().tolist() for t in trees]
    [[0, 0], [0, 1]]
    """
    if bkd is None:
        bkd = NumpyBkd()

    if tree_depth < 2 or len(children) == 0:
        # Base case: all children are direct children of root
        yield ModelTree(root, list(children), bkd)
    else:
        # For each combination of (leaf, sub-root) decisions
        for prod in product((0, 1), repeat=len(children)):
            if not any(prod):
                # Need at least one child to proceed (skip empty selection)
                continue

            # Separate into leaves (nexts) and sub-roots
            nexts: List[int] = []  # Children staying as leaves
            sub_roots: List[int] = []  # Children becoming sub-roots
            for i, child in enumerate(children):
                if prod[i] == 0:
                    nexts.append(child)
                else:
                    sub_roots.append(child)

            # For each way to assign leaves to sub-roots
            for assignment in product(range(len(sub_roots)), repeat=len(nexts)):
                # Build children lists for each sub-root
                sub_children: List[List[int]] = [[] for _ in sub_roots]
                for leaf_idx, sub_root_idx in enumerate(assignment):
                    sub_children[sub_root_idx].append(nexts[leaf_idx])

                # Recursively generate sub-trees for each sub-root
                sub_tree_generators = [
                    generate_all_trees(sc, sr, tree_depth - 1, bkd)
                    for sr, sc in zip(sub_roots, sub_children)
                ]

                # Yield all combinations of sub-trees
                for sub_trees in product(*sub_tree_generators):
                    yield ModelTree(root, list(sub_trees), bkd)


def get_acv_recursion_indices(
    nmodels: int,
    depth: Optional[int] = None,
    bkd: Optional[Backend[Array]] = None,
) -> Iterator[Array]:
    """Generate all valid recursion indices for given number of models.

    A recursion index defines the parent-child relationships in the ACV
    estimator tree structure. index[m-1] specifies which model's samples
    are used as control variates for model m.

    Parameters
    ----------
    nmodels : int
        Total number of models (including HF model at index 0).
    depth : int, optional
        Maximum tree depth. If None, uses nmodels-1 (full depth).
        Lower values limit the complexity of recursion patterns.
    bkd : Backend[Array], optional
        Computational backend. If None, uses NumpyBkd.

    Yields
    ------
    Array
        Each valid recursion index. Shape: (nmodels-1,)

    Raises
    ------
    ValueError
        If depth exceeds nmodels-1.

    Examples
    --------
    For 3 models (HF + 2 LF):
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> indices = list(get_acv_recursion_indices(3, bkd=bkd))
    >>> len(indices)
    2
    >>> [idx.tolist() for idx in indices]
    [[0, 0], [0, 1]]

    Interpretation:
    - [0, 0]: Both LF models coupled with HF (star topology/MFMC)
    - [0, 1]: LF1 coupled with HF, LF2 coupled with LF1 (chain/MLMC)

    With depth limit:
    >>> indices = list(get_acv_recursion_indices(4, depth=2, bkd=bkd))
    >>> len(indices)  # Fewer than full depth
    """
    if bkd is None:
        bkd = NumpyBkd()

    if depth is None:
        depth = nmodels - 1

    if depth > nmodels - 1:
        raise ValueError(
            f"Depth {depth} exceeds number of lower-fidelity models "
            f"({nmodels - 1})"
        )

    # Generate trees with LF models as children of root (HF = model 0)
    children = list(range(1, nmodels))
    for tree in generate_all_trees(children, 0, depth, bkd):
        yield tree.to_index()


def count_recursion_indices(
    nmodels: int,
    depth: Optional[int] = None,
) -> int:
    """Count total number of valid recursion indices.

    Useful for progress reporting and complexity estimation before
    running a full search.

    Parameters
    ----------
    nmodels : int
        Total number of models (including HF).
    depth : int, optional
        Maximum tree depth. If None, uses nmodels-1 (full depth).

    Returns
    -------
    int
        Number of valid recursion indices.

    Notes
    -----
    The count can grow rapidly with nmodels:
    - 3 models: 2 indices
    - 4 models: ~26 indices (full depth)
    - 5 models: ~150+ indices (full depth)
    - 6 models: ~1000+ indices (full depth)

    Examples
    --------
    >>> count_recursion_indices(3)
    2
    >>> count_recursion_indices(4)
    26
    >>> count_recursion_indices(4, depth=2)  # Limited depth
    8
    """
    return sum(1 for _ in get_acv_recursion_indices(nmodels, depth))
