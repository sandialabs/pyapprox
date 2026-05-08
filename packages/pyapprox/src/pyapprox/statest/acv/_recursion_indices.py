"""ACV recursion index generation.

Ported from pyapprox.multifidelity._optim to remove legacy dependency.
"""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import Any, Generator, List, Optional

import numpy as np
from numpy.typing import NDArray


class ModelTree:
    def __init__(self, root: int, children: list[Any] | NDArray[Any] = []) -> None:
        if isinstance(children, np.ndarray):
            children = list(children)
        self.children: list[Any] = children
        for ii in range(len(self.children)):
            if not isinstance(self.children[ii], ModelTree):
                self.children[ii] = ModelTree(self.children[ii])
        self.root = root

    def num_nodes(self) -> int:
        nnodes = 1
        for child in self.children:
            if isinstance(child, ModelTree):
                nnodes += child.num_nodes()
            else:
                nnodes += 1
        return nnodes

    def to_index(self) -> NDArray[Any]:
        index: List[Optional[int]] = [None for ii in range(self.num_nodes())]
        index[0] = self.root
        self._to_index_recusive(index, self)
        return np.array(index)

    def _to_index_recusive(self, index: List[Optional[int]], root: ModelTree) -> None:
        for child in root.children:
            index[child.root] = root.root
            self._to_index_recusive(index, child)

    def __repr__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, self.to_index())


def _update_list_for_reduce(
    mylist: List[List[Any]], indices: tuple[int, int]
) -> List[List[Any]]:
    mylist[indices[0]].append(indices[1])
    return mylist


def _generate_all_trees(
    children: NDArray[Any] | list[Any], root: int, tree_depth: int
) -> Generator[ModelTree, None, None]:
    if tree_depth < 2 or len(children) == 0:
        yield ModelTree(root, children)
    else:
        for prod in product((0, 1), repeat=len(children)):
            if not any(prod):
                continue
            acc: List[List[Any]] = [[], []]
            res = reduce(
                _update_list_for_reduce, zip(prod, children), acc
            )
            nexts: List[Any] = res[0]
            sub_roots: List[Any] = res[1]
            for q in product(range(len(sub_roots)), repeat=len(nexts)):
                sub_children: List[List[Any]] = reduce(
                    _update_list_for_reduce,
                    zip(q, nexts),
                    [[] for ii in sub_roots],
                )
                yield from [
                    ModelTree(root, list(children))
                    for children in product(
                        *(
                            _generate_all_trees(sc, sr, tree_depth - 1)
                            for sr, sc in zip(sub_roots, sub_children)
                        )
                    )
                ]


def _get_acv_recursion_indices(
    nmodels: int, depth: int | None = None
) -> Generator[NDArray[Any], None, None]:
    if depth is None:
        depth = nmodels - 1
    if depth > nmodels - 1:
        msg = f"Depth {depth} exceeds number of lower-fidelity models"
        raise ValueError(msg)
    for index in _generate_all_trees(np.arange(1, nmodels), 0, depth):
        yield index.to_index()[1:]
