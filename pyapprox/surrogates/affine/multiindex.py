from abc import ABC, abstractmethod
import itertools
from typing import List, Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.template import Array, BackendMixin


def hash_index(array: Array, bkd: BackendMixin) -> int:
    np_array = bkd.to_numpy(array)
    return hash(np_array.tobytes())


def _unique_values_per_row(a: Array) -> Array:
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)


def _compute_hyperbolic_level_indices(
    nvars: int, level: int, pnorm: float, bkd: BackendMixin
) -> Array:
    eps = 1000 * np.finfo(np.double).eps
    if level == 0:
        return bkd.zeros((nvars, 1), dtype=int)
    # must use np here as torch does not play well with
    # combinations_with_replacement. I can see no reason
    # why one would want to differentiate through this functino
    tmp = np.asarray(
        list(itertools.combinations_with_replacement(np.arange(nvars), level))
    )
    # count number of times each element appears in tmp1
    indices = _unique_values_per_row(tmp).T
    p_norms = np.sum(indices**pnorm, axis=0) ** (1.0 / pnorm)
    II = np.where(p_norms <= level + eps)[0]
    return bkd.asarray(indices[:, II], dtype=int)


def compute_hyperbolic_indices(
    nvars: int,
    max_level: int,
    pnorm: float,
    bkd: BackendMixin = NumpyMixin,
) -> Array:
    indices = bkd.empty((nvars, 0), dtype=int)
    for dd in range(max_level + 1):
        new_indices = _compute_hyperbolic_level_indices(nvars, dd, pnorm, bkd)
        indices = bkd.hstack((indices, new_indices))
    return indices


def argsort_indices_lexiographically(
    indices: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
    np_indices = bkd.to_numpy(indices)
    index_tuple = (indices[0, :],)
    for ii in range(1, np_indices.shape[0]):
        index_tuple = index_tuple + (np_indices[ii, :],)
    index_tuple = index_tuple + (np_indices.sum(axis=0),)
    return bkd.asarray(np.lexsort(index_tuple), dtype=int)


def sort_indices_lexiographically(
    indices: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
    r"""
    Sort by level then lexiographically
    The last key in the sequence is used for the primary sort order,
    the second-to-last key for the secondary sort order, and so on
    """
    return indices[:, argsort_indices_lexiographically(indices, bkd)]


def _plot_2d_index(ax, index: Array, color: str = "gray"):
    box = (
        np.array(
            [
                [index[0] - 1, index[1] - 1],
                [index[0], index[1] - 1],
                [index[0], index[1]],
                [index[0] - 1, index[1]],
                [index[0] - 1, index[1] - 1],
            ]
        ).T
        + 0.5
    )
    ax.plot(box[0, :], box[1, :], "-k", lw=1)
    ax.fill(box[0, :], box[1, :], color=color, alpha=0.5, edgecolor="k")


def _plot_index_voxels(ax, data: Array, color: Tuple = [1, 1, 1, 0.9]):
    # color: [r,g,b,alpha]
    # ax.patch.set_alpha(0.5)  # Set semi-opacity
    colors = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))
    colors[np.where(data)] = color
    ax.voxels(data, facecolors=colors, edgecolor="gray")


class IndexGenerator(ABC):
    def __init__(self, nvars: int, backend: BackendMixin = NumpyMixin):
        self._bkd = backend
        self._nvars = nvars
        self._indices = self._bkd.zeros((nvars, 0), dtype=int)

    def nvars(self) -> int:
        return self._nvars

    def nindices(self) -> int:
        return self._indices.shape[1]

    def _hash_index(self, array: Array) -> int:
        np_array = self._bkd.to_numpy(array)
        return hash(np_array.tobytes())

    @abstractmethod
    def _get_indices(self) -> Array:
        raise NotImplementedError

    def get_indices(self) -> Array:
        indices = self._get_indices()
        if indices.dtype != self._bkd.int():
            raise RuntimeError("indices must be integers")
        return indices

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nindices={2})".format(
            self.__class__.__name__, self.nvars(), self.nindices()
        )

    def _plot_indices_2d(self, ax, indices: Array, color: str = "gray"):
        for index in indices.T:
            _plot_2d_index(ax, index, color)
        lim = self._bkd.to_numpy(self._bkd.max(indices))
        ax.set_xticks(np.arange(0, lim + 1))
        ax.set_yticks(np.arange(0, lim + 1))
        ax.set_xlim(-0.5, lim + 1)
        ax.set_ylim(-0.5, lim + 1)

    def _plot_indices_3d(self, ax, indices: Array, color: str = "gray"):
        if not isinstance(ax, Axes3D):
            raise ValueError(
                "ax must be an instance of  mpl_toolkits.mplot3d.Axes3D"
            )
        shape = tuple(self._bkd.max(indices, axis=1) + 1)
        filled = self._bkd.zeros(shape, dtype=int)
        for nn in range(indices.shape[1]):
            ii, jj, kk = indices[:, nn]
            filled[ii, jj, kk] = 1
        _plot_index_voxels(ax, filled)
        angle = 45
        ax.view_init(30, angle)
        ax.set_axis_off()

    def plot_indices(self, ax):
        if self.nvars() < 2 or self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars not in [2, 3].")

        if self.nvars() == 2:
            im1 = self._plot_indices_2d(ax, self.get_selected_indices())
            if self.get_candidate_indices() is None:
                return (im1,)
            im2 = self._plot_indices_2d(ax, self.get_candidate_indices(), "r")
            return (im1, im2)

        im1 = self._plot_indices_3d(ax, self.get_selected_indices())
        if self.get_candidate_indices() is None:
            return (im1,)
        im2 = self._plot_indices_3d(
            ax, self.get_candidate_indices(), [0.5, 0.5, 0.5, 0.9]
        )
        return (im1, im2)


class IterativeIndexGenerator(IndexGenerator):
    def __init__(self, nvars: int, backend: BackendMixin = NumpyMixin):
        super().__init__(nvars, backend)
        self._verbosity = 0
        self._sel_indices_dict = dict()
        self._cand_indices_dict = dict()
        self._admis_fun = None

    def _index_on_margin(self, index: Array) -> bool:
        for dim_id in range(self.nvars()):
            neighbor = self._get_forward_neighbor(index, dim_id)
            if self._hash_index(neighbor) in self._sel_indices_dict:
                return False
        return True

    def _find_candidate_indices(self) -> Array:
        # generate candidate indices from selected indices
        cand_indices = []
        idx = self.nselected_indices()
        for index in self._indices.T:
            if not self._index_on_margin(index):
                continue
            new_cand_indices = self._get_new_candidate_indices(index)
            for index in new_cand_indices.T:
                key = self._hash_index(index)
                if key not in self._cand_indices_dict:
                    self._cand_indices_dict[key] = idx
                    cand_indices.append(index)
                    idx += 1
        if len(cand_indices) > 0:
            return self._bkd.stack(cand_indices, axis=1)
        return self._bkd.zeros((index.shape[0], 0), dtype=int)

    def set_selected_indices(self, selected_indices: Array):
        self._sel_indices_dict = dict()
        self._cand_indices_dict = dict()

        if selected_indices.dtype != self._bkd.int():
            raise RuntimeError("selected_indices must be integers")

        if (
            selected_indices.ndim != 2
            or selected_indices.shape[0] != self.nvars()
        ):
            raise ValueError(
                "selected_indices must be a 2D array with nrows=nvars"
            )
        self._indices = self._bkd.copy(selected_indices)
        idx = 0
        for index in self._indices.T:
            self._sel_indices_dict[self._hash_index(index)] = idx
            idx += 1

        if not self._indices_are_downward_closed(self._indices):
            raise ValueError("selected indices were not downward closed")

        cand_indices = self._find_candidate_indices()
        self._indices = self._bkd.hstack((self._indices, cand_indices))

    def _indices_are_downward_closed(self, indices: Array) -> bool:
        for index in indices.T:
            for dim_id in range(self.nvars()):
                if index[dim_id] > 0:
                    neighbor = self._get_backward_neighbor(index, dim_id)
                    if (
                        self._hash_index(neighbor)
                        not in self._sel_indices_dict
                    ):
                        return False
        return True

    def set_verbosity(self, verbosity: int):
        self._verbosity = verbosity

    def set_admissibility_function(self, fun: "AdmissibilityCriteria"):
        if not isinstance(fun, AdmissibilityCriteria):
            raise ValueError(
                "fun must be an instance of AdmissibilityCriteria"
            )
        self._admis_fun = fun

    def _get_forward_neighbor(self, index: Array, dim_id: int) -> Array:
        neighbor = self._bkd.copy(index)
        neighbor[dim_id] += 1
        return neighbor

    def _get_backward_neighbor(self, index: Array, dim_id: int) -> Array:
        neighbor = self._bkd.copy(index)
        neighbor[dim_id] -= 1
        return neighbor

    def _is_admissible(self, index: Array) -> bool:
        fail_msg = f"Index {index} is not admissible: "
        if self._hash_index(index) in self._sel_indices_dict:
            if self._verbosity > 1:
                print(fail_msg + "already in the selected index set")
            return False
        if self._hash_index(index) in self._cand_indices_dict:
            if self._verbosity > 1:
                print(fail_msg + "already in the candidate index set")
            return False
        for dim_id in range(self.nvars()):
            if index[dim_id] > 0:
                neighbor_index = self._get_backward_neighbor(index, dim_id)
                if (
                    self._hash_index(neighbor_index)
                    not in self._sel_indices_dict
                ):
                    if self._verbosity > 1:
                        print(fail_msg + "Index is not downward_closed")
                    return False
        is_admisible = self._admis_fun(index)
        if not is_admisible and self._verbosity > 1:
            print(fail_msg + self._admis_fun.failure_message())
        return is_admisible

    def _get_new_candidate_indices(self, index: Array) -> Array:
        if self._admis_fun is None:
            raise RuntimeError("Must call set_admissibility_function")
        new_cand_indices = []
        for dim_id in range(self.nvars()):
            neighbor_index = self._get_forward_neighbor(index, dim_id)
            if self._is_admissible(neighbor_index):
                new_cand_indices.append(neighbor_index)
                if self._verbosity > 1:
                    msg = f"Adding candidate index {neighbor_index}"
                    print(msg)
        if len(new_cand_indices) > 0:
            return self._bkd.stack(new_cand_indices, axis=1)
        return self._bkd.zeros((self.nvars(), 0))

    def refine_index(self, index: Array) -> Array:
        if self._verbosity > 0:
            print(f"Refining index {index}")
        key = self._hash_index(index)
        self._sel_indices_dict[key] = self._cand_indices_dict[key]
        del self._cand_indices_dict[key]
        new_cand_indices = self._get_new_candidate_indices(index)
        idx = self._indices.shape[1]
        for new_index in new_cand_indices.T:
            self._cand_indices_dict[self._hash_index(new_index)] = idx
            idx += 1
        if new_cand_indices.shape[1] > 0:
            self._indices = self._bkd.hstack((self._indices, new_cand_indices))
        return new_cand_indices

    def nselected_indices(self) -> int:
        return len(self._sel_indices_dict)

    def ncandidate_indices(self) -> int:
        return len(self._cand_indices_dict)

    def _get_selected_idx(self) -> Array:
        return self._bkd.asarray(
            [item for key, item in self._sel_indices_dict.items()], dtype=int
        )

    def get_selected_indices(self) -> Array:
        idx = self._get_selected_idx()
        return self._indices[:, idx]

    def _get_candidate_idx(self) -> Array:
        # return  elements in self._indices that contain candidate indices
        return self._bkd.asarray(
            [item for key, item in self._cand_indices_dict.items()], dtype=int
        )

    def get_candidate_indices(self) -> Array:
        if self.ncandidate_indices() > 0:
            return self._indices[:, self._get_candidate_idx()]
        return None

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nsel_indices={2}, ncand_indices={3})".format(
            self.__class__.__name__,
            self.nvars(),
            self.nselected_indices(),
            self.ncandidate_indices(),
        )

    def step(self):
        self._indices = self._bkd.hstack(
            (self._indices, self._find_candidate_indices())
        )
        for key, item in self._cand_indices_dict.items():
            self._sel_indices_dict[key] = item
        for key in list(self._cand_indices_dict.keys()):
            del self._cand_indices_dict[key]

    def _get_indices(self) -> Array:
        return self._indices


class AdmissibilityCriteria(ABC):
    @abstractmethod
    def __call__(self, index: Array) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class MaxLevelAdmissibilityCriteria(AdmissibilityCriteria):
    def __init__(
        self,
        max_level: int,
        pnorm: int,
        max_1d_levels: List[int],
        backend: BackendMixin = NumpyMixin,
    ):
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend
        self._max_level = max_level
        if max_1d_levels is not None:
            max_1d_levels = self._bkd.asarray(max_1d_levels, dtype=int)
        self._max_1d_levels = max_1d_levels
        self._pnorm = pnorm

    def _indices_norm(self, indices: Array) -> float:
        return self._bkd.sum(indices**self._pnorm, axis=0) ** (
            1.0 / self._pnorm
        )

    def __call__(self, index: Array) -> bool:
        if self._indices_norm(index) > self._max_level:
            return False
        if self._max_1d_levels is not None and self._bkd.any(
            index > self._max_1d_levels
        ):
            return False
        if (
            self._max_1d_levels is not None
            and self._bkd.where(
                index[index > 0] == self._max_1d_levels[index > 0]
            )[0].shape[0]
            > 1
        ):
            return False
        return True


class MaxIndicesAdmissibilityCriteria(AdmissibilityCriteria):
    def __init__(
        self,
        max_nindices: int,
        backend: BackendMixin = NumpyMixin,
    ):
        self._bkd = backend
        self._max_nindices = max_nindices

    def set_max_nindices(self, max_nindices: int):
        # make sure to only add a subset of possible indices
        # so take current count from generator and reset it
        # here whenever gen.step is called
        self._cnt = self._gen.nindices()
        self._max_nindices = self._gen.nindices() + max_nindices

    def set_index_generator(self, gen: IterativeIndexGenerator):
        self._gen = gen

    def __call__(self, index: Array) -> bool:
        if not hasattr(self, "_gen"):
            raise AttributeError("must call set_index_generator")
        if self._gen.nindices() + self._cnt < self._max_nindices:
            self._cnt += 1
            return True
        return False


class HyperbolicIndexGenerator(IterativeIndexGenerator):
    def __init__(
        self,
        nvars: int,
        max_level: int,
        pnorm: float,
        max_1d_levels: List[int] = None,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(nvars, backend=backend)
        self.set_admissibility_function(
            MaxLevelAdmissibilityCriteria(
                max_level, pnorm, max_1d_levels, backend=backend
            )
        )
        self._get_init_indices()

    def _next_index_to_refine(self) -> int:
        return self._bkd.argmin(
            self._admis_fun._indices_norm(self.get_candidate_indices())
        )

    def _compute_indices(self):
        while len(self._cand_indices_dict) > 0:
            # get any index of smallest norm
            idx = self._next_index_to_refine()
            index = self._indices[:, self._get_candidate_idx()[idx]]
            self.refine_index(index)

    def _get_init_indices(self):
        if self.nindices() != 0:
            raise ValueError("can only call if nindices()==0")
        self.set_selected_indices(
            self._bkd.zeros((self.nvars(), 1), dtype=int)
        )
        self._compute_indices()
        return self._get_indices()

    def _get_indices(self) -> Array:
        return self._indices

    def step(self):
        """Increment max_level by 1"""
        self._admis_fun._max_level += 1
        super().step()


class ExpandingMarginGenerator(HyperbolicIndexGenerator):
    def __init__(
        self,
        nvars: int,
        max_level: int,
        pnorm: float,
        nindices_increment: int,
        backend: BackendMixin = NumpyMixin,
    ):
        self._nindices_increment = nindices_increment
        super().__init__(nvars, max_level, pnorm, None, backend)
        self.set_admissibility_function(
            MaxIndicesAdmissibilityCriteria(self.nindices(), self._bkd)
        )
        self._admis_fun.set_index_generator(self)

    def step(self):
        """Increment max_level by 1"""
        self._admis_fun.set_max_nindices(
            self.nindices() + self._nindices_increment
        )
        IterativeIndexGenerator.step(self)

    def _next_index_to_refine(self) -> int:
        if not hasattr(self, "_bexp"):
            return super()._next_index_to_refine()
        candidate_idx = self._get_candidate_idx()
        return self._bkd.argmax(
            self._bkd.abs(self._bexp.get_coefficients()[candidate_idx, :])
        )


class IndexGrowthRule(ABC):
    @abstractmethod
    def __call__(self, level: int) -> int:
        raise NotImplementedError


class LinearGrowthRule(IndexGrowthRule):
    def __init__(self, scale: int, shift: int):
        self._scale = scale
        self._shift = shift

    def __call__(self, level: int) -> int:
        if level == 0:
            return 1
        return self._scale * level + self._shift


class DoublePlusOneIndexGrowthRule(IndexGrowthRule):
    def __call__(self, level: int) -> int:
        if level == 0:
            return 1
        return 2**level + 1


class BasisIndexGenerator:
    def __init__(self, nvars: int, nrefinement_vars: int, gen, growth_rules):
        if not isinstance(gen, IterativeIndexGenerator):
            raise ValueError(
                "gen must be an instance of IterativeIndexGenerator"
            )
        self._nvars = nvars
        self._nrefinement_vars = nrefinement_vars
        self._subspace_gen = gen
        self._bkd = self._subspace_gen._bkd
        self._subspace_indices = None
        self._basis_indices = self._bkd.zeros(
            (self.nindex_vars(), 0), dtype=int
        )
        self._basis_indices_dict = dict()
        self._unique_subspace_basis_idx = []
        self._subspace_basis_idx = []
        self._hash_index = self._subspace_gen._hash_index

        if isinstance(growth_rules, IndexGrowthRule):
            growth_rules = [growth_rules] * self.nvars()

        if len(growth_rules) != self.nvars():
            raise ValueError(
                "Must specify a single growth rule or one per dimension"
            )

        for dim_id in range(self.nvars()):
            if not isinstance(growth_rules[dim_id], IndexGrowthRule):
                raise ValueError(
                    "growth_rule must be an instance of IndexGrowthRule"
                )
        self._growth_rules = growth_rules

    def nindex_vars(self) -> int:
        return self.nvars() + self.nrefinement_vars()

    def nrefinement_vars(self) -> int:
        return self._nrefinement_vars

    def nvars(self) -> int:
        return self._nvars

    def nindices(self) -> int:
        # must call get_indices so that if step has been called nindices
        # is updated
        return self.get_indices().shape[1]

    def nunivariate_basis(self, subspace_index: Array) -> List[int]:
        return [
            self._growth_rules[dim_id](subspace_index[dim_id])
            for dim_id in range(self.nvars())
        ]

    def _set_selected_subspace_indices(self, subspace_indices: Array):
        self._subspace_gen.set_selected_indices(subspace_indices)
        for subspace_index in subspace_indices.T:
            self._set_unique_subspace_basis_indices(subspace_index, False)
        if self._subspace_gen.ncandidate_indices() == 0:
            # needed if no new subspaces satsify admissibility criteria
            return
        for subspace_index in self._subspace_gen.get_candidate_indices().T:
            self._set_unique_subspace_basis_indices(subspace_index, True)

    def _set_unique_subspace_basis_indices(
        self, subspace_index: Array, cand_subspace: Array
    ):
        # if cand_subsapce is true: search for subspace_index in
        # self._subspace_gen._cand_indices_dict
        # otherwise search in   self._subspace_gen._sel_indices_dict
        # this is needed because first set of subspaces are set to selected
        # before these subspace indices are refined and the new subspaces
        # added to the candidate dictionary

        # The basis indices not already in basis_indices_dict
        unique_basis_indices = []
        # The array index of the subspace samples corresponding to
        # each the unique_basis_indices
        unique_subspace_sample_idx = []
        # The array index associated with each basis index of the subspace
        # in self._basis_indices
        global_basis_idx = []
        idx = len(self._basis_indices_dict)
        basis_indices = self._subspace_basis_indices(subspace_index)
        subspace_key = self._hash_index(subspace_index)
        for sample_idx, basis_index in enumerate(basis_indices.T):
            basis_key = self._hash_index(basis_index)
            if basis_key not in self._basis_indices_dict:
                if cand_subspace:
                    subspace_idx = self._subspace_gen._cand_indices_dict[
                        subspace_key
                    ]
                else:
                    subspace_idx = self._subspace_gen._sel_indices_dict[
                        subspace_key
                    ]
                self._basis_indices_dict[basis_key] = (idx, subspace_idx)
                unique_basis_indices.append(basis_index)
                unique_subspace_sample_idx.append(sample_idx)
                global_basis_idx.append(idx)
                idx += 1
            else:
                global_basis_idx.append(self._basis_indices_dict[basis_key][0])
        unique_basis_indices = self._bkd.stack(unique_basis_indices, axis=1)
        self._unique_subspace_basis_idx.append(unique_subspace_sample_idx)
        self._basis_indices = self._bkd.hstack(
            (self._basis_indices, unique_basis_indices)
        )
        self._subspace_basis_idx.append(
            self._bkd.array(global_basis_idx, dtype=int)
        )

    def _subspace_basis_indices(self, subspace_index: Array) -> Array:
        basis_indices_1d = []
        nbasis_1d = self.nunivariate_basis(subspace_index)
        basis_indices_1d = [
            self._bkd.arange(n_1d, dtype=int) for n_1d in nbasis_1d
        ]
        basis_indices = self._bkd.cartesian_product(basis_indices_1d)
        if self.nrefinement_vars() == 0:
            return basis_indices
        return self._bkd.vstack(
            (
                basis_indices,
                self._bkd.tile(
                    subspace_index[-self._nrefinement_vars :][:, None],
                    (basis_indices.shape[1],),
                ),
            )
        )

    def _get_basis_indices(
        self, subspace_indices: Array, return_all: bool = False
    ) -> Array:
        basis_indices = []
        basis_indices_dict = dict()
        basis_idx, subspace_idx = 0, 0
        for subspace_index in subspace_indices.T:
            # All unique basis indices can be found using subspace indices
            # on the margin, so avoid extra work by ignoring other subspaces
            if self._subspace_gen._index_on_margin(subspace_index):
                subspace_basis_indices = self._subspace_basis_indices(
                    subspace_index
                )
                # get basis indices not already collected
                for basis_index in subspace_basis_indices.T:
                    key = self._hash_index(basis_index)
                    if key not in basis_indices_dict:
                        basis_indices_dict[key] = (basis_idx, subspace_idx)
                        basis_indices.append(basis_index)
                        basis_idx += 1
            subspace_idx += 1
        basis_indices = self._bkd.stack(basis_indices, axis=1)
        if not return_all:
            return basis_indices
        return basis_indices, basis_indices_dict

    def _get_all_basis_indices(self, return_all=False) -> Array:
        return self._get_basis_indices(
            self._subspace_gen.get_indices(), return_all=return_all
        )

    def _get_all_basis_indices_of_selected_subspaces(
        self, return_all: bool = False
    ) -> Array:
        return self._get_basis_indices(
            self._subspace_gen.get_selected_indices(), return_all=return_all
        )

    def _subspace_indices_changed(self) -> bool:
        subspace_indices = self._subspace_gen.get_indices()
        if (
            self._subspace_indices is None
            or subspace_indices.shape != self._subspace_indices.shape
            or not self._bkd.allclose(
                subspace_indices, self._subspace_indices, atol=1e-15
            )
        ):
            self._subspace_indices = subspace_indices
            return True
        return False

    def get_indices(self) -> Array:
        if self._basis_indices is None or self._subspace_indices_changed():
            self._basis_indices, self._basis_indices_dict = (
                self._get_all_basis_indices(True)
            )
        return self._basis_indices

    def plot_indices(self, ax):
        if self.nvars() < 2 or self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars not in [2, 3].")

        if self.nvars() == 2:
            return self._subspace_gen._plot_indices_2d(ax, self.get_indices())

        return self._subspace_gen._plot_indices_3d(ax, self.get_indices())

    def refine_subspace_index(self, subspace_index: Array) -> Array:
        new_subspace_indices = self._subspace_gen.refine_index(subspace_index)
        for new_subspace_index in new_subspace_indices.T:
            self._set_unique_subspace_basis_indices(new_subspace_index, True)
        return new_subspace_indices


class IsotropicSGIndexGenerator(BasisIndexGenerator):
    def __init__(
        self,
        nvars: int,
        max_level: int,
        growth_rules: List[IndexGrowthRule],
        nrefinement_vars: int = 0,
        backend: BackendMixin = NumpyMixin,
    ):
        gen = HyperbolicIndexGenerator(nvars, max_level, 1.0, backend=backend)
        super().__init__(nvars, nrefinement_vars, gen, growth_rules)

    def step(self):
        self._subspace_gen.step()
        # call get_indices to update basis indices once subspace indices
        # have been changed
        self.get_indices()


def anova_level_indices(
    nvars: int, level: int, bkd: BackendMixin = NumpyMixin
):
    if level > nvars:
        raise ValueError(f"level {level} is larger than nvars {nvars}")
    return list(itertools.combinations(bkd.arange(nvars, dtype=int), level))
