from abc import ABC, abstractmethod
import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def _unique_values_per_row(a):
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:, None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1, N)


def _compute_hyperbolic_level_indices(nvars, level, pnorm, bkd):
    eps = 1000 * np.finfo(np.double).eps
    if level == 0:
        return bkd.zeros((nvars, 1), dtype=int)
    # must use np here as torch does not play well with
    # combinations_with_replacement. I can see no reason
    # why one would want to differentiate through this functino
    tmp = np.asarray(
        list(
            itertools.combinations_with_replacement(
                np.arange(nvars), level
            )
        )
    )
    # count number of times each element appears in tmp1
    indices = _unique_values_per_row(tmp).T
    p_norms = np.sum(indices**pnorm, axis=0)**(1.0/pnorm)
    II = np.where(p_norms <= level+eps)[0]
    return bkd.asarray(indices[:, II], dtype=int)


def compute_hyperbolic_indices(
        nvars, max_level, pnorm, bkd=NumpyLinAlgMixin
):
    indices = bkd.empty((nvars, 0), dtype=int)
    for dd in range(max_level+1):
        new_indices = _compute_hyperbolic_level_indices(
            nvars, dd, pnorm, bkd)
        indices = bkd.hstack((indices, new_indices))
    return indices


def sort_indices_lexiographically(indices, bkd=NumpyLinAlgMixin):
    r"""
    Sort by level then lexiographically
    The last key in the sequence is used for the primary sort order,
    the second-to-last key for the secondary sort order, and so on
    """
    np_indices = bkd.to_numpy(indices)
    index_tuple = (indices[0, :],)
    for ii in range(1, np_indices.shape[0]):
        index_tuple = index_tuple+(np_indices[ii, :],)
    index_tuple = index_tuple+(np_indices.sum(axis=0),)
    II = np.lexsort(index_tuple)
    return indices[:, II]


def _plot_2d_index(ax, index):
    box = np.array([[index[0]-1, index[1]-1],
                    [index[0], index[1]-1],
                    [index[0], index[1]],
                    [index[0]-1, index[1]],
                    [index[0]-1, index[1]-1]]).T + 0.5
    ax.plot(box[0, :], box[1, :], '-k', lw=1)
    ax.fill(box[0, :], box[1, :], color='gray', alpha=0.5, edgecolor='k')


def _plot_index_voxels(ax, data):
    # color: [r,g,b,alpha]
    color = [1, 1, 1, .9]
    # ax.patch.set_alpha(0.5)  # Set semi-opacity
    colors = np.zeros((data.shape[0], data.shape[1], data.shape[2], 4))
    colors[np.where(data)] = color
    ax.voxels(data, facecolors=colors, edgecolor='gray')


class IndexGenerator(ABC):
    def __init__(self, nvars, backend=NumpyLinAlgMixin):
        self._bkd = backend
        self._nvars = nvars
        self._indices = self._bkd.zeros((nvars, 0), dtype=int)

    def nvars(self):
        return self._nvars

    def nindices(self):
        return self._indices.shape[1]

    def _hash_index(self, array):
        np_array = self._bkd.to_numpy(array)
        return hash(np_array.tobytes())

    @abstractmethod
    def _get_indices(self):
        raise NotImplementedError

    def get_indices(self):
        indices = self._get_indices()
        if indices.dtype != self._bkd.int():
            raise RuntimeError("indices must be integers")
        return indices

    def __repr__(self):
        return "{0}(nvars={1}, nindices={2})".format(
            self.__class__.__name__, self.nvars(), self.nindices()
        )

    def _plot_indices_2d(self, ax, indices):
        for index in indices.T:
            _plot_2d_index(ax, index)
        lim = self._bkd.max(indices)
        ax.set_xticks(np.arange(0, lim + 1))
        ax.set_yticks(np.arange(0, lim + 1))
        ax.set_xlim(-0.5, lim + 1)
        ax.set_ylim(-0.5, lim + 1)

    def _plot_indices_3d(self, ax, indices):
        if not isinstance(ax, Axes3D):
            raise ValueError(
                "ax must be an instance of  mpl_toolkits.mplot3d.Axes3D"
            )
        shape = tuple(self._bkd.max(indices, axis=1)+1)
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
            return self._plot_indices_2d(ax, self.get_indices())

        return self._plot_indices_3d(ax, self.get_indices())


class IterativeIndexGenerator(IndexGenerator):
    def __init__(self, nvars, backend=NumpyLinAlgMixin):
        super().__init__(nvars, backend)
        self._verbosity = 0
        self._sel_indices_dict = dict()
        self._cand_indices_dict = dict()
        self._admis_fun = None

    def _index_on_margin(self, index):
        for dim_id in range(self.nvars()):
            neighbor = self._get_forward_neighbor(index, dim_id)
            if self._hash_index(neighbor) in self._sel_indices_dict:
                return False
        return True

    def _get_candidate_indices(self):
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
        return self._bkd.stack(cand_indices, axis=1)

    def set_selected_indices(self, selected_indices):
        self._sel_indices_dict = dict()
        self._cand_indices_dict = dict()
        self._indices = self._bkd.zeros((self.nvars(), 0))

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

        cand_indices = self._get_candidate_indices()
        self._indices = self._bkd.hstack((self._indices, cand_indices))

    def _indices_are_downward_closed(self, indices):
        for index in indices.T:
            for dim_id in range(self.nvars()):
                if index[dim_id] > 0:
                    neighbor = self._get_backward_neighbor(index, dim_id)
                    if self._hash_index(neighbor) not in self._sel_indices_dict:
                        return False
        return True

    def set_verbosity(self, verbosity):
        self._verbosity = verbosity

    def set_admissibility_function(self, fun):
        self._admis_fun = fun

    def _get_forward_neighbor(self, index, dim_id):
        neighbor = self._bkd.copy(index)
        neighbor[dim_id] += 1
        return neighbor

    def _get_backward_neighbor(self, index, dim_id):
        neighbor = self._bkd.copy(index)
        neighbor[dim_id] -= 1
        return neighbor

    def _is_admissible(self, index):
        if self._hash_index(index) in self._sel_indices_dict:
            return False
        if self._hash_index(index) in self._cand_indices_dict:
            return False
        for dim_id in range(self.nvars()):
            if index[dim_id] > 0:
                neighbor_index = self._get_backward_neighbor(index, dim_id)
                if self._hash_index(neighbor_index) not in self._sel_indices_dict:
                    return False
        return True

    def _get_new_candidate_indices(self, index):
        if self._admis_fun is None:
            raise RuntimeError("Must call set_admissibility_function")
        new_cand_indices = []
        for dim_id in range(self.nvars()):
            neighbor_index = self._get_forward_neighbor(index, dim_id)
            if (
                    self._is_admissible(neighbor_index) and
                    self._admis_fun(neighbor_index)
            ):
                new_cand_indices.append(neighbor_index)
                if self._verbosity > 2:
                    msg = f"Adding candidate index {neighbor_index}"
                    print(msg)
            else:
                if self._verbosity > 2:
                    msg = f"Index {neighbor_index} is not admissible"
                    print(msg)
        if len(new_cand_indices) > 0:
            return self._bkd.stack(new_cand_indices, axis=1)
        return self._bkd.zeros((self.nvars(), 0))

    def refine_index(self, index):
        if self._verbosity > 2:
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
            self._indices = self._bkd.hstack(
                (self._indices, new_cand_indices)
            )
        return new_cand_indices

    def nselected_indices(self):
        return len(self._sel_indices_dict)

    def ncandidate_indices(self):
        return len(self._cand_indices_dict)

    def get_selected_indices(self):
        idx = self._bkd.hstack(
            [item for key, item in self._sel_indices_dict.items()])
        return self._indices[:, idx]

    def _get_candidate_idx(self):
        # return  elements in self._indices that contain candidate indices
        return self._bkd.asarray(
            [item for key, item in self._cand_indices_dict.items()], dtype=int)

    def get_candidate_indices(self):
        if self.ncandidate_indices() > 0:
            return self._indices[:, self._get_candidate_idx()]
        return None

    def __repr__(self):
        return "{0}(nvars={1}, nsel_indices={2}, ncand_indices={3})".format(
            self.__class__.__name__, self.nvars(), self.nselected_indices(),
            self.ncandidate_indices()
        )

    # @abstractmethod
    # def step(self):
    #     raise NotImplementedError()

    def step(self):
        self._indices = self._bkd.hstack(
            (self._indices, self._get_candidate_indices())
        )
        for key, item in self._cand_indices_dict.items():
            self._sel_indices_dict[key] = item
        for key in list(self._cand_indices_dict.keys()):
            del self._cand_indices_dict[key]


class AdmissibilityCriteria(ABC):
    @abstractmethod
    def __call__(self, index):
        raise NotImplementedError


class MaxLevelAdmissibilityCriteria(AdmissibilityCriteria):
    def __init__(self, max_level, pnorm, backend=None):
        self._max_level = max_level
        self._pnorm = pnorm
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend

    def _indices_norm(self, indices):
        return (
            self._bkd.sum(indices**self._pnorm, axis=0)**(1.0/self._pnorm)
        )

    def __call__(self, index):
        if self._indices_norm(index) <= self._max_level:
            return True
        return False


class HyperbolicIndexGenerator(IterativeIndexGenerator):
    def __init__(self, nvars, max_level, pnorm, backend=NumpyLinAlgMixin):
        super().__init__(nvars, backend=backend)
        self.set_admissibility_function(
            MaxLevelAdmissibilityCriteria(max_level, pnorm, backend)
        )

    def _compute_indices(self):
        while len(self._cand_indices_dict) > 0:
            # get any index of smallest norm
            idx = self._bkd.argmin(
                self._admis_fun._indices_norm(self.get_candidate_indices())
            )
            index = self._indices[:,  self._get_candidate_idx()[idx]]
            self.refine_index(index)

    def _get_indices(self):
        if self.nindices() == 0:
            self.set_selected_indices(self._bkd.zeros(
                (self.nvars(), 1), dtype=int))
            self._compute_indices()
        return self._indices

    def step(self):
        """Increment max_level by 1"""
        self._admis_fun._max_level += 1
        super().step()


class IndexGrowthRule(ABC):
    @abstractmethod
    def __call__(self, level):
        raise NotImplementedError


class LinearGrowthRule(IndexGrowthRule):
    def __init__(self, scale, shift):
        self._scale = scale
        self._shift = shift

    def __call__(self, level):
        if level == 0:
            return 1
        return self._scale*level + self._shift


class DoublePlusOneIndexGrowthRule(IndexGrowthRule):
    def __call__(self, level):
        if level == 0:
            return 1
        return 2**level + 1


class BasisIndexGenerator:
    def __init__(self, gen, growth_rules):
        if not isinstance(gen, IterativeIndexGenerator):
            raise ValueError(
                "gen must be an instance of IterativeIndexGenerator"
            )
        self._subspace_gen = gen
        self._bkd = self._subspace_gen._bkd
        self._subspace_indices = None
        self._basis_indices = None

        if isinstance(growth_rules, IndexGrowthRule):
            growth_rules = [growth_rules]*self.nvars()

        if len(growth_rules) != self.nvars():
            raise ValueError(
                "Must specify a single growth rule or one per dimension")

        for dim_id in range(self.nvars()):
            if not isinstance(growth_rules[dim_id], IndexGrowthRule):
                raise ValueError(
                    "growth_rule must be an instance of IndexGrowthRule"
                )
        self._growth_rules = growth_rules

    def nvars(self):
        return self._subspace_gen.nvars()

    def nindices(self):
        # must call get_indices so that if step has been called nindices
        # is updated
        return self.get_indices().shape[1]

    def nunivariate_basis(self, subspace_index):
        return [
            self._growth_rules[dim_id](subspace_index[dim_id])
            for dim_id in range(self.nvars())
        ]

    def _subspace_basis_indices(self, subspace_index):
        basis_indices_1d = []
        nbasis_1d = self.nunivariate_basis(subspace_index)
        basis_indices_1d = [
            self._bkd.arange(n_1d, dtype=int) for n_1d in nbasis_1d]
        return self._bkd.cartesian_product(basis_indices_1d)

    def _get_basis_indices(self):
        subspace_indices = self._subspace_gen.get_indices()
        basis_indices = []
        basis_indices_set = set()
        for subspace_index in subspace_indices.T:
            # All unique basis indices can be found using subspace indices
            # on the margin, so avoid extra work by ignoring other subspaces
            if self._subspace_gen._index_on_margin(subspace_index):
                subspace_basis_indices = self._subspace_basis_indices(
                    subspace_index
                )
                # get basis indices not already collected
                for basis_index in subspace_basis_indices.T:
                    key = self._subspace_gen._hash_index(basis_index)
                    if key not in basis_indices_set:
                        basis_indices_set.add(key)
                        basis_indices.append(basis_index)
        return self._bkd.stack(basis_indices, axis=1)

    def _subspace_indices_changed(self):
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

    def get_indices(self):
        if self._basis_indices is None or self._subspace_indices_changed():
            self._basis_indices = self._get_basis_indices()
        return self._basis_indices

    def plot_indices(self, ax):
        if self.nvars() < 2 or self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars not in [2, 3].")

        if self.nvars() == 2:
            return self._subspace_gen._plot_indices_2d(ax, self.get_indices())

        return self._subspace_gen._plot_indices_3d(ax, self.get_indices())


class IsotropicSGIndexGenerator(BasisIndexGenerator):
    def __init__(
            self, nvars, max_level, growth_rules, backend=NumpyLinAlgMixin
    ):
        gen = HyperbolicIndexGenerator(nvars, max_level, 1., backend)
        super().__init__(gen, growth_rules)

    def step(self):
        self._subspace_gen.step()
        # call get_indices to update basis indices once subspace indices
        # have been changed
        self.get_indices()
