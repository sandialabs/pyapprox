import heapq
from abc import ABC, abstractmethod

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.bases.multiindex import (
    HyperbolicIndexGenerator,
    BasisIndexGenerator,
    DoublePlusOneIndexGrowthRule,
)
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.bases.basisexp import TensorProductInterpolant
from pyapprox.surrogates.regressor import Regressor


class PriorityQueue:
    def __init__(self):
        self.list = []

    def empty(self):
        return len(self.list) == 0

    def put(self, item):
        if len(item) != 3:
            raise ValueError("must provide list with 3 entries")
        heapq.heappush(self.list, item)

    def get(self):
        item = heapq.heappop(self.list)
        return item

    def __eq__(self, other):
        return other.list == self.list

    def __neq__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self.list)

    def __len__(self):
        return len(self.list)


def _compute_smolyak_coefficients(sg, indices):
    smolyak_coefs = sg._bkd.zeros((indices.shape[1]), dtype=float)
    # the implementation below is simple but is slow in high-dimensions
    # because it checks all (2**nvars) shifts. TODO implement
    # way of enumerating shifts in index set
    shifts = sg._bkd.cartesian_product(
        [sg._bkd.asarray([0, 1], dtype=int)] * sg.nvars()
    )
    for idx, subspace_index in enumerate(indices.T):
        for shift in shifts.T:
            neighbor = subspace_index + shift
            if (
                sg._subspace_gen._hash_index(neighbor)
                in sg._subspace_gen._sel_indices_dict
            ):
                smolyak_coefs[idx] += (-1.0) ** (sg._bkd.sum(shift))
    return smolyak_coefs


class CombinationSparseGrid(Regressor):
    def __init__(self, nqoi, backend=NumpyLinAlgMixin):
        super().__init__(backend=backend)
        self._nqoi = nqoi

        self._subspace_gen = None
        self._basis_gen = None

        # unique sparse grid samples
        self._train_samples = None
        # training valus at unique sparse grid samples
        self._train_values = None
        # univariate interpolation samples that are resused
        # for each tensor product to save time by avoiding
        # thier construction for each tensor product
        self._univariate_samples = None

        self._smolyak_coefs = None
        self._basis = None
        self._subspace_surrogates = []
        self._cand_subspace_queue = None
        self._basis_indices_dict = dict()
        # for each subspace store the array indices of
        # samples and values needed to create subspace surrogate
        self._subspace_global_data_idx = []
        # keep track of how many unique samples are used by each subspace
        self._nunique_subspace_samples = []

    def _set_basis_index_generator(self, growth_rules):
        self._basis_gen = BasisIndexGenerator(self._subspace_gen, growth_rules)

    def set_subspace_generator(self, subspace_gen, growth_rules):
        self._subspace_gen = subspace_gen
        self._set_basis_index_generator(growth_rules)
        self._bkd = self._subspace_gen._bkd
        self._smolyak_coefs = self._bkd.zeros(0)
        self._train_samples = self._bkd.zeros((self.nvars(), 0))
        self._train_values = self._bkd.zeros((0, self.nqoi()))

    def set_basis(self, basis):
        if not isinstance(basis, TensorProductInterpolatingBasis):
            raise ValueError(
                "basis must be an instance of TensorProductInterpolatingBasis"
            )
        self._basis = basis

    def nvars(self):
        return self._subspace_gen.nvars()

    def nqoi(self):
        return self._nqoi

    def _unique_subspace_samples(self, subspace_index, basis):
        basis_indices = self._basis_gen._subspace_basis_indices(subspace_index)
        grid_samples = basis.tensor_product_grid()
        unique_samples, data_idx = [], []
        global_idx = self._train_samples.shape[1]
        for basis_index, sample in zip(basis_indices.T, grid_samples.T):
            key = self._subspace_gen._hash_index(basis_index)
            if key not in self._basis_indices_dict:
                self._basis_indices_dict[key] = len(self._basis_indices_dict)
                unique_samples.append(sample)
                data_idx.append(global_idx)
                global_idx += 1
            else:
                data_idx.append(self._basis_indices_dict[key])
        self._subspace_global_data_idx.append(
            self._bkd.asarray(data_idx, dtype=int)
        )
        return self._bkd.stack(unique_samples, axis=1)

    def _setup_tensor_product_interpolant(self, subspace_index):
        nsubspace_nodes_1d = self._basis_gen.nunivariate_basis(subspace_index)
        basis = self._basis._semideep_copy()
        basis.set_tensor_product_indices(nsubspace_nodes_1d)
        basis.set_indices(
            self._basis_gen._subspace_basis_indices(subspace_index)
        )
        self._subspace_surrogates.append(TensorProductInterpolant(basis))
        unique_samples = self._unique_subspace_samples(subspace_index, basis)
        self._nunique_subspace_samples.append(unique_samples.shape[1])
        self._train_samples = self._bkd.hstack(
            (self._train_samples, unique_samples)
        )
        return unique_samples

    def _values(self, samples):
        values = 0
        for subspace_idx in range(self._subspace_gen.nindices()):
            # TODO store smolyak coefficients for candidate & selected indices
            # but set can_indices coefs to zero. Have option to evaluate sparse
            # grid with and without can indices
            if abs(self._smolyak_coefs[subspace_idx]) <= np.finfo(float).eps:
                continue

            values += self._smolyak_coefs[
                subspace_idx
            ] * self._subspace_surrogates[subspace_idx](samples)
        return values

    def __repr__(self):
        return "{0}(nvars={1})".format(self.__class__.__name__, self.nvars())

    def _plot_grid_1d(self, ax):
        ax.plot(self.train_samples()[0], self.train_samples()[0], "o")

    def _plot_grid_2d(self, ax):
        ax.plot(*self.train_samples(), "o")

    def _plot_grid_3d(self, ax):
        if not isinstance(ax, Axes3D):
            raise ValueError(
                "ax must be an instance of  mpl_toolkits.mplot3d.Axes3D"
            )

    def plot_grid(self, ax):
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars >= 3.")

        plot_grid_funs = {
            1: self._plot_grid_1d,
            2: self._plot_grid_2d,
            3: self._plot_grid_3d,
        }
        plot_grid_funs[self.nvars()](ax)

    def integrate(self):
        values = 0
        for subspace_idx in range(self._subspace_gen.nindices()):
            if abs(self._smolyak_coefs[subspace_idx]) <= np.finfo(float).eps:
                continue

            values += (
                self._smolyak_coefs[subspace_idx]
                * self._subspace_surrogates[subspace_idx].integrate()
            )
        return values

    def train_samples(self):
        return self._train_samples

    def _fit(self, iterate):
        for subspace_idx in range(self._subspace_gen.nindices()):
            subspace_values = self._ctrain_values[
                self._subspace_global_data_idx[subspace_idx], :
            ]
            self._subspace_surrogates[subspace_idx].fit(subspace_values)
            self._set_smolyak_coefficients()


class IsotropicCombinationSparseGrid(CombinationSparseGrid):
    def __init__(
        self,
        nqoi,
        nvars,
        max_level,
        growth_rules,
        basis,
        backend=NumpyLinAlgMixin,
    ):
        super().__init__(nqoi, backend=backend)
        self._set_basis(basis)
        self._set_subspace_generator(
            HyperbolicIndexGenerator(nvars, max_level, 1.0, self._bkd),
            growth_rules,
        )

    def _set_basis(self, basis):
        super().set_basis(basis)

    def set_basis(self, basis):
        raise RuntimeError("Do not set basis as it has already been set")

    def set_subspace_generator(self, subspace_gen, growth_rules):
        raise RuntimeError(
            "Do not call set_subspace_generator as it has already been set"
        )

    def _set_subspace_generator(self, subspace_gen, growth_rules):
        super().set_subspace_generator(subspace_gen, growth_rules)
        # must call get_indices to generate basis indices
        self._subspace_gen.get_indices()
        subspace_indices = self._subspace_gen.get_indices()
        for subspace_index in subspace_indices.T:
            self._setup_tensor_product_interpolant(subspace_index)

    def _set_smolyak_coefficients(self):
        self._smolyak_coefs = _compute_smolyak_coefficients(
            self, self._subspace_gen.get_indices()
        )

    def plot_subspace_indices(self, ax):
        self._subspace_gen.plot_indices(ax)
        if self.nvars() != 2:
            return
        for ii, index in enumerate(self._subspace_gen.get_indices().T):
            ax.text(
                index[0],
                index[1],
                r"${0}$".format(int(self._smolyak_coefs[ii])),
                fontsize=24,
                horizontalalignment="center",
                verticalalignment="center",
            )


class RefinementCriteria(ABC):
    def __init__(self):
        self._sg = None
        self._bkd = None

    def set_sparse_grid(self, sg):
        self._sg = sg
        self._bkd = self._sg._bkd

    def cost_per_sample(self, subspace_index):
        return 1

    def cost(self, subspace_index):
        """Computational cost of collecting the unique training data."""
        key = self._sg._subspace_gen._hash_index(subspace_index)
        subspace_idx = self._sg._subspace_gen._cand_indices_dict[key]
        return self._sg._nunique_subspace_samples[
            subspace_idx
        ] * self.cost_per_sample(subspace_index)

    @abstractmethod
    def _priority(self, subspace_index):
        raise NotImplementedError

    def __call__(self, subspace_index):
        # divide priority by cost so to choose subspaces
        # that require smaller training data collection costs first
        error, priority = self._priority(subspace_index)
        priority /= self.cost_per_sample(subspace_index)
        return error, priority

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class LevelRefinementCriteria(RefinementCriteria):
    def _priority(self, subspace_index):
        # Avoid computing error which requires specifying a metric
        # and can slow criteria evaluation down
        error = np.inf
        # priority queue gives higher priority to smaler values
        # so this will add subspaces wih lower l1 norm first
        priority = self._bkd.sum(subspace_index)
        return error, priority


class VarianceRefinementCriteria(RefinementCriteria):
    def _priority(self, subspace_index):
        # priority queue gives higher priority to smaler values
        # so this will add subspaces wih lower l1 norm first
        priority = self._bkd.sum(subspace_index)
        error = np.inf
        return error, priority


class AdaptiveCombinationSparseGrid(CombinationSparseGrid):
    def __init__(self, nqoi, backend=NumpyLinAlgMixin):
        super().__init__(nqoi, backend)
        self._last_subspace_indices = None
        self._refine_criteria = None
        self._first_step = True

    def set_refinement_criteria(self, refine_criteria):
        if not isinstance(refine_criteria, RefinementCriteria):
            raise ValueError(
                "refine_criteria must be and instance of RefinementCriteria"
            )
        self._refine_criteria = refine_criteria
        self._refine_criteria.set_sparse_grid(self)

    def set_initial_subspace_indices(self, indices=None):
        if indices is None:
            indices = self._bkd.zeros((self.nvars(), 1), dtype=int)
        # TODO change set selected indices to only allow generation of
        # candidate indices accorging to variable ordering.
        # allow group based ordering, where dimensions in a group
        # are unordered
        self._subspace_gen.set_selected_indices(indices)

    def _setup_subspaces(self, subspace_indices):
        unique_samples = []
        for subspace_index in subspace_indices.T:
            unique_samples.append(
                self._setup_tensor_product_interpolant(subspace_index)
            )
        return self._bkd.hstack(unique_samples)

    def _first_step_samples(self):
        if self._subspace_gen is None:
            raise RuntimeError("Must call set_subspace_generator")
        if self._refine_criteria is None:
            raise RuntimeError("Must call set_refinement_criteria")
        if self._subspace_gen.nselected_indices() == 0:
            raise RuntimeError("must call set_initial_subspace_indices")
        # get samples for both selected and candidate subspace indices
        unique_samples = self._setup_subspaces(
            self._subspace_gen.get_indices()
        )
        # only set smolyak coefficients to use selected indices
        self._smolyak_coefs = _compute_smolyak_coefficients(
            self, self._subspace_gen.get_selected_indices()
        )
        return unique_samples

    def _prioritize_candidate_subspaces(self):
        # reset priority of all candidate subspace indices
        self._cand_subspace_queue = PriorityQueue()
        for subspace_index in self._subspace_gen.get_candidate_indices().T:
            subspace_indicator, subspace_error = self._refine_criteria(
                subspace_index
            )
            key = self._subspace_gen._hash_index(subspace_index)
            subspace_idx = self._subspace_gen._cand_indices_dict[key]
            self._cand_subspace_queue.put(
                (subspace_indicator, subspace_error, subspace_idx)
            )

    def _get_best_cand_subspace(self):
        priority, error, best_subspace_idx = self._cand_subspace_queue.get()
        best_cand_subspace_index = self._subspace_gen._indices[
            :, best_subspace_idx
        ]
        return best_cand_subspace_index

    def _update_smolyak_coefficients(self, new_index):
        new_smolyak_coeffs = self._bkd.hstack(
            (self._smolyak_coefs, self._bkd.zeros((1,)))
        )
        selected_indices = self._subspace_gen.get_selected_indices()
        for ii in range(selected_indices.shape[1]):
            diff = new_index - selected_indices[:, ii]
            if self._bkd.all(diff >= 0) and self._bkd.max(diff) <= 1:
                new_smolyak_coeffs[ii] += (-1.0) ** self._bkd.sum(diff)
        return new_smolyak_coeffs

    def _step_samples(self):
        while len(self._subspace_gen._cand_indices_dict) > 0:
            best_subspace_index = self._get_best_cand_subspace()
            new_subspace_indices = self._subspace_gen.refine_index(
                best_subspace_index
            )
            # update smolyak coefs must occur in here so that even
            # if no candidates are added
            self._smolyak_coefs = self._update_smolyak_coefficients(
                best_subspace_index
            )
            if new_subspace_indices.shape[1] > 0:
                self._last_subspace_indices = new_subspace_indices
                return self._setup_subspaces(new_subspace_indices)
        return None

    def step_samples(self):
        if self._first_step:
            return self._first_step_samples()
        return self._step_samples()

    def _step_values(self, values, subspace_idxs):
        self._train_values = self._bkd.vstack((self._train_values, values))
        # set training data sets ctrain samples and values
        self._set_training_data(self._train_samples, self._train_values)
        for subspace_idx in subspace_idxs:
            subspace_values = self._ctrain_values[
                self._subspace_global_data_idx[subspace_idx], :
            ]
            self._subspace_surrogates[subspace_idx].fit(subspace_values)
        self._prioritize_candidate_subspaces()

    def step_values(self, values):
        if self._first_step:
            # need to setup tensor product interpolants for both
            # selected and candidate subspace indices
            self._step_values(
                values, self._bkd.arange(self._subspace_gen.nindices())
            )
            self._first_step = False
        else:
            subspace_idxs = []
            for subspace_index in self._last_subspace_indices.T:
                key = self._subspace_gen._hash_index(subspace_index)
                subspace_idxs.append(
                    self._subspace_gen._cand_indices_dict[key]
                )
            self._step_values(values, subspace_idxs)

    def step(self, fun):
        unique_samples = self.step_samples()
        if unique_samples is None:
            return False
        unique_values = fun(unique_samples)
        self.step_values(unique_values)
        return True

    def build(self, fun):
        while self.step(fun):
            pass


class LocalIndexGenerator(BasisIndexGenerator):
    def __init__(self, gen, growth_rules):
        super().__init__(gen, growth_rules)
        self._cand_basis_indices_dict = None
        self._sel_basis_indices_dict = None
        self._basis_global_subspace_idx = None

    def _subspace_index_from_basis_index(self, basis_index):
        subspace_idx = self._basis_global_subspace_idx[
                self._sel_basis_indices_dict[self._hash_index(basis_index)]
            ]
        return self._subspace_gen._indices[:, subspace_idx]

    def _left_neighbor(self, basis_index, dim_id):
        if basis_index[dim_id] == 1:
            return None
        neighbor = self._bkd.copy(basis_index)
        if basis_index[dim_id] == 0:
            neighbor[dim_id] = 1
        elif basis_index[dim_id] == 2:
            neighbor[dim_id] = 4
        else:
            neighbor[dim_id] = 2 * basis_index[dim_id] - 1
        return neighbor

    def _right_neighbor(self, basis_index, dim_id):
        if basis_index[dim_id] == 2:
            return None
        neighbor = self._bkd.copy(basis_index)
        if basis_index[dim_id] == 0:
            neighbor[dim_id] = 2
        elif basis_index[dim_id] == 1:
            neighbor[dim_id] = 3
        else:
            neighbor[dim_id] = 2 * basis_index[dim_id]
        return neighbor

    def _parent(self, basis_index, dim_id):
        parent = self._bkd.copy(basis_index)
        parent[dim_id] = (basis_index[dim_id] + (basis_index[dim_id] % 2)) / 2
        return parent

    def _is_admissible(self, basis_index, subspace_index):
        if basis_index is None:
            return False
        if not self._subspace_gen._is_admissible(subspace_index) and (
            self._subspace_gen._hash_index(subspace_index)
            not in self._subspace_gen._cand_indices_dict
        ):
            # admissible return False if subspace is a candidate, but
            # we can add point in candidate subspaces
            return False
        if self._hash_index(basis_index) in self._sel_basis_indices_dict:
            return False
        if self._hash_index(basis_index) in self._cand_basis_indices_dict:
            return False
        for dim_id in range(self.nvars()):
            if basis_index[dim_id] == 0:
                return True
            parent_basis_index = self._parent(basis_index, dim_id)
            key = self._subspace_gen._hash_index(parent_basis_index)
            if key in self._sel_basis_indices_dict:
                return True
        return True

    def get_candidate_basis_indices(self, basis_index, subspace_index):
        new_cand_indices = []
        for dim_id in range(self.nvars()):
            left_neighbor = self._left_neighbor(basis_index, dim_id)
            print("L", basis_index, left_neighbor, dim_id)
            if self._is_admissible(left_neighbor, subspace_index):
                new_cand_indices.append(left_neighbor)
                print("Ad")
            right_neighbor = self._right_neighbor(basis_index, dim_id)
            if self._is_admissible(right_neighbor, subspace_index):
                new_cand_indices.append(right_neighbor)
        if len(new_cand_indices) > 0:
            return self._bkd.stack(new_cand_indices, axis=1)
        return self._bkd.zeros((self.nvars(), 0))

    def refine_basis_index(self, basis_index):
        #if self._verbosity > 0:
        print(f"Refining basis index {basis_index}")
        key = self._hash_index(basis_index)
        # todo combine self._basis_indices_dict of this class
        # and self._basis_indices_dict of combinationsparsegrid
        self._sel_basis_indices_dict[key] = self._cand_basis_indices_dict[key]
        del self._cand_basis_indices_dict[key]
        subspace_index = self._subspace_index_from_basis_index(basis_index)
        new_cand_basis_indices = self.get_candidate_basis_indices(
            basis_index, subspace_index
        )
        idx = self._basis_indices.shape[1]
        for new_index in new_cand_basis_indices.T:
            key = self._hash_index(new_index)
            if key not in self._cand_basis_indices_dict:
                self._cand_basis_indices_dict[key] = idx
                idx += 1
            else:
                raise RuntimeError("This should not happen")
        if new_cand_basis_indices.shape[1] > 0:
            self._basis_indices = self._bkd.hstack(
                (self._basis_indices, new_cand_basis_indices)
            )
        return new_cand_basis_indices

    def get_indices(self):
        return self._basis_indices

    def nselected_indices(self):
        return len(self._sel_basis_indices_dict)

    def ncandidate_indices(self):
        return len(self._cand_basis_indices_dict)

    def _get_candidate_idx(self):
        # return  elements in self._indices that contain candidate indices
        return self._bkd.asarray(
            [item for key, item in self._cand_basis_indices_dict.items()],
            dtype=int,
        )

    def get_candidate_indices(self):
        if self.ncandidate_indices() > 0:
            return self._basis_indices[:, self._get_candidate_idx()]
        return None

    def _find_candidate_indices(self, basis_indices):
        new_cand_basis_indices = []
        idx = basis_indices.shape[1]
        for basis_index in basis_indices.T:
            subspace_index = self._subspace_index_from_basis_index(basis_index)
            cand_basis_indices = self.get_candidate_basis_indices(
                basis_index, subspace_index
            )
            for new_index in cand_basis_indices.T:
                key = self._hash_index(new_index)
                if key not in self._cand_basis_indices_dict:
                    self._cand_basis_indices_dict[key] = idx
                    new_cand_basis_indices.append(new_index)
                    idx += 1
        return self._bkd.stack(new_cand_basis_indices, axis=1)

    def set_selected_indices(self, sel_basis_indices, basis_subspace_idx):
        self._sel_basis_indices_dict = dict()
        self._cand_basis_indices_dict = dict()
        self._basis_global_subspace_idx = basis_subspace_idx
        self._basis_indices = self._bkd.copy(sel_basis_indices)
        idx = 0
        for basis_index, subspace_idx in zip(
            self._basis_indices.T, basis_subspace_idx
        ):
            key = self._hash_index(basis_index)
            if key in self._sel_basis_indices_dict:
                raise ValueError("sel_basis_indices has a repeated index")
            self._sel_basis_indices_dict[key] = idx
            idx += 1
        cand_basis_indices = self._find_candidate_indices(self._basis_indices)
        self._basis_indices = self._bkd.hstack(
            (self._basis_indices, cand_basis_indices)
        )


class LocalRefinementCriteria(ABC):
    def __init__(self):
        self._sg = None
        self._bkd = None

    def set_sparse_grid(self, sg):
        self._sg = sg
        self._bkd = self._sg._bkd

    def cost_per_sample(self, subspace_index):
        return 1

    def cost(self, basis_index):
        """Computational cost of collecting the single training data."""
        key = self._basis_gen._hash_index(basis_index)
        subspace_index = self._sg._basis_gen._subspace_index_from_basis_index(
            basis_index
        )
        return self.cost_per_sample(subspace_index)

    @abstractmethod
    def _priority(self, basis_index):
        raise NotImplementedError

    def __call__(self, basis_index):
        # divide priority by cost so to choose subspaces
        # that require smaller training data collection costs first
        error, priority = self._priority(basis_index)
        priority /= self.cost_per_sample(basis_index)
        return error, priority

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class LocalHierarchicalRefinementCriteria(LocalRefinementCriteria):
    def _priority(self, basis_index):
        # todo consider prioritizing all candidate basis indices at once
        # for local hierarhical surplus.
        key = self._sg._basis_gen._hash_index(basis_index)
        basis_idx = self._sg._basis_gen._cand_basis_indices_dict[key]
        error = self._sg._train_values[basis_idx] - self._sg(
            self._sg._train_samples[:, basis_idx]
        )
        priority = -error
        return error, priority


class LocallyAdaptiveCombinationSparseGrid(AdaptiveCombinationSparseGrid):
    def __init__(self, nqoi, backend=NumpyLinAlgMixin):
        super().__init__(nqoi, backend)
        self._cand_basis_queue = None
        self._last_basis_indices = None
        self._last_nunique_samples = None
        self._refine_criteria = None

    def set_refinement_criteria(self, refine_criteria):
        if refine_criteria is None:
            refine_criteria = LocalHierarchicalRefinementCriteria()
        print(refine_criteria)
        if not isinstance(refine_criteria, LocalRefinementCriteria):
            raise ValueError(
                "refine_criteria must be and instance of "
                "LocalRefinementCriteria"
            )
        self._refine_criteria = refine_criteria
        self._refine_criteria.set_sparse_grid(self)

    def _set_basis_index_generator(self, growth_rules):
        self._basis_gen = LocalIndexGenerator(self._subspace_gen, growth_rules)

    def set_subspace_generator(self, subspace_gen):
        super().set_subspace_generator(
            subspace_gen, DoublePlusOneIndexGrowthRule()
        )

    def set_initial_subspace_indices(self, indices=None):
        super().set_initial_subspace_indices(indices)
        basis_indices, basis_subspace_idx = self._basis_gen._get_basis_indices(
            return_all=True
        )
        self._basis_gen.set_selected_indices(basis_indices, basis_subspace_idx)

    def _get_best_cand_basis(self):
        priority, error, best_basis_idx = self._cand_basis_queue.get()
        best_cand_basis_index = self._basis_gen._basis_indices[
            :, best_basis_idx
        ]
        return best_cand_basis_index

    def _first_step_samples(self):
        unique_samples = super()._first_step_samples()
        self._last_nunique_samples = unique_samples.shape[1]
        self._last_global_data_idx = self._bkd.hstack(
            self._subspace_global_data_idx
        )
        return unique_samples

    def _local_idx(self, subspace_index, new_basis_indices):
        key = self._subspace_gen._hash_index(subspace_index)
        subspace_idx = self._sg._subspace_gen._cand_indices_dict[key]
        global_data_idx = self._subspace_global_data_idx[subspace_idx]
        basis_idx = [
            self._basis_gen._cand_basis_indices_dict[
                self._subspace_gen._hash_index(basis_index)
            ]
            for basis_index in new_basis_indices
        ]
        return global_data_idx, basis_idx

    def _step_samples(self):
        while len(self._basis_gen._cand_basis_indices_dict) > 0:
            best_basis_index = self._get_best_cand_basis()
            new_basis_indices = self._basis_gen.refine_basis_index(
                best_basis_index
            )
            best_subspace_index
            self._smolyak_coefs = self._update_smolyak_coefficients(
                best_subspace_index
            )
            if new_basis_indices.shape[1] > 0:
                continue
            new_subspace_indices = self._subspace_gen.refine_index(
                best_subspace_index
            )
            unique_samples = self._setup_subspaces(new_subspace_indices)
            self._last_basis_indices = new_basis_indices
            self._last_nunique_samples = unique_samples.shape[1]
            global_data_idx, basis_idx = self._local_idx(
                best_subspace_index, new_basis_indices
            )
            self._last_global_data_idx = global_data_idx
            # only return samples that have been identified as important
            return self._train_samples[:, basis_idx]
        return None

    def _prioritize_candidate_bases(self):
        # reset priority of all candidate subspace indices
        self._cand_basis_queue = PriorityQueue()
        for basis_index in self._basis_gen.get_candidate_indices().T:
            basis_indicator, subspace_error = self._refine_criteria(
                basis_index
            )
            key = self._subspace_gen._hash_index(basis_index)
            basis_idx = self._basis_gen._cand_basis_indices_dict[key]
            self._cand_basis_queue.put(
                (basis_indicator, subspace_error, basis_idx)
            )

    def _step_values(self, values, subspace_idxs):
        # nunique_samples is the number of all samples in recently
        # added subspaces
        subspace_values = self._bkd.zeros(
            (self._last_nunique_samples, self.nqoi())
        )
        self._train_values = self._bkd.vstack(
            (self._train_values, subspace_values)
        )
        # Overwrite zeros for all points corresponding to the samples
        # associated with values
        self._train_values[self._last_global_data_idx] = values
        # set training data sets ctrain samples and values
        self._set_training_data(self._train_samples, self._train_values)
        # Loop over all subspaces effected by data update above
        # For now just loop over all subspaces.
        # TODO only loop over effected subspaces
        for subspace_idx in range(self._subspace_gen.nindices()):
            subspace_values = subspace_values = self._ctrain_values[
                self._subspace_global_data_idx[subspace_idx], :
            ]
            self._subspace_surrogates[subspace_idx].fit(subspace_values)
        self._prioritize_candidate_bases()

    def step_values(self, values):
        if self._first_step:
            # need to setup tensor product interpolants for both
            # selected and candidate subspace indices
            self._step_values(
                values, self._bkd.arange(self._subspace_gen.nindices())
            )
            self._first_step = False
        else:
            subspace_idxs = []
            for subspace_index in self._last_subspace_indices.T:
                key = self._subspace_gen._hash_index(subspace_index)
                subspace_idxs.append(
                    self._subspace_gen._cand_indices_dict[key]
                )
            self._step_values(values, subspace_idxs)


# TODO mix locally adaptive basis with global polynomial basis in another
# dimension. Do this by adding all left and right neigbours in dimensions
# with global basis
