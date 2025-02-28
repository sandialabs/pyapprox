import heapq
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.bases.multiindex import (
    HyperbolicIndexGenerator,
    BasisIndexGenerator,
    DoublePlusOneIndexGrowthRule,
    AdmissibilityCriteria,
    IndexGrowthRule,
    LinearGrowthRule,
    IterativeIndexGenerator,
)
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.bases.basisexp import (
    TensorProductInterpolant,
    TensorProductLagrangeInterpolantToPolynomialChaosExpansionConverter,
    PolynomialChaosExpansion,
    TensorProductQuadratureRule,
)
from pyapprox.surrogates.regressor import Regressor, AdaptiveRegressorMixin
from pyapprox.interface.model import Model, MultiIndexModelEnsemble
from pyapprox.surrogates.bases.univariate.base import UnivariateBasis
from pyapprox.surrogates.bases.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.univariate.leja import (
    TwoPointChristoffelLejaQuadratureRule,
    LejaQuadratureRule,
)


class PriorityQueue:
    def __init__(self):
        self.list = []

    def empty(self) -> bool:
        return len(self.list) == 0

    def put(self, item):
        if len(item) != 3:
            raise ValueError("must provide list with 3 entries")
        heapq.heappush(self.list, item)

    def get(self):
        item = heapq.heappop(self.list)
        return item

    def __eq__(self, other: "PriorityQueue") -> bool:
        return other.list == self.list

    def __neq__(self, other: "PriorityQueue") -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return str(self.list)

    def __len__(self) -> int:
        return len(self.list)

    def items(self) -> List:
        """
        Get shallow copy of all items in the queue
        """
        # cannot extract items without destroying queue so destroy the recreate
        items = []
        while not self.empty():
            items.append(self.get())
        # restore queue
        for item in items:
            self.put(item)
        return items


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


class SparseGridAdmissibilityCriteria(AdmissibilityCriteria):
    def __init__(self):
        self._sg = None
        self._bkd = None

    def set_sparse_grid(self, sg: "CombinationSparseGrid"):
        self._sg = sg
        self._bkd = self._sg._bkd

    @abstractmethod
    def __call__(self, index: Array) -> bool:
        raise NotImplementedError

    def failure_message(self) -> str:
        return "{0} not met".format(self.__class__.__name__)


class SparseGridSubSpaceAdmissibilityCriteria(SparseGridAdmissibilityCriteria):
    pass


class CombinationSparseGrid(Regressor):
    def __init__(
        self, nqoi: int, nvars: int, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        super().__init__(backend=backend)
        self._verbosity = 0
        self._nqoi = nqoi
        self._nvars = nvars

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
        # TODO: Move queue to subspace_gen
        self._cand_subspace_queue = None
        self._subspace_errors = []

    def _set_basis_index_generator(self, growth_rules: List[IndexGrowthRule]):
        self._basis_gen = BasisIndexGenerator(
            self.nvars(),
            self.nrefinement_vars(),
            self._subspace_gen,
            growth_rules,
        )

    def nsubspace_vars(self) -> int:
        return self.nvars() + self.nrefinement_vars()

    def nrefinement_vars(self) -> int:
        return 0

    def set_subspace_generator(
        self, subspace_gen, growth_rules: List[IndexGrowthRule]
    ):
        if subspace_gen.nvars() != self.nsubspace_vars():
            raise ValueError(
                "subspace_gen has the wrong nvars {0} should be {1}".format(
                    subspace_gen.nvars(), self.nsubspace_vars()
                )
            )
        self._subspace_gen = subspace_gen
        self._bkd = self._subspace_gen._bkd
        self._set_basis_index_generator(growth_rules)
        self._smolyak_coefs = self._bkd.zeros(0)
        self._train_samples = self._bkd.zeros((self.nsubspace_vars(), 0))
        self._train_values = self._bkd.zeros((0, self.nqoi()))

    def set_verbosity(self, verbosity: int):
        self._verbosity = verbosity
        if self._subspace_gen is None:
            raise ValueError(
                "Cannot set verbosity until self.set_subspace_generator()"
                " is called"
            )
        self._subspace_gen.set_verbosity(verbosity - 1)

    def set_basis(self, basis: TensorProductInterpolatingBasis):
        if not isinstance(basis, TensorProductInterpolatingBasis):
            raise ValueError(
                "basis must be an instance of TensorProductInterpolatingBasis"
            )
        self._basis = basis

    # TODO change name of setup_subspaces.
    # A first setup phase has already been performed, this is really just
    # the second setup phase
    def _setup_subspaces(
        self, subspace_indices: Array, is_cand_subspace: bool
    ):
        unique_samples = []
        for subspace_index in subspace_indices.T:
            subspace_unique_samples = self._setup_tensor_product_interpolant(
                subspace_index, is_cand_subspace
            )
            unique_samples.append(subspace_unique_samples)
        self._subspace_errors += [
            0.0 for ii in range(subspace_indices.shape[1])
        ]
        return self._bkd.hstack(unique_samples)

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def _add_refinement_id(
        self, subspace_index: Array, samples: Array
    ) -> Array:
        if self.nrefinement_vars() == 0:
            return samples

        return self._bkd.vstack(
            (
                samples,
                self._bkd.tile(
                    subspace_index[-self._nrefinement_vars :][:, None],
                    (samples.shape[1],),
                ),
            )
        )

    def _setup_tensor_product_interpolant(
        self, subspace_index: Array, is_cand_subspace: bool
    ):
        nsubspace_nodes_1d = self._basis_gen.nunivariate_basis(subspace_index)
        basis = self._basis._semideep_copy()
        basis.set_tensor_product_indices(nsubspace_nodes_1d)
        self._subspace_surrogates.append(TensorProductInterpolant(basis))
        subspace_key = self._basis_gen._hash_index(subspace_index)
        if is_cand_subspace:
            subspace_idx = self._subspace_gen._cand_indices_dict[subspace_key]
        else:
            subspace_idx = self._subspace_gen._sel_indices_dict[subspace_key]
        unique_samples_idx = self._basis_gen._unique_subspace_basis_idx[
            subspace_idx
        ]
        unique_samples = basis.tensor_product_grid()[:, unique_samples_idx]
        unique_samples = self._add_refinement_id(
            subspace_index, unique_samples
        )
        self._train_samples = self._bkd.hstack(
            (self._train_samples, unique_samples)
        )
        return unique_samples

    def _values_using_smolyak_coefs(
        self, samples: Array, smolyak_coefs: Array
    ) -> Array:
        if samples.shape[0] != self.nvars():
            raise ValueError("samples have the wrong shape")
        values = 0
        for subspace_idx in range(self._subspace_gen.nindices()):
            # TODO store smolyak coefficients for candidate & selected indices
            # but set can_indices coefs to zero. Have option to evaluate sparse
            # grid with and without can indices
            if abs(smolyak_coefs[subspace_idx]) <= np.finfo(float).eps:
                continue
            values += smolyak_coefs[subspace_idx] * self._subspace_surrogates[
                subspace_idx
            ](samples)
        return values

    def _values_using_only_selected_subspaces(self, samples: Array) -> Array:
        return self._values_using_smolyak_coefs(samples, self._smolyak_coefs)

    def _values_using_all_subspaces(self, samples: Array) -> Array:
        queue_items = self._cand_subspace_queue.items()
        smolyak_coefs = self._smolyak_coefs
        selected_idx = self._subspace_gen._get_selected_idx()
        for item in queue_items:
            subspace_idx = item[2]
            subspace_index = self._subspace_gen._indices[:, subspace_idx]
            selected_idx = self._bkd.hstack((selected_idx, subspace_idx))
            smolyak_coefs = self._adjust_smolyak_coefficients(
                smolyak_coefs, subspace_index, selected_idx
            )
        return self._values_using_smolyak_coefs(samples, smolyak_coefs)

    def _values(self, samples: Array) -> Array:
        if self._cand_subspace_queue is None:
            return self._values_using_only_selected_subspaces(samples)
        return self._values_using_all_subspaces(samples)

    def __repr__(self) -> str:
        return "{0}(nvars={1})".format(self.__class__.__name__, self.nvars())

    def _plot_grid_1d(self, ax):
        ax.plot(self.train_samples()[0], self.train_samples()[0] * 0, "o")

    def _plot_grid_2d(self, ax):
        ax.plot(*self.train_samples(), "o")

    def _plot_grid_3d(self, ax):
        if not isinstance(ax, Axes3D):
            raise ValueError(
                "ax must be an instance of  mpl_toolkits.mplot3d.Axes3D"
            )
        ax.plot(*self.train_samples(), "o")

    def plot_grid(self, ax):
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars >= 3.")

        plot_grid_funs = {
            1: self._plot_grid_1d,
            2: self._plot_grid_2d,
            3: self._plot_grid_3d,
        }
        plot_grid_funs[self.nvars()](ax)

    def _compute_moment(self, moment: str, smolyak_coefs: Array) -> Array:
        values = 0
        for subspace_idx in range(self._subspace_gen.nindices()):
            if abs(smolyak_coefs[subspace_idx]) <= np.finfo(float).eps:
                continue

            subspace_moment = getattr(
                self._subspace_surrogates[subspace_idx], moment
            )()
            values += smolyak_coefs[subspace_idx] * subspace_moment
        return values

    def mean(self) -> Array:
        return self._compute_moment("mean", self._smolyak_coefs)

    def variance(self) -> Array:
        return self._compute_moment("variance", self._smolyak_coefs)

    def train_samples(self) -> Array:
        return self._train_samples

    def train_values(self) -> Array:
        return self._train_values

    def set_subspace_admissibility_criteria(
        self, admis_criteria: SparseGridSubSpaceAdmissibilityCriteria
    ):
        if not isinstance(
            admis_criteria, SparseGridSubSpaceAdmissibilityCriteria
        ):
            raise ValueError(
                "admis_criteria must be an instance of "
                "SparseGridSubspaceAdmissibilityCriteria"
            )
        if self._subspace_gen is None:
            raise ValueError("must first call set_subspace_generator")
        admis_criteria.set_sparse_grid(self)
        self._subspace_gen.set_admissibility_function(admis_criteria)


class IsotropicCombinationSparseGrid(CombinationSparseGrid):
    def __init__(
        self,
        nqoi: int,
        nvars: int,
        max_level: int,
        growth_rules: List[IndexGrowthRule],
        basis: TensorProductInterpolatingBasis,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(nqoi, nvars, backend=backend)
        self._set_basis(basis)
        self._set_subspace_generator(
            HyperbolicIndexGenerator(nvars, max_level, 1.0, backend=self._bkd),
            growth_rules,
        )

    def _set_basis(self, basis: TensorProductInterpolatingBasis):
        super().set_basis(basis)

    def set_basis(self, basis):
        raise RuntimeError("Do not set basis as it has already been set")

    def set_subspace_generator(
        self, subspace_gen, growth_rules: List[IndexGrowthRule]
    ):
        raise RuntimeError(
            "Do not call set_subspace_generator as it has already been set"
        )

    def set_subspace_admissibility_criteria(self, admis_criteria):
        raise RuntimeError(
            "Do not call set_subspace_admissibility_criteria as it has "
            "already been set"
        )

    def _set_subspace_generator(
        self, subspace_gen, growth_rules: List[IndexGrowthRule]
    ):
        super().set_subspace_generator(subspace_gen, growth_rules)
        # must call get_indices to generate basis indices
        subspace_indices = self._subspace_gen.get_indices()
        for subspace_index in subspace_indices.T:
            self._basis_gen._set_unique_subspace_basis_indices(
                subspace_index, False
            )
        self._setup_subspaces(subspace_indices, False)

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

    def _fit(self, iterate: Array):
        self._train_values = self._out_trans.map_from_canonical(
            self._ctrain_values
        )
        # set training data sets ctrain samples and values
        self._set_training_data(self._train_samples, self._train_values)
        for subspace_idx in range(self._subspace_gen.nindices()):
            subspace_values = self._ctrain_values[
                self._basis_gen._subspace_basis_idx[subspace_idx], :
            ]
            self._subspace_surrogates[subspace_idx].fit(subspace_values)
            self._set_smolyak_coefficients()


class Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(
    SparseGridSubSpaceAdmissibilityCriteria
):
    def __init__(self, max_levels_1d: List[int]):
        super().__init__()
        self._max_levels_1d = max_levels_1d

    def __call__(self, index: Array) -> bool:
        if self._bkd.any(index > self._bkd.asarray(self._max_levels_1d)):
            return False
        return True


class MaxLevelSparseGridSubSpaceAdmissibilityCriteria(
    SparseGridSubSpaceAdmissibilityCriteria
):
    def __init__(self, max_level: int, pnorm: float):
        super().__init__()
        self._max_level = max_level
        self._pnorm = pnorm

    def _indices_norm(self, indices: Array) -> float:
        return self._bkd.sum(indices**self._pnorm, axis=0) ** (
            1.0 / self._pnorm
        )

    def __call__(self, index: Array) -> bool:
        if self._indices_norm(index) <= self._max_level:
            return True
        return False


class MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(
    SparseGridSubSpaceAdmissibilityCriteria
):
    def __init__(self, max_nsamples: int):
        super().__init__()
        self._max_nsamples = max_nsamples

    def __call__(self, index: Array) -> bool:
        if self._sg._train_samples.shape[1] < self._max_nsamples:
            return True
        return False


class MaxErrorSparseGridSubspaceAdmissibilityCriteria(
    SparseGridSubSpaceAdmissibilityCriteria
):
    def __init__(self, max_error: float):
        super().__init__()
        self._max_error = max_error

    def __call__(self, index: Array) -> bool:
        if len(self._sg._subspace_errors) == 0:
            return True
        if self._sg.error() > self._max_error:
            return True
        return False


class MultipleSparseGridSubSpaceAdmissibilityCriteria(
    SparseGridSubSpaceAdmissibilityCriteria
):
    def __init__(
        self, criterion: List[SparseGridSubSpaceAdmissibilityCriteria]
    ):
        if not isinstance(criterion, list):
            raise ValueError("criterion must be a list")
        for criteria in criterion:
            if not isinstance(
                criteria, SparseGridSubSpaceAdmissibilityCriteria
            ):
                raise ValueError(
                    "Criteria must be an instance of "
                    "SparseGridSubSpaceAdmissibilityCriteria"
                )
        self._criterion = criterion

    def set_sparse_grid(self, sg: CombinationSparseGrid):
        for criteria in self._criterion:
            criteria.set_sparse_grid(sg)

    def __call__(self, index: Array) -> bool:
        for criteria in self._criterion:
            if not criteria(index):
                self._failed_criteria = criteria
                return False
        return True

    def failure_message(self) -> str:
        return "{0} not met".format(self._failed_criteria)

    def __repr__(self) -> str:
        return "{0}({1})".format(
            self.__class__.__name__,
            ", ".join([criteria.__repr__() for criteria in self._criterion]),
        )


class SparseGridBasisAdmissibilityCriteria(SparseGridAdmissibilityCriteria):
    pass


class MaxCostSparseGridBasisAdmissibilityCriteria(
    SparseGridBasisAdmissibilityCriteria
):
    def __init__(self, max_cost: float):
        super().__init__()
        self._max_cost = max_cost

    def cost_per_sample(self, subspace_index: Array) -> float:
        return 1

    def _sparse_grid_cost(self) -> float:
        return self._bkd.sum(
            self._bkd.asarray(
                [
                    self.cost_per_sample(
                        self._sg._basis_gen._subspace_index_from_basis_index(
                            basis_index
                        )
                    )
                    for basis_index in self._sg._basis_gen.get_indices().T
                ]
            )
        )

    def __call__(self, basis_index: Array, subspace_index: Array) -> bool:
        # We cannot guarantee that this bound will be satisfied exactly
        # as we canno take into account other points that may be added
        # as candidates from the same best_basis_index
        return (
            self.cost_per_sample(subspace_index) + self._sparse_grid_cost()
            <= self._max_cost
        )


class RefinementCriteria(ABC):
    def __init__(self):
        self._sg = None
        self._bkd = None

    def set_sparse_grid(self, sg: CombinationSparseGrid):
        self._sg = sg
        self._bkd = self._sg._bkd

    def cost_per_sample(self, subspace_index: Array) -> float:
        return 1

    def cost(self, subspace_index: Array) -> float:
        """Computational cost of collecting the unique training data."""
        key = self._sg._subspace_gen._hash_index(subspace_index)
        subspace_idx = self._sg._subspace_gen._cand_indices_dict[key]
        return self._sg._nunique_subspace_samples[
            subspace_idx
        ] * self.cost_per_sample(subspace_index)

    @abstractmethod
    def _priority(self, subspace_index: Array) -> float:
        raise NotImplementedError

    def __call__(self, subspace_index: Array) -> Tuple[float, float]:
        # divide priority by cost so to choose subspaces
        # that require smaller training data collection costs first
        error, priority = self._priority(subspace_index)
        priority /= self.cost_per_sample(subspace_index)
        return error, priority

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class LevelRefinementCriteria(RefinementCriteria):
    def _priority(self, subspace_index: Array) -> Tuple[float, float]:
        # Avoid computing error which requires specifying a metric
        # and can slow criteria evaluation down
        error = np.inf
        # priority queue gives higher priority to smaler values
        # so this will add subspaces wih lower l1 norm first
        priority = self._bkd.sum(
            self._bkd.asarray(subspace_index, dtype=self._bkd.double_type())
        )
        return priority, error


class L2NormRefinementCriteria(RefinementCriteria):
    def _priority(self, subspace_index: Array) -> Tuple[float, float]:
        subspace_key = self._sg._basis_gen._hash_index(subspace_index)
        subspace_idx = self._sg._subspace_gen._cand_indices_dict[subspace_key]
        # This computes l2 norm over samples already in sparse grid
        # as well as new unique samples
        subspace_values = self._sg._train_values[
            self._sg._basis_gen._subspace_basis_idx[subspace_idx], :
        ]
        subspace_samples = self._sg._train_samples[
            : self._sg.nvars(),  # ignore refinement vars when evaluating
            self._sg._basis_gen._subspace_basis_idx[subspace_idx],
        ]
        error = (
            self._bkd.norm(
                subspace_values
                - self._sg._values_using_only_selected_subspaces(
                    subspace_samples
                )
            )
            / subspace_values.shape[0]
        )
        # print("####", subspace_index)
        # print(error)
        # print(subspace_values[:, 0])
        # print(self._sg(subspace_samples)[:, 0])
        # priority queue gives higher priority to smaler values so set
        # priority = -error
        return -error, error


class VarianceRefinementCriteria(RefinementCriteria):
    def _priority(self, subspace_index: Array) -> Tuple[float, float]:
        current_mean = self._sg.mean()
        current_variance = self._sg.variance()
        new_smolyak_coefs = (
            self._sg._adjust_smolyak_coefficients_with_candidate_index(
                self._sg._smolyak_coefs, subspace_index
            )
        )
        new_mean = self._sg._compute_moment("mean", new_smolyak_coefs)
        new_variance = self._sg._compute_moment("variance", new_smolyak_coefs)
        error = (
            self._bkd.abs(new_mean - current_mean)
            + self._bkd.sqrt(self._bkd.abs(new_variance - current_variance))
        ).max()
        # print("####", subspace_index)
        # print(self._sg._smolyak_coefs)
        # print(new_smolyak_coefs)
        # print(
        #     current_mean, new_mean, current_variance, new_variance, "V", error
        # )
        return -error, error


class AdaptiveCombinationSparseGrid(
    CombinationSparseGrid, AdaptiveRegressorMixin
):
    def __init__(
        self, nqoi: int, nvars: int, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        super().__init__(nqoi, nvars, backend)
        self._last_subspace_indices = None
        self._refine_criteria = None
        self._first_step = True

    def _fit(self):
        # TODO make adaptive regressor class
        raise NotImplementedError("Adaptive regressors do not call fit")

    def set_refinement_criteria(self, refine_criteria: RefinementCriteria):
        if not isinstance(refine_criteria, RefinementCriteria):
            raise ValueError(
                "refine_criteria must be and instance of RefinementCriteria"
            )
        self._refine_criteria = refine_criteria
        self._refine_criteria.set_sparse_grid(self)

    def set_initial_subspace_indices(self, subspace_indices: Array = None):
        if subspace_indices is None:
            subspace_indices = self._bkd.zeros(
                (self.nsubspace_vars(), 1), dtype=int
            )
        # TODO change set selected indices to only allow generation of
        # candidate indices accorging to variable ordering.
        # allow group based ordering, where dimensions in a group
        # are unordered
        self._basis_gen._set_selected_subspace_indices(subspace_indices)
        # only set smolyak coefficients to use selected indices
        self._smolyak_coefs = _compute_smolyak_coefficients(
            self, self._subspace_gen.get_selected_indices()
        )
        self._smolyak_coefs = self._bkd.hstack(
            (
                self._smolyak_coefs,
                self._bkd.zeros((self._subspace_gen.ncandidate_indices(),)),
            )
        )

    def _setup_first_selected_subspaces(self) -> Array:
        unique_sel_samples = self._setup_subspaces(
            self._subspace_gen.get_selected_indices(), False
        )
        return unique_sel_samples

    def _setup_first_candidate_subspaces(self) -> Array:
        unique_cand_samples = self._setup_subspaces(
            self._subspace_gen.get_candidate_indices(), True
        )
        return unique_cand_samples

    def _first_step_samples(self) -> Array:
        if self._subspace_gen is None:
            raise RuntimeError("Must call set_subspace_generator")
        if self._refine_criteria is None:
            raise RuntimeError("Must call set_refinement_criteria")
        if self._subspace_gen.nselected_indices() == 0:
            raise RuntimeError("must call set_initial_subspace_indices")
        unique_sel_samples = self._setup_first_selected_subspaces()
        unique_cand_samples = self._setup_first_candidate_subspaces()
        unique_samples = self._bkd.hstack(
            (unique_sel_samples, unique_cand_samples)
        )
        self._last_subspace_indices = self._subspace_gen.get_indices()
        self._first_step = False
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
            self._subspace_errors[subspace_idx] = subspace_error

    def _get_best_cand_subspace(self) -> Tuple[Array, int]:
        priority, error, best_subspace_idx = self._cand_subspace_queue.get()
        best_cand_subspace_index = self._subspace_gen._indices[
            :, best_subspace_idx
        ]
        if self._verbosity > 0:
            print(
                f"Refining subspace {best_cand_subspace_index} with "
                f"{priority=}"
            )
        # self._subspace_errors[best_subspace_idx] *= 0.0
        return best_cand_subspace_index, best_subspace_idx

    def _adjust_smolyak_coefficients(
        self, smolyak_coefs: Array, new_index: Array, selected_idx: Array
    ) -> Array:
        new_smolyak_coefs = self._bkd.copy(smolyak_coefs)
        for idx in selected_idx:
            diff = new_index - self._subspace_gen._indices[:, idx]
            if self._bkd.all(diff >= 0) and self._bkd.max(diff) <= 1:
                new_smolyak_coefs[idx] += (-1.0) ** self._bkd.sum(diff)
        return new_smolyak_coefs

    def _adjust_smolyak_coefficients_with_selected_index(
        self, new_index: Array
    ):
        """
        For use when smolyak_coefficient has been added to the selected
        set
        """
        key = self._subspace_gen._hash_index(new_index)
        if key not in self._subspace_gen._sel_indices_dict:
            raise RuntimeError("new_index has not been selected")
        selected_idx = self._subspace_gen._get_selected_idx()
        return self._adjust_smolyak_coefficients(
            self._smolyak_coefs, new_index, selected_idx
        )

    def _adjust_smolyak_coefficients_with_candidate_index(
        self, smolyak_coefs: Array, new_index: Array
    ):
        """
        For use when smolyak_coefficient is still in candidate set.
        Typically used when computing subspace priorities
        """
        key = self._subspace_gen._hash_index(new_index)
        if key not in self._subspace_gen._cand_indices_dict:
            raise RuntimeError("new_index is not a candidate index")
        new_index_idx = self._subspace_gen._cand_indices_dict[key]
        selected_idx = self._bkd.hstack(
            (self._subspace_gen._get_selected_idx(), new_index_idx)
        )
        return self._adjust_smolyak_coefficients(
            smolyak_coefs, new_index, selected_idx
        )

    def _step_samples(self) -> Array:
        while len(self._subspace_gen._cand_indices_dict) > 0:
            best_subspace_index, best_subspace_idx = (
                self._get_best_cand_subspace()
            )
            new_subspace_indices = self._basis_gen.refine_subspace_index(
                best_subspace_index
            )
            self._smolyak_coefs = self._bkd.hstack(
                (
                    self._smolyak_coefs,
                    self._bkd.zeros((new_subspace_indices.shape[1],)),
                )
            )
            # update smolyak coefs must occur here even
            # if no candidates are added
            self._smolyak_coefs = (
                self._adjust_smolyak_coefficients_with_selected_index(
                    best_subspace_index
                )
            )
            self._subspace_errors[best_subspace_idx] *= 0.0
            if new_subspace_indices.shape[1] > 0:
                self._last_subspace_indices = new_subspace_indices
                return self._setup_subspaces(new_subspace_indices, True)
        return None

    def step_samples(self) -> Array:
        if self._first_step:
            return self._first_step_samples()
        return self._step_samples()

    def step_values(self, values: Array):
        # set training data sets ctrain samples and values
        self._train_values = self._bkd.vstack((self._train_values, values))
        self._set_training_data(self._train_samples, self._train_values)
        for subspace_index in self._last_subspace_indices.T:
            key = self._subspace_gen._hash_index(subspace_index)
            if key in self._subspace_gen._sel_indices_dict:
                subspace_idx = self._subspace_gen._sel_indices_dict[key]
            else:
                subspace_idx = self._subspace_gen._cand_indices_dict[key]
            subspace_values = self._ctrain_values[
                self._basis_gen._subspace_basis_idx[subspace_idx], :
            ]
            self._subspace_surrogates[subspace_idx].fit(subspace_values)
        self._prioritize_candidate_subspaces()

    def step(self, fun: Model) -> bool:
        unique_samples = self.step_samples()
        if unique_samples is None:
            return False
        unique_values = fun(unique_samples)
        self.step_values(unique_values)
        return True

    def error(self):
        return self._bkd.sum(self._bkd.asarray(self._subspace_errors))

    def setup(
        self,
        subspace_admissibility_criteria: SparseGridSubSpaceAdmissibilityCriteria,
        refinement_criteria: RefinementCriteria = L2NormRefinementCriteria(),
        univariate_bases: List[UnivariateBasis] = None,
        growth_rule: IndexGrowthRule = LinearGrowthRule(2, 1),
    ):
        basis = TensorProductInterpolatingBasis(univariate_bases)
        self.set_basis(basis)
        subspace_gen = IterativeIndexGenerator(
            self.nsubspace_vars(), backend=self._bkd
        )
        self.set_subspace_generator(subspace_gen, growth_rule)
        self.set_subspace_admissibility_criteria(
            subspace_admissibility_criteria
        )
        self.set_refinement_criteria(refinement_criteria)
        self.set_initial_subspace_indices()


class LocalIndexGenerator(BasisIndexGenerator):
    def __init__(
        self, nvars: int, nrefinement_vars: int, gen, growth_rules, verbosity=0
    ):
        super().__init__(nvars, nrefinement_vars, gen, growth_rules)
        self._sel_basis_indices_dict = dict()
        self._cand_basis_indices_dict = dict()
        self._admis_fun = None
        self._verbosity = verbosity

    def _subspace_index_from_basis_index(
        self, basis_index: Array
    ) -> Tuple[Array, int]:
        subspace_idx = self._basis_indices_dict[self._hash_index(basis_index)][
            1
        ]
        return self._subspace_gen._indices[:, subspace_idx], subspace_idx

    def _set_selected_subspace_indices(self, subspace_indices: Array):
        super()._set_selected_subspace_indices(subspace_indices)
        for _, subspace_idx in self._subspace_gen._sel_indices_dict.items():
            basis_idxs = self._subspace_basis_idx[subspace_idx][
                self._unique_subspace_basis_idx[subspace_idx]
            ]
            for basis_idx in basis_idxs:
                basis_index = self._basis_indices[:, basis_idx]
                basis_key = self._hash_index(basis_index)
                self._sel_basis_indices_dict[basis_key] = (
                    self._basis_indices_dict[basis_key][0]
                )
        for _, subspace_idx in self._subspace_gen._cand_indices_dict.items():
            basis_idxs = self._subspace_basis_idx[subspace_idx][
                self._unique_subspace_basis_idx[subspace_idx]
            ]
            for basis_idx in basis_idxs:
                basis_index = self._basis_indices[:, basis_idx]
                basis_key = self._hash_index(basis_index)
                self._cand_basis_indices_dict[basis_key] = (
                    self._basis_indices_dict[basis_key][0]
                )

    # def _subspace_index_from_basis_index(self, basis_index):
    #     subspace_index = []
    #     for dim_id in range(self.nvars()):
    #         if basis_index[dim_id] == 0:
    #             subspace_index.append(0)
    #         elif basis_index[dim_id] <= 2:
    #             subspace_index.append(1)
    #         else:
    #             subspace_index.append(
    #                 int(round(math.log(basis_index[dim_id], 2), 0)))
    #     return self._bkd.array(subspace_index, dtype=int)

    def _left_neighbor(
        self, basis_index: Array, subspace_index: Array, dim_id: int
    ) -> Array:
        subspace_neighbor = self._bkd.copy(subspace_index)
        subspace_neighbor[dim_id] += 1
        if basis_index[dim_id] == 1:
            return None, subspace_neighbor
        basis_neighbor = self._bkd.copy(basis_index)
        if basis_index[dim_id] == 0:
            basis_neighbor[dim_id] = 1
        elif basis_index[dim_id] == 2:
            basis_neighbor[dim_id] = 4
        else:
            basis_neighbor[dim_id] = 2 * basis_index[dim_id] - 1
        return basis_neighbor, subspace_neighbor

    def _right_neighbor(
        self, basis_index: Array, subspace_index: Array, dim_id: int
    ) -> Array:
        subspace_neighbor = self._bkd.copy(subspace_index)
        subspace_neighbor[dim_id] += 1
        if basis_index[dim_id] == 2:
            return None, subspace_neighbor
        basis_neighbor = self._bkd.copy(basis_index)
        if basis_index[dim_id] == 0:
            basis_neighbor[dim_id] = 2
        elif basis_index[dim_id] == 1:
            basis_neighbor[dim_id] = 3
        else:
            basis_neighbor[dim_id] = 2 * basis_index[dim_id]
        return basis_neighbor, subspace_neighbor

    def _parent(self, basis_index: Array, dim_id: int) -> Array:
        parent = self._bkd.copy(basis_index)
        parent[dim_id] = (basis_index[dim_id] + (basis_index[dim_id] % 2)) / 2
        return parent

    def _is_admissible(
        self, basis_index: Array, subspace_index: Array
    ) -> bool:
        if basis_index is None:
            return False
        key = self._subspace_gen._hash_index(subspace_index)
        if (
            not self._subspace_gen._is_admissible(subspace_index)
            and key not in self._subspace_gen._cand_indices_dict
            and key not in self._subspace_gen._sel_indices_dict
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
                break
            parent_basis_index = self._parent(basis_index, dim_id)
            key = self._subspace_gen._hash_index(parent_basis_index)
            if key in self._sel_basis_indices_dict:
                break
        return self._admis_fun(basis_index, subspace_index)

    def get_candidate_basis_indices(
        self, basis_index: Array, subspace_index: Array
    ) -> Tuple[Array, Array]:
        new_cand_basis_indices, subspace_indices_of_cand_bases = [], []
        for dim_id in range(self.nvars()):
            left_basis, left_subspace = self._left_neighbor(
                basis_index, subspace_index, dim_id
            )
            if self._is_admissible(left_basis, left_subspace):
                new_cand_basis_indices.append(left_basis)
                subspace_indices_of_cand_bases.append(left_subspace)
                if self._verbosity > 1:
                    print("Add left", basis_index, left_basis, dim_id)
            right_basis, right_subspace = self._right_neighbor(
                basis_index, subspace_index, dim_id
            )
            if self._is_admissible(right_basis, right_subspace):
                if self._verbosity > 1:
                    print("Add right", basis_index, right_basis, dim_id)
                new_cand_basis_indices.append(right_basis)
                subspace_indices_of_cand_bases.append(right_subspace)
        if len(new_cand_basis_indices) > 0:
            return (
                self._bkd.stack(new_cand_basis_indices, axis=1),
                self._bkd.stack(subspace_indices_of_cand_bases, axis=1),
            )
        return (
            self._bkd.zeros((self.nvars(), 0)),
            self._bkd.zeros((self.nvars(), 0)),
        )

    def refine_basis_index(
        self, basis_index: Array
    ) -> Tuple[Array, Array, Array]:
        subspace_index = self._subspace_index_from_basis_index(basis_index)[0]
        if self._verbosity > 0:
            print(
                f"Refining basis index {basis_index} from subspace"
                f" {subspace_index}"
            )
        basis_key = self._hash_index(basis_index)
        # update candidate subspaces
        if (
            self._subspace_gen._hash_index(subspace_index)
            in self._subspace_gen._cand_indices_dict
        ):
            # this step adds all susbpace basis indices to self._basis_indices
            # and self._basis_indices_dict
            new_cand_subspace_indices = self.refine_subspace_index(
                subspace_index
            )
            # new basis_index was in a subspace not in sel_subspace_indices
            # so must update smolyak coefficients
            sel_subspace_added = True
        else:
            new_cand_subspace_indices = self._bkd.zeros((self.nvars(), 0))
            # new basis_index was already in a subspace in sel_subspace_indices
            # so do not need to update smolyak coefficients
            sel_subspace_added = False

        self._sel_basis_indices_dict[basis_key] = (
            self._cand_basis_indices_dict[basis_key]
        )
        del self._cand_basis_indices_dict[basis_key]
        cand_basis_indices, subspace_indices_of_cand_bases = (
            self.get_candidate_basis_indices(basis_index, subspace_index)
        )
        # There may be repeated candidate basis indices,
        # so find unique indices
        new_cand_basis_indices, new_basis_subspace_idx = [], []
        for cand_basis_index, new_subspace_index in zip(
            cand_basis_indices.T, subspace_indices_of_cand_bases.T
        ):
            cand_basis_key = self._hash_index(cand_basis_index)
            if cand_basis_key not in self._cand_basis_indices_dict:
                self._cand_basis_indices_dict[cand_basis_key] = (
                    self._basis_indices_dict[cand_basis_key][0]
                )
                new_cand_basis_indices.append(cand_basis_index)
                subspace_key = self._subspace_gen._hash_index(
                    new_subspace_index
                )
                if subspace_key in self._subspace_gen._cand_indices_dict:
                    cand_subspace_idx = self._subspace_gen._cand_indices_dict[
                        subspace_key
                    ]
                else:
                    cand_subspace_idx = self._subspace_gen._sel_indices_dict[
                        subspace_key
                    ]
                new_basis_subspace_idx.append(cand_subspace_idx)
            else:
                raise RuntimeError("This should not happen")

        if len(new_cand_basis_indices) == 0:
            return (
                self._bkd.zeros((self.nvars(), 0), dtype=int),
                self._bkd.zeros((self.nvars(), 0), dtype=int),
                sel_subspace_added,
            )

        new_cand_basis_indices = self._bkd.stack(
            new_cand_basis_indices, axis=1
        )
        return (
            new_cand_basis_indices,
            new_cand_subspace_indices,
            sel_subspace_added,
        )

    def get_indices(self) -> Array:
        return self._basis_indices

    def nselected_indices(self) -> int:
        return len(self._sel_basis_indices_dict)

    def ncandidate_indices(self) -> int:
        return len(self._cand_basis_indices_dict)

    def _get_candidate_idx(self) -> Array:
        # return  elements in self._indices that contain candidate indices
        return self._bkd.asarray(
            [item for key, item in self._cand_basis_indices_dict.items()],
            dtype=int,
        )

    def _get_selected_idx(self) -> Array:
        # return  elements in self._indices that contain selected indices
        return self._bkd.asarray(
            [item for key, item in self._sel_basis_indices_dict.items()],
            dtype=int,
        )

    def get_candidate_indices(self) -> Array:
        if self.ncandidate_indices() > 0:
            return self._basis_indices[:, self._get_candidate_idx()]
        return None

    def _find_candidate_indices(self, basis_indices: Array) -> Array:
        new_cand_basis_indices = []
        idx = basis_indices.shape[1]
        for basis_index in basis_indices.T:
            subspace_index, subspace_idx = (
                self._subspace_index_from_basis_index(basis_index)
            )
            cand_basis_indices, subspace_indices_of_cand_bases = (
                self.get_candidate_basis_indices(basis_index, subspace_index)
            )
            for new_basis_index, new_subspace_index in zip(
                cand_basis_indices.T, subspace_indices_of_cand_bases.T
            ):
                basis_key = self._hash_index(new_basis_index)
                if basis_key not in self._cand_basis_indices_dict:
                    self._cand_basis_indices_dict[basis_key] = idx
                    new_cand_basis_indices.append(new_basis_index)
                    idx += 1
        return self._bkd.stack(new_cand_basis_indices, axis=1)

    # def set_selected_indices(self, sel_basis_indices, sel_basis_indices_dict):
    #     self._sel_basis_indices_dict = dict()
    #     self._cand_basis_indices_dict = dict()
    #     self._basis_indices = self._bkd.copy(sel_basis_indices)
    #     self._basis_indices_dict = sel_basis_indices_dict
    #     self._sel_basis_indices_dict = sel_basis_indices_dict.copy()
    #     cand_basis_indices = (
    #         self._find_candidate_indices(self._basis_indices)
    #     )
    #     self._basis_indices = self._bkd.hstack(
    #         (self._basis_indices, cand_basis_indices)
    #     )

    def set_admissibility_function(self, admis_criteria):
        self._admis_fun = admis_criteria


class LocalRefinementCriteria(ABC):
    def __init__(self):
        self._sg = None
        self._bkd = None

    def set_sparse_grid(self, sg: CombinationSparseGrid):
        self._sg = sg
        self._bkd = self._sg._bkd

    def cost_per_sample(self, subspace_index: Array) -> float:
        return 1

    def cost(self, basis_index: Array) -> float:
        """Computational cost of collecting the single training data."""
        subspace_index = self._sg._basis_gen._subspace_index_from_basis_index(
            basis_index
        )[0]
        return self.cost_per_sample(subspace_index)

    @abstractmethod
    def _priority(self, basis_index: Array) -> float:
        raise NotImplementedError

    def __call__(self, basis_index: Array) -> Tuple[float, float]:
        # divide priority by cost so to choose subspaces
        # that require smaller training data collection costs first
        error, priority = self._priority(basis_index)
        if priority > 0:
            raise RuntimeError("priority must be negative")
        priority /= self.cost_per_sample(basis_index)
        return error, priority

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class LocalHierarchicalRefinementCriteria(LocalRefinementCriteria):
    def _priority(self, basis_index: Array) -> Tuple[float, float]:
        # todo consider prioritizing all candidate basis indices at once
        # for local hierarhical surplus.
        key = self._sg._basis_gen._hash_index(basis_index)
        basis_idx = self._sg._basis_gen._cand_basis_indices_dict[key]
        error = self._bkd.abs(
            self._sg._train_values[basis_idx : basis_idx + 1]
            - self._sg(self._sg._train_samples[:, basis_idx : basis_idx + 1])
        )
        priority = -self._bkd.max(error)
        return error, priority


class LocallyAdaptiveCombinationSparseGrid(AdaptiveCombinationSparseGrid):
    def __init__(
        self, nqoi: int, nvars: int, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        super().__init__(nqoi, nvars, backend)
        self._cand_basis_queue = None
        self._last_basis_indices = None
        self._last_nunique_samples = None
        self._refine_criteria = None

    def set_refinement_criteria(
        self, refine_criteria: LocalRefinementCriteria
    ):
        if refine_criteria is None:
            refine_criteria = LocalHierarchicalRefinementCriteria()
        if not isinstance(refine_criteria, LocalRefinementCriteria):
            raise ValueError(
                "refine_criteria must be and instance of "
                "LocalRefinementCriteria"
            )
        self._refine_criteria = refine_criteria
        self._refine_criteria.set_sparse_grid(self)

    def _set_basis_index_generator(self, growth_rules: List[IndexGrowthRule]):
        self._basis_gen = LocalIndexGenerator(
            self.nvars(),
            self.nrefinement_vars(),
            self._subspace_gen,
            growth_rules,
            self._verbosity,
        )

    def set_subspace_generator(self, subspace_gen):
        super().set_subspace_generator(
            subspace_gen, DoublePlusOneIndexGrowthRule()
        )

    def _get_best_cand_basis(self) -> Array:
        priority, best_basis_idx, error = self._cand_basis_queue.get()
        best_cand_basis_index = self._basis_gen._basis_indices[
            :, best_basis_idx
        ]
        return best_cand_basis_index

    def _first_step_samples(self) -> Array:
        unique_samples = super()._first_step_samples()
        self._last_nunique_samples = unique_samples.shape[1]
        self._last_global_data_idx = self._bkd.arange(
            self._last_nunique_samples
        )
        return unique_samples

    def _global_data_idx(self, new_basis_indices: Array) -> List[int]:
        global_data_idx = [
            self._basis_gen._cand_basis_indices_dict[
                self._subspace_gen._hash_index(basis_index)
            ]
            for basis_index in new_basis_indices.T
        ]
        return global_data_idx

    def _step_samples(self) -> Array:
        while len(self._basis_gen._cand_basis_indices_dict) > 0:
            best_basis_index = self._get_best_cand_basis()
            new_basis_indices, new_subspace_indices, sel_subspace_added = (
                self._basis_gen.refine_basis_index(best_basis_index)
            )
            best_subspace_index = (
                self._basis_gen._subspace_index_from_basis_index(
                    best_basis_index
                )[0]
            )
            if sel_subspace_added:
                self._smolyak_coefs = self._bkd.hstack(
                    (
                        self._smolyak_coefs,
                        self._bkd.zeros((new_subspace_indices.shape[1],)),
                    )
                )
                self._smolyak_coefs = (
                    self._adjust_smolyak_coefficients_with_selected_index(
                        best_subspace_index
                    )
                )
            if new_basis_indices.shape[1] == 0:
                continue
            if new_subspace_indices.shape[1] > 0:
                unique_samples = self._setup_subspaces(
                    new_subspace_indices, True
                )
                self._last_nunique_samples = unique_samples.shape[1]
            else:
                self._last_nunique_samples = 0
            self._last_basis_indices = new_basis_indices
            global_data_idx = self._global_data_idx(new_basis_indices)
            self._last_global_data_idx = global_data_idx
            # only return samples that have been identified as important
            return self._train_samples[:, global_data_idx]
        return None

    def _prioritize_candidate_bases(self):
        # reset priority of all candidate subspace indices
        self._cand_basis_queue = PriorityQueue()
        for basis_index in self._basis_gen.get_candidate_indices().T:
            subspace_error, basis_indicator = self._refine_criteria(
                basis_index
            )
            key = self._basis_gen._hash_index(basis_index)
            basis_idx = self._basis_gen._cand_basis_indices_dict[key]
            self._cand_basis_queue.put(
                (basis_indicator, basis_idx, subspace_error)
            )

    def step_values(self, values: Array):
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
        self._train_values[self._last_global_data_idx, :] = values
        # set training data sets ctrain samples and values
        self._set_training_data(self._train_samples, self._train_values)
        # Loop over all subspaces effected by data update above
        # For now just loop over all subspaces.
        # TODO only loop over effected subspaces
        for subspace_idx in range(self._subspace_gen.nindices()):
            subspace_values = self._ctrain_values[
                self._basis_gen._subspace_basis_idx[subspace_idx], :
            ]
            self._subspace_surrogates[subspace_idx].fit(subspace_values)
        self._prioritize_candidate_bases()

    def _candidate_train_samples(self) -> Array:
        idx = self._basis_gen._get_candidate_idx()
        return self.train_samples()[:, idx]

    def _selected_train_samples(self) -> Array:
        idx = self._basis_gen._get_selected_idx()
        return self.train_samples()[:, idx]

    def _plot_grid_1d(self, ax):
        super()._plot_grid_1d(ax)
        sel_samples = self._selected_train_samples()
        ax.plot(sel_samples[0], sel_samples[0] * 0, "o")
        cand_samples = self._candidate_train_samples()
        ax.plot(cand_samples[0], cand_samples[0] * 0, "X")

    def _plot_grid_2d(self, ax):
        sel_samples = self._selected_train_samples()
        ax.plot(*sel_samples, "o")
        cand_samples = self._candidate_train_samples()
        ax.plot(*cand_samples, "X")

    def _plot_grid_3d(self, ax):
        if not isinstance(ax, Axes3D):
            raise ValueError(
                "ax must be an instance of  mpl_toolkits.mplot3d.Axes3D"
            )
        sel_samples = self._selected_train_samples()
        ax.plot(*sel_samples, "o")
        cand_samples = self._candidate_train_samples()
        ax.plot(*cand_samples, "X")

    def set_basis_admissibility_criteria(
        self, admis_criteria: SparseGridBasisAdmissibilityCriteria
    ):
        if not isinstance(
            admis_criteria, SparseGridBasisAdmissibilityCriteria
        ):
            raise ValueError(
                "admis_criteria must be an instance of "
                "SparseGridBasisAdmissibilityCriteria"
            )
        if self._subspace_gen is None:
            raise ValueError("must first call set_subspace_generator")
        admis_criteria.set_sparse_grid(self)
        self._basis_gen.set_admissibility_function(admis_criteria)


class SparseGridToOrthonormalPolynomialChaosExpansionConverter:
    def __init__(self, quad_rule: TensorProductQuadratureRule):
        self._tp_converter = TensorProductLagrangeInterpolantToPolynomialChaosExpansionConverter(
            quad_rule
        )

    def _check_sparse_grid(self, sg: CombinationSparseGrid):
        if not isinstance(sg, CombinationSparseGrid):
            print(sg)
            raise Exception("Can only convert a sparse grid")

    def _univariate_lagrange_basis_to_orthopoly(
        self, sg: CombinationSparseGrid
    ):
        for basis1d in sg._basis._bases_1d:
            basis1d.orthonormal_polynomial_coefficients()

    def convert(self, sg: CombinationSparseGrid) -> PolynomialChaosExpansion:
        self._check_sparse_grid(sg)
        pce = 0.0
        for subspace_idx in range(1, sg._subspace_gen.nindices()):
            pce += sg._smolyak_coefs[
                subspace_idx
            ] * self._tp_converter.convert(
                sg._subspace_surrogates[subspace_idx]
            )
        return pce


class LejaLagrangeAdaptiveCombinationSparseGrid(AdaptiveCombinationSparseGrid):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nqoi: int,
    ):
        self._variable = variable
        super().__init__(nqoi, variable.nvars(), variable._bkd)

    def unique_univariate_leja_quadrature_rules(
        self,
        init_sequences: List[Array] = None,
        leja_quad_rule_cls: LejaQuadratureRule = TwoPointChristoffelLejaQuadratureRule,
    ) -> List[TwoPointChristoffelLejaQuadratureRule]:
        # Constructing Leja sequence has a non-trivial cost so just create
        # once per unique marginal
        if init_sequences is not None:
            for ii in range(self._variable._nunique_vars):
                kk = self._variable._unique_variable_indices[ii][0]
                for jj in self._variable._unique_variable_indices[ii][1:]:
                    if not self._variable._bkd.allclose(
                        init_sequences[kk], init_sequences[jj], atol=1e-15
                    ):
                        raise ValueError(
                            "init_sequences differed for the same marginal"
                        )
        else:
            init_sequences = [None] * self._variable.nvars()

        unique_quad_rules = [
            leja_quad_rule_cls(
                marginal, init_sequence=init_seq, backend=self._bkd, store=True
            )
            for marginal, init_seq in zip(
                self._variable._unique_marginals, init_sequences
            )
        ]
        quad_rules = [None for marginal in self._variable.marginals()]
        for ii in range(len(unique_quad_rules)):
            for jj in self._variable._unique_variable_indices[ii]:
                quad_rules[jj] = unique_quad_rules[ii]
        return quad_rules

    def _check_quad_rules(
        self, univariate_quad_rules: List[LejaQuadratureRule]
    ):
        for quad_rule in univariate_quad_rules:
            if not isinstance(quad_rule, LejaQuadratureRule):
                raise ValueError(
                    "Univariate quadrature rules must be an instance of "
                    "LejaQuadratureRule"
                )

    def setup(
        self,
        subspace_admissibility_criteria: SparseGridSubSpaceAdmissibilityCriteria,
        refinement_criteria: RefinementCriteria = L2NormRefinementCriteria(),
        univariate_quad_rules: List[LejaQuadratureRule] = None,
        growth_rule: IndexGrowthRule = LinearGrowthRule(2, 1),
    ):
        if univariate_quad_rules is None:
            univariate_quad_rules = (
                self.unique_univariate_leja_quadrature_rules()
            )
        self._check_quad_rules(univariate_quad_rules)
        bases_1d = [
            UnivariateLagrangeBasis(quad_rule, 1)
            for quad_rule in univariate_quad_rules
        ]
        super().setup(
            subspace_admissibility_criteria,
            refinement_criteria,
            bases_1d,
            growth_rule,
        )


# class MultiIndexBasisGenerator(BasisIndexGenerator):


class MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid(
    LejaLagrangeAdaptiveCombinationSparseGrid
):
    # TODO make MultiIndex Mixin so can use with other types of
    # sparse grids
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nqoi: int,
        nrefinement_vars: int,
        refinement_bounds: Array,
    ):
        self._nrefinement_vars = nrefinement_vars
        self._refinement_bounds = refinement_bounds
        super().__init__(variable, nqoi)

    def nrefinement_vars(self) -> int:
        return self._nrefinement_vars

    def default_max_level_1d_admissibility_criteria(
        self,
    ) -> Max1DLevelSparseGridSubSpaceAdmissibilityCriteria:
        return Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(
            self._bkd.hstack(
                (
                    self._bkd.asarray([np.inf] * self.nvars()),
                    self._refinement_bounds,
                ),
            )
        )

    def step(self, model_ensemble: MultiIndexModelEnsemble) -> bool:
        unique_ensemble_samples_per_model = self.step_samples()
        if unique_ensemble_samples_per_model is None:
            return False
        # print(unique_ensemble_samples_per_model)
        model_ids, unique_samples_per_model, sample_idx_per_model = (
            model_ensemble.split_ensemble_samples(
                unique_ensemble_samples_per_model
            )
        )
        unique_values_per_model = []
        for ii, model_id in enumerate(model_ids.T):
            unique_values_per_model.append(
                model_ensemble.get_model(model_id)(
                    unique_samples_per_model[ii]
                )
            )
        ensemble_values = model_ensemble.combine_values(
            unique_values_per_model, sample_idx_per_model
        )
        self.step_values(ensemble_values)
        return True

    def set_subspace_admissibility_criteria(
        self, admis_criteria: SparseGridSubSpaceAdmissibilityCriteria
    ):
        # Make sure there is an upper bound on the refinement variables
        if isinstance(
            admis_criteria, MultipleSparseGridSubSpaceAdmissibilityCriteria
        ):
            criterion = admis_criteria._criterion
        elif isinstance(
            admis_criteria, SparseGridSubSpaceAdmissibilityCriteria
        ):
            criterion = [admis_criteria]
        else:
            raise ValueError(
                "criteria must be an instance of "
                "SparseGridSubSpaceAdmissibilityCriteria"
            )
        found = False
        for criteria in criterion:
            if isinstance(
                admis_criteria,
                Max1DLevelSparseGridSubSpaceAdmissibilityCriteria,
            ):
                found = True
                break
        print(found)
        print(criterion)
        if not found:
            criterion.append(
                self.default_max_level_1d_admissibility_criteria()
            )
        print(criterion)
        admis_criteria = MultipleSparseGridSubSpaceAdmissibilityCriteria(
            criterion
        )
        print(admis_criteria)
        super().set_subspace_admissibility_criteria(admis_criteria)


# TODO mix locally adaptive basis with global polynomial basis in another
# dimension. Do this by adding all left and right neigbours in dimensions
# with global basis
