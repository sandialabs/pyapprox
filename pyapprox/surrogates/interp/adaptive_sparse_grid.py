import pickle
from functools import partial
import heapq
import numpy as np

from pyapprox.variables.marginals import (
    variable_shapes_equivalent, is_bounded_discrete_variable,
    get_probability_masses
)
from pyapprox.util.utilities import (
    lists_of_lists_of_arrays_equal,
    lists_of_arrays_equal, partial_functions_equal, hash_array
)
from pyapprox.surrogates.interp.indexing import (
    get_forward_neighbor, get_backward_neighbor
)
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_rule_growth, constant_increment_growth_rule
)
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule
)
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.util.visualization import plot_2d_indices, plot_3d_indices, plt
from matplotlib.pyplot import MaxNLocator
from pyapprox.surrogates.interp.sparse_grid import (
    get_sparse_grid_samples, integrate_sparse_grid,
    evaluate_sparse_grid, get_subspace_samples,
    get_hierarchical_sample_indices, get_subspace_values,
    get_subspace_weights,
    get_subspace_polynomial_indices,
    get_1d_samples_weights, update_1d_samples_weights,
    integrate_sparse_grid_from_subspace_moments,
    evaluate_sparse_grid_from_subspace_values,
    integrate_sparse_grid_subspace, evaluate_sparse_grid_subspace
 )
from pyapprox.surrogates.interp.tensorprod import (
    piecewise_univariate_linear_quad_rule,
    piecewise_univariate_quadratic_quad_rule
)


class mypriorityqueue():
    def __init__(self):
        self.list = []

    def empty(self):
        return len(self.list) == 0

    def put(self, item):
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


def extract_items_from_priority_queue(pqueue):
    """
    Return the items in a priority queue. The items will only be shallow copies
    of items in queue

    Priority queue is thread safe so does not support shallow or deep copy
    One can copy this queue by pushing and popping by original queue will
    be destroyed. Return a copy of the original queue that can be used to
    replace the destroyed queue
    """

    pqueue1 = mypriorityqueue()
    items = []
    while not pqueue.empty():
        item = pqueue.get()
        items.append(item)
        pqueue1.put(item)
    return items, pqueue1


def update_smolyak_coefficients(new_index, subspace_indices, smolyak_coeffs):
    assert new_index.ndim == 1
    assert subspace_indices.ndim == 2

    new_smolyak_coeffs = smolyak_coeffs.copy()

    try:
        from pyapprox.cython.adaptive_sparse_grid import \
            update_smolyak_coefficients_pyx
        return update_smolyak_coefficients_pyx(new_index, subspace_indices,
                                               new_smolyak_coeffs)
        # from pyapprox.weave.adaptive_sparse_grid import \
        #     c_update_smolyak_coefficients failed
        # # new_index.copy is needed
        # return c_update_smolyak_coefficients(
        #    new_index.copy(),subspace_indices,smolyak_coeffs)
    except ImportError:
        print('update_smolyak_coefficients extension failed')

    num_vars, num_subspace_indices = subspace_indices.shape
    for ii in range(num_subspace_indices):
        diff = new_index-subspace_indices[:, ii]
        if np.all(diff >= 0) and diff.max() <= 1:
            new_smolyak_coeffs[ii] += (-1.)**diff.sum()
    return new_smolyak_coeffs


def add_unique_poly_indices(poly_indices_dict, new_poly_indices):
    unique_poly_indices = []
    num_unique_poly_indices = len(poly_indices_dict)
    array_indices = np.empty((new_poly_indices.shape[1]), dtype=int)
    for jj in range(new_poly_indices.shape[1]):
        poly_index = new_poly_indices[:, jj]
        key = hash_array(poly_index)
        if key not in poly_indices_dict:
            unique_poly_indices.append(poly_index)
            poly_indices_dict[key] = num_unique_poly_indices
            array_indices[jj] = num_unique_poly_indices
            num_unique_poly_indices += 1
        else:
            array_indices[jj] = poly_indices_dict[key]
    return poly_indices_dict, np.asarray(unique_poly_indices).T, array_indices


def subspace_index_is_admissible(subspace_index, subspace_indices_dict):
    if hash_array(subspace_index) in subspace_indices_dict:
        return False

    if subspace_index.sum() <= 1:
        return True

    num_vars = subspace_index.shape[0]
    for ii in range(num_vars):
        if subspace_index[ii] > 0:
            neighbor_index = get_backward_neighbor(subspace_index, ii)
            if hash_array(neighbor_index) not in subspace_indices_dict:
                return False
    return True


def max_level_admissibility_function(max_level, max_level_1d,
                                     max_num_sparse_grid_samples, error_tol,
                                     sparse_grid, subspace_index, verbose=0):

    if subspace_index.sum() > max_level:
        return False

    if error_tol is not None:
        if sparse_grid.error.sum() < error_tol*np.absolute(
                sparse_grid.values[0, 0]):
            if len(sparse_grid.active_subspace_queue.list) > 0:
                msg = 'Desired accuracy %1.2e obtained. Error: %1.2e' % (
                    error_tol*np.absolute(sparse_grid.values[0, 0]),
                    sparse_grid.error.sum())
                msg += "\nNo. active subspaces remaining %d" % len(
                    sparse_grid.active_subspace_queue.list)
                msg += f'\nNo. samples {sparse_grid.samples.shape[1]}'
                if verbose > 0:
                    print(msg)
            else:
                print(subspace_index,  sparse_grid.error.sum())
                msg = 'Accuracy misleadingly appears reached because '
                msg += 'admissibility  criterion is preventing new subspaces '
                msg += 'from being added to the active set'
                print(msg)
            return False

    if max_level_1d is not None:
        for dd in range(subspace_index.shape[0]):
            if subspace_index[dd] > max_level_1d[dd]:
                if verbose > 0:
                    msg = f'Cannot add subspace {subspace_index}\n'
                    msg += f'Max level of {max_level_1d[dd]} reached in '
                    msg += f'variable {dd}'
                    print(msg)
                return False

    if (max_num_sparse_grid_samples is not None and
        (sparse_grid.num_equivalent_function_evaluations >
         max_num_sparse_grid_samples)):
        if verbose > 0:
            print(
                f'Max num evaluations ({max_num_sparse_grid_samples}) reached')
            print(f'Error estimate {sparse_grid.error.sum()}')
        return False

    if verbose > 1:
        msg = f'Subspace {subspace_index} is admissible'
        print(msg)
    return True


def default_combination_sparse_grid_cost_function(x):
    return np.ones(x.shape[1])


def get_active_subspace_indices(active_subspace_indices_dict,
                                sparse_grid_subspace_indices):
    II = []
    for key in active_subspace_indices_dict:
        II.append(active_subspace_indices_dict[key])
    return sparse_grid_subspace_indices[:, II], II


def partition_sparse_grid_samples(sparse_grid):
    num_vars = sparse_grid.samples.shape[0]

    active_subspace_indices, active_subspace_idx = get_active_subspace_indices(
        sparse_grid.active_subspace_indices_dict,
        sparse_grid.subspace_indices)

    sparse_grid_subspace_idx = np.ones(
        (sparse_grid.subspace_indices.shape[1]), dtype=bool)
    sparse_grid_subspace_idx[active_subspace_idx] = False

    samples = np.empty((num_vars, 0), dtype=float)
    samples_dict = dict()
    kk = 0
    for ii in np.arange(
            sparse_grid_subspace_idx.shape[0])[sparse_grid_subspace_idx]:
        subspace_poly_indices = \
            sparse_grid.subspace_poly_indices_list[ii]
        subspace_samples = get_sparse_grid_samples(
            subspace_poly_indices,
            sparse_grid.samples_1d,
            sparse_grid.config_variables_idx)
        for jj in range(subspace_samples.shape[1]):
            key = hash_array(subspace_samples[:, jj])
            if key not in samples_dict:
                samples_dict[key] = kk
                samples = np.hstack((samples, subspace_samples[:, jj:jj+1]))
                kk += 1

    num_active_samples = sparse_grid.samples.shape[1] -\
        samples.shape[1]
    active_samples_idx = np.empty((num_active_samples), dtype=int)
    kk = 0
    for ii in range(sparse_grid.samples.shape[1]):
        sample = sparse_grid.samples[:, ii]
        key = hash_array(sample)
        if key not in samples_dict:
            active_samples_idx[kk] = ii
            kk += 1
    assert kk == num_active_samples

    active_samples = sparse_grid.samples[:, active_samples_idx]
    return samples, active_samples


def plot_adaptive_sparse_grid_2d(sparse_grid, plot_grid=True, axs=None,
                                 samples_marker=('k', 'o', None),
                                 active_samples_marker=('r', 'o', None)):
    active_subspace_indices, active_subspace_idx = get_active_subspace_indices(
        sparse_grid.active_subspace_indices_dict,
        sparse_grid.subspace_indices)

    # get subspace indices that have been added to the sparse grid,
    # i.e are not active
    sparse_grid_subspace_idx = np.ones(
        (sparse_grid.subspace_indices.shape[1]), dtype=bool)
    sparse_grid_subspace_idx[active_subspace_idx] = False

    if plot_grid:
        if axs is None:
            f, axs = plt.subplots(1, 2, sharey=False, figsize=(16, 6))
    else:
        if axs is None:
            f, axs = plt.subplots(1, 1, sharey=False, figsize=(8, 6))
            axs = [axs]

    plot_2d_indices(
        sparse_grid.subspace_indices[:, sparse_grid_subspace_idx],
        coeffs=sparse_grid.smolyak_coefficients[sparse_grid_subspace_idx],
        other_indices=active_subspace_indices, ax=axs[0])

    if sparse_grid.config_variables_idx is not None:
        axs[0].set_xlabel(r'$\beta_1$', rotation=0)
        axs[0].set_ylabel(r'$\alpha_1$')  # ,rotation=0)

    if plot_grid:
        samples, active_samples = partition_sparse_grid_samples(sparse_grid)
        samples = sparse_grid.var_trans.map_from_canonical(
            samples)
        active_samples = \
            sparse_grid.var_trans.map_from_canonical(
                active_samples)
        if sparse_grid.config_variables_idx is None:
            axs[1].plot(samples[0, :], samples[1, :], samples_marker[1],
                        color=samples_marker[0], ms=samples_marker[2])
            axs[1].plot(active_samples[0, :], active_samples[1, :],
                        active_samples_marker[1],
                        color=active_samples_marker[0],
                        ms=active_samples_marker[2])
        else:
            for ii in range(samples.shape[1]):
                axs[1].plot(samples[0, ii], samples[1, ii],  samples_marker[1],
                            color=samples_marker[0], ms=samples_marker[2])
            for ii in range(active_samples.shape[1]):
                axs[1].plot(active_samples[0, ii], active_samples[1, ii],
                            active_samples_marker[1],
                            color=active_samples_marker[0],
                            ms=active_samples_marker[2])
            ya = axs[1].get_yaxis()
            ya.set_major_locator(MaxNLocator(integer=True))
            # axs[1].set_ylabel(r'$\alpha_1$',rotation=0)
            axs[1].set_xlabel('$z_1$', rotation=0,)


def isotropic_refinement_indicator(subspace_index,
                                   num_new_subspace_samples,
                                   sparse_grid):
    return float(subspace_index.sum()), np.inf


def tensor_product_refinement_indicator(
        subspace_index, num_new_subspace_samples, sparse_grid):
    return float(subspace_index.max()), np.inf


def variance_refinement_indicator_old(subspace_index, num_new_subspace_samples,
                                      sparse_grid, normalize=True, mean_only=False):
    """
    when config index is increased but the other indices are 0 the
    subspace will only have one random sample. Thus the variance
    contribution will be zero regardless of value of the function value
    at that sample
    """

    # return subspace_index.sum()
    moments = sparse_grid.moments()
    smolyak_coeffs = update_smolyak_coefficients(
        subspace_index, sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())
    new_moments = integrate_sparse_grid(
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        smolyak_coeffs, sparse_grid.weights_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    if mean_only:
        error = np.absolute(new_moments[0]-moments[0])
    else:
        error = np.absolute(new_moments[0]-moments[0])**2 +\
            np.absolute(new_moments[1]-moments[1])

    indicator = error.copy()

    # relative error will not work if value at first grid point is close to
    # zero
    if normalize:
        assert np.all(np.absolute(sparse_grid.values[0, :]) > 1e-6)
        indicator /= np.absolute(sparse_grid.values[0, :])**2

    qoi_chosen = np.argmax(indicator)
    # print (qoi_chosen)

    indicator = indicator.max()

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:, np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    # compute marginal benefit
    indicator /= cost

    return -indicator, error[qoi_chosen]


def variance_refinement_indicator(subspace_index, num_new_subspace_samples,
                                  sparse_grid, normalize=True,
                                  mean_only=False, convex_param=1):
    """
    when config index is increased but the other indices are 0 the
    subspace will only have one random sample. Thus the variance
    contribution will be zero regardless of value of the function value
    at that sample
    """

    # return subspace_index.sum()
    moments = sparse_grid.moments()
    smolyak_coeffs = update_smolyak_coefficients(
        subspace_index, sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())

    new_moments = sparse_grid.moments_(smolyak_coeffs)

    if mean_only:
        error = np.absolute(new_moments[0]-moments[0])
    else:
        # old version from misc paper
        # error = np.absolute(new_moments[0]-moments[0])**2 +\
        #    np.absolute(new_moments[1]-moments[1])
        # new version from integrated surrogates paper
        error = np.absolute(new_moments[0]-moments[0]) +\
            np.sqrt(np.absolute(new_moments[1]-moments[1]))

    indicator = error.copy()

    if normalize:
        # relative error will not work if value at first grid point is
        # close to zero
        # assert np.all(np.absolute(sparse_grid.values[0,:])>1e-6)
        denom = np.absolute(sparse_grid.values[0, :])
        II = np.where(denom < 1e-6)[0]
        denom[II] = 1
        #old version from misc paper
        #indicator /= denom**2
        # new version from integrated surrogates paper
        indicator /= denom

    qoi_chosen = np.argmax(indicator)
    #print (qoi_chosen)

    indicator = indicator.max()

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:, np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    # compute marginal benefit
    indicator /= cost

    # always keep in list
    indicator = convex_param*indicator+(1-convex_param)/subspace_index.sum()

    return -indicator, error[qoi_chosen]


def cv_refinement_indicator(validation_samples, validation_values,
                            subspace_index, num_new_subspace_samples,
                            sparse_grid):
    smolyak_coefficients = update_smolyak_coefficients(
        subspace_index, sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())
    approx_values = evaluate_sparse_grid(
        validation_samples[:sparse_grid.config_variables_idx, :],
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        smolyak_coefficients, sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:, np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    error = np.linalg.norm(
        approx_values-validation_values)/np.std(validation_values)
    current_approx_values = sparse_grid(validation_samples)
    current_error = np.linalg.norm(
        current_approx_values-validation_values)/np.std(validation_values)
    error = abs(error-current_error)
    indicator = error.max()/cost
    return -indicator, error


def compute_surpluses(subspace_index, sparse_grid, hierarchical=False):
    key = hash_array(subspace_index)
    ii = sparse_grid.active_subspace_indices_dict[key]

    subspace_samples = get_subspace_samples(
        subspace_index,
        sparse_grid.subspace_poly_indices_list[ii],
        sparse_grid.samples_1d, sparse_grid.config_variables_idx,
        unique_samples_only=False)

    if hierarchical:
        hier_indices = get_hierarchical_sample_indices(
            subspace_index,
            sparse_grid.subspace_poly_indices_list[ii],
            sparse_grid.samples_1d, sparse_grid.config_variables_idx)
        subspace_samples = subspace_samples[:, hier_indices]
    else:
        hier_indices = None

    current_approx_values = evaluate_sparse_grid(
        subspace_samples,
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        sparse_grid.smolyak_coefficients,
        sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    smolyak_coefficients = update_smolyak_coefficients(
        subspace_index, sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())

    new_approx_values = evaluate_sparse_grid(
        subspace_samples,
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        smolyak_coefficients,
        sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    return new_approx_values-current_approx_values, hier_indices


def compute_hierarchical_surpluses_direct(subspace_index, sparse_grid):

    # only works if not used in multilevel setting
    assert sparse_grid.config_variables_idx is None
    key = hash_array(subspace_index)
    ii = sparse_grid.active_subspace_indices_dict[key]

    subspace_samples = get_subspace_samples(
        subspace_index,
        sparse_grid.subspace_poly_indices_list[ii],
        sparse_grid.samples_1d, sparse_grid.config_variables_idx,
        unique_samples_only=False)

    hier_indices = get_hierarchical_sample_indices(
        subspace_index, sparse_grid.subspace_poly_indices_list[ii],
        sparse_grid.samples_1d, sparse_grid.config_variables_idx)
    # hier_indices = np.arange(subspace_samples.shape[1])

    subspace_samples = subspace_samples[:, hier_indices]

    current_approx_values = evaluate_sparse_grid(
        subspace_samples,
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        sparse_grid.smolyak_coefficients,
        sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    subspace_values = get_subspace_values(
        sparse_grid.values,
        sparse_grid.subspace_values_indices_list[ii])
    subspace_values = subspace_values[hier_indices, :]

    surpluses = subspace_values-current_approx_values
    return surpluses


def surplus_refinement_indicator(subspace_index, num_new_subspace_samples,
                                 sparse_grid, output=False, hierarchical=False,
                                 norm_order=np.inf, normalize=True):

    surpluses, hier_indices = compute_surpluses(
        subspace_index, sparse_grid, hierarchical=hierarchical)

    subspace_weights = get_subspace_weights(
        subspace_index, sparse_grid.weights_1d,
        sparse_grid.config_variables_idx)
    if hier_indices is not None:
        subspace_weights = subspace_weights[hier_indices]

    if norm_order == np.inf:
        error = np.max(np.abs(surpluses), axis=0)
    elif norm_order == 1:
        error = np.abs(np.dot(subspace_weights, surpluses))
    else:
        raise Exception("ensure norm_order in [np.inf,1]")

    assert error.shape[0] == surpluses.shape[1]

    # relative error will not work if value at first grid point is close to
    # zero
    if normalize:
        assert np.all(np.absolute(sparse_grid.values[0, :]) > 1e-6)
        error /= np.absolute(sparse_grid.values[0, :])

    error = np.max(error)

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:, np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    indicator = error/cost

    return -indicator, error


def extract_sparse_grid_quadrature_rule(asg):
    """
    Returns samples in canonical space
    """
    num_sparse_grid_points = (
        asg.poly_indices.shape[1])
    # must initialize to zero
    weights = np.zeros((num_sparse_grid_points), dtype=float)
    samples = get_sparse_grid_samples(
        asg.poly_indices, asg.samples_1d)
    for ii in range(asg.subspace_indices.shape[1]):
        if (abs(asg.smolyak_coefficients[ii]) > np.finfo(float).eps):
            subspace_index = asg.subspace_indices[:, ii]
            subspace_poly_indices = asg.subspace_poly_indices_list[ii]
            subspace_weights = get_subspace_weights(
                subspace_index, asg.weights_1d)*asg.smolyak_coefficients[ii]
            for jj in range(subspace_poly_indices.shape[1]):
                poly_index = subspace_poly_indices[:, jj]
                key = hash_array(poly_index)
                if key in asg.poly_indices_dict:
                    weights[asg.poly_indices_dict[key]] += subspace_weights[jj]
                else:
                    raise Exception('index not found')
    return samples, weights


class SubSpaceRefinementManager(object):
    def __init__(self, num_vars):
        self.verbose = 0
        self.num_vars = num_vars
        self.num_config_vars = 0
        self.subspace_indices_dict = dict()
        self.subspace_indices = np.zeros((self.num_vars, 0), dtype=int)
        self.active_subspace_indices_dict = dict()
        self.active_subspace_queue = mypriorityqueue()
        self.admissibility_function = None
        self.refinement_indicator = None
        self.univariate_growth_rule = None
        self.subspace_poly_indices_list = []
        self.poly_indices = np.zeros((self.num_vars, 0), dtype=int)
        self.subspace_values_indices_list = []
        self.config_variables_idx = None
        self.samples = None
        self.values = None
        self.num_equivalent_function_evaluations = 0
        self.function = None
        self.var_trans = None
        self.work_qoi_index = None
        self.config_var_trans = None
        self.unique_quadrule_indices = None
        self.compact_univariate_growth_rule = None
        self.unique_poly_indices_idx = np.zeros((0), dtype=int)
        self.enforce_variable_ordering = False

    def initialize(self):
        self.poly_indices_dict = dict()
        self.samples = np.zeros((self.num_vars, 0))
        self.add_new_subspaces(np.zeros((self.num_vars, 1), dtype=int))
        self.error = np.zeros((0))
        # self.prioritize_active_subspaces(
        #    self.subspace_indices, np.asarray([self.samples.shape[1]]))
        # self.active_subspace_queue.list[0] = (np.inf,self.error[0],0)
        self.error = np.concatenate([self.error, [np.inf]])
        self.active_subspace_queue.put((-np.inf, self.error[0], 0))

    def refine(self):
        if self.subspace_indices.shape[1] == 0:
            self.initialize()
        priority, error, best_subspace_idx = self.active_subspace_queue.get()
        best_active_subspace_index = self.subspace_indices[
            :, best_subspace_idx]
        if self.verbose > 1:
            msg = f'refining index {best_active_subspace_index} '
            msg += f'with priority {priority}\n'
            msg += 'The current number of equivalent function evaluations is '
            msg += f'{self.num_equivalent_function_evaluations}'
            print(msg)

        new_active_subspace_indices, num_new_subspace_samples = \
            self.refine_and_add_new_subspaces(best_active_subspace_index)

        self.prioritize_active_subspaces(
            new_active_subspace_indices, num_new_subspace_samples)

        self.error[best_subspace_idx] = 0.0

    def postpone_subspace_refinement(self, new_active_subspace_indices):
        """
        used to enforce variable ordering
        """
        if not hasattr(self, 'postponed_subspace_indices'):
            self.postponed_subspace_indices = dict()

        reactivated_subspace_indices_set = set()
        reactivated_subspace_indices = []
        activated_keys = []

        # get maximum level of subspace_indices in sparse grid
        # self.subspace_indices contains active indices as well so exclude
        # by only considering index front for which non-zero smolyak
        # coefficients is a proxy
        I = np.where(np.abs(self.smolyak_coefficients)
                     > np.finfo(float).eps)[0]
        if I.shape[0] > 0:
            max_level_1d = np.max(self.subspace_indices[:, I], axis=1)
        else:
            max_level_1d = np.zeros(new_active_subspace_indices.shape[0])

        for key, new_subspace_index in self.postponed_subspace_indices.items():
            use = True
            for dd in range(1, new_active_subspace_indices.shape[0]):
                if new_subspace_index[dd] > 0:
                    # if conditions correspond respectively to decreasing
                    # conservatism. On problem I have tested there seems to
                    # be no difference in final answer
                    # temp_index = new_subspace_index.copy()
                    # temp_index[dd-1]=new_subspace_index[dd]
                    # temp_index[dd]-=1
                    # temp_key = hash_array(temp_index)
                    # if temp_key not in self.subspace_indices_dict:
                    if np.any(max_level_1d[:dd] < new_subspace_index[dd]):
                        # if np.any(max_level_1d[:dd]==0):
                        use = False
                        break
            if use and self.admissibility_function(self, new_subspace_index):
                reactivated_subspace_indices_set.add(key)
                reactivated_subspace_indices.append(new_subspace_index)
                activated_keys.append(key)

        for key in activated_keys:
            del self.postponed_subspace_indices[key]

        use_idx = []
        for jj in range(new_active_subspace_indices.shape[1]):
            use = True
            new_subspace_index = new_active_subspace_indices[:, jj]
            for dd in range(1, new_active_subspace_indices.shape[0]):
                if new_subspace_index[dd] > 0:
                    # temp_index = new_subspace_index.copy()
                    # temp_index[dd-1]=new_subspace_index[dd]
                    # temp_index[dd]-=1
                    # temp_key = hash_array(temp_index)
                    # if temp_key not in self.subspace_indices_dict:
                    if np.any(max_level_1d[:dd] < new_subspace_index[dd]):
                        # if np.any(max_level_1d[:dd]==0):
                        self.postponed_subspace_indices[
                            hash_array(new_subspace_index)] = \
                                new_subspace_index
                        use = False
                        break
            if use:
                key = hash_array(new_subspace_index)
                if key not in reactivated_subspace_indices_set:
                    use_idx.append(jj)
                if key in self.postponed_subspace_indices:
                    del self.postponed_subspace_indices[key]

        new_active_subspace_indices = new_active_subspace_indices[:, use_idx]
        if len(reactivated_subspace_indices) > 0:
            new_active_subspace_indices = np.hstack([
                new_active_subspace_indices,
                np.asarray(reactivated_subspace_indices).T])

        return new_active_subspace_indices

    def refine_subspace(self, subspace_index):
        new_active_subspace_indices = np.zeros((self.num_vars, 0), dtype=int)
        for ii in range(self.num_vars):
            neighbor_index = get_forward_neighbor(subspace_index, ii)
            if (subspace_index_is_admissible(
                neighbor_index, self.subspace_indices_dict) and
                    hash_array(neighbor_index) not in
                    self.active_subspace_indices_dict and
                    self.admissibility_function(self, neighbor_index)):
                new_active_subspace_indices = np.hstack(
                    (new_active_subspace_indices,
                     neighbor_index[:, np.newaxis]))
            else:
                if self.verbose > 2:
                    msg = f'Subspace {neighbor_index} is not admissible'
                    print(msg)

        if self.enforce_variable_ordering:
            new_active_subspace_indices = self.postpone_subspace_refinement(
                new_active_subspace_indices)

        return new_active_subspace_indices

    def build(self, callback=None):
        """
        """
        while (not self.active_subspace_queue.empty() or
               self.subspace_indices.shape[1] == 0):
            self.refine()
            if callback is not None:
                callback(self)

    def refine_and_add_new_subspaces(self, best_active_subspace_index):
        key = hash_array(best_active_subspace_index)
        self.subspace_indices_dict[key] =\
            self.active_subspace_indices_dict[key]

        # get all new active subspace indices
        new_active_subspace_indices = self.refine_subspace(
            best_active_subspace_index)
        del self.active_subspace_indices_dict[key]

        if new_active_subspace_indices.shape[1] > 0:
            num_new_subspace_samples = self.add_new_subspaces(
                new_active_subspace_indices)
        else:
            num_new_subspace_samples = 0
        return new_active_subspace_indices, num_new_subspace_samples

    def get_subspace_samples(self, subspace_index, unique_poly_indices):
        """
        Must be implemented by derived class
        This function should only be called when updating grid not interogating
        grid
        """
        msg = "get_subspace_samples must be implemented by derived class"
        raise Exception(msg)

    def initialize_subspace(self, subspace_index):
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, self.univariate_growth_rule,
            self.config_variables_idx)

        self.subspace_poly_indices_list.append(subspace_poly_indices)
        self.poly_indices_dict, unique_poly_indices, \
            subspace_values_indices = add_unique_poly_indices(
                self.poly_indices_dict, subspace_poly_indices)
        self.unique_poly_indices_idx = np.concatenate(
            [self.unique_poly_indices_idx, [self.poly_indices.shape[1]]])
        self.subspace_values_indices_list.append(subspace_values_indices)
        self.poly_indices = np.hstack((self.poly_indices, unique_poly_indices))
        return unique_poly_indices

    def initialize_subspaces(self, new_subspace_indices):
        num_vars, num_new_subspaces = new_subspace_indices.shape
        num_current_subspaces = self.subspace_indices.shape[1]
        cnt = num_current_subspaces
        for ii in range(num_new_subspaces):
            subspace_index = new_subspace_indices[:, ii]
            self.initialize_subspace(subspace_index)
            self.active_subspace_indices_dict[hash_array(subspace_index)] = cnt
            cnt += 1

    def create_new_subspaces_data(self, new_subspace_indices):
        num_current_subspaces = self.subspace_indices.shape[1]
        self.initialize_subspaces(new_subspace_indices)
        num_vars, num_new_subspaces = new_subspace_indices.shape
        new_samples = np.empty((num_vars, 0), dtype=float)
        # num_current_subspaces = self.subspace_indices.shape[1]
        # cnt = num_current_subspaces
        num_new_subspace_samples = np.empty((num_new_subspaces), dtype=int)
        for ii in range(num_new_subspaces):
            subspace_index = new_subspace_indices[:, ii]
            # unique_poly_indices = self.initialize_subspace(subspace_index)
            idx1 = self.unique_poly_indices_idx[num_current_subspaces+ii]
            if ii < num_new_subspaces-1:
                idx2 = self.unique_poly_indices_idx[num_current_subspaces+ii+1]
            else:
                idx2 = self.poly_indices.shape[1]
            unique_poly_indices = self.poly_indices[:, idx1:idx2]
            unique_subspace_samples = self.get_subspace_samples(
                subspace_index, unique_poly_indices)
            new_samples = np.hstack(
                (new_samples, unique_subspace_samples))
            num_new_subspace_samples[ii] = unique_subspace_samples.shape[1]
            # self.active_subspace_indices_dict[hash_array(subspace_index)]=cnt
            # cnt += 1
        return new_samples, num_new_subspace_samples

    def add_new_subspaces(self, new_subspace_indices):
        new_samples, num_new_subspace_samples = self.create_new_subspaces_data(
            new_subspace_indices)

        new_values = self.eval_function(new_samples)
        self.subspace_indices = np.hstack(
            (self.subspace_indices, new_subspace_indices))
        self.samples = np.hstack((self.samples, new_samples))

        if self.values is None:
            msg = "New values cannot have NaN!"
            assert np.any(np.isnan(new_values)) == False, msg
            self.values = new_values
        else:
            self.values = np.vstack((self.values, new_values))

        self.num_equivalent_function_evaluations += self.get_cost(
            new_subspace_indices, num_new_subspace_samples)

        return num_new_subspace_samples

    def prioritize_active_subspaces(self, new_active_subspace_indices,
                                    num_new_subspace_samples):
        cnt = self.subspace_indices.shape[1] -\
            new_active_subspace_indices.shape[1]
        for ii in range(new_active_subspace_indices.shape[1]):
            subspace_index = new_active_subspace_indices[:, ii]

            priority, error = self.refinement_indicator(
                subspace_index, num_new_subspace_samples[ii], self)
            self.active_subspace_queue.put((priority, error, cnt))
            self.error = np.concatenate([self.error, [error]])

            if self.verbose > 2:
                msg = f'adding new index {subspace_index} '
                msg += f'with priority {priority}'
                print(msg)
            cnt += 1

    def set_function(self, function, variable_transformation=None):
        self.function = function
        self.var_trans = variable_transformation

    def map_config_samples_from_canonical_space(self, samples):
        if self.config_variables_idx is None:
            config_variables_idx = self.num_vars
        else:
            config_variables_idx = self.config_variables_idx
        config_samples = samples[config_variables_idx:, :]
        if self.config_var_trans is not None:
            config_samples = self.config_var_trans.map_from_canonical(
                config_samples)
        return config_samples

    def map_random_samples_from_canonical_space(self, canonical_samples):
        if self.config_variables_idx is None:
            config_variables_idx = self.num_vars
        else:
            config_variables_idx = self.config_variables_idx
        random_samples = canonical_samples[:config_variables_idx, :]
        if self.var_trans is not None:
            random_samples = \
                self.var_trans.map_from_canonical(
                    random_samples)
        return random_samples

    def eval_function(self, canonical_samples):
        random_samples = self.map_random_samples_from_canonical_space(
            canonical_samples)
        config_samples = self.map_config_samples_from_canonical_space(
            canonical_samples)
        samples = np.vstack((random_samples, config_samples))

        values = self.function(samples)

        return values

    def set_univariate_growth_rules(self, univariate_growth_rule,
                                    unique_quadrule_indices):
        """
        self.config_variable_idx must be set if univariate_growth_rule is
        a callable function and not a lisf of callable functions. Otherwise
        errors such as assert len(growth_rule_1d)==config_variables_idx will
        be thrown

        TODO: eventually retire self.univariate_growth rule and just pass
        around compact_growth_rule. When doing this change from storing
        samples_1d for each dimension to only storing for unique quadrature
        rules
        """
        self.unique_quadrule_indices = unique_quadrule_indices

        if self.config_variables_idx is None:
            dd = self.num_vars
        else:
            dd = self.config_variables_idx
        if callable(univariate_growth_rule):
            self.compact_univariate_growth_rule = [univariate_growth_rule]
            self.unique_quadrule_indices = [np.arange(dd)]
        else:
            self.compact_univariate_growth_rule = univariate_growth_rule

        if self.unique_quadrule_indices is not None:
            cnt = 0
            for ii in self.unique_quadrule_indices:
                cnt += len(ii)
            if cnt != dd:
                msg = 'unique_quad_rule_indices is inconsistent with num_vars'
                raise Exception(msg)
            assert len(self.compact_univariate_growth_rule) == len(
                self.unique_quadrule_indices)
            self.univariate_growth_rule = [[] for dd in range(dd)]
            for ii in range(len(self.unique_quadrule_indices)):
                jj = self.unique_quadrule_indices[ii]
                for kk in jj:
                    self.univariate_growth_rule[kk] =\
                        self.compact_univariate_growth_rule[ii]
        else:
            if len(self.compact_univariate_growth_rule) != dd:
                msg = 'length of growth_rules is inconsitent with num_vars.'
                msg += 'Maybe you need to set unique_quadrule_indices'
                raise Exception(msg)
            self.univariate_growth_rule = self.compact_univariate_growth_rule

        assert len(self.univariate_growth_rule) == dd

    def set_refinement_functions(self, refinement_indicator,
                                 admissibility_function,
                                 univariate_growth_rule,
                                 cost_function=None, work_qoi_index=None,
                                 unique_quadrule_indices=None):
        """
        cost_function : callable (or object is work_qoi_index is not None)
            Return the cost of evaluating a function with a unique indentifier.
            Identifiers can be strings, integers, etc. The identifier
            is found by mapping the sparse grid canonical_config_samples which
            are consecutive integers 0,1,... using self.config_var_trans

        work_qoi_index : integer (default None)
            If provided self.function is assumed to return the work (typically
            measured in wall time) taken to evaluate each sample. The work
            for each sample return as a QoI in the column indexed by
            work_qoi_index. The work QoI is ignored by the sparse grid
            eval_function() member function. If work_qoi_index is provided
            cost_function() must be a class with a member function
            update(config_samples,costs). config_samples is a 2d array whose
            columns are unique identifiers of the model being evaluated and
            costs is the work needed to evaluate that model. If building single
            fidelity sparse grid then config vars is set to be (0,...,0) for
            each sample
        """
        self.refinement_indicator = refinement_indicator
        self.admissibility_function = admissibility_function
        self.set_univariate_growth_rules(
            univariate_growth_rule, unique_quadrule_indices)
        if cost_function is None:
            cost_function = default_combination_sparse_grid_cost_function
        self.cost_function = cost_function
        if work_qoi_index is not None:
            raise Exception('This option is deprecated and will be removed')

    def set_config_variable_index(self, idx, config_var_trans=None):
        if self.function is None:
            msg = 'Must call set_function before entry'
            raise Exception(msg)
        self.config_variables_idx = idx
        self.config_var_trans = config_var_trans
        self.num_config_vars = self.num_vars-self.config_variables_idx
        if self.var_trans is not None:
            assert (self.var_trans.num_vars() ==
                    self.config_variables_idx)
        if self.config_var_trans is not None:
            assert self.num_config_vars == self.config_var_trans.num_vars()

    def eval_cost_function(self, samples):
        config_samples = self.map_config_samples_from_canonical_space(
            samples)
        if self.config_variables_idx is None:
            # single fidelity so make up dummy unique key for work tracker
            config_samples = np.zeros((1, samples.shape[1]), dtype=int)
        costs = self.cost_function(config_samples)
        return costs

    def get_cost(self, subspace_indices, num_new_subspace_samples):
        assert subspace_indices.shape[1] == num_new_subspace_samples.shape[0]
        # the cost of a single evaluate of each function
        function_costs = self.eval_cost_function(subspace_indices)
        assert function_costs.ndim == 1
        # the cost of evaluating the unique points of each subspace
        subspace_costs = function_costs*num_new_subspace_samples
        # the cost of evaluating the unique points of all subspaces in
        # subspace_indices
        cost = np.sum(subspace_costs)
        return cost

    def __neq__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        """
        This function will compare all attributes of the derived class and this
        base class.
        """
        member_names = [
            m[0] for m in vars(self).items() if not m[0].startswith("__")]
        for m in member_names:
            attr = getattr(other, m)
            # print(m)
            # print(type(attr))
            # print(attr)
            if type(attr) == partial:
                if not partial_functions_equal(attr, getattr(self, m)):
                    return False
            elif (type(attr) == list and len(attr) == 0):
                assert len(getattr(self, m)) == 0
            elif (type(attr) == list and type(attr[0]) == np.ndarray):
                if not lists_of_arrays_equal(attr, getattr(self, m)):
                    return False
            elif type(attr) == list and type(attr[0]) == list and type(
                    attr[0][0]) == np.ndarray:
                if not lists_of_lists_of_arrays_equal(attr, getattr(self, m)):
                    return False
            elif np.any(getattr(other, m) != getattr(self, m)):
                return False
        return True

    def recompute_active_subspace_priorities(self):
        if self.active_subspace_queue.empty():
            return
        items = extract_items_from_priority_queue(
            self.active_subspace_queue)[0]
        self.active_subspace_queue = mypriorityqueue()
        for item in items:
            count = item[2]  # index of grid.subspace_indices
            # find num_samples for subspace
            subspace_index = self.subspace_indices[:, count]
            idx1 = self.unique_poly_indices_idx[count]
            if count < self.unique_poly_indices_idx.shape[0]-1:
                idx2 = self.unique_poly_indices_idx[count+1]
            else:
                idx2 = self.poly_indices.shape[1]
            num_new_subspace_samples = self.poly_indices[:, idx1:idx2].shape[1]
            # compute priority and error for subspace
            priority, error = self.refinement_indicator(
                subspace_index, num_new_subspace_samples, self)
            new_item = (priority, error, count)
            self.active_subspace_queue.put(new_item)
            self.error[count] = error

    def get_total_work(self):
        return self.num_equivalent_function_evaluations


def get_unique_quadrule_variables(var_trans):
    """
    This function will create a quad rule for each variable type with different
    scaling. This can cause redundant computation of quad rules which
    may be significant when using leja sequences
    """
    unique_quadrule_variables = [var_trans.variable.unique_variables[0]]
    unique_quadrule_indices = [
        var_trans.variable.unique_variable_indices[0].copy()]
    for ii in range(1, var_trans.variable.nunique_vars):
        var = var_trans.variable.unique_variables[ii]
        var_indices = var_trans.variable.unique_variable_indices[ii].copy()
        found = False
        for jj in range(len(unique_quadrule_variables)):
            if variable_shapes_equivalent(var, unique_quadrule_variables[jj]):
                unique_quadrule_indices[jj] = np.concatenate(
                    [unique_quadrule_indices[jj], var_indices])
                found = True
                break
        if not found:
            unique_quadrule_variables.append(var)
            unique_quadrule_indices.append(var_indices)

    return unique_quadrule_variables, unique_quadrule_indices


def get_unique_max_level_1d(var_trans, growth_rules):
    unique_quadrule_variables, unique_quadrule_indices = \
        get_unique_quadrule_variables(var_trans)
    # print(len(growth_rules), unique_quadrule_indices)
    if len(growth_rules) != len(unique_quadrule_indices):
        msg = 'growth rules and unique_quadrule_indices'
        msg += ' (derived from var_trans) are inconsistent'
        raise Exception(msg)

    max_level_1d = []
    for ii in range(len(unique_quadrule_indices)):
        if is_bounded_discrete_variable(unique_quadrule_variables[ii]):
            max_nsamples_ii = get_probability_masses(
                unique_quadrule_variables[ii])[0].shape[0]
            ll = 0
            while True:
                if growth_rules[ii](ll) > max_nsamples_ii-1:
                    max_level_1d_ii = ll-1
                    break
                ll += 1
        else:
            max_level_1d_ii = np.inf

        max_level_1d.append(max_level_1d_ii)
    return np.asarray(max_level_1d)


def get_sparse_grid_univariate_leja_quadrature_rules_economical(
        var_trans, growth_rules=None, method='pdf', growth_incr=2):
    """
    Return a list of unique quadrature rules. If each dimension has the same
    rule then list will only have one entry.
    """
    assert var_trans is not None

    if growth_rules is None:
        if growth_incr == 'clenshaw':
            growth_rules = clenshaw_curtis_rule_growth
        else:
            growth_rules = partial(constant_increment_growth_rule, growth_incr)

    unique_quadrule_variables, unique_quadrule_indices = \
        get_unique_quadrule_variables(var_trans)

    if callable(growth_rules):
        growth_rules = [growth_rules]*len(unique_quadrule_indices)

    if len(growth_rules) != len(unique_quadrule_indices):
        msg = 'growth rules and unique_quadrule_indices'
        msg += ' (derived from var_trans) are inconsistent'
        raise Exception(msg)

    quad_rules = []
    for ii in range(len(unique_quadrule_indices)):
        quad_rule = get_univariate_leja_quadrature_rule(
            unique_quadrule_variables[ii], growth_rules[ii], method)
        quad_rules.append(quad_rule)

    max_level_1d = get_unique_max_level_1d(var_trans, growth_rules)

    return quad_rules, growth_rules, unique_quadrule_indices, max_level_1d


def get_sparse_grid_univariate_leja_quadrature_rules(
        var_trans, growth_rules=None):
    """
    Return a list of quadrature rules for every variable
    """
    (unique_quad_rules, unique_growth_rules, unique_quadrule_indices,
     unique_max_level_1d) = (
         get_sparse_grid_univariate_leja_quadrature_rules_economical(
             var_trans, growth_rules=None))
    quad_rules = [None for ii in var_trans.num_vars()]
    growth_rules = [None for ii in var_trans.num_vars()]
    max_level_1d = [None for ii in var_trans.num_vars()]
    for quad_rule, growth_rule, indices, max_level in zip(
            unique_quad_rules, unique_growth_rules, unique_quadrule_indices,
            unique_max_level_1d):
        quad_rules[indices] = quad_rule
        growth_rules[indices] = growth_rule
        max_level_1d[indices] = max_level
    return quad_rules, growth_rules, max_level_1d


class CombinationSparseGrid(SubSpaceRefinementManager):
    """
    Adaptive sparse grid that uses the combination technique.
    """

    def __init__(self, num_vars, basis_type="barycentric"):
        """
        num_vars : integer
            The number of variables

        basis_type : string (default="barycentric")
           Specify the basis type to use. Currently the same basis must be used
           for all dimensions. Options "barycentric", "linear", "quadratic"
        """
        super(CombinationSparseGrid, self).__init__(num_vars)

        # to allow for mixed barycentric and piecwise poly basis
        # if type(basis_type) == str:
        #    basis_type = [basis_type]*num_vars

        self.basis_type = basis_type
        self.univariate_quad_rule = None
        self.samples_1d, self.weights_1d = [None, None]
        self.smolyak_coefficients = np.empty((0), np.float64)
        self.var_trans = None
        self.compact_univariate_quad_rule = None

        # extra storage to reduce cost of repeated interrogation
        self.subspace_moments = None
        self.subspace_interrogation_values = []
        self.canonical_interrogation_samples = None

    def setup(self, function, config_variables_idx, refinement_indicator,
              admissibility_function, univariate_growth_rule,
              univariate_quad_rule,
              variable_transformation=None, config_var_trans=None,
              cost_function=None, work_qoi_index=None,
              unique_quadrule_indices=None,
              verbose=0):
        self.set_function(function, variable_transformation)
        if config_variables_idx is not None:
            self.set_config_variable_index(
                config_variables_idx, config_var_trans)
        self.set_refinement_functions(
            refinement_indicator, admissibility_function,
            univariate_growth_rule, cost_function, work_qoi_index,
            unique_quadrule_indices)
        self.set_univariate_rules(univariate_quad_rule)
        self.verbose = verbose

    def set_univariate_rules(self, univariate_quad_rule, max_level=2):
        if self.univariate_growth_rule is None:
            msg = "Must call set_refinement_functions before set_univariate "
            msg += "rules"
            raise Exception(msg)
        self.univariate_quad_rule = univariate_quad_rule

        if self.config_variables_idx is None:
            dd = self.num_vars
        else:
            dd = self.config_variables_idx

        num_random_vars = 0
        for ii in range(len(self.unique_quadrule_indices)):
            num_random_vars += len(self.unique_quadrule_indices[ii])
        if num_random_vars != dd:
            msg = 'unique_quadrule_indices is inconsistent with '
            msg += 'self.config_variables_idx. If using config_variables try'
            msg += 'calling the following functions in this order'
            msg += """
                   set_function()
                   set_config_variable_index()
                   set_refinement_functions()
                   set_univariate_rules()
                   """
            raise Exception(msg)

        if callable(univariate_quad_rule):
            self.compact_univariate_quad_rule = [self.univariate_quad_rule]
        else:
            self.compact_univariate_quad_rule = univariate_quad_rule

        if self.unique_quadrule_indices is None:
            self.univariate_quad_rule = self.compact_univariate_quad_rule
        else:
            assert len(self.compact_univariate_quad_rule) == len(
                self.unique_quadrule_indices)
            self.univariate_quad_rule = [[] for dd in range(dd)]
            for ii in range(len(self.unique_quadrule_indices)):
                jj = self.unique_quadrule_indices[ii]
                for kk in jj:
                    self.univariate_quad_rule[kk] = \
                        self.compact_univariate_quad_rule[ii]

        assert len(self.univariate_quad_rule) == dd

        self.samples_1d, self.weights_1d = get_1d_samples_weights(
            self.compact_univariate_quad_rule,
            self.compact_univariate_growth_rule,
            [max_level]*dd, self.config_variables_idx,
            self.unique_quadrule_indices)

    def refine_and_add_new_subspaces(self, best_active_subspace_index):
        new_active_subspace_indices, num_new_subspace_samples = super(
            CombinationSparseGrid, self).refine_and_add_new_subspaces(
            best_active_subspace_index)
        self.smolyak_coefficients = update_smolyak_coefficients(
            best_active_subspace_index, self.subspace_indices,
            self.smolyak_coefficients)
        return new_active_subspace_indices, num_new_subspace_samples

    def get_subspace_samples(self, subspace_index, unique_poly_indices):
        samples_1d, weights_1d = update_1d_samples_weights(
            self.compact_univariate_quad_rule,
            self.compact_univariate_growth_rule,
            subspace_index, self.samples_1d, self.weights_1d,
            self.config_variables_idx, self.unique_quadrule_indices)

        self.smolyak_coefficients = np.hstack(
            (self.smolyak_coefficients, np.zeros(1)))

        return get_sparse_grid_samples(
            unique_poly_indices, self.samples_1d, self.config_variables_idx)

    def __call__(self, samples, return_grad=False):
        """
        config values are ignored. The sparse grid just returns its best
        approximation of the highest fidelity model. TODO: consider enforcing
        that samples do not have configure variables
        """
        if self.var_trans is not None:
            canonical_samples = \
                self.var_trans.map_to_canonical(
                    samples[:self.config_variables_idx, :])
        else:
            canonical_samples = samples[:self.config_variables_idx, :]

        result = evaluate_sparse_grid(
            canonical_samples[:self.config_variables_idx, :],
            self.values,
            self.poly_indices_dict, self.subspace_indices,
            self.subspace_poly_indices_list,
            self.smolyak_coefficients, self.samples_1d,
            self.subspace_values_indices_list,
            self.config_variables_idx, return_grad=return_grad,
            basis_type=self.basis_type)
        if not return_grad:
            return result
        vals, jacs = result
        if samples.shape[1] == 1:
            jacs = jacs[0]
        return vals, jacs

    def moments_(self, smolyak_coefficients):
        return integrate_sparse_grid_from_subspace_moments(
            self.subspace_indices, smolyak_coefficients,
            self.subspace_moments)

        # return integrate_sparse_grid(
        #     self.values,
        #     self.poly_indices_dict,self.subspace_indices,
        #     self.subspace_poly_indices_list,
        #     smolyak_coefficients,self.weights_1d,
        #     self.subspace_values_indices_list,
        #     self.config_variables_idx)

    def moments(self):
        return self.moments_(self.smolyak_coefficients)

    def set_interrogation_samples(self, samples):
        """
        Set samples which are used to evaluate a sparse grid repeatedly.
        If provided each time a subspace is added the subspace is evaluated
        at these points so that when self.evaluate_at_interrogation_samples
        is called no major computations are required.
        Note the reduced time complexity requires more storage

        Parameters
        ----------
        samples : np.ndarray (num_vars) or (num_vars-num_config_vars)
             Samples at which to evaluate the sparae grid. If config values
             are provided they are ignored.
        """
        if self.var_trans is not None:
            canonical_samples = \
                self.var_trans.map_to_canonical(
                    samples[:self.config_variables_idx, :])
        else:
            canonical_samples = samples[:self.config_variables_idx, :]
        self.canonical_interrogation_samples = canonical_samples

    def evaluate_at_interrogation_samples(self):
        """
        Evaluate the sparse grid at self.canonical_interrogation_samples.

        Note, this fuction only uses subspaces which are not active
        """
        return evaluate_sparse_grid_from_subspace_values(
            self.subspace_indices, self.smolyak_coefficients,
            self.subspace_interrogation_values)

    def evaluate_using_all_data(self, samples):
        """
        Evaluate sparse grid using all subspace indices including
        active subspaces. __call__ only uses subspaces which are not active
        """
        # extract active subspaces from queue without destroying queue
        pairs, self.active_subspace_queue = \
            extract_items_from_priority_queue(self.active_subspace_queue)
        # copy smolyak coefficients so as not affect future refinement
        smolyak_coefficients = self.smolyak_coefficients.copy()
        # add all active subspaces to sparse grid by updating smolyak
        # coefficients
        for ii in range(len(pairs)):
            subspace_index = self.subspace_indices[:, pairs[ii][-1]]
            smolyak_coefficients = update_smolyak_coefficients(
                subspace_index, self.subspace_indices,
                smolyak_coefficients)

        if self.var_trans is not None:
            canonical_samples = \
                self.var_trans.map_to_canonical(
                    samples[:self.config_variables_idx, :])
        else:
            canonical_samples = samples[:self.config_variables_idx, :]

        # evaluate sparse grid includding active subspaces
        approx_values = evaluate_sparse_grid(
            canonical_samples,
            self.values, self.poly_indices_dict,
            self.subspace_indices,
            self.subspace_poly_indices_list,
            smolyak_coefficients, self.samples_1d,
            self.subspace_values_indices_list,
            self.config_variables_idx)
        return approx_values

    def add_new_subspaces(self, new_subspace_indices):
        num_new_subspaces = new_subspace_indices.shape[1]
        num_current_subspaces = self.subspace_indices.shape[1]
        num_new_subspace_samples = super(
            CombinationSparseGrid, self).add_new_subspaces(
                new_subspace_indices)

        cnt = num_current_subspaces
        new_subspace_moments = np.empty(
            (num_new_subspaces, self.values.shape[1], 2), dtype=float)
        for ii in range(num_new_subspaces):
            subspace_index = new_subspace_indices[:, ii]
            subspace_values = get_subspace_values(
                self.values, self.subspace_values_indices_list[cnt])
            subspace_moments = integrate_sparse_grid_subspace(
                subspace_index, subspace_values, self.weights_1d,
                self.config_variables_idx)
            new_subspace_moments[ii, :, :] = subspace_moments.T
            if self.canonical_interrogation_samples is not None:
                # if storage becomes a problem may need to remove subspace
                # values when they have a non-zero smolyak coefficient and
                # recompute it if needed again
                self.subspace_interrogation_values.append(
                    evaluate_sparse_grid_subspace(
                        self.canonical_interrogation_samples, subspace_index,
                        subspace_values, self.samples_1d,
                        self.config_variables_idx))
            cnt += 1

        if self.subspace_moments is None:
            self.subspace_moments = new_subspace_moments
        else:
            self.subspace_moments = np.vstack(
                (self.subspace_moments, new_subspace_moments))

        return num_new_subspace_samples

    def save(self, filename):
        try:
            with open(filename, 'wb') as file_object:
                pickle.dump(self, file_object)
        except RuntimeError:
            msg = 'Initial attempt to save failed. Likely self.function '
            msg += 'cannot be pickled. Trying to save setting function to None'
            print(msg)
            function = self.function
            self.function = None
            with open(filename, 'wb') as file_object:
                pickle.dump(self, file_object)
            self.function = function
            msg = 'Second save was successful'
            print(msg)

    def get_samples(self):
        return self.var_trans.map_from_canonical(self.samples)

    def __repr__(self):
        return "{0}(nvars={1})".format(
            self.__class__.__name__, self.num_vars)


def plot_adaptive_sparse_grid_3d(sparse_grid, plot_grid=True):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    active_subspace_indices, active_subspace_idx = get_active_subspace_indices(
        sparse_grid.active_subspace_indices_dict,
        sparse_grid.subspace_indices)

    # get subspace indices that have been added to the sparse grid,
    # i.e are not active
    sparse_grid_subspace_idx = np.ones(
        (sparse_grid.subspace_indices.shape[1]), dtype=bool)
    sparse_grid_subspace_idx[active_subspace_idx] = False

    nn = 1
    if plot_grid:
        nn = 2
    ax = fig.add_subplot(1, nn, 1, projection='3d')
    if active_subspace_indices.shape[1] == 0:
        active_subspace_indices = None
    plot_3d_indices(sparse_grid.subspace_indices, ax, active_subspace_indices)

    if plot_grid:
        samples, active_samples = partition_sparse_grid_samples(sparse_grid)
        ax = fig.add_subplot(1, nn, 2, projection='3d')
        ax.plot(samples[0, :], samples[1, :], samples[2, :], 'ko')
        ax.plot(active_samples[0, :], active_samples[1, :],
                active_samples[2, :], 'ro')

        angle = 45
        ax.view_init(30, angle)
        # ax.set_axis_off()
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def insitu_update_sparse_grid_quadrature_rule(sparse_grid,
                                              quadrule_variables,
                                              method='pdf'):
    num_vars = sparse_grid.num_vars
    num_random_vars = num_vars-sparse_grid.num_config_vars
    assert len(quadrule_variables) == num_random_vars
    # univariate_growth_rules = []
    unique_quadrule_indices = [[ii] for ii in range(num_random_vars)]
    new_var_trans = AffineTransform(
        quadrule_variables)
    quad_rules = []
    max_levels = sparse_grid.subspace_indices.max(axis=1)
    # initial_points_list = []
    growth_rules = []
    all_variable = sparse_grid.var_trans.variable.marginals()
    for ii in range(num_random_vars):
        for jj, inds in enumerate(sparse_grid.unique_quadrule_indices):
            if ii in inds:
                break
        growth_rule = sparse_grid.compact_univariate_growth_rule[jj]
        growth_rules.append(growth_rule)
        if all_variable[ii] == quadrule_variables[ii]:
            quad_rules.append(sparse_grid.compact_univariate_quad_rule[jj])
            continue
        canonical_initial_points = \
            sparse_grid.samples_1d[ii][max_levels[ii]][None, :]
        # samples_1d are in the canonical domain map to old user domain
        initial_points_old = \
            sparse_grid.var_trans.map_from_canonical_1d(
                canonical_initial_points, ii)
        # map to new canonical domain
        canonical_initial_points_new = new_var_trans.map_to_canonical_1d(
                initial_points_old, ii)
        # initial_points_list.append(canonical_initial_points_new)

        quad_rules.append(get_univariate_leja_quadrature_rule(
            quadrule_variables[ii], growth_rule, method,
            initial_points=canonical_initial_points_new))

    sparse_grid.set_univariate_growth_rules(
        growth_rules, unique_quadrule_indices)
    max_level = sparse_grid.subspace_indices.max()
    sparse_grid.set_univariate_rules(quad_rules, max_level)
    sparse_grid_samples = sparse_grid.samples.copy()
    sparse_grid_samples = \
        sparse_grid.var_trans.map_from_canonical(
            sparse_grid_samples)
    sparse_grid_samples = new_var_trans.map_to_canonical(
        sparse_grid_samples)
    sparse_grid.samples = sparse_grid_samples
    sparse_grid.var_trans = new_var_trans
    return sparse_grid

"""
Notes if use combination technique to manage only adaptive refinement in configure variables and another strategy (e.g. another independent combination technique to refine in stochastic space) then this will remove downward index constraint
# between subspaces that vary both models and parameters.

An Adaptive PCE can only use this aforementioned case. I do not see a way to
let each subspace still be a tensor product index and build an approximation only over tha subspace and then combine.
"""
