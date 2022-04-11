import os
import numpy as np
from functools import partial
from scipy import stats

from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    CombinationSparseGrid,
    variance_refinement_indicator, max_level_admissibility_function,
    get_sparse_grid_univariate_leja_quadrature_rules_economical,
    update_smolyak_coefficients, insitu_update_sparse_grid_quadrature_rule
)
from pyapprox.expdesign.low_discrepancy_sequences import (
    transformed_halton_sequence
)


def get_coupling_variables_via_sampling(
        network, random_variables, nsamples, expansion_factor=0.1,
        filename=None):
    """
    Compute the bounds on the coupling variables (output of upstream models)
    using Monte Carlo sampling. Return uniform variables over a slgithly larger
    range
    """
    if filename is not None and os.path.exists(filename):
        print(f'loading file {filename}')
        values = np.load(filename)['values']
    else:
        samples = generate_independent_random_samples(
            random_variables, nsamples)
        component_ids = np.arange(len(network.graph.nodes))
        values = network(samples, component_ids)
        if filename is not None:
            np.savez(filename, values=values, samples=samples)

    coupling_bounds = [[v.min(axis=0), v.max(axis=0)] for v in values]
    coupling_variables = {}

    graph = network.graph
    for jj in graph.nodes:
        coupling_variables[jj] = []
        indices = graph.nodes[jj]['global_coupling_component_indices']
        for ii in range(len(indices)//2):
            lb = coupling_bounds[indices[2*ii]][0][indices[2*ii+1]]
            ub = coupling_bounds[indices[2*ii]][1][indices[2*ii+1]]
            diff = ub-lb
            lb = lb - diff*expansion_factor/2
            ub = ub + diff*expansion_factor/2
            coupling_variables[jj].append(stats.uniform(lb, ub-lb))
    return coupling_variables


def get_coupling_variables_from_specified_ranges(network):
    """
    Assumes ranges have been added to graph.
    """
    coupling_variables = {}
    graph = network.graph
    for jj in graph.nodes:
        coupling_variables[jj] = []
        indices = graph.nodes[jj]['global_coupling_component_indices']
        ranges = graph.nodes[jj]['coupling_variables_ranges']
        for ii in range(len(indices)//2):
            lb, ub = ranges[ii]
            coupling_variables[jj].append(stats.uniform(lb, ub-lb))
    return coupling_variables


class DecoupledSystemSurrogate(object):
    """
    Object to manage the refinement of decoupled multi-physics
    models.

    Parameters
    ----------
    system_network : :class:SystemNetwork
        Object describing the connections between components of a system model.

    variables : :class:`pyapprox.IndependentMarginalsVariable`
        The system level variables

    estimate_coupling_ranges : boolean
        True - estimate ranges of coupling variables
        False - use set_coupling_variables to define ranges

    Attributes
    ----------

    surogate_network : :class:SystemNetwork
        Object containing the surrogates of each system component

    """
    def __init__(self, system_network, variables, nrefinement_samples=1e6,
                 verbose=0, estimate_coupling_ranges=False):
        self.system_network = system_network
        # self.surrogate_network = copy.deepcopy(self.system_network)
        self.surrogate_network = self.system_network.copy()
        self.variables = variables
        self.nrefinement_samples = int(nrefinement_samples)
        self.verbose = verbose
        self.estimate_coupling_ranges = estimate_coupling_ranges

        self.config_var_indices = []
        surr_graph = self.surrogate_network.graph
        for nid in surr_graph.nodes:
            # remove config variables which are not used when evaluating
            # the surrogate
            if 'local_config_var_indices' in surr_graph.nodes[nid]:
                self.config_var_indices.append(
                    surr_graph.nodes[nid]['global_config_var_indices'])
                del surr_graph.nodes[nid]['local_config_var_indices']
                del surr_graph.nodes[nid]['global_config_var_indices']

    def initialize_component_surrogates(self, component_options):
        """
        Initialize a surrogate of each system component.

        Parameters
        ----------
        component_options : iterable
            List of dictionary of options containing the arguments necessary to
            initialize each surrogate

            See documentation of
           :func:`pyapprox.surrogates.approximate.adaptive_approximate_sparse_grid`
        """
        # self.random_samples_for_refinement_test = \
        #     generate_independent_random_samples(
        #         self.variables, self.nrefinement_samples)
        marginal_icdfs = [v.ppf for v in self.variables.marginals()]
        self.random_samples_for_refinement_test = \
            transformed_halton_sequence(
                marginal_icdfs, len(marginal_icdfs), self.nrefinement_samples)

        surr_graph = self.surrogate_network.graph
        functions = []
        for nid in surr_graph.nodes:
            options = component_options[nid]
            functions.append(
                self.initialize_surrogate(surr_graph.nodes[nid], **options))
        self.surrogate_network.set_functions(functions)

        # Add first index of each variable to active set of respective grid
        self.component_output_ranges = []
        for nid in surr_graph.nodes:
            if self.verbose > 0:
                print('------------------------------------')
                print(f'Refining component {surr_graph.nodes[nid]["label"]}')
            surr_graph.nodes[nid]['functions'].refine()
            if self.verbose > 0:
                print('------------------------------------')

    def set_coupling_variables(self, coupling_variables):
        surr_graph = self.surrogate_network.graph
        for component_id, variables in coupling_variables.items():
            surr_graph.nodes[component_id]['coupling_variables'] = \
                variables

    def get_node_variables(self, node):
        var_indices = node['global_random_var_indices']
        global_variables = self.variables.marginals()
        local_nvars = len(node['local_random_var_indices']) + \
            len(node['local_coupling_var_indices_in'])
        local_variables = [None for ii in range(local_nvars)]
        random_variables = [global_variables[v] for v in var_indices]
        for ii, idx in enumerate(node['local_random_var_indices']):
            local_variables[idx] = random_variables[ii]
        coupling_variables = node['coupling_variables']
        for ii, idx in enumerate(node['local_coupling_var_indices_in']):
            local_variables[idx] = coupling_variables[ii]
        return local_variables

    def get_univariate_quadrature_rules(self, variables,
                                        enforce_variable_bounds,
                                        univariate_quad_rule_info,
                                        quad_method, growth_incr=2):
        var_trans = AffineTransform(
            variables, enforce_variable_bounds)

        if univariate_quad_rule_info is None:
            (quad_rules, growth_rules, unique_quadrule_indices,
             unique_max_level_1d) = \
                 get_sparse_grid_univariate_leja_quadrature_rules_economical(
                     var_trans, method=quad_method, growth_incr=growth_incr)
        else:
            quad_rules, growth_rules = univariate_quad_rule_info
            unique_quadrule_indices = None
        return var_trans, quad_rules, growth_rules, unique_quadrule_indices

    def initialize_surrogate(
            self, node, refinement_indicator=None,
            univariate_quad_rule_info=None, max_nsamples=100, tol=0,
            verbose=0, config_variables_idx=None, config_var_trans=None,
            cost_function=None, max_level_1d=None, quad_method='pdf',
            enforce_variable_bounds=False, growth_incr=2):
        """
        Initialize an adaptive sparse grid surrogate of a component.

        Parameters
        ----------
        node : :class:`networkx.Node`
            The node repesenting the component of interest

        options : dict
            Arguments necessary to initialize each surrogate.
            See documentation of
            :func:`pyapprox.surrogates.approximate.adaptive_approximate_sparse_grid`

        Returns
        -------
        approx : :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid`
            The sparse grid approximation
        """
        variables = self.get_node_variables(node)
        var_trans, quad_rules, growth_rules, unique_quadrule_indices = \
            self.get_univariate_quadrature_rules(
                variables, enforce_variable_bounds, univariate_quad_rule_info,
                quad_method, growth_incr)

        nvars = var_trans.num_vars()
        if config_var_trans is not None:
            nvars += config_var_trans.num_vars()

        if max_level_1d is None:
            max_level_1d = [np.inf]*nvars
        elif np.isscalar(max_level_1d):
            max_level_1d = [max_level_1d]*nvars
        assert len(max_level_1d) == nvars
        admissibility_function = partial(
            max_level_admissibility_function, np.inf, max_level_1d,
            max_nsamples, tol, verbose=verbose)
        if (self.estimate_coupling_ranges is True and
                refinement_indicator is not None):
            msg = "Can only estimate ranges if default refinement indicator "
            msg += "is used"
            raise ValueError(msg)
        if refinement_indicator is None:
            refinement_indicator = self.__refinement_indicator

        sparse_grid = CombinationSparseGrid(nvars)
        sparse_grid.setup(
            node['functions'], config_variables_idx, refinement_indicator,
            admissibility_function, growth_rules, quad_rules,
            var_trans, unique_quadrule_indices=unique_quadrule_indices,
            verbose=verbose, cost_function=cost_function,
            config_var_trans=config_var_trans)

        if self.verbose > 0:
            print('------------------------------------')
            print(f'Initializing component {node["label"]}')
            print('--')
        sparse_grid.initialize()
        if self.verbose > 0:
            print('------------------------------------')
        return sparse_grid

    def extract_coupling_ranges_from_samples(self, values_all_nodes):
        surr_graph_nodes = self.surrogate_network.graph.nodes
        for nid in surr_graph_nodes:
            coupling_mins = values_all_nodes[nid].min(axis=0)
            coupling_maxs = values_all_nodes[nid].max(axis=0)
            if len(self.component_output_ranges) <= nid:
                self.component_output_ranges.append([
                    coupling_mins, coupling_maxs])
            else:
                for kk in range(len(coupling_mins)):
                    self.component_output_ranges[nid][0][kk] = min(
                        self.component_output_ranges[nid][0][kk],
                        coupling_mins[kk])
                    self.component_output_ranges[nid][1][kk] = max(
                        self.component_output_ranges[nid][1][kk],
                        coupling_maxs[kk])

    def __refinement_indicator(
            self, subspace_index, num_new_subspace_samples, surrogate):
        values_old = self(self.random_samples_for_refinement_test)

        old_smolyak_coeffs = surrogate.smolyak_coefficients.copy()
        new_smolyak_coeffs = update_smolyak_coefficients(
            subspace_index, surrogate.subspace_indices,
            old_smolyak_coeffs.copy())

        surrogate.smolyak_coefficients = new_smolyak_coeffs
        # try:
        values_new_all = self(self.random_samples_for_refinement_test,
                              np.arange(self.system_network.ncomponents()))
        values_new = self(self.random_samples_for_refinement_test)
        self.extract_coupling_ranges_from_samples(values_new_all)
        # except:
        # values_new = None

        surrogate.smolyak_coefficients = old_smolyak_coeffs

        # default to variance refinement indicator when downstream
        # models have not been refined in the coupling directions.
        if (values_new is None or values_old is None):
            return variance_refinement_indicator(
                subspace_index, num_new_subspace_samples, surrogate)

        # moments = [np.mean(values_old, axis=0), np.var(values_old, axis=0)]
        # new_moments = [np.mean(values_new, axis=0), np.var(values_new, axis=0)]
        # error = np.absolute(new_moments[0] - moments[0])**2 + np.absolute(
        #    new_moments[1] - moments[1])

        error = np.linalg.norm(values_old-values_new, axis=0)
        indicator = error.copy()

        qoi_chosen = np.argmax(indicator)
        indicator = indicator[qoi_chosen]

        cost_per_sample = surrogate.eval_cost_function(
            subspace_index[:, np.newaxis])
        cost = cost_per_sample*num_new_subspace_samples

        indicator /= -cost
        error = error[qoi_chosen]

        if (subspace_index.sum() == 1 and indicator == 0):
            # indicator == 0 imples downstream models are constant functions
            indicator, error = variance_refinement_indicator(
                subspace_index, num_new_subspace_samples, surrogate)

        return indicator, error

    def __get_priorities(self, debug=False):
        surr_graph = self.surrogate_network.graph
        priorities = np.empty(surr_graph.number_of_nodes())
        subspace_indices = [None for ii in range(surr_graph.number_of_nodes())]
        for nid in surr_graph.nodes:
            surrogate = surr_graph.nodes[nid]['functions']
            if (not surrogate.active_subspace_queue.empty()):
                priorities[nid] = surrogate.active_subspace_queue.list[0][0]
                subspace_indices[nid] = surrogate.subspace_indices[
                    :, surrogate.active_subspace_queue.list[0][2]]
            else:
                priorities[nid], subspace_indices[nid] = np.inf, None
        return priorities, subspace_indices

    def refine(self):
        """
        Refine the surrogate with the highest priority (most negative value)
        """
        priorities, subspace_indices = self.__get_priorities()
        kk = np.argmin(priorities)
        surr_graph = self.surrogate_network.graph
        if self.verbose > 0:
            print('------------------------------------')
            print(f'Refining component {surr_graph.nodes[kk]["label"]}')
            print(f'By adding subspace {subspace_indices[kk]}')
            print(f'Total work {self.get_total_work()}')
            print('--')
        self.component_output_ranges = []
        best_surrogate_to_refine = surr_graph.nodes[kk]['functions']
        best_surrogate_to_refine.refine()
        if self.verbose > 0:
            print('------------------------------------')

        self.component_output_ranges = []
        for nid in surr_graph.nodes:
            # print(nid)
            surrogate = surr_graph.nodes[nid]['functions']
            surrogate.recompute_active_subspace_priorities()
            # tmp_priorities, tmp_subspace_indices = self.__get_priorities()
            # print(tmp_priorities, tmp_subspace_indices)

        self.best_component = kk
        self.best_subspace_index = subspace_indices[kk]
        if self.estimate_coupling_ranges:
            self.update_coupling_variables()

    def update_coupling_variables(self):
        surr_graph = self.surrogate_network.graph
        surr_graph_nodes = surr_graph.nodes
        for nid in surr_graph_nodes:
            update_coupling_variables = False
            node = surr_graph_nodes[nid]
            # noutputs = len(self.component_output_ranges[nid])
            coupling_inds = node['global_coupling_component_indices']
            cp_vars = node['coupling_variables']
            for kk, v in enumerate(cp_vars):
                out_component_id = coupling_inds[2*kk]
                out_qoi_id = coupling_inds[2*kk+1]
                v_range = v.interval(1)
                new_range = [v_range[0], v_range[1]]
                if v_range[0] > self.component_output_ranges[
                        out_component_id][0][out_qoi_id]:
                    new_range[0] = self.component_output_ranges[
                        out_component_id][0][out_qoi_id]
                if v_range[1] < self.component_output_ranges[
                        out_component_id][1][out_qoi_id]:
                    new_range[1] = self.component_output_ranges[
                        out_component_id][1][out_qoi_id]
                if new_range != list(v_range):
                    update_coupling_variables = True
                    node['coupling_variables'][kk] = \
                        stats.uniform(new_range[0], new_range[1]-new_range[0])
                    if self.verbose > 0:
                        msg = 'Adjusting ranges of local coupling variable '
                        msg += f' {kk} of component {nid} from '
                        msg += f'{v_range} to {new_range}'
                        print(msg)
            if update_coupling_variables is True:
                surrogate = surr_graph.nodes[nid]['functions']
                coupling_inds = node['local_coupling_var_indices_in']
                quadrule_variables = \
                    surrogate.var_trans.variable.marginals()
                for kk, ind in enumerate(coupling_inds):
                    quadrule_variables[ind] = node['coupling_variables'][kk]
                surr_graph.nodes[nid]['functions'] = \
                    insitu_update_sparse_grid_quadrature_rule(
                        surrogate, quadrule_variables)

    def __terminate_build(self, terminate_test=None):
        """
        Return flag indicating the need to terminate the build.

        The default termination condition checks if all surrogate subspace
        queues are empty. Thus defaulting to conditions specified to each
        surrogate in the system.

        Parameters
        ----------
        terminate_test : callable
            Function with signature

            `terminate_test(approx) -> boolean`

            where self is passed as approx. If true the build will be
            terminated. If False it will be allowed to continue. If provided
            this function overides the default termination which is only
            evaluated if terminate_test is False
        """
        surr_graph = self.surrogate_network.graph
        if (terminate_test is not None and terminate_test(self)):
            return True

        for nid in surr_graph.nodes:
            surrogate = surr_graph.nodes[nid]['functions']
            if ((not surrogate.active_subspace_queue.empty()) or
                    (surrogate.subspace_indices.shape[1] == 0)):
                # At least one surrogate can be refined
                return False

        # all surrogates can no longer be refined
        # component sparse grid exit criteria has stopped subspaces
        # being added to the grid
        if self.verbose > 0:
            print('Exiting: no components can be refined')
        return True

    def build(self, callback=None, terminate_test=None):
        """
        Build surrogates of the system components until the termination
        criteria are reached
        """
        while (not self.__terminate_build(terminate_test)):
            self.refine()
            if callback is not None:
                callback(self)

    def __call__(self, samples, component_ids=None):
        """
        Evaluate the surrogate of the entire system.

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Samples of the system parameters

        component_ids : iterable
            The ids of the components whose evaluation is requested

        Returns
        -------
        values : list
            Evaluation of each component in component_ids at the samples
            Each entry of the list is np.ndarray (nsamples, nlocal_qoi)
        """
        return self.surrogate_network(samples, component_ids)

    def __get_work(self):
        """
        Return the number of equivalent function evaluations for each grid
        """
        surr_graph = self.surrogate_network.graph
        num_evals = np.zeros(surr_graph.number_of_nodes())
        for nid in surr_graph.nodes:
            surrogate = surr_graph.nodes[nid]['functions']
            num_evals[nid] = surrogate.num_equivalent_function_evaluations
        return num_evals

    def get_total_work(self):
        """Return the total work used to construct the current state."""
        return np.sum(self.__get_work())


class TerminateTest(object):
    def __init__(self, max_iters=np.inf, max_work=np.inf, verbosity=0):
        self.verbosity = verbosity
        self.max_iters = max_iters
        self.max_work = max_work

        self.iters = 0

    def __call__(self, approx):
        self.iters += 1
        if (approx.get_total_work() >= self.max_work):
            if self.verbosity > 0:
                print('End refinement: maximum work budget exceeded.')
            return True

        if (self.iters > self.max_iters):
            if self.verbosity > 0:
                print('End refinement: maximum iterations reached.')
            return True

        return False


def get_coupling_variable_bounds(coupling_variables):
    coupling_bounds = []
    for jj in coupling_variables.keys():
        coupling_bounds_jj = []
        for var in coupling_variables[jj]:
            coupling_bounds_jj.append(var.interval(1))
        coupling_bounds.append(np.asarray(coupling_bounds_jj))
    return coupling_bounds
