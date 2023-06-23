import numpy as np
import networkx as nx
from scipy import sparse as scipy_sparse
import copy

from pyapprox.util.utilities import (
    flatten_2D_list, equality_constrained_linear_least_squares
)


def get_extraction_matrices(system_labels, component_labels):
    """
    Compute the extraction matrics of each component used to extract
    quantities from a global list

    Note
    ----
    Should use get_extraction_indices instead

    Parameters
    ----------
    system_labels : list
        The collection of all component labels

    component_labels : list of lists
        The labels of each component

    Returns
    -------
    extraction_matrices : list
        List of extraction matrices with shape below

    extraction_matrix : np.ndarray (ncomponent_labels, nsystem_labels)
        Matrix of zeros and ones. If the (i,j) entry is one that indicates
        the ith component quantity is the jth system quantity
    """
    ncomponents = len(component_labels)
    ncols = len(system_labels)
    adj_matrices = []
    for ii in range(ncomponents):
        nrows = len(component_labels[ii])
        adj_matrix = np.zeros((nrows, ncols), dtype=int)
        for jj in range(nrows):
            kk = system_labels.index(component_labels[ii][jj])
            adj_matrix[jj, kk] = 1
        adj_matrices.append(adj_matrix)
    return adj_matrices


def get_extraction_indices(system_labels, component_labels):
    """
    Compute the indices indicating where each component label sits in the
    system_labels

    Parameters
    ----------
    system_labels : list
        The collection of all component labels

    component_labels : list of lists
        The labels of each component


    Returns
    -------
    extraction_indices : list
        List of extraction indices with shape (ncomponent_labels)
        extraction_indices[ii][jj] indicates that the ith component quantity
        is the jth system quantity
    """
    ncomponents = len(component_labels)
    adj_indices = []
    for ii in range(ncomponents):
        ncoupling_vars_ii = len(component_labels[ii])
        adj_indices_ii = np.empty(ncoupling_vars_ii, dtype=int)
        for jj in range(ncoupling_vars_ii):
            adj_indices_ii[jj] = system_labels.index(component_labels[ii][jj])
        adj_indices.append(adj_indices_ii)
    return adj_indices


def get_adjacency_matrix_from_labels(
        component_output_labels, component_coupling_labels):
    """
    Compute the adjacency matrix of a coupled system from the component
    output and coupling labels

    Adjacency matrices index system outputs by staking the outputs of
    each component
    sequentially and ordered by the index of the component. Similarly
    the coupling variables are the component coupling variables sequentially
    stacked. If a component does not have a coupling variable it is skipped
    E.g. Consider 3 models each with one output and coupled in a chain.
    The number of coupling variables are [0, 1, 1] and the number of outputs
    are [1,1,]. The adjacency matrix will be shape (2, 3) with entries
    [[0, 1, 0]
    [0, 0, 1]]

    Parameters
    ----------
    component_output_labels : list
        The labels of each component output

    component_coupling_labels : list of lists
        The labels of each component coupling variable

    Returns
    -------
    adjacency_matrix : np.ndarray (ncoupling_vars, noutputs)
        Matrix of zeros and ones.  If the (i,j) entry is one that indicates
        the ith system coupling variable is the jth system output.

    all_output_labels : list
        The labels of the system outputs (collection of all component outputs)

    all_coupling_labels : list
        The labels of the system coupling variables
        (collection of all component outputs)
    """
    ncomponent_outputs = [len(ol) for ol in component_output_labels]
    ncomponent_coupling_vars = [len(cl) for cl in component_coupling_labels]
    ncoupling_vars = np.sum(ncomponent_coupling_vars)
    noutputs = np.sum(ncomponent_outputs)
    adjacency_matrix = np.zeros((ncoupling_vars, noutputs))
    all_output_labels, all_coupling_labels = [], []
    all_output_labels = flatten_2D_list(component_output_labels)
    all_coupling_labels = flatten_2D_list(component_coupling_labels)
    for ii in range(len(all_coupling_labels)):
        cl = all_coupling_labels[ii]
        for jj in range(len(all_output_labels)):
            ol = all_output_labels[jj]
            if cl == ol:
                adjacency_matrix[ii, jj] = 1
    adjacency_matrix = scipy_sparse.csr_matrix(adjacency_matrix)
    return adjacency_matrix, all_output_labels, all_coupling_labels


def get_local_coupling_variables_indices_in(component_random_variable_labels,
                                            component_coupling_labels,
                                            local_random_var_indices=None):
    """
    Get the indices of the coupling variables of a component as they
    appear in the system output variables.

    Parameters
    ----------
    local_random_var_indices : np.ndarray (nrandom_vars)
        The indices of the random variables as they appaer in the component
        model. If None assume that model is parameterized by all random
        variables first and coupling variables second.
    """
    ncomponents = len(component_random_variable_labels)
    if local_random_var_indices is None:
        local_random_var_indices = [
            np.arange(len(ll)) for ll in component_random_variable_labels]
    local_coupling_var_indices_in = []
    for ii in range(ncomponents):
        assert len(component_random_variable_labels[ii]) == len(
            local_random_var_indices[ii])
        local_coupling_var_indices_in.append(
            np.delete(np.arange(
                len(component_random_variable_labels[ii]) +
                len(component_coupling_labels[ii])),
                      local_random_var_indices[ii]))
    return local_coupling_var_indices_in


def get_global_coupling_indices_in(component_coupling_labels,
                                   component_output_labels):
    """
    Get the index pairs (ii, jj) denoting the jjth qoi of the iith component
    which correspond to each system coupling variable

    Notes
    -----
    The user does not need to use this function
    """
    ncomponents = len(component_coupling_labels)
    global_coupling_component_indices = []
    for ii in range(ncomponents):
        inds = []
        for label in component_coupling_labels[ii]:
            for jj in range(ncomponents):
                try:
                    kk = component_output_labels[jj].index(label)
                    inds += [jj, kk]
                    break
                except:
                    pass
        global_coupling_component_indices.append(inds)
    return global_coupling_component_indices


def get_adjacency_matrix_component_info(
        component_output_labels, component_coupling_labels):
    """
    Get the number of coupling variables and outputs of each component

    Parameters
    ----------
    component_output_labels : list
        The labels of each component output

    component_coupling_labels : list of lists
        The labels of each component coupling variable

    Returns
    -------
    component_info : tuple (ncomponents)
        Tuple (ncomponent_coupling_vars, ncomponent_outputs) where
        ncomponent_coupling_vars is a np.ndarray with shape (ncomponents) and
        ncomponent_outputs is a np.ndarray with shape (ncomponents)
    """
    return (np.array([len(cl) for cl in component_coupling_labels]),
            np.array([len(cl) for cl in component_output_labels]))


def plot_adjacency_matrix(adjacency_matrix, component_info=None, ax=None,
                          xticklabels=None, yticklabels=None):
    r"""
    Plot an adjacency matrix

    Parameters
    ----------
    adjacency_matrix : np.ndarray (ncoupling_vars, noutputs)
        The adjacency matrix with entries that are either zero or one.
        The entry (i,j) is one if :math:`\xi_i=y_j` zero otherwise, where
        :math:`\xi_i` are coupling variables and :math:`y_j` output variables

    component_info : tuple (ncomponents)
        Tuple (ncomponent_coupling_vars, ncomponent_outputs) where
        ncomponent_coupling_vars is a np.ndarray with shape (ncomponents) and
        ncomponent_outputs is a np.ndarray with shape (ncomponents)
        Plot code assumes that adjacency matrix stores the outputs
        (and coupling vars) of each component sequentially, e.g. all of outputs
        from component 1 then all outputs of component 2 and so on.
    """
    if type(adjacency_matrix) != np.ndarray:
        adjacency_matrix = adjacency_matrix.todense()

    for row in adjacency_matrix:
        if np.count_nonzero(row) > 1:
            raise ValueError("Must have at most one non zero entry per row")

    if component_info is not None:
        component_shapes = [
            (jj, ii) for ii, jj in zip(component_info[0], component_info[1])]
    else:
        component_shapes = None

    from matplotlib import pyplot as plt, patches
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

    ax.imshow(adjacency_matrix, interpolation="none", aspect=1, cmap="Greys")
    # major ticks
    ax.set_xticks(np.arange(0, adjacency_matrix.shape[1], dtype=np.int64))
    ax.set_yticks(np.arange(0, adjacency_matrix.shape[0], dtype=np.int64))
    # minor ticks
    ax.set_xticks(
        np.arange(-0.5, adjacency_matrix.shape[1], 1), minor=True)
    ax.set_yticks(
        np.arange(-0.5, adjacency_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth="1")

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=90)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if component_shapes is None:
        return

    ncoupling_vars, noutputs = adjacency_matrix.shape
    current_idx, current_jdx = 0, 0
    from itertools import cycle
    colors = cycle(["grey", "b"])
    for component_shape in component_shapes:
        # plot rectangle that separates coupling variables
        # belonging to different components
        ax.add_patch(
            patches.Rectangle(
                (-0.5, current_jdx-0.5),
                noutputs, component_shape[1],
                facecolor=next(colors),  # edgecolor="r",
                linewidth="3", alpha=0.3, zorder=100))
        # plot line that separates output belonging to different components
        # skip line at 0
        if current_idx > 0:
            ax.axvline(x=current_idx-0.5, color='r', lw=5)
        current_idx += component_shape[0]
        current_jdx += component_shape[1]


def extract_sub_samples_using_matrix(extraction_matrix, samples):
    """
    Extract component quantities from the system quantities

    Notes
    -----
    This is inefficient. Use extract_sub_samples instead

    Parameters
    ----------
    extraction_matrix : np.ndarray (ncomponent_vars, nsystem_vars)
        The indices of component variables as they are in the system variables

    samples : np.ndarray (nsystem_vars, nsamples)
        Samples of the system variables (collection of all component variables)

    Returns
    -------
    sub_samples : np.ndarray (ncomponent_vars, nsamples)
        Samples of the component variables
    """
    return extraction_matrix.dot(samples)


def extract_sub_samples(extraction_indices, samples):
    """
    Extract component variables from the system varaibles

    Parameters
    ----------
    extraction_indices : np.ndarray (ncomponent_vars)
        The indices of component variables as they are in the system variables

    samples : np.ndarray (nsystem_vars, nsamples)
        Samples of the system variables (collection of all component variables)

    Returns
    -------
    sub_samples : np.ndarray (ncomponent_vars, nsamples)
        Samples of the component variables
    """
    sub_samples = samples[extraction_indices, :]
    return sub_samples


def evaluate_component_functions(
        component_funs, exog_extraction_indices, coup_extraction_indices,
        exog_samples, coup_samples):
    """
    Evaluate a set of component functions at a set of samples

    component_funs : list [callable]
        List of funcions with signature

        ``fun(component_samples)->np.ndarray (nsamples, ncomponent_outputs)``

        where component_samples has shape (ncomponent_vars, nsamples).
        WARNING: component_samples must be
        np.vstack(exog_samples_ii, coup_samples_ii) where
        exog_samples_ii, coup_samples_ii are the exogeneous and coupling
        variable samples specific to the ii-th component

    exog_extraction_indices : list [np.ndarray]
        List of extraction indices that select component exogeneous variables
        from the list of all system exogeneous varaiables

    coup_extraction_indices : list [np.ndarray]
        List of extraction indices that select component coupling variables
        from the list of all system coupling variables

    exog_samples : np.ndarray (nexog_vars, nsamples)
        Samples of the system exogeneous variables

    coup_samples : np.ndarray (nexog_vars, nsamples)
        Sample of the system coupling variables

    Returns
    -------
    values : np.ndarray (nsamples, nsystem_outputs)
        The outputs of each component evaluated at the exogeneous and coupling
        samples. The outputs of each component are stacked sequentially
        using np.hstack

    Notes
    -----
    This could be parallelized if necessary.
    """
    ncomponents = len(component_funs)
    values = []
    for ii in range(ncomponents):
        exog_samples_ii = extract_sub_samples(
            exog_extraction_indices[ii], exog_samples)
        coup_samples_ii = extract_sub_samples(
            coup_extraction_indices[ii], coup_samples)
        # assume component funs take all exog variables then all coup variables
        component_samples_ii = np.vstack((exog_samples_ii, coup_samples_ii))
        component_values = component_funs[ii](component_samples_ii)
        values.append(component_values)
    return np.hstack(values)


def gauss_jacobi_fixed_point_iteration(adjacency_matrix,
                                       exog_extraction_indices,
                                       coup_extraction_indices,
                                       component_funs,
                                       init_coup_samples, exog_samples,
                                       tol=1e-15,
                                       max_iters=100, verbose=0,
                                       anderson_memory=0):
    r"""
    Solve a set of coupled equations using Gauss-Jacobi fixed point iteration

    Parameters
    ----------
    adjacency_matrix : np.ndarray (ncoupling_vars, noutputs)
        The adjacency matrix with entries that are either zero or one.
        The entry (i,j) is one if :math:`\xi_i=y_j` zero otherwise, where
        :math:`\xi_i` are coupling variables and :math:`y_j` output variables

    exog_extraction_indices : list [np.ndarray]
        List of extraction indices that select component exogeneous variables
        from the list of all system exogeneous varaiables

    coup_extraction_indices : list [np.ndarray]
        List of extraction indices that select component coupling variables
        from the list of all system coupling variables

    component_funs : list [callable]
        List of funcions with signature

        ``fun(component_samples)->np.ndarray (nsamples, ncomponent_outputs)``

        where component_samples has shape (ncomponent_vars, nsamples).
        WARNING: component_samples must be
        np.vstack(exog_samples_ii, coup_samples_ii) where
        exog_samples_ii, coup_samples_ii are the exogeneous and coupling
        variable samples specific to the ii-th component

    init_coup_samples : np.ndarray (nsystem_coupling_vars, nsamples)
        The initial samples of the coupling variables used to start the
        fixed point iteration

    exog_samples : np.ndarray (nexog_vars, nsamples)
        Samples of the system exogeneous variables

    tol : float
        The error tolerance used to terminate the fixed point interation

    max_iters : integer
        The maximum number of fixed point iterations

    verbose : integer
        The amount of information to print to standard output

    anderson_memory : integer
        The amount of memory used in the Anderson extrapolation

    Returns
    -------
    outputs : np.ndarray (nsamples, nsystem_outputs)
        The outputs of each component evaluated at the exogeneous and coupling
        samples. The outputs of each component are stacked sequentially
        using np.hstack

    niters_per_sample : np.ndarray(nsamples)
        the number of iterations needed for each sample
    """
    if init_coup_samples.shape[1] != exog_samples.shape[1]:
        raise ValueError("Must provide initial guess for every sample")
    ncomponents = len(component_funs)
    if len(exog_extraction_indices) != ncomponents:
        raise ValueError("Must provide extraction matrix for each component")

    niters_per_sample = -np.ones(exog_samples.shape[1])
    it = 0
    coup_samples = init_coup_samples
    coup_history = np.ones(
        (coup_samples.shape[1], coup_samples.shape[0], anderson_memory+1))
    residuals_history = np.empty(
        (coup_samples.shape[1], coup_samples.shape[0], anderson_memory+1))
    while True:
        outputs = evaluate_component_functions(
            component_funs, exog_extraction_indices, coup_extraction_indices,
            exog_samples, coup_samples)
        new_coup_samples = adjacency_matrix.dot(outputs.T)

        residuals = new_coup_samples-coup_samples

        diff_norms = np.max(residuals, axis=0)
        error = np.max(diff_norms)

        if verbose > 0:
            msg = f"Iter  {it} : {error}"
            print(msg)

        if error <= tol or it+1 >= max_iters:
            break

        idx = min(anderson_memory+1, it+1)
        if it > anderson_memory:
            # delete oldest entry
            coup_history[:, :, :anderson_memory] = \
                coup_history[:, :, 1:].copy()
            residuals_history[:, :, :anderson_memory] = \
                residuals_history[:, :, 1:].copy()
        coup_history[:, :, idx-1] = new_coup_samples.T
        residuals_history[:, :, idx-1] = residuals.T

        # Anderson acceleration
        # based alpha on first sample
        A = residuals_history[0, :, :idx]
        B = np.ones((1, A.shape[1]), dtype=float)
        y = np.zeros((A.shape[0]), dtype=float)
        z = np.ones((1), dtype=float)
        alpha = equality_constrained_linear_least_squares(A, B, y, z)
        coup_samples = coup_history[:, :, :idx].dot(alpha).T

        # standard Gauss Jacobi
        # coup_samples = (
        #    relax_factor*new_coup_samples+(1-relax_factor)*coup_samples)
        it += 1
        niters_per_sample[(diff_norms <= tol) & (niters_per_sample < 0)] = it
    return outputs, niters_per_sample


def evaluate_function(graph, node_id, global_samples):
    """
    Evaluate a component of a system of coupled components with only feed
    forward coupling

    Parameters
    ----------
    graph : nx.DiGraph
        A directed ayclic graph encoding the coupling between components

    node_id : integer
        The component to be evaluated. No downstream components will be
        evaluated, only components that produce outputs that effect
        the output of the chosen component will be evaluated.

    global_samples : np.ndarray (nexog_vars, nsamples)
        Samples of the system exogeneous variables

    Returns
    -------
    values : np.ndarray (nsamples, ncomponent_outputs)
        The values of the component evaluated at the set of samples
    """

    node = graph.nodes[node_id]
    global_random_var_indices = node['global_random_var_indices']
    local_random_samples = global_samples[global_random_var_indices, :]

    node_local_random_var_indices = node['local_random_var_indices']
    node_local_coupling_var_indices_in = node['local_coupling_var_indices_in']
    if 'global_config_var_indices' in node:
        node_local_config_var_indices = node['local_config_var_indices']
        local_config_samples = \
            global_samples[node['global_config_var_indices'], :]
    else:
        node_local_config_var_indices = []
        local_config_samples = None

    children_ids = list(graph.predecessors(node_id))
    if len(children_ids) == 0 or node['values'] is not None:
        if node['values'] is None:
            nlocal_vars = len(node_local_random_var_indices) +\
                len(node_local_config_var_indices)
            local_samples = np.empty((nlocal_vars, global_samples.shape[1]))
            local_samples[node_local_random_var_indices, :] = \
                local_random_samples
            if len(node_local_config_var_indices) > 0:
                local_samples[node_local_config_var_indices, :] = \
                    local_config_samples
            values = node['functions'](local_samples)
            node['values'] = values
        return node['values']

    child_id = children_ids[0]
    children_values = evaluate_function(graph, child_id, global_samples)
    children_nqoi = [children_values.shape[1]]
    for child_id in children_ids[1:]:
        child_values = evaluate_function(graph, child_id, global_samples)
        children_nqoi.append(child_values.shape[1])
        children_values = np.hstack(
            [children_values, child_values])

    nlocal_vars = len(node_local_random_var_indices) +\
        len(node_local_coupling_var_indices_in) + \
        len(node_local_config_var_indices)
    local_samples = np.empty((nlocal_vars, global_samples.shape[1]))

    # populate random samples
    local_samples[node_local_random_var_indices, :] = local_random_samples

    # populate coupling samples
    node_global_coupling_component_indices = \
        node['global_coupling_component_indices']
    strides = np.hstack([[0], np.cumsum(children_nqoi)[:-1]])
    selected_indices = []  # indices into children_values
    for ii in range(len(node_global_coupling_component_indices)//2):
        component_id, qoi_id = \
            node_global_coupling_component_indices[2*ii:2*ii+2]
        local_child_id = children_ids.index(component_id)
        selected_indices.append(strides[local_child_id]+qoi_id)
    local_samples[node_local_coupling_var_indices_in, :] = \
        children_values[:, selected_indices].T

    # populate configuration samples
    if len(node_local_config_var_indices) > 0:
        local_samples[node_local_config_var_indices, :] = local_config_samples

    values = node['functions'](local_samples)
    node['values'] = values
    return node['values']


def evaluate_functions(graph, global_samples, node_ids=None):
    """
    Evaluate a set of components in a system of coupled components with
    only feed forward coupling

    Parameters
    ----------
    graph : nx.DiGraph
        A directed ayclic graph encoding the coupling between components

    global_samples : np.ndarray (nexog_vars, nsamples)
        Samples of the system exogeneous variables

    node_ids : list[integer]
        The components to be evaluated. If None all component will be evalauted

    Returns
    -------
    values : np.ndarray (nsamples, ncomponent_outputs)
        The values of the component evaluated at the set of samples
    """
    values = []
    for nid in graph.nodes:
        graph.nodes[nid]['values'] = None

    if node_ids is None:
        node_ids = [np.max([nid for nid in graph.nodes])]
    for node_id in node_ids:
        values.append(evaluate_function(graph, node_id, global_samples))
    return values


class SystemNetwork(object):
    """
    Object describing the connections between components of a system model
    with only feed-forward coupling.

    Parameters
    ----------
    graph : :class:`networkx.DiGrapgh`
        Graph representing the coupled system


    Node Attributes
    ---------------
    Each node must have the following attributes

    label : string
        A unique string identifier

    functions: callable
        A funciton with the signature

        `f(z) -> np.ndarray(nsamples, nlocal_qoi)`

        where z is a np.ndarray (nlocal_vars, nsamples) and
        nlocal_vars = nlocal_random_vars + nlocal_config_vars

    local_random_var_indices : np.ndarray (nlocal_random_vars)
        The index to the arguments of the local function which are
        random variables

    local_coupling_var_indices_in : np.ndarray (nlocal_coupling_var_indices_in)
        The index to the arguments of the local function which are
        coupling variables

    global_random_var_indices : np.ndarray (nlocal_random_vars)
        The index to the arguments of the global ssytem function which are
        random variables

    global_config_component_indices : np.ndarray (nlocal_coupling_var_indices_in)
        The index to the arguments of the global system function which are
        coupling variables

    Notes
    -----
    Currently this code assumes that all variables are independent.
    TODO: break global variables into groups of independent variables then
    use indexing to reference these groups. For example group 1 may have a 2D
    correlated variable tensor product with a univariate marginal which are
    inputs to component 1. Second component may be tensor product of 2
    additional independent variables. In this case there will be four groups
    1) the two correlated variables 2), 3), 4) each being a unique independent
    variable. An important test case here will be when the variables of a
    multivariate group, e.g. the 2D correlated variables, are arguments of
    different components, e.g. the first correlated variable is an input
    for component 1 and the second and input for component 2.
    """
    def __init__(self, graph):
        self.graph = graph
        self.__validate_graph()

    def set_functions(self, functions):
        surr_graph = self.graph
        for nid in surr_graph.nodes:
            surr_graph.nodes[nid]['functions'] = functions[nid]

    def copy_graph(self):
        copy_graph = nx.DiGraph()
        copy_graph.add_nodes_from(self.graph)
        copy_graph.add_edges_from(self.graph.edges)
        for nid in self.graph.nodes:
            node = self.graph.nodes[nid]
            for key, item in node.items():
                if key != 'functions':
                    copy_graph.nodes[nid][key] = copy.deepcopy(item)
                else:
                    # shallow copy of functions. Deep copy will try to pickle
                    # the functions which cannot be used with PoolModel or
                    # WorkTracker model
                    copy_graph.nodes[nid][key] = item
        return copy_graph

    def copy(self):
        """
        Return deep copy of this object except for shallow copies of functions.
        Deep copy will try to pickle the functions which cannot be used with
        PoolModel or WorkTracker model
        """
        copy_graph = self.copy_graph()
        copy_network = type(self)(copy_graph)
        return copy_network

    def __validate_graph(self):
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            assert ((len(node['local_coupling_var_indices_in']) ==
                     len(node['global_coupling_component_indices'])//2) or
                    (len(node['local_coupling_var_indices_in']) ==
                    len(node['global_coupling_component_indices']) == 0))

    def component_nvars(self):
        """
        Return the total number of independent variables of each component.
        The total number of variables is the sum of the number of random and
        coupling variables.
        """
        nvars = []
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            nvars.append(
                len(node['local_random_var_indices']) +
                len(node['local_coupling_var_indices_in']))
        return nvars

    def __call__(self, samples, component_ids=None):
        """
        Evaluate the system at a set of samples.

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Samples of the system parameters

        component_ids : iterable
            The ids of the components whose evaluation is requested

        Returns
        -------
        values : list or np.ndarray
            Evaluation of each component in component_ids at the samples
            Each entry of the list is np.ndarray (nsamples, nlocal_qoi)
            If component_ids is None then values will be the
            np.ndarray (nsamples, nlocal_qoi) containing only the output
            of the most downsream component
        """
        values = evaluate_functions(self.graph, samples, component_ids)
        if component_ids is None:
            # values are outputs of most downstream component
            # assumes these are QoI and return as np.ndarray
            return values[0]
        return values

    def ncomponents(self):
        return len(self.graph.nodes)

    def get_graph_attribute(self, name):
        vals = []
        for node in self.graph.nodes:
            vals.append(self.graph.nodes[node][name])
        return vals


class GaussJacobiSystemNetwork(SystemNetwork):
    """
    Use Gauss Jacobi iteration to evaluate a coupled system with either
    feed-forward or feed-back coupling or both.
    """

    def __init__(self, graph):
        super().__init__(graph)
        self.adjacency_matrix = None
        self.exog_ext_indices = None
        self.coup_ext_indices = None
        self.qoi_ext_indices = None
        self.ncomponent_outputs = None
        self.output_indices = None
        self.init_coup_sample = None
        self.opts = {"tol": 1e-15, "max_iters": 100, "verbose": 0,
                     "anderson_memory": 0}
        self.functions = None
        self.set_functions(self.get_graph_attribute("functions"))
        self.eval_iteration_count_history = []

    def set_functions(self, functions):
        super().set_functions(functions)
        self.functions = functions

    def set_adjacency_matrix(self, adjacency_matrix):
        """
        Set the system adjacency matrix

        Parameters
        ----------
        adjacency_matrix : np.ndarray or csr_matrix (ncoupling_vars, noutputs)
            Matrix of zeros and ones.  If the (i,j) entry is one that indicates
            the ith system coupling variable is the jth system output.
            self.adjacency_matrix = adjacency_matrix
        """
        self.adjacency_matrix = adjacency_matrix

    def set_extraction_indices(
            self, exog_ext_indices, coup_ext_indices, qoi_ext_indices,
            ncomponent_outputs):
        """
        Parameters
        ----------
        exog_ext_indices : list [np.ndarray]
            List of extraction indices that select component exogeneous
            variables from the list of all system exogeneous varaiables

        coup_ext_indices : list [np.ndarray]
            List of extraction indices that select component coupling variables
            from the list of all system coupling variables

        coup_ext_indices : np.ndarray(nqoi)
            List of extraction indices that select the quantities of interest
            from the list of all system outputs

        ncomponent_outputs : iterable
            The number of ouputs of each component
        """
        if ((type(qoi_ext_indices) != np.ndarray or qoi_ext_indices.ndim != 1)
            and not (type(qoi_ext_indices) == list and
                     type(qoi_ext_indices[0]) == int)):
            raise ValueError(
                "qoi_ext_indices, must be a 1D np.ndarray or list")
        self.exog_ext_indices = exog_ext_indices
        self.coup_ext_indices = coup_ext_indices
        self.qoi_ext_indices = qoi_ext_indices
        self.ncomponent_outputs = ncomponent_outputs
        self.output_indices = np.hstack(
            (0, np.cumsum(self.ncomponent_outputs)))

    def set_initial_coupling_sample(self, init_coup_sample):
        """
        The same coupling sample will be used to initialize the Gauss
        Jacobi iteration

        init_coup_sample : np.ndarary (nsystem_coupling_vars, 1)
            The initial value of the coupling vars
        """
        if init_coup_sample.ndim != 2 or init_coup_sample.shape[1] != 1:
            raise ValueError("init_coup sample must be a 2D np.ndarray")
        self.init_coup_sample = init_coup_sample

    def ncoupling_vars(self):
        return self.adjacency_matrix.shape[0]

    def nsystem_outputs(self):
        return self.adjacency_matrix.shape[1]

    def set_gauss_jacobi_options(self, opts):
        self.opts = opts

    def copy(self):
        """
        Return deep copy of this object except for shallow copies of functions.
        Deep copy will try to pickle the functions which cannot be used with
        PoolModel or WorkTracker model
        """
        copy_graph = self.copy_graph()
        copy_network = type(self)(copy_graph)
        copy_network.set_adjacency_matrix(self.adjacency_matrix)
        copy_network.set_extraction_indices(
            self.exog_ext_indices, self.coup_ext_indices, self.qoi_ext_indices,
            self.ncomponent_outputs)
        copy_network.set_initial_coupling_sample(self.init_coup_sample)
        return copy_network

    def __call__(self, exog_samples, component_ids=None,
                 init_coup_samples=None):
        """
        Evaluate the system at a set of samples.

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            Samples of the system parameters

        component_ids : iterable
            The ids of the components whose evaluation is requested

        init_coup_samples : np.ndarray (nsystem_coupling_vars, nsamples)
            The initial samples of the coupling variables used to start the
            fixed point iteration

        Returns
        -------
        values : list
            Evaluation of each component in component_ids at the samples
            Each entry of the list is np.ndarray (nsamples, nlocal_qoi)
        """
        if (self.adjacency_matrix is None):
            raise ValueError("Must set adjacency matrix")
        if (self.exog_ext_indices is None or self.coup_ext_indices is None):
            raise ValueError("Must set extraction indices")

        if init_coup_samples is None:
            if (self.init_coup_sample is None):
                msg = "Must set provide initial coupling sample for each"
                msg += " exog_sample or use set_initial_coupling_sample"
                raise ValueError(msg)
            init_coup_samples = np.tile(
                self.init_coup_sample, (1, exog_samples.shape[1]))
        # init_output_samples = self.expand_initial_coupling_samples(
        #     init_coup_samples)

        outputs, iters = gauss_jacobi_fixed_point_iteration(
            self.adjacency_matrix, self.exog_ext_indices,
            self.coup_ext_indices, self.functions,
            # init_output_samples,
            init_coup_samples,
            exog_samples, **self.opts)
        self.eval_iteration_count_history.append(iters)
        if component_ids is None:
            if self.qoi_ext_indices is None:
                raise ValueError("Must set QoI extraction indices")
            qoi = outputs[:, self.qoi_ext_indices]
            return qoi
        # assumes output of each component are returned in sequential order
        return [outputs[:, self.output_indices[ii]:self.output_indices[ii+1]]
                for ii in component_ids]


def extract_node_data(node_id, data):
    """
    Extract the values of the attributes of a node from a dictionary
    containing node data for each component in a system
    """
    node_data = dict()
    for key, item in data.items():
        node_data[key] = item[node_id]
    return node_data


def build_chain_graph(nmodels, data=dict()):
    """
    Build a directed a cyclic graph that consists of a chain of components.

    0 -> 1 -> 2 -> ... -> n-2 -> n-1
    """
    g = nx.DiGraph()
    for ii in range(nmodels):
        node_data = extract_node_data(ii, data)
        g.add_node(ii, **node_data)

    edges = [[ii, ii+1]for ii in range(nmodels-1)]
    g.add_edges_from(edges)

    return g


def build_peer_graph(nmodels, data=dict()):
    """
    Build a directed a cyclic graph that consists of a single leaf node and
    N-1 root nodes.

    0   ->
    1   ->
    ...    n-1
    n-3 ->
    n-2 ->
    """
    g = nx.DiGraph()
    for ii in range(nmodels):
        node_data = extract_node_data(ii, data)
        g.add_node(ii, **node_data)

    edges = [[ii, nmodels-1]for ii in range(nmodels-1)]
    g.add_edges_from(edges)

    return g


def build_3_component_full_graph(data=dict()):
    """
    Build a graph with three components where components 0 and 1 both connect
    to component 2 and component 0 also connects to component 1.

    0
    | \
    |  \
    v   v
    1 --> 2
    """
    g = nx.DiGraph()
    for ii in range(3):
        node_data = extract_node_data(ii, data)
        g.add_node(ii, **node_data)
    edges = [[0, 1], [0, 2], [1, 2]]
    g.add_edges_from(edges)
    return g
