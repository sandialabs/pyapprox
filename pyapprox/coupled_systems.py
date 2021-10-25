import numpy as np
import networkx as nx


def get_extraction_matrices(system_labels, component_labels):
    ncomponents = len(component_labels)
    ncols = len(system_labels)
    adj_matrices = []
    for ii in range(ncomponents):
        nrows = len(component_labels[ii])
        adj_matrix = np.zeros((nrows, ncols))
        for jj in range(nrows):
            kk = system_labels.index(component_labels[ii][jj])
            adj_matrix[jj, kk] = 1
        adj_matrices.append(adj_matrix)
    return adj_matrices


def plot_adjacency_matrix(adjacency_matrix, component_info=None, ax=None):
    r"""
    Parameters
    ----------
    adjacency_matrix : np.ndarray (ncoupling_vars, noutputs)
        The adjacency matrix with entries that are either zero or one.
        The entry (i,j) is one if :math:`\xi_i=y_j` zero otherwise, where
        :math:`\xi_i` are coupling variables and :math:`y_j` output variables

    component_info : tuple (ncomponents)
        Tuple (ncomponent_coupling_vars, ncomponent_outputs,) where
        ncomponent_coupling_vars is a np.ndarray with shape (ncomponents) and
        ncomponent_outputs is a np.ndarray with shape (ncomponents,)
    """
    if type(adjacency_matrix) != np.ndarray:
        adjacency_matrix = adjacency_matrix.todense()

    for row in adjacency_matrix:
        if np.count_nonzero(row) > 1:
            raise ValueError("Must have at most one non zero entry per row")

    component_shapes = [
        (jj, ii) for ii, jj in zip(component_info[0], component_info[1])]

    from matplotlib import pyplot as plt, patches
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

    ax.imshow(adjacency_matrix, interpolation="none", aspect=1, cmap="Greys")
    # major ticks
    ax.set_xticks(np.arange(0, adjacency_matrix.shape[1], dtype=np.int))
    ax.set_yticks(np.arange(0, adjacency_matrix.shape[0], dtype=np.int))
    # minor ticks
    ax.set_xticks(
        np.arange(-0.5, adjacency_matrix.shape[1], 1), minor=True)
    ax.set_yticks(
        np.arange(-0.5, adjacency_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth="1")

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


def extract_sub_samples(extraction_matrix, samples):
    # The following does not take account of sparsity
    # sub_samples = extraction_matrix.dot(samples)
    # so instead assume extraction matrix is a vector of indices
    sub_samples = samples[extraction_matrix, :]
    return sub_samples


def evaluate_component_functions(
        component_funs, exog_extraction_matrices,
        coup_extraction_matrices,
        exog_samples, coup_samples):
    """
    Notes:
    This could be parallelized if necessary
    """
    ncomponents = len(component_funs)
    values = []
    for ii in range(ncomponents):
        exog_samples_ii = extract_sub_samples(
            exog_extraction_matrices[ii], exog_samples)
        coup_samples_ii = extract_sub_samples(
            coup_extraction_matrices[ii], coup_samples)
        # assume component funs take all exog variables then all coup variables
        component_samples_ii = np.vstack((exog_samples_ii, coup_samples_ii))
        component_values = component_funs[ii](component_samples_ii)
        values.append(component_values)
    return np.hstack(values)


def equality_constrained_linear_least_squares(A, B, y, z):
    from scipy.linalg import lapack
    return lapack.dgglse(A, B, y, z)[3]


def gauss_jacobi_fixed_point_iteration(adjacency_matrix,
                                       exog_extraction_matrices,
                                       coup_extraction_matrices,
                                       component_funs,
                                       init_coup_samples, exog_samples,
                                       tol=1e-15,
                                       max_iters=100, verbose=0,
                                       # relax_factor=1,
                                       anderson_memory=0):

    if init_coup_samples.shape[1] != exog_samples.shape[1]:
        raise ValueError("Must provide initial guess for every sample")
    ncomponents = len(component_funs)
    if len(exog_extraction_matrices) != ncomponents:
        raise ValueError("Must provide extraction matrix for each component")

    it = 0
    coup_samples = init_coup_samples
    coup_history = np.empty(
        (coup_samples.shape[1], coup_samples.shape[0], anderson_memory+1))
    residuals_history = np.empty(
        (coup_samples.shape[1], coup_samples.shape[0], anderson_memory+1))
    while True:
        outputs = evaluate_component_functions(
            component_funs, exog_extraction_matrices, coup_extraction_matrices,
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
        if it+1 >= anderson_memory:
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
    return outputs


def get_local_coupling_variables_indices_in(component_random_variable_labels,
                                            component_coupling_labels,
                                            local_random_var_indices=None):
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


def evaluate_function(graph, node_id, global_samples):
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
    Object describing the connections between components of a system model.

    Parameters
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
        values : list
            Evaluation of each component in component_ids at the samples
            Each entry of the list is np.ndarray (nsamples, nlocal_qoi)
        """
        return evaluate_functions(self.graph, samples, component_ids)

    def ncomponents(self):
        return len(self.graph.nodes)

    def get_graph_attribute(self, name):
        vals = []
        for node in self.graph.nodes:
            vals.append(self.graph.nodes[node][name])
        return vals


def extract_node_data(node_id, data):
    node_data = dict()
    for key, item in data.items():
        node_data[key] = item[node_id]
    return node_data


def build_chain_graph(nmodels, data=dict()):
    """
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
