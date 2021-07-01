#!/usr/bin/env python
import numpy as np
import networkx as nx


def get_adjacency_matrices(system_labels, component_labels):
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
                len(component_random_variable_labels[ii])+
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
            nlocal_vars = len(node_local_random_var_indices)+\
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
        
    nlocal_vars = len(node_local_random_var_indices)+\
        len(node_local_coupling_var_indices_in) + \
        len(node_local_config_var_indices)
    local_samples = np.empty((nlocal_vars, global_samples.shape[1]))

    # populate random samples
    local_samples[node_local_random_var_indices, :] = local_random_samples

    # populate coupling samples
    node_global_coupling_component_indices = \
        node['global_coupling_component_indices']
    strides = np.hstack([[0], np.cumsum(children_nqoi)[:-1]])
    selected_indices = [] # indices into children_values
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
                    (len(node['local_coupling_var_indices_in'])==
                    len(node['global_coupling_component_indices'])==0))

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
                len(node['local_random_var_indices'])+
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
    


