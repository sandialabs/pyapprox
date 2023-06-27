import numpy as np
import networkx as nx
from functools import partial
from scipy.optimize import minimize

try:
    import queue.SimpleQueue as SimpleQueue
except ImportError:
    from queue import Queue as SimpleQueue


def get_graph_attribute_vector(graph, attribute):
    nodes = graph.nodes
    if attribute not in nodes[1]:
        raise ValueError(f"The attribute ({attribute}) has not been set")
    node_attr_dict = nx.get_node_attributes(graph, attribute)
    vec = np.concatenate([node_attr_dict[n] for n in nodes])
    return vec


def get_graph_params(graph):
    """Extract the multifidelity surrogate parameters from the graph

    Parameters
    ----------
    graph : networkx.graph
        The graphical representation of the MF network

    Returns
    -------
        vec : np.ndarray (nparams)
        A flattened array containing all the parameters of the MF network
    """
    return get_graph_attribute_vector(graph, "params")


def set_graph_attribute(graph, attribute, vec, use_params_shape=False):
    nodes = graph.nodes
    ind = 0
    for node_id in nodes:
        if use_params_shape:
            offset = graph.nodes[node_id]["params"].shape[0]
        else:
            offset = graph.nodes[node_id][attribute].shape[0]
        graph.nodes[node_id][attribute] = vec[ind:ind + offset]
        ind = ind + offset


def set_graph_parameters(vec, graph):
    """
    Update the parameters of a multifidelity surrogate

    Parameters
    ----------
    vec : np.ndarray (nparams)
        A flattened array containing all the parameters of the MF network

    graph : networkx.graph
        The graphical representation of the MF network

    Returns
    -------
    graph : networkx.graph
        The updated graphical representation of the MF network with the
        parameter values given by ``vec``.
    """
    set_graph_attribute(graph, "params", vec)


def monomial_1d(params, input_samples, return_grad):
    """Linear Model with Monomial basis functions

    p[0]+sum(x**p[1:])

    Parameters
    ----------
    params : np.ndarray (nparams)
       The parameters of the model

    input_samples : np.ndarray (nsamples,nparams)
       The independent variables of the model

    Returns
    -------
    vals : np.ndarray (nsamples)
        Evaluation of the linear model

    grad : np.ndarray (nsamples,nparams)
      gradient of the linear model with respect to the model parameters
    """
    basis = input_samples.T**np.arange(params.shape[0])[None, :]
    vals = basis.dot(params)
    if not return_grad:
        return vals

    grad = basis
    return vals, grad


def monomial_nd(indices, params, input_samples, return_grad):
    """Linear Model with Monomial basis functions

    p[0]+sum(x**p[1:])

    Parameters
    ----------
    params : np.ndarray (nparams)
       The parameters of the model

    input_samples : np.ndarray (nsamples,nparams)
       The independent variables of the model

    Returns
    -------
    vals : np.ndarray (nsamples)
        Evaluation of the linear model

    grad : np.ndarray (nsamples,nparams)
      gradient of the linear model with respect to the model parameters
    """
    from pyapprox.surrogates.interp.monomial import monomial_basis_matrix
    if indices.shape[1] != params.shape[0]:
        raise ValueError("indices and params are inconsistent")
    basis = monomial_basis_matrix(indices, input_samples)
    vals = basis.dot(params)
    if not return_grad:
        return vals

    grad = basis
    return vals, grad


def least_squares_objective(obs_vals, pred_vals, noise_std=1,
                            return_grad=True):
    r"""Evaluate the least squares objective function

    Parameters
    ----------
    obs_vals : np.ndarray (nobs, 1)
        The observations

    predicted : np.ndarray (nobs, 1)
        The model predictions of the observations

    noise_std : float
        The standard deviation of the I.I.D noise

    return_grad : boolean
        False - only return objective value
        True - return objective and gradient

    Returns
    -------
    obj : float
        The value of the least squares objective function

    grad : np.ndarray (nobs)
        The gradient of ``obj`` with respect to pred_vals
    """
    residuals = obs_vals-pred_vals
    obj_val = residuals.T.dot(residuals)/(2*noise_std**2)
    if not return_grad:
        return obj_val

    grad = -residuals/(noise_std**2)
    return obj_val, grad


class MFNets(object):
    """
    Terminology
    -----------
    For graph 1 -> 2 -> 3
    root of graph is 1 (most upstream node)
    leaf of graph is 3 (most downstream node)
    predecessor (ancestor/parent) of 3 is 2
    sucessor (decendent/child) of 2 is 3
    """
    def __init__(self, graph):
        self.graph = graph
        params = get_graph_params(self.graph)
        if params.ndim != 2 or params.shape[1] != 1:
            raise ValueError(
                "parameters must be 2D np.ndarray with one column")
        self.nparams = params.shape[0]

    def get_nparams(self):
        return self.nparams

    def get_nnodes(self):
        return len(self.graph.nodes)

    def set_parameters(self, params):
        if params.ndim != 2 or params.shape[1] != 1:
            raise ValueError(
                "parameters must be 2D np.ndarray with one column")
        set_graph_parameters(params, self.graph)

    def forward_pass(self, input_samples, node_id, update_grad=False):
        """Evaluate the surrogate output at node_id by considering the
        subgraph of all ancestors of this node

        Parameters
        ----------
        input_samples : np.ndarray (nsamples, nparams)
            The independent variables of the model

        node_id : integer
            The id of the node under consideration

        Returns
        -------
        This function adds the following attributes to the underlying graph

        eval : np.ndarray (nsamples, nqoi)
            Stores the values of the function represented by the
            particular node / edge the evaluations at the nodes are
            cumulative (summing up all ancestors) whereas the edges are local

        Noes
        -------
        To avoid recomputation during recusion __forward_pass stores values
        locally. This wrapper calls self.clear_data() to make sure that
        these stored values are not used for new input_samples.
        """
        self.clear_data()
        return self.__forward_pass(input_samples, node_id, update_grad)

    def __forward_pass(self, input_samples, node_id, update_grad=False):
        if input_samples.ndim != 2:
            raise ValueError("input samples must be a 2d np.ndarray")
        node = self.graph.nodes[node_id]
        children_ids = list(self.graph.predecessors(node_id))

        if "values" in node and node["values"] is not None:
            if update_grad is False:
                return node["values"]

        elif len(children_ids) == 0:
            # WARNING: this could be potentially problematic
            # Currrently root node funs take update_grad as True or False
            # But all other nodes take False, "params" or "inputs"
            # for root nodes inputs is never an option. So we could
            # make true equivalent to params or just make all functions
            # take params as a valid value
            result = node["fun"](
                node["params"], input_samples, update_grad)
            if update_grad is False:
                node["values"] = result
            else:
                node["values"], node["params_grad"] = result
            return node["values"]

        if update_grad is True:
            update_params = "params"
        else:
            update_params = False

        child_id = children_ids[0]
        children_values = self.__forward_pass(
            input_samples, child_id, update_grad)
        children_nqoi = [children_values.shape[1]]
        for child_id in children_ids[1:]:
            child_values = self.__forward_pass(
                input_samples, child_id, update_grad)
            children_nqoi.append(child_values.shape[1])
            children_values = np.hstack(
                [children_values, child_values])

        node_input_samples = np.vstack((input_samples, children_values.T))
        result = node["fun"](
            node["params"], node_input_samples, update_params)
        if not update_grad:
            node["values"] = result
        else:
            node["values"], node["params_grad"] = result
            node["inputs_grad"] = node["fun"](
                node["params"], node_input_samples, "inputs")[1]
        return node["values"]

    def backward_pass(self, node_id, obj_grad):
        """
        Parameters
        ----------
        node_id : integer
            The id of the node under consideration

        Returns
        -------
        gradient : np.ndarray(nparams)
            A vector containing the gradient with respect to all parameters
            of the network
        """
        ancestors = nx.ancestors(self.graph, node_id)
        node_and_ancestors = ancestors.union(set([node_id]))

        # store gradient of objective with respect to predicted
        # output of network
        node = self.graph.nodes[node_id]
        node["pass_down"] = obj_grad
        # gradient with respect to node parameters
        node["grad"] = (obj_grad.T.dot(node["params_grad"])).T

        # print(node["grad"], 'node_grad')

        queue = SimpleQueue()
        queue.put(node_id)

        for ancestor_node_id in ancestors:
            self.graph.nodes[ancestor_node_id]["children_remain"] = set(
                self.graph.successors(ancestor_node_id)).intersection(
                    node_and_ancestors)
            self.graph.nodes[ancestor_node_id]["pass_down"] = 0.0
            self.graph.nodes[ancestor_node_id]["grad"] = 0.0
        # print([self.graph.nodes[n] for n in self.graph.nodes])
        while not queue.empty():
            current_node_id = queue.get()
            current_node = self.graph.nodes[current_node_id]
            current_node_pass_down = (
                self.graph.nodes[current_node_id]["pass_down"])

            # print('###', current_node_id)
            # print(current_node_pass_down, 'passdown')
            for kk, parent_node_id in enumerate(
                    self.graph.predecessors(current_node_id)):
                # print("@@@", parent_node_id)
                parent_node = self.graph.nodes[parent_node_id]
                # grad with respect to inputs which are outputs of this parent
                # node
                parent_node["pass_down"] = current_node_pass_down.T.dot(
                    current_node["inputs_grad"][kk]).T
                # print(parent_node)
                # print(current_node["inputs_grad"], parent_node["params_grad"])

                # grad with respect to network parameters of this parent node
                # WARNING: There is potentially a problem with ordering
                # of input_grad  and parent_node_ids returned from
                # predecessors function
                parent_node["grad"] = current_node_pass_down.T.dot(
                    current_node["inputs_grad"][kk].dot(
                        parent_node["params_grad"])).T
                # print(current_node["params_grad"], 'pre_grad')
                # print(parent_node["grad"], 'parentgrad')

                children_remain = parent_node["children_remain"]
                children_remain.remove(current_node_id)
                if len(children_remain) == 0:
                    queue.put(parent_node_id)

        return get_graph_attribute_vector(self.graph, "grad")

    def clear_data(self):
        attribute_list = [
            "values", "params_grad", "inputs_grad", "children_remain"]
        for node_id in self.graph.nodes:
            node_attr = self.graph.nodes[node_id]
            for attribute in attribute_list:
                node_attr[attribute] = None

    def zero_derivatives(self):
        set_graph_attribute(
            self.graph, "grad", np.zeros((self.get_nparams(), 1)), True)

    def fit_objective(self, params, return_grad=True):
        """
        params must be 1d np.ndarray for scipy optimizer to work
        """
        if params.ndim == 1:
            params = params[:, None]
        result = mfnets_graph_objective(
            self, self.node_id_list, self.train_samples_list,
            self.train_values_list,  self.noise_std_list, self.obj_fun,
            params, return_grad)
        # scipy requies 1D np.ndarray for gradient
        return result[0], result[1][:, 0]

    def fit(self, train_samples_list, train_values_list, noise_std_list,
            node_id_list, init_params, obj_fun=least_squares_objective,
            opts={}):
        self.node_id_list = node_id_list
        self.train_samples_list = train_samples_list
        self.train_values_list = train_values_list
        self.noise_std_list = noise_std_list
        self.obj_fun = obj_fun

        method = opts.get("method", 'BFGS')
        copy_opts = opts.copy()
        if "method" in copy_opts:
            del copy_opts["method"]

        res = minimize(
            self.fit_objective, init_params.squeeze(), method=method, jac=True,
            options=copy_opts)
        self.set_parameters(res.x[:, None])

    def __call__(self, input_samples, node_id=None):
        if node_id is None:
            node_id = self.get_nnodes()
        return self.forward_pass(input_samples, node_id, False)


def mfnets_node_objective(
        mfnets, node_id, train_samples, train_values, obj_fun, noise_std,
        params, return_grad=True):
    r"""
    Evaluate the learning objective function using data from one node

    Parameters
    ----------
    mfnets : MFNets
        A MFNets surrogate

    node_id : integer
        The id of the node

    train_samples : np.ndarray(ninputs, ntrain_samples)
        The training samples used for the current node

    train_values : np.ndarray(ninputs, ntrain_samples)
        The training values used for the current node

    obj_fun : callable
        Function that computes the fit of the surrogate to the training data
        with signature

        ``obj_fun(train_values, pred_values, noise_std, return_grad) ->
            (float, np.ndarray(ntrain_samples, 1)``

    noise_std : float
        The standard deviation of the noise in the training data

    params : np.ndarray (nparams)
        The parameter values at which to compute the objective value and
        gradient

    return_grad : boolean
        False - only return objective value
        True - return objective and gradient

    Returns
    -------
    obj_val : float
        The value of the mfnets objective function for the current node

    grad : np.ndarray (nparams, 1)
        The gradient of the mfnets objective function for the current node
        with resect to all the parameters of the graph
    """
    mfnets.zero_derivatives()
    mfnets.set_parameters(params)
    pred_values = mfnets.forward_pass(train_samples, node_id, return_grad)
    obj_result = obj_fun(train_values, pred_values, noise_std, return_grad)

    if not return_grad:
        obj_val = obj_result
        return obj_val

    obj_val, obj_grad = obj_result
    grad = mfnets.backward_pass(node_id, obj_grad)
    return obj_val, grad


def mfnets_graph_objective(
        mfnets, node_id_list, train_samples_list, train_values_list,
        noise_std_list, obj_fun, params, return_grad):
    r"""
    Evaluate the learning objective function using data from a set of nodes

    Parameters
    ----------
    mfnets : MFNets
        A MFNets surrogate

    node_id_list : list(integer)
        The ids of all nodes for which data is available

    train_samples_list : list(np.ndarray(ninputs, ntrain_samples_ii))
        List of the training samples for each node

    train_values_list : list(np.ndarray(ninputs, ntrain_samples_ii))
        List of the training values for each node

    obj_fun : callable
        Function that computes the fit of the surrogate to the training data
        with signature

        ``obj_fun(train_values, pred_values, noise_std, return_grad) ->
            (float, np.ndarray(ntrain_samples, 1)``

    noise_std_list : list(float)
        The standard deviation of the noise in the training data of each node

    params : np.ndarray (nparams)
        The parameter values at which to compute the objective value and
        gradient

    return_grad : boolean
        False - only return objective value
        True - return objective and gradient

    Returns
    -------
    obj_val : float
        The value of the mfnets objective function for the current node

    grad : np.ndarray (nparams, 1)
        The gradient of the mfnets objective function for the current node
        with resect to all the parameters of the graph
    """
    grad = np.zeros_like(params)
    obj_val = 0.0
    for node_id, train_samples, train_values, noise_std in zip(
            node_id_list, train_samples_list, train_values_list,
            noise_std_list):
        mfnets.clear_data()
        node_result = mfnets_node_objective(
            mfnets, node_id, train_samples, train_values, obj_fun, noise_std,
            params, return_grad)
        if not return_grad:
            obj_val += node_result
        else:
            obj_val += node_result[0]
            grad += node_result[1]
    return obj_val, grad

# -------------------------------------------------------------------------- #
# Functions for specific MFNets, e.g. different objective functions
# different node functions
# -------------------------------------------------------------------------- #

def populate_functions_multiplicative_additive_graph(
        graph, discrepancy_fun, scaling_fun, ndiscrepancy_params,
        nscaling_params, ninputs):
    """
    Parameters
    ----------
    """
    # initialize empty graph
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        nparents = len(list(graph.predecessors(node_id)))
        if nparents == 0:
            node["fun"] = discrepancy_fun
            node["params"] = np.zeros((ndiscrepancy_params, 1))
        else:
            node["fun"] = partial(
                multiplicative_additive_discrepancy_fun, discrepancy_fun,
                [scaling_fun]*nparents, ndiscrepancy_params,
                [nscaling_params]*nparents, ninputs)
            node["params"] = np.zeros(
                (ndiscrepancy_params+nscaling_params*nparents, 1))


def multiplicative_additive_discrepancy_fun(
        discrepancy_fun, scaling_funs, ndiscrepancy_params, nscaling_params,
        ndiscrepancy_inputs, params, input_samples, return_jac):
    """
    Parameters
    ----------
    scaling_funs : list(callable)
        The scaling function applied to each upstream function
        nparents = len(scaling_funs)

    ndiscrepancy_params : integer
        The number of parameters of the discrepancy function

    nscaling_params : list(integer)
        The number of parameters of the scaling functions

    ndiscrepancy_inputs : integer
        The number of input variables to the discrepancy function

    params : np.ndarray (nparams, 1)
        The concatenation of all discrepancy and scaling function parameters.
        Given S parents params concatenates as follows
        [discrepancy_params, scaling_fun_1_params, ... , scaling_fun_S_params]

    input_samples : np.ndarray(ndiscrepancy_inputs+nparents, ninput_samples)
        The input samples to the multiplicative additive function.
        ``input_samples = np.vstack(
            (discrepancy_samples, parent_function_values.T))``

    return_jac : boolean or string
        False - do not retun Jacobian
        "params" - return jacobian with respec to parameters
        "inputs" - return jacobian with respec to inputs

    Returns
    -------
    vals : np.ndarray(ninputs_samples, 1)
        The value of the function

    The jac type depends on the argument return_jac

    If return_jac == False nothing only vals is returned

    If return_jac == "params"

    jac : np.ndarray (ninput_samples, nparams)
        The jacobian of function with resect to all the parameters of the
        function

    If return_jac == "inputs"

    jac : list(np.ndarray (ninput_samples, ninput_samples))
        The list of jacobian of the function with resect to each input of the
        function

    Notes
    -----
    The number of quantities of interest (QoI) of each scaling fun
    is assumed to be the same as the number of outputs of the discrepancy fun.

    Assumes the number of inputs to the discrepancy and scaling funs are all
    the same.
    """
    nparents = len(scaling_funs)
    if (input_samples.shape[0]-ndiscrepancy_inputs) % nparents != 0:
        raise ValueError("scaling functions must have the same number of QoI")
    nscaling_qoi = int((input_samples.shape[0]-ndiscrepancy_inputs)/nparents)
    # at the moment I think the above only works for one qoi
    assert nscaling_qoi == 1, (
        nscaling_qoi, input_samples.shape, nparents, ndiscrepancy_inputs)
    if nscaling_qoi == 0:
        msg = "inputs_samples shape, scaling funs and ndiscrepancy_inputs are "
        msg += "inconsistent"
        print(input_samples.shape, ndiscrepancy_inputs, nparents)
        raise ValueError(msg)
    discrepancy_params = params[:ndiscrepancy_params]
    discrepancy_input_samples = input_samples[:ndiscrepancy_inputs, :]
    return_params_jac = (return_jac == "params")
    result = discrepancy_fun(
        discrepancy_params, discrepancy_input_samples, return_params_jac)
    if return_jac is False:
        vals = result
    elif return_params_jac is True:
        jac = np.empty(
            (result[1].shape[0], ndiscrepancy_params+np.sum(nscaling_params)))
        vals, jac[:, :ndiscrepancy_params] = result
    else:
        vals = result
        jac = []

    idx1, jdx1 = ndiscrepancy_params, ndiscrepancy_inputs
    for ii in range(nparents):
        idx2, jdx2 = idx1 + nscaling_params[ii], jdx1 + nscaling_qoi
        scaling_input_ii = input_samples[jdx1:jdx2, :].T
        result = scaling_funs[ii](
            params[idx1:idx2], discrepancy_input_samples, return_params_jac)
        if return_jac is False:
            vals += result * scaling_input_ii
        elif return_params_jac is True:
            vals += result[0] * scaling_input_ii
            jac[:, idx1:idx2] = result[1] * scaling_input_ii
        else:
            vals += result[0] * scaling_input_ii
            jac.append(np.diag(result[:, 0]))
            # below is faster than creating diagonal matrix
            # jac.flat[::result.shape[0]+1] = result[:, 0]
        idx1, jdx1 = idx2, jdx2
    if return_jac is False:
        return vals
    elif return_params_jac is True:
        return vals, jac
    else:
        return vals, jac
