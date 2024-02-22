import numpy as np
import networkx as nx
import copy
import itertools

from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.variables.gaussian import (
    convert_conditional_probability_density_to_canonical_form, GaussianFactor,
    compute_gaussian_pdf_canonical_form_normalization,
    convert_gaussian_from_canonical_form
)


def get_var_ids_to_eliminate_from_node_query(
        network_node_var_ids, network_labels, query_labels,
        evidence_node_ids=None):
    r"""
    Get the ids of all variables in a network not associated with nodes being
    queried. A given node can consist of multiple variables.
    This function will always exclude from elimination any ids which are in
    evidence_ids.

    Parameters
    ----------
    network_node_var_ids: list
        List with entries containing  containing the integer ids
        np.ndarray (nnetwork_vars) of all variables of a node.

    network_labels: list (nnodes)
       A list of strings containing the names of each node in the network.
       ith network labels must be associated with the ith node in the network
       ordering matters.

    query_labels : list (nqueries)
        A list of strings containing the node labels which must remain
        after elimination. nqueries<=nnodes. The network labels not in this
        list will be identified for elimination from the scope of the network.
        This list must not contain nodes associated with data.

    evidence_node_ids : np.ndarray (ndata)
       The node ids of the data nodes in the network.

    Returns
    -------
    eliminate_var_ids : list
       A list of labels of variables to be eliminated. Labels associated with
       evidence (data) will not be eliminated. Variables associated with data
       will never be identified for elimination
    """
    assert len(network_node_var_ids) == len(network_labels)
    nvars_to_query = len(query_labels)
    query_ids = np.empty(nvars_to_query, dtype=int)
    for ii in range(nvars_to_query):
        found = False
        label = query_labels[ii]
        for jj in range(len(network_labels)):
            if label == network_labels[jj]:
                query_ids[ii] = jj
                found = True
                break
        if not found:
            msg = f'query label {label} was not found in network_labels '
            msg += f'{network_labels}'
            raise Exception(msg)

    if evidence_node_ids is not None:
        assert np.intersect1d(query_ids, evidence_node_ids).shape[0] == 0
        query_ids = np.concatenate((query_ids, evidence_node_ids))

    eliminate_node_ids = np.setdiff1d(
        np.arange(len(network_labels)), query_ids)

    if eliminate_node_ids.shape[0] > 0:
        eliminate_var_ids = np.concatenate(
            [network_node_var_ids[idx] for idx in eliminate_node_ids])
    else:
        eliminate_var_ids = np.empty(0, dtype=int)
    return eliminate_var_ids


def get_var_ids_to_eliminate(network_ids, network_labels, query_labels,
                             evidence_ids=None):
    r"""
    Get the ids of variables in a network which are not being queried.
    This function will always exclude from elimination any ids which are in
    evidence_ids.

    Parameters
    ----------
    network_ids : np.ndarray (nnodes)
        The integer ids of each nodes in the network.

    network_labels: list (nnodes)
       A list of strings containing the names of each node in the network

    query_labels : list (nqueries)
        A list of strings containing the variable labels which must remain
        after elimination. nqueries<=nnodes. The network labels not in this
        list will be identified for elimination from the scope of the network.
        This list must not contain nodes associated with data.

    evidence_ids : np.ndarray (ndata)
       The variable ids of the data in the network. Note the variable ids
       are not the same as the node ids. Each node can have multiple variables.


    Returns
    -------
    eliminate_ids : list
       A list of labels of variables to be eliminated. Labels associated with
       evidence (data) will not be eliminated. Variables associated with data
       will never be identified for elimination
    """
    assert len(network_ids) == len(network_labels)
    nvars_to_query = len(query_labels)
    query_ids = np.empty(nvars_to_query, dtype=int)
    for ii in range(nvars_to_query):
        found = False
        label = query_labels[ii]
        for jj in range(len(network_labels)):
            if label == network_labels[jj]:
                query_ids[ii] = network_ids[jj]
                found = True
                break
        if not found:
            msg = f'query label {label} was not found in network_labels '
            msg += f'{network_labels}'
            raise Exception(msg)

    if evidence_ids is not None:
        assert np.intersect1d(query_ids, evidence_ids).shape[0] == 0
        query_ids = np.concatenate((query_ids, evidence_ids))

    mask = np.isin(network_ids, query_ids)
    eliminate_ids = np.asarray(network_ids)[~mask]
    return eliminate_ids


def get_nparams_of_nodes(vands):
    num_nodes = len(vands)
    nparams = np.asarray([vands[ii].shape[1] for ii in range(num_nodes)])
    return nparams


def basis_matrix_cols(nvars, degree):
    from pyapprox.util.utilities import total_degree_space_dimension
    ncols = total_degree_space_dimension(nvars, degree)
    return ncols


def get_cpd_block_diagonal_linear_matrix(graph, node_index):
    r"""
    Get the mathrix :math:`A` from the conditional probability density

    .. math:: \mathbb{P}(\theta_i\mid \mathrm{pa}(theta_i))\sim\mathcal{A\theta+b,\Sigma_v}

    The graph must have edges with the attribute ``cpd_scale`` which is
    either a constant or a np.ndarray (nparams) or a
    np.ndarray(nparams,nchild_params)

    if ``cpd_scale`` is a constant or 1D array
    ``A=cpd_scale*np.eye(nparams,nchild_params)`` otherwise ``A=cpd_scale``

    Parameters
    ----------
    graph : nx.DiGraph
        A networkx directed acyclic graph

    node_index : integer
        The index of the node in the graph corresponding to the variable
        :math:`\theta_i`

    Returns
    -------
    Amat : np.ndarray (nparams,nchild_params)
        A matrix relating the
    """
    assert len(nx.get_edge_attributes(graph, 'cpd_scale')) > 0
    child_indices = list(graph.predecessors(node_index))
    if len(child_indices) == 0:
        return None

    Amat_blocks = []
    node_nparams = graph.nodes[node_index]["nparams"]
    for child in child_indices:
        cpd_scale = graph.edges[child, node_index]['cpd_scale']
        child_nparams = graph.nodes[child]["nparams"]
        if np.isscalar(cpd_scale) or cpd_scale.ndim == 1:
            Amat_blocks.append(
                cpd_scale*np.eye(node_nparams, child_nparams))
        else:
            assert cpd_scale.shape == (node_nparams, child_nparams)
            Amat_blocks.append(cpd_scale)

    Amat = np.hstack(Amat_blocks)
    return Amat


def get_cpd_linear_matrix(graph, node_index):
    child_indices = list(graph.predecessors(node_index))
    if len(child_indices) == 0:
        return None, None

    Amat_blocks = []
    node_nparams = graph.nodes[node_index]["nparams"]
    for child in child_indices:
        cpd_scale = graph.edges[child, node_index]['cpd_scale']
        child_nparams = graph.nodes[child]["nparams"]
        if np.isscalar(cpd_scale) or cpd_scale.ndim == 1:
            Amat_blocks.append(
                cpd_scale*np.eye(node_nparams, child_nparams))
        else:
            assert cpd_scale.shape == (node_nparams, child_nparams)
            Amat_blocks.append(cpd_scale)

    Amat = np.hstack(Amat_blocks)
    return Amat, child_indices


def get_cpd_prior_covariance(graph, node_index):
    prior_scale = graph.nodes[node_index]["prior_scale"]
    nparams = graph.nodes[node_index]["nparams"]
    if np.isscalar(prior_scale) or prior_scale.ndim == 1:
        prior_cov = np.eye(nparams)*prior_scale**2
    else:
        assert prior_scale.shape == (nparams, nparams)
        prior_cov = prior_scale
    return prior_cov


def get_gaussian_factor_in_canonical_form(Amat, bvec, cov2g1,
                                          var1_ids, nvars_per_var1,
                                          var2_ids, nvars_per_var2):
    r"""
    Todo consider massing inv(cov2g1) to function so can leverage structure
    in matrix and not to inversion inside convert_conditional function
    """
    if bvec is None:
        bvec = np.zeros(Amat.shape[0])
    precision_matrix, shift, normalization, var_ids, nvars_per_var = \
        convert_conditional_probability_density_to_canonical_form(
            Amat, bvec, cov2g1, var1_ids, nvars_per_var1,
            var2_ids, nvars_per_var2)
    return GaussianFactor(
        precision_matrix, shift, normalization, var_ids, nvars_per_var)


def build_hierarchical_polynomial_network(
        prior_covs, cpd_scales, basis_matrix_funcs,
        nparams, model_labels=None):
    r"""
    prior_scales : list
        List of diagonal matrices (represented by either a scalar for a
        constant diagonal or a vector)

    cpd_scales : list
        List of diagonal matrices (represented by either a scalar for a
        constant diagonal or a vector)

    """
    nmodels = len(nparams)
    assert len(basis_matrix_funcs) == nmodels
    assert len(prior_covs) == nmodels
    assert len(cpd_scales) == nmodels-1

    if model_labels is None:
        model_labels = ['M_%d' % ii for ii in range(nmodels)]

    graph = nx.DiGraph()
    ii = 0
    graph.add_node(
        ii, label=model_labels[ii], prior_scale=np.sqrt(prior_covs[ii]),
        nparams=nparams[ii],
        basis_matrix_func=basis_matrix_funcs[ii])
    # todo pass in list of nparams
    for ii in range(1, nmodels):
        prior_scale = np.sqrt(
            max(1e-8, prior_covs[ii] - cpd_scales[ii-1]**2*prior_covs[ii-1]))
        graph.add_node(
            ii, label=model_labels[ii], prior_scale=prior_scale,
            nparams=nparams[ii], basis_matrix_func=basis_matrix_funcs[ii])

    graph.add_edges_from(
        [(ii, ii+1, {'cpd_scale': cpd_scales[ii]}) for ii in range(nmodels-1)])

    network = GaussianNetwork(graph)
    return network


class GaussianNetwork(object):
    r"""
    A Bayesian network of linear Gaussian models.
    """

    # Note
    # Currently only entire (dataless) nodes can be marginalized.
    # self.evidence_ids is defined to be [k-1,k-1+ndata] where k is the number
    # of nodes (variable indexing starts at 0). k<= number of vars associated
    # with dataless nodes. For example if k=2 and each node has two
    # vars number of dataless variables is 4, yet self.evidence_ids starts at 2.
    def __init__(self, graph):
        """
        Constructor.

        Parameters
        ----------
        graph : py:class:`networkx.DiGraph`
            A directed acyclic graph
        """
        self.graph = copy.deepcopy(graph)
        self.construct_dataless_network()

    def construct_dataless_network(self):
        nnodes = len(self.graph.nodes)
        if len(self.graph.nodes) > 1:
            assert (np.max(self.graph.nodes) == nnodes-1 and
                    np.min(self.graph.nodes) == 0)

        self.bvecs = [None]*nnodes
        self.Amats, self.cpd_covs = [None]*nnodes, [None]*nnodes
        self.node_childs, self.node_nvars = [None]*nnodes, [None]*nnodes
        self.node_labels, self.node_ids = [
            None]*nnodes, list(np.arange(nnodes))
        self.node_var_ids = []
        self.nnetwork_vars = 0
        for ii in self.graph.nodes:
            # Extract node information from graph
            # nparams = self.graph.nodes[ii]['nparams']
            self.Amats[ii] = self.graph.nodes[ii]['cpd_mat']
            self.bvecs[ii] = self.graph.nodes[ii]['cpd_mean'].squeeze()
            self.node_childs[ii] = list(self.graph.predecessors(ii))
            self.cpd_covs[ii] = self.graph.nodes[ii]['cpd_cov']
            self.node_labels[ii] = self.graph.nodes[ii]['label']
            self.node_nvars[ii] = self.cpd_covs[ii].shape[0]
            self.node_var_ids += [list(
                range(self.nnetwork_vars,
                      self.nnetwork_vars+self.node_nvars[ii]))]
            self.nnetwork_vars += self.node_nvars[ii]

            # check the validity of the graph
            if self.node_childs[ii] is not None:
                for child in self.node_childs[ii]:
                    assert child < ii

    def num_vars(self):
        r"""
        Return number of uncertain variables in the network

        Returns
        -------
        nnetwork_vars : integer
            The number of uncertain variables in the network
        """
        return self.nnetwork_vars

    def convert_to_compact_factors(self):
        r"""
        Compute the factors of the network
        """
        self.factors = []
        for ii in self.graph.nodes:
            if len(self.node_childs[ii]) > 0:
                var_ids1 = np.concatenate(
                    [self.node_var_ids[jj] for jj in self.node_childs[ii]])
                nvars_per_var1 = [1 for jj in range(var_ids1.shape[0])]
                nvars_per_var2 = [1 for kk in range(self.node_nvars[ii])]
                cpd = get_gaussian_factor_in_canonical_form(
                    self.Amats[ii], self.bvecs[ii], self.cpd_covs[ii],
                    var_ids1, nvars_per_var1,
                    self.node_var_ids[ii], nvars_per_var2)
                self.factors.append(cpd)
            else:
                # Leaf nodes - no children (predecessors in networkx)
                # TODO: replace inverse by method that takes advantage of matrix
                # structure, e.g. diagonal, constant diagonal
                precision_matrix = np.linalg.inv(self.cpd_covs[ii])
                mean = self.bvecs[ii]
                shift = precision_matrix.dot(mean)
                normalization = \
                    compute_gaussian_pdf_canonical_form_normalization(
                        self.bvecs[ii], shift, precision_matrix)
                nvars_per_var = [1 for kk in range(self.node_nvars[ii])]
                self.factors.append(GaussianFactor(
                    precision_matrix, shift, normalization,
                    self.node_var_ids[ii], nvars_per_var))

    def add_data_to_network(self, data_cpd_mats, data_cpd_vecs,
                            noise_covariances):
        r"""
        Todo pass in argument containing nodes which have data for situations
        when not all nodes have data
        """
        nnodes = len(self.graph.nodes)
        assert len(data_cpd_mats) == nnodes
        assert len(noise_covariances) == nnodes
        # self.build_matrix_functions = build_matrix_functions
        self.ndata = [data_cpd_mats[ii].shape[0] for ii in range(nnodes)]

        # retain copy of old dataless graph
        dataless_graph = copy.deepcopy(self.graph)
        kk = len(self.graph.nodes)
        dataless_nodes_nvars = np.sum(self.node_nvars)
        jj = dataless_nodes_nvars
        for ii in dataless_graph.nodes:
            # Note: this assumes that every node has data. This can be relaxed
            assert data_cpd_mats[ii].shape[1] == \
                self.graph.nodes[ii]['nparams']
            assert data_cpd_vecs[ii].shape[0] == self.ndata[ii]
            assert noise_covariances[ii].shape[0] == self.ndata[ii]
            self.node_ids.append(kk)
            label = self.graph.nodes[ii]['label']+'_data'
            self.graph.add_node(kk, label=label)
            self.graph.add_edge(ii, kk)
            self.Amats.append(data_cpd_mats[ii])
            self.bvecs.append(data_cpd_vecs[ii].squeeze())
            self.node_childs.append([ii])
            self.cpd_covs.append(noise_covariances[ii])
            self.node_labels.append(label)
            self.node_nvars.append(self.ndata[ii])
            self.node_var_ids.append(np.arange(jj, jj+self.ndata[ii]))
            jj += self.ndata[ii]
            kk += 1
        self.evidence_var_ids = np.arange(dataless_nodes_nvars, jj, dtype=int)
        self.evidence_node_ids = np.arange(
            len(dataless_graph.nodes), kk, dtype=int)

    def assemble_evidence(self, data):
        r"""
        Assemble the evidence in the form needed to condition the network

        Returns
        -------
        evidence : np.ndarray (nevidence)
            The data used to condition the network

        evidence_var_ids : np.ndarray (nevidence)
            The variable ids containing each data

        Notes
        -----
        Relies on order vandermondes are added in network.add_data_to_network
        """
        assert len(data) == len(self.ndata)
        nevidence = np.sum([d.shape[0] for d in data])
        assert nevidence == len(self.evidence_var_ids)
        evidence = np.empty((nevidence))
        kk = 0
        for ii in range(len(data)):
            assert (data[ii].ndim == 1 or data[ii].shape[1] == 1), (
                ii, data[ii].shape)
            for jj in range(data[ii].shape[0]):
                evidence[kk] = data[ii][jj][0]
                kk += 1
        return evidence, self.evidence_var_ids


def sum_product_eliminate_variable(factors, var_id_to_eliminate):
    r"""
    Marginalize out a variable from a multivariate Gaussian defined by
    the product of the gaussian variables in factors.

    Algorithm 9.1 in Koller

    Parameters
    ----------
    factors : list (num_factors)
        List of gaussian variables in CanonicalForm

    var_id_to_eliminate : integer
        The variable to eliminate from each of the factors

    Returns
    -------
    fpp_tau : list
        List of gaussian variables in CanonicalForm. The first entries are
        all the factors that did not contain the variable_to_eliminate on
        entry to this function. The last entry is the multivariate gaussian
        which is based upon the product of all factors that did contain the
        elimination variable for which the elimination var is then marginalized
        out
    """

    # Get list of factors which contain the variable to eliminate
    fp, fpp = [], []
    for factor in factors:
        if var_id_to_eliminate in factor.var_ids:
            fp.append(factor)
        else:
            fpp.append(factor)

    if len(fp) == 0:
        return fpp

    # Of the factors which contain the variable to eliminate marginalize
    # out that variable from a multivariate Gaussian of all factors
    # containing that variable

    # construct multivariate Gaussian distribution in canonical form
    psi = copy.deepcopy(fp[0])
    for jj in range(1, len(fp)):
        psi *= fp[jj]

    # marginalize out all data associated with var_to_eliminate
    tau = copy.deepcopy(psi)
    tau.marginalize([var_id_to_eliminate])
    # tau = tau.reorder(ordering)

    # Combine the marginalized factors and the factors which did
    # not originally contain the variable to eliminate
    return fpp+[tau]


def sum_product_variable_elimination(factors, var_ids_to_eliminate):
    r"""
    Marginalize out a list of variables from the multivariate Gaussian variable
    which is the product of all factors.
    """
    # nvars_to_eliminate = len(var_ids_to_eliminate)

    fup = copy.deepcopy(factors)
    for var_id in var_ids_to_eliminate:
        fup = sum_product_eliminate_variable(fup, var_id)

    assert len(fup) > 0, "no factors left after elimination"
    assert len(fup[0].var_ids) != 0, "factor k = {0}".format(
        fup[0].precision_matrix)

    factor_ret = fup[0]
    for jj in range(1, len(fup)):
        factor_ret *= fup[jj]
    return factor_ret


def cond_prob_variable_elimination(network, query_labels, evidence_ids=None,
                                   evidence=None):
    r"""
    Marginalize out variables not in query labels.
    """
    eliminate_ids = get_var_ids_to_eliminate_from_node_query(
        network.node_var_ids, network.node_labels, query_labels, evidence_ids)

    factors = copy.deepcopy(network.factors)

    # Condition each node on available data
    if evidence is not None:
        for factor in factors:
            factor.condition(evidence_ids, evidence)

    # Marginalize out all unrequested variables
    factor_ret = sum_product_variable_elimination(factors, eliminate_ids)

    return factor_ret


def build_peer_polynomial_network(prior_covs, cpd_scales, basis_matrix_funcs,
                                  nparams, model_labels=None):
    r"""
    All list arguments must contain high-fidelity info in last entry
    """
    graph = nx.DiGraph()
    nnodes = len(prior_covs)
    if model_labels is None:
        model_labels = [f'M{ii}' for ii in range(nnodes)]
    assert len(model_labels) == nnodes
    cpd_mats = [None]+[
        cpd_scales[ii]*np.eye(nparams[ii+1], nparams[ii])
        for ii in range(nnodes-1)]
    prior_means = np.zeros(nnodes)

    for ii in range(nnodes-1):
        graph.add_node(
            ii, label=model_labels[ii],
            cpd_cov=prior_covs[ii]*np.eye(nparams[ii]),
            nparams=nparams[ii], cpd_mat=cpd_mats[ii],
            cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))

    ii = nnodes-1
    cov = np.eye(nparams[ii])*max(1e-8, prior_covs[ii]-np.dot(
        np.asarray(cpd_scales)**2, prior_covs[:ii]))
    graph.add_node(
        ii, label=model_labels[ii], cpd_cov=cov, nparams=nparams[ii],
        cpd_mat=cpd_mats[ii],
        cpd_mean=(prior_means[ii]-np.dot(cpd_scales[:ii], prior_means[:ii])) *
        np.ones((nparams[ii], 1)))

    graph.add_edges_from(
        [(ii, nnodes-1, {'cpd_cov': np.eye(nparams[ii])*cpd_scales[ii]})
         for ii in range(nnodes-1)])

    network = GaussianNetwork(graph)
    return network


def nonlinear_constraint_peer(covs, scales):
    r"""
    All list arguments must contain high-fidelity info in last entry
    """
    cpd_cov = [covs[-1]-np.dot(scales**2, covs[:-1])-1e-7]
    return cpd_cov  # must be > 0 to ensure cpd_cov is positive


def nonlinear_constraint_hierarchical(covs, scales):
    r"""
    All list arguments must contain model info ordered lowest-highest fidelity
    """
    cpd_cov = [None]*len(scales)
    for dim in range(len(scales)):
        cpd_cov[dim] = covs[dim+1] - scales[dim]**2 * covs[dim] - 1e-8
    return cpd_cov  # must be > 0 to ensure cpd_cov is positive


def infer(build_network, scales, samples_train, data_train, noise_std):

    network = build_network(scales)
    network.add_data_to_network(samples_train, noise_std**2)
    network.convert_to_compact_factors()
    evidence, evidence_ids = network.assemble_evidence(data_train)

    # high fidelity model is always last label of models. It will not
    # be last lable in graph. These will be data
    hf_label = network.graph.nodes[len(samples_train)-1]['label']

    factor_post = cond_prob_variable_elimination(
        network, [hf_label], evidence_ids=evidence_ids, evidence=evidence)
    gauss_post = convert_gaussian_from_canonical_form(
        factor_post.precision_matrix, factor_post.shift)

    factor_prior = cond_prob_variable_elimination(network, [hf_label], None)
    gauss_prior = convert_gaussian_from_canonical_form(
        factor_prior.precision_matrix, factor_prior.shift)

    return gauss_post, gauss_prior, network


def get_heterogeneous_data(ndata, noise_std):
    def f1(x): return np.cos(3*np.pi*x[0, :]+0.1*x[1, :])[:, np.newaxis]
    def f2(x): return np.exp(-(x-.5)**2/0.5).T
    def f3(x): return (f2(x).T+np.cos(3*np.pi*x)).T
    XX = [np.random.uniform(0, 1, (2, ndata[0]))]+[
        np.random.uniform(0, 1, (1, n)) for n in ndata[1:]]

    funcs = [f1, f2, f3]
    data = [f(xx) + e*np.random.normal(0, 1, (n, 1))
            for f, xx, n, e in zip(funcs, XX, ndata, noise_std)]
    samples = [x for x in XX]

    validation_samples = np.linspace(0, 1, 10001)
    validation_data = funcs[-1](validation_samples)

    ranges = [[0, 1, 0, 1], [0, 1], [0, 1]]

    for ii in range(3):
        assert data[ii].shape == (
            ndata[ii], 1), (ii, data[ii].shape, ndata[ii])
    return samples, data, validation_samples, validation_data, ranges


def get_total_degree_polynomials(univariate_variables, degrees):
    assert type(univariate_variables[0]) == list
    assert len(univariate_variables) == len(degrees)
    polys, nparams = [], []
    for ii in range(len(degrees)):
        poly = PolynomialChaosExpansion()
        var_trans = AffineTransform(
            univariate_variables[ii])
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        indices = compute_hyperbolic_indices(
            var_trans.num_vars(), degrees[ii], 1.0)
        poly.set_indices(indices)
        polys.append(poly)
        nparams.append(indices.shape[1])
    return polys, np.array(nparams)


def plot_1d_lvn_approx(xx, nmodels, hf_vandermonde, gauss_post, gauss_prior,
                       axs, samples, data, labels, ranges, hf_data_mean=None,
                       colors=None, mean_label=r'$\mathrm{MFNet}$'):
    if samples[-1].ndim != 1 and samples[-1].shape[0] > 1:
        print('Cannot plot num_vars>1')
        return
    xx = np.linspace(0, 1, 101)
    xx = (ranges[1]-ranges[0])*xx+ranges[0]
    basis_matrix = hf_vandermonde(xx[np.newaxis, :])
    approx_post_covariance = basis_matrix.dot(
        gauss_post[1].dot(basis_matrix.T))
    assert (np.diag(approx_post_covariance).min() >
            0), np.diag(approx_post_covariance).min()
    approx_prior_covariance = basis_matrix.dot(
        gauss_prior[1].dot(basis_matrix.T))
    # print (approx_covariance.shape,gauss_post.get_covariance().shape)

    approx_post_mean = np.dot(basis_matrix, gauss_post[0]).squeeze()
    approx_prior_mean = np.dot(basis_matrix, gauss_prior[0]).squeeze()
    if hf_data_mean is not None:
        approx_post_mean += hf_data_mean
        approx_prior_mean += hf_data_mean

    approx_post_std = np.sqrt(np.diag(approx_post_covariance))
    axs.plot(xx, approx_post_mean, '--g', label=mean_label)
    axs.fill_between(xx, approx_post_mean-2*approx_post_std,
                     approx_post_mean+2*approx_post_std, color='green',
                     alpha=0.5)

    approx_prior_std = np.sqrt(np.diag(approx_prior_covariance))
    axs.fill_between(xx, approx_prior_mean-2*approx_prior_std,
                     approx_prior_mean+2*approx_prior_std, color='black',
                     alpha=0.2)

    if labels is None:
        labels = [None]*len(samples)
    for ii in range(len(samples)):
        samples[ii] = np.atleast_2d(samples[ii])
        if colors is not None:
            axs.plot(samples[ii][0, :], data[ii], 'o', label=labels[ii],
                     c=colors[ii])
        else:
            axs.plot(samples[ii][0, :], data[ii], 'o', label=labels[ii])
            # axs.ylim([(approx_post_mean-2*approx_post_std).min(),
            #          (approx_post_mean+2*approx_post_std).max()])
            # axs.set_ylim([(-2*approx_prior_std).min(),(2*approx_prior_std).max()])
    axs.set_xlim([xx.min(), xx.max()])
    axs.legend()


def plot_peer_network(nmodels, ax):
    # Best use with axis of (8,3) or (8,6)
    options = {'node_size': 2000, 'width': 3, 'arrowsize': 20, 'ax': ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels-1)/64, (nmodels-1)/64, nmodels-1), [0]))
    coords += [[0, 1/16]]
    pos = dict(zip(np.arange(nmodels, dtype=int), coords))

    graph = nx.DiGraph()
    for ii in range(nmodels):
        graph.add_node(ii)

    for ii in range(nmodels-1):
        graph.add_edge(ii, nmodels-1)

    nx.draw(graph, pos, **options)
    labels_str = [r'$\theta_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels = dict(zip(np.arange(nmodels, dtype=int), labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)


def plot_diverging_network(nmodels, ax):
    # Best use with axis of (8,3) or (8,6)
    options = {'node_size': 2000, 'width': 3, 'arrowsize': 20, 'ax': ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels-1)/64, (nmodels-1)/64, nmodels-1), [0]))
    coords += [[0, -1/16]]
    pos = dict(zip(np.arange(nmodels, dtype=int), coords))

    graph = nx.DiGraph()
    for ii in range(nmodels):
        graph.add_node(ii)

    for ii in range(nmodels-1):
        graph.add_edge(nmodels-1, ii)

    nx.draw(graph, pos, **options)
    labels_str = [r'$\theta_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels = dict(zip(np.arange(nmodels, dtype=int), labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)


def plot_peer_network_with_data(graph, ax):
    nmodels = len(graph.nodes)//2
    # Best use with axis of (8,3) or (8,6)
    options = {'node_size': 2000, 'width': 3, 'arrowsize': 20, 'ax': ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels-1)/64, (nmodels-1)/64, nmodels-1), [0]))
    coords += [[0, 1/16]]
    coords += list(itertools.product(
        np.linspace(-(nmodels-1)/64, (nmodels-1)/64, nmodels-1), [-1/16]))
    coords += [[coords[nmodels//2][0], 1/16]]
    pos = dict(zip(np.arange(2*nmodels, dtype=int), coords))

    nx.draw(graph, pos, **options)
    labels_str = [r'$\theta_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels_str += [r'$y_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels = dict(zip(np.arange(2*nmodels, dtype=int), labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)


def plot_hierarchical_network(nmodels, ax):
    # Best use with axis of (8,3) or (8,6)
    options = {'node_size': 2000, 'width': 3, 'arrowsize': 20, 'ax': ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels)/64, (nmodels)/64, nmodels), [0]))
    pos = dict(zip(np.arange(nmodels, dtype=int), coords))

    graph = nx.DiGraph()
    for ii in range(nmodels):
        graph.add_node(ii)

    for ii in range(nmodels-1):
        graph.add_edge(ii, ii+1)

    nx.draw(graph, pos, **options)
    labels_str = [r'$\theta_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels = dict(zip(np.arange(nmodels, dtype=int), labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)


def plot_hierarchical_network_network_with_data(graph, ax):
    nmodels = len(graph.nodes)//2
    # Best use with axis of (8,3) or (8,6)
    options = {'node_size': 2000, 'width': 3, 'arrowsize': 20, 'ax': ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels)/64, (nmodels)/64, nmodels), [0]))
    coords += list(itertools.product(
        np.linspace(-(nmodels)/64, (nmodels)/64, nmodels), [-1/16]))
    pos = dict(zip(np.arange(2*nmodels, dtype=int), coords))

    nx.draw(graph, pos, **options)
    labels_str = [r'$\theta_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels_str += [r'$y_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels = dict(zip(np.arange(2*nmodels, dtype=int), labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)


def set_polynomial_ensemble_coef_from_flattened(polys, coefs):
    idx1, idx2 = 0, 0
    for ii in range(len(polys)):
        idx2 += polys[ii].num_terms()
        polys[ii].set_coefficients(coefs[idx1:idx2])
        idx1 = idx2
    return polys


def plot_3_node_fully_connected_network(ax, labels_str=None):
    nmodels = 3
    options = {'node_size': 2000, 'width': 3, 'arrowsize': 20, 'ax': ax}
    coords = list(itertools.product(
        np.linspace(-(nmodels-1)/64, (nmodels-1)/64, nmodels-1), [0]))
    coords += [[0, 1/16]]
    pos = dict(zip(np.arange(nmodels, dtype=int), coords))

    graph = nx.DiGraph()
    for ii in range(nmodels):
        graph.add_node(ii)

    for ii in range(nmodels-1):
        graph.add_edge(ii, nmodels-1)

    graph.add_edge(0, 1)

    nx.draw(graph, pos, **options)
    if labels_str is None:
        labels_str = [r'$\theta_{%d}$' % (ii+1) for ii in range(nmodels)]
    labels = dict(zip(np.arange(nmodels, dtype=int), labels_str))
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, ax=ax)
